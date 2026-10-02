/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Custom solver for wingMotion2D co-simulation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Description
    See multiInterpolatingSolidBodyMotionSolver.H

\*---------------------------------------------------------------------------*/

#include "multiInterpolatingSolidBodyMotionSolver.H"
#include "addToRunTimeSelectionTable.H"
#include "cellZoneMesh.H"
#include "pointPatchDist.H"
#include "pointConstraints.H"
#include "mathematicalConstants.H"
#include "syncTools.H"
#include "fvCFD.H"
#include "volPointInterpolation.H"
#include "zeroGradientFvPatchFields.H"
#include "fixedValueFvPatchFields.H"
#include "patchWave.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(multiInterpolatingSolidBodyMotionSolver, 0);

    addToRunTimeSelectionTable
    (
        motionSolver,
        multiInterpolatingSolidBodyMotionSolver,
        dictionary
    );
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::multiInterpolatingSolidBodyMotionSolver::
multiInterpolatingSolidBodyMotionSolver
(
    const polyMesh& mesh,
    const dictionary& dict
)
:
    points0MotionSolver(mesh, dict, typeName),
    di_(readScalar(coeffDict().lookup("innerDistance"))),
    do_(readScalar(coeffDict().lookup("outerDistance"))),
    enableBgLaplacian_
    (
        coeffDict().lookupOrDefault<bool>("enableBgLaplacian", false)
    )
{
    // ------------------------------------------------------------------
    // Build per-zone point lists, read zone CofG + patch names,
    // then compute per-zone SLERP scale fields.
    // ------------------------------------------------------------------

    const label nEntries = coeffDict().size();
    zoneIDs_.setSize(nEntries);
    SBMFs_.setSize(nEntries);
    zoneCofGs_.setSize(nEntries);
    zonePatchIDs_.setSize(nEntries, -1);
    scales_.setSize(nEntries);
    zoneInnerDist_.setSize(nEntries, di_);
    zoneOuterDist_.setSize(nEntries, do_);
    rigidOnly_.setSize(nEntries, false);
    rigidPointIDs_.setSize(nEntries);
    label zonei = 0;

    const pointMesh& pMesh = pointMesh::New(mesh);

    forAllConstIter(dictionary, coeffDict(), iter)
    {
        if (!iter().isDict())
            continue;

        const word& zoneName = iter().keyword();

        // Skip non-zone scalar/word entries
        if
        (
            zoneName == "diffusivity"
         || zoneName == "patches"
         || zoneName == "CofG"
         || zoneName == "innerDistance"
         || zoneName == "outerDistance"
        )
        {
            continue;
        }

        const dictionary& subDict = iter().dict();

        // ---- Cell zone ID ----
        zoneIDs_[zonei] = mesh.cellZones().findZoneID(zoneName);
        if (zoneIDs_[zonei] == -1)
        {
            FatalIOErrorInFunction(coeffDict())
                << "Cannot find cellZone named " << zoneName
                << ". Valid zones: " << mesh.cellZones().names()
                << exit(FatalIOError);
        }

        // ---- Motion function ----
        SBMFs_.set
        (
            zonei,
            solidBodyMotionFunction::New(subDict, mesh.time())
        );

        // ---- CofG (read from SBMF coeffs sub-dict) ----
        const word sbmfType =
            subDict.lookup("solidBodyMotionFunction");
        const word coeffsName = sbmfType + "Coeffs";
        if (subDict.found(coeffsName))
        {
            zoneCofGs_[zonei] =
                vector(subDict.subDict(coeffsName).lookup("CofG"));
        }
        else
        {
            zoneCofGs_[zonei] =
                subDict.found("CofG")
              ? vector(subDict.lookup("CofG"))
              : vector::zero;
        }

        // ---- Surface patch for this zone ----
        if (subDict.found("patch"))
        {
            const word patchName(subDict.lookup("patch"));
            const label patchi =
                mesh.boundaryMesh().findPatchID(patchName);
            if (patchi == -1)
            {
                FatalIOErrorInFunction(subDict)
                    << "Cannot find patch named " << patchName
                    << exit(FatalIOError);
            }
            zonePatchIDs_[zonei] = patchi;
        }
        else
        {
            FatalIOErrorInFunction(subDict)
                << "Zone " << zoneName
                << " missing required 'patch' entry"
                << exit(FatalIOError);
        }

        // ---- Per-zone morphing mode ----
        rigidOnly_[zonei] = subDict.lookupOrDefault<bool>("rigidOnly", false);

        // ---- Per-zone SLERP distances (override global values if specified) ----
        // (ignored when rigidOnly=true)
        zoneInnerDist_[zonei] =
            subDict.lookupOrDefault<scalar>("innerDistance", di_);
        zoneOuterDist_[zonei] =
            subDict.lookupOrDefault<scalar>("outerDistance", do_);

        // ---- Scale field for this zone ----
        {
            scales_.set
            (
                zonei,
                new pointScalarField
                (
                    IOobject
                    (
                        "motionScale_" + zoneName,
                        mesh.time().timeName(),
                        mesh,
                        IOobject::NO_READ,
                        IOobject::NO_WRITE,
                        false
                    ),
                    pMesh,
                    dimensionedScalar(dimless, 0)
                )
            );

            if (rigidOnly_[zonei])
            {
                // Collect zone cell-points using MPI-safe syncTools pattern
                // (mirrors multiSolidBodyMotionSolver exactly).
                const cellZone& cz = mesh.cellZones()[zoneIDs_[zonei]];

                boolList movePts(mesh.nPoints(), false);
                forAll(cz, i)
                {
                    label celli = cz[i];
                    const cell& c = mesh.cells()[celli];
                    forAll(c, j)
                    {
                        const face& f = mesh.faces()[c[j]];
                        forAll(f, k)
                        {
                            movePts[f[k]] = true;
                        }
                    }
                }
                syncTools::syncPointList(mesh, movePts, orEqOp<bool>(), false);

                DynamicList<label> ptIDs(mesh.nPoints());
                forAll(movePts, i)
                {
                    if (movePts[i])
                    {
                        ptIDs.append(i);
                        scales_[zonei][i] = scalar(1);
                    }
                }
                rigidPointIDs_[zonei].transfer(ptIDs);
            }
            else
            {
                // SLERP blend: distance from zone patch → cosine scale in [0,1].
                // Scale=1 within innerDistance (full rigid motion),
                // tapers to 0 at outerDistance (stationary far field).

                labelHashSet patchSet;
                patchSet.insert(zonePatchIDs_[zonei]);

                // Optional wake stretch: divide effective distance by wakeStretch
                // for points at x > wakeStartX, extending the blend further
                // behind the trailing edge without affecting the LE side.
                const scalar wakeStretch =
                    subDict.lookupOrDefault<scalar>("wakeStretch", scalar(1));
                const scalar wakeStartX =
                    subDict.lookupOrDefault<scalar>("wakeStartX", scalar(vGreat));

                // Distance field mode:
                //   euclidean (default): pointPatchDist from surface patch
                //   chebyshev: max(|x-cx|/sx, |y-cy|/sy)
                //   bbox: distance from bounding box = sqrt(dx²+dy²)
                const word distMode =
                    subDict.lookupOrDefault<word>("distMode", word("euclidean"));

                scalarField effectiveDist(mesh.nPoints(), scalar(0));
                if (distMode == "bbox")
                {
                    const scalar cx =
                        subDict.lookupOrDefault<scalar>("distCentreX", scalar(0));
                    const scalar cy =
                        subDict.lookupOrDefault<scalar>("distCentreY", scalar(0));
                    const scalar halfX =
                        subDict.lookupOrDefault<scalar>("distHalfX", scalar(0.5));
                    const scalar halfY =
                        subDict.lookupOrDefault<scalar>("distHalfY", scalar(0.05));
                    const pointField& pts0 = points0();
                    forAll(effectiveDist, pointi)
                    {
                        const scalar dx =
                            max(scalar(0), Foam::mag(pts0[pointi].x() - cx) - halfX);
                        const scalar dy =
                            max(scalar(0), Foam::mag(pts0[pointi].y() - cy) - halfY);
                        effectiveDist[pointi] = Foam::sqrt(dx*dx + dy*dy);
                    }
                }
                else if (distMode == "chebyshev")
                {
                    const scalar cx =
                        subDict.lookupOrDefault<scalar>("distCentreX", scalar(0.5));
                    const scalar cy =
                        subDict.lookupOrDefault<scalar>("distCentreY", scalar(0));
                    const scalar sx =
                        subDict.lookupOrDefault<scalar>("distScaleX",  scalar(1));
                    const scalar sy =
                        subDict.lookupOrDefault<scalar>("distScaleY",  scalar(1));
                    const pointField& pts0 = points0();
                    forAll(effectiveDist, pointi)
                    {
                        effectiveDist[pointi] = max(
                            Foam::mag(pts0[pointi].x() - cx) / sx,
                            Foam::mag(pts0[pointi].y() - cy) / sy
                        );
                    }
                }
                else
                {
                    pointPatchDist pDist(pMesh, patchSet, points0());
                    effectiveDist = pDist.primitiveField();
                }

                if (wakeStretch > 1 + small)
                {
                    const pointField& pts0 = points0();
                    forAll(effectiveDist, pointi)
                    {
                        if (pts0[pointi].x() > wakeStartX)
                        {
                            effectiveDist[pointi] /= wakeStretch;
                        }
                    }
                }

                const scalar di = zoneInnerDist_[zonei];
                const scalar doo = zoneOuterDist_[zonei];

                // Linear ramp: 1 at di, 0 at doo
                scales_[zonei].primitiveFieldRef() =
                    min
                    (
                        max
                        (
                            (doo - effectiveDist) / (doo - di),
                            scalar(0)
                        ),
                        scalar(1)
                    );

                // Cosine smoothing (C1 at endpoints)
                scales_[zonei].primitiveFieldRef() =
                    min
                    (
                        max
                        (
                            0.5
                          - 0.5 * cos
                            (
                                scales_[zonei].primitiveField()
                              * Foam::constant::mathematical::pi
                            ),
                            scalar(0)
                        ),
                        scalar(1)
                    );

                // Apply front x-ramp: suppress SLERP influence upstream of frontEndX.
                const scalar frontEndX =
                    subDict.lookupOrDefault<scalar>("frontEndX", scalar(-vGreat));
                const scalar frontStartX =
                    subDict.lookupOrDefault<scalar>("frontStartX", frontEndX - scalar(0.05));

                if (frontEndX > -vGreat/2)
                {
                    const pointField& pts0 = points0();
                    scalarField& sf = scales_[zonei].primitiveFieldRef();

                    const scalar frontRampYStart =
                        subDict.lookupOrDefault<scalar>("frontRampYStart", scalar(0));
                    const scalar frontRampYEnd =
                        subDict.lookupOrDefault<scalar>("frontRampYEnd", scalar(vGreat));

                    forAll(sf, pointi)
                    {
                        const scalar px  = pts0[pointi].x();
                        const scalar apy = Foam::mag(pts0[pointi].y());

                        scalar xRampStrength = scalar(1);
                        if (frontRampYEnd < vGreat/2)
                        {
                            if (apy >= frontRampYEnd)
                            {
                                xRampStrength = scalar(0);
                            }
                            else if (apy > frontRampYStart)
                            {
                                const scalar ty =
                                    (apy - frontRampYStart)
                                  / (frontRampYEnd - frontRampYStart);
                                xRampStrength =
                                    0.5*(1 + Foam::cos(ty*Foam::constant::mathematical::pi));
                            }
                        }

                        if (xRampStrength < small) continue;

                        if (px <= frontStartX)
                        {
                            sf[pointi] *= (scalar(1) - xRampStrength);
                        }
                        else if (px < frontEndX)
                        {
                            const scalar t =
                                (px - frontStartX) / (frontEndX - frontStartX);
                            const scalar xFactor =
                                0.5*(1 - Foam::cos(t*Foam::constant::mathematical::pi));
                            sf[pointi] *=
                                (scalar(1) - xRampStrength) + xRampStrength * xFactor;
                        }
                    }
                }

                // Apply rear x-ramp: suppress SLERP beyond rearEndX.
                const scalar rearStartX =
                    subDict.lookupOrDefault<scalar>("rearStartX", scalar(vGreat));
                const scalar rearEndX =
                    subDict.lookupOrDefault<scalar>("rearEndX", rearStartX + scalar(0.10));

                if (rearStartX < vGreat/2)
                {
                    const pointField& pts0 = points0();
                    scalarField& sf = scales_[zonei].primitiveFieldRef();
                    forAll(sf, pointi)
                    {
                        const scalar px = pts0[pointi].x();
                        if (px > rearEndX)
                        {
                            sf[pointi] = scalar(0);
                        }
                        else if (px > rearStartX)
                        {
                            const scalar t =
                                (px - rearStartX) / (rearEndX - rearStartX);
                            sf[pointi] *=
                                0.5*(1 + Foam::cos(t*Foam::constant::mathematical::pi));
                        }
                    }
                }

                // Apply top y-ramp: taper scale to 0 above topEndY.
                const scalar topEndY =
                    subDict.lookupOrDefault<scalar>("topEndY", scalar(vGreat));
                const scalar topStartY =
                    subDict.lookupOrDefault<scalar>("topStartY", topEndY - scalar(0.05));

                if (topEndY < vGreat/2)
                {
                    const pointField& pts0 = points0();
                    scalarField& sf = scales_[zonei].primitiveFieldRef();
                    forAll(sf, pointi)
                    {
                        const scalar py = pts0[pointi].y();
                        if (py >= topEndY)
                        {
                            sf[pointi] = scalar(0);
                        }
                        else if (py > topStartY)
                        {
                            const scalar t =
                                (py - topStartY) / (topEndY - topStartY);
                            sf[pointi] *=
                                0.5*(1 + Foam::cos(t*Foam::constant::mathematical::pi));
                        }
                    }
                }

                // Apply bottom y-ramp: taper scale to 0 below botEndY.
                const scalar botEndY =
                    subDict.lookupOrDefault<scalar>("botEndY", scalar(-vGreat));
                const scalar botStartY =
                    subDict.lookupOrDefault<scalar>("botStartY", botEndY + scalar(0.05));

                if (botEndY > -vGreat/2)
                {
                    const pointField& pts0 = points0();
                    scalarField& sf = scales_[zonei].primitiveFieldRef();
                    forAll(sf, pointi)
                    {
                        const scalar py = pts0[pointi].y();
                        if (py <= botEndY)
                        {
                            sf[pointi] = scalar(0);
                        }
                        else if (py < botStartY)
                        {
                            const scalar t =
                                (py - botEndY) / (botStartY - botEndY);
                            sf[pointi] *=
                                0.5*(1 - Foam::cos(t*Foam::constant::mathematical::pi));
                        }
                    }
                }

                pointConstraints::New(pMesh).constrain(scales_[zonei]);

                // Explicitly enforce scale=1 on surface patch points.
                {
                    const pointBoundaryMesh& pbm = pMesh.boundary();
                    const labelList& patchPts =
                        pbm[zonePatchIDs_[zonei]].meshPoints();
                    forAll(patchPts, pi)
                    {
                        scales_[zonei][patchPts[pi]] = scalar(1);
                    }
                }
            }
        }

        {
            Info<< "multiInterpolatingSolidBodyMotionSolver: "
                << "zone " << zoneName
                << "  CofG=" << zoneCofGs_[zonei]
                << "  patch=" << mesh.boundaryMesh()[zonePatchIDs_[zonei]].name();
            if (rigidOnly_[zonei])
            {
                Info<< "  mode=rigidOnly"
                    << "  nPts=" << rigidPointIDs_[zonei].size();
            }
            else
            {
                Info<< "  mode=SLERP"
                    << "  di=" << zoneInnerDist_[zonei]
                    << "  do=" << zoneOuterDist_[zonei];
            }
            Info<< endl;
        }

        zonei++;
    }

    // Trim to actual zone count
    zoneIDs_.setSize(zonei);
    SBMFs_.setSize(zonei);
    zoneCofGs_.setSize(zonei);
    zonePatchIDs_.setSize(zonei);
    scales_.setSize(zonei);
    zoneInnerDist_.setSize(zonei);
    zoneOuterDist_.setSize(zonei);
    rigidOnly_.setSize(zonei);
    rigidPointIDs_.setSize(zonei);

    if (zonei == 0)
    {
        FatalIOErrorInFunction(coeffDict())
            << "No cellZone sub-dictionaries found in "
            << typeName << "Coeffs."
            << exit(FatalIOError);
    }

    Info<< "multiInterpolatingSolidBodyMotionSolver: "
        << "innerDistance=" << di_ << "  outerDistance=" << do_ << endl;

    // ------------------------------------------------------------------
    // Build inverse-distance face diffusivity for weighted Laplacian.
    // D = 1 / max(wallDist, eps) at cell centres, interpolated to faces.
    // Computed once from points0 — valid for small flap deflections (<10°).
    // ------------------------------------------------------------------
    if (enableBgLaplacian_)
    {
        const fvMesh& fvm = refCast<const fvMesh>(mesh);
        const scalar eps =
            coeffDict().lookupOrDefault<scalar>("bgLaplacianEps", scalar(1e-3));

        labelHashSet patchSet;
        forAll(zonePatchIDs_, zi)
        {
            if (zonePatchIDs_[zi] >= 0)
                patchSet.insert(zonePatchIDs_[zi]);
        }

        volScalarField wDist
        (
            IOobject
            (
                "bgWallDist",
                fvm.time().timeName(),
                fvm,
                IOobject::NO_READ,
                IOobject::NO_WRITE,
                false
            ),
            fvm,
            dimensionedScalar(dimLength, eps)
        );

        if (patchSet.size())
        {
            patchWave pw(fvm, patchSet, false);
            const scalarField& dist = pw.distance();
            forAll(dist, ci)
            {
                wDist[ci] = max(dist[ci], eps);
            }
        }

        bgDiffusivityPtr_.reset
        (
            new surfaceScalarField
            (
                IOobject
                (
                    "bgDiffusivity",
                    fvm.time().timeName(),
                    fvm,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE,
                    false
                ),
                fvc::interpolate(scalar(1.0) / wDist)
            )
        );

        Info<< "multiInterpolatingSolidBodyMotionSolver: "
            << "bgLaplacian diffusivity=inverseDistance  eps=" << eps << endl;
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::multiInterpolatingSolidBodyMotionSolver::
~multiInterpolatingSolidBodyMotionSolver()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::pointField>
Foam::multiInterpolatingSolidBodyMotionSolver::curPoints() const
{
    const pointField& points0 = this->points0();

    tmp<pointField> tpts(new pointField(points0));
    pointField& pts = tpts.ref();

    // ------------------------------------------------------------------
    // Two-pass point update (tutorial multiSolidBodyMotionSolver architecture):
    //
    // Pass 1 — SLERP zones (rigidOnly=false):
    //   Dominant-zone SLERP blending from points0.
    //   For each point, the zone with the highest scale wins.
    //   Handles smooth background morphing (e.g. wing heave+pitch).
    //
    // Pass 2 — Rigid zones (rigidOnly=true):
    //   Overwrite zone cell-points with exact rigid transform from points0.
    //   Applied last — binary, no blend, exact surface conformity.
    //   (e.g. flap: composed transform = wing heave+pitch + flap delta)
    // ------------------------------------------------------------------

    // Pass 1: SLERP zones
    forAll(pts, pointi)
    {
        scalar bestScale = small;
        label  bestZi    = -1;

        forAll(zoneIDs_, zi)
        {
            if (!rigidOnly_[zi] && scales_[zi][pointi] > bestScale)
            {
                bestScale = scales_[zi][pointi];
                bestZi    = zi;
            }
        }

        if (bestZi >= 0)
        {
            const septernion s   = SBMFs_[bestZi].transformation();
            const point&     p0i = points0[pointi];

            if (bestScale > 1 - small)
            {
                pts[pointi] = s.transformPoint(p0i);
            }
            else
            {
                const septernion ss(slerp(septernion::I, s, bestScale));
                pts[pointi] = ss.transformPoint(p0i);
            }
        }
    }

    // Pass 2: rigid zones — overwrite with exact transform from points0
    forAll(zoneIDs_, zi)
    {
        if (!rigidOnly_[zi]) continue;

        const septernion   s   = SBMFs_[zi].transformation();
        const labelList&   ids = rigidPointIDs_[zi];

        forAll(ids, i)
        {
            const label pointi = ids[i];
            pts[pointi] = s.transformPoint(points0[pointi]);
        }
    }

    twoDCorrectPoints(pts);

    // ------------------------------------------------------------------
    // Pass 4 (optional): Laplacian background diffusion.
    //
    // After the analytical SLERP/overlay/rigid passes, the background
    // mesh displacement is piece-wise smooth but has a hard gradient at
    // the refinement box boundary (scale transitions from 1 to <1 inside
    // the fine cells → compression).
    //
    // This pass replaces the background point positions with the result
    // of ∇²(cellDisplacement) = 0 solved with:
    //   - Dirichlet (fixedValue) BCs on wing/flap patches
    //     (values taken from the SLERP-computed displacements at those faces)
    //   - zeroGradient on all other patches (far-field, inlet, outlet)
    //
    // The Laplacian naturally distributes the surface displacements
    // smoothly into the background mesh — no gradients at the refinement
    // box boundary.  After the solve, surface patch points are
    // re-enforced from the original rigid transforms to maintain exact
    // surface conformity.
    // ------------------------------------------------------------------
    if (enableBgLaplacian_)
    {
        const fvMesh& fvm = refCast<const fvMesh>(mesh());
        const pointField& p0 = points0;   // captured at top of curPoints()
        const polyBoundaryMesh& pbm = mesh().boundaryMesh();

        // -- Build BC type list for cellDisplacement --
        // wing/flap patches → fixedValue (Dirichlet from SLERP)
        // everything else   → fixedValue(zero) to keep far-field boundaries stationary
        const label nPatches = fvm.boundary().size();
        wordList bcTypes(nPatches, fixedValueFvPatchVectorField::typeName);
        forAll(zoneIDs_, zi)
        {
            const label patchi = zonePatchIDs_[zi];
            if (patchi >= 0 && patchi < nPatches)
            {
                bcTypes[patchi] = fixedValueFvPatchVectorField::typeName;
            }
        }

        // -- Create cellDisplacement field with those BCs --
        volVectorField cellDisp
        (
            IOobject
            (
                "bgCellDisp",
                fvm.time().timeName(),
                fvm,
                IOobject::NO_READ,
                IOobject::NO_WRITE,
                false  // do not register
            ),
            fvm,
            dimensionedVector("zero", dimLength, vector::zero),
            bcTypes
        );

        // -- Set fixedValue BCs on zone patches from SLERP displacements --
        // For each face on the patch, average the displacement of its points.
        forAll(zoneIDs_, zi)
        {
            const label patchi = zonePatchIDs_[zi];
            if (patchi < 0 || patchi >= nPatches) continue;

            const polyPatch& pp = pbm[patchi];
            vectorField& patchVal =
                cellDisp.boundaryFieldRef()[patchi];

            forAll(pp, facei)
            {
                const face& f = pp[facei];
                vector avgDisp(vector::zero);
                forAll(f, pi)
                {
                    avgDisp += pts[f[pi]] - p0[f[pi]];
                }
                patchVal[facei] = avgDisp / scalar(f.size());
            }
        }

        // -- Solve diffusivity-weighted Laplacian for smooth interior distribution --
        // D = 1/wallDist (precomputed in constructor): high diffusivity near wall
        // → BL cells are pulled strongly toward the surface BC → no shear.
        // 1 solve is sufficient: GAMG converges in ~16 inner iterations.
        {
            fvVectorMatrix eqn(fvm::laplacian(*bgDiffusivityPtr_, cellDisp));
            eqn.solve();
        }

        // -- Interpolate cell displacement back to points --
        pointVectorField pointDisp
        (
            IOobject
            (
                "bgPointDisp",
                fvm.time().timeName(),
                fvm,
                IOobject::NO_READ,
                IOobject::NO_WRITE,
                false
            ),
            pointMesh::New(fvm),
            dimensionedVector("zero", dimLength, vector::zero)
        );
        volPointInterpolation::New(fvm).interpolate(cellDisp, pointDisp);

        // -- Apply smoothed displacement to all points --
        forAll(pts, pointi)
        {
            pts[pointi] = p0[pointi] + pointDisp[pointi];
        }

        // -- Re-enforce exact rigid transforms on surface patch points --
        // These were overwritten by the vol→point interpolation above.
        forAll(zoneIDs_, zi)
        {
            const septernion s = SBMFs_[zi].transformation();
            const label patchi = zonePatchIDs_[zi];
            if (patchi < 0 || patchi >= nPatches) continue;
            const polyPatch& pp = pbm[patchi];
            forAll(pp, facei)
            {
                const face& f = pp[facei];
                forAll(f, pi)
                {
                    const label pointi = f[pi];
                    pts[pointi] = s.transformPoint(p0[pointi]);
                }
            }
        }

        twoDCorrectPoints(pts);
    }

    return tpts;
}


// ************************************************************************* //
