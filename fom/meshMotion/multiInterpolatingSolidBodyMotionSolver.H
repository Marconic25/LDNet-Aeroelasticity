/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Custom solver for wingMotion2D co-simulation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Class
    Foam::multiInterpolatingSolidBodyMotionSolver

Description
    Two-step mesh motion solver for wing+flap aeroelastic co-simulation.

    Architecture (tutorial multiSolidBodyMotionSolver extended with SLERP):

    Pass 1 — SLERP zones:
      Dominant-zone SLERP blending from points0 over the whole domain.
      Used for the wing heave+pitch motion: smooth gradient from patch to far-field.

    Pass 2 — Rigid zones (rigidOnly=true):
      Binary overwrite of zone cell-points with exact rigid transform from points0.
      Used for the flap: composed transform = wing heave+pitch + flap delta.
      Applied last — exact surface conformity, no blending distortion.

    Scale field (computed once in constructor from patch distances):
      - 1 at distance < innerDistance  (rigid motion region)
      - cosine taper from innerDistance to outerDistance
      - 0 at distance > outerDistance  (stationary far field)
      - surface patch points explicitly forced to 1

    solve() is a no-op — no PDE needed.

    Dictionary format:
    \verbatim
    motionSolver    multiInterpolatingSolidBodyMotionSolver;

    multiInterpolatingSolidBodyMotionSolverCoeffs
    {
        innerDistance   0.3;
        outerDistance   4.0;

        // Pass 2: exact rigid body (flapZone cellZone from topoSetDict)
        flapZone
        {
            rigidOnly               true;
            patch                   flap;
            solidBodyMotionFunction tabulated6DoFMotion;
            tabulated6DoFMotionCoeffs
            {
                CofG             (0.779 0.0 0.125);
                timeDataFileName "constant/flapMotion.dat";
            }
        }

        // Pass 1: SLERP blend over whole domain
        wingZone
        {
            innerDistance           0.3;
            outerDistance           4.0;
            distMode                bbox;
            distCentreX             0.36;
            distCentreY             0.0;
            distHalfX               0.36;
            distHalfY               0.075;
            patch                   wing_main;
            solidBodyMotionFunction tabulated6DoFMotion;
            tabulated6DoFMotionCoeffs
            {
                CofG             (0.25 0.0 0.125);
                timeDataFileName "constant/wingMotion.dat";
            }
        }
    }
    \endverbatim

 SourceFiles
    multiInterpolatingSolidBodyMotionSolver.C

\*---------------------------------------------------------------------------*/

#ifndef multiInterpolatingSolidBodyMotionSolver_H
#define multiInterpolatingSolidBodyMotionSolver_H

#include "points0MotionSolver.H"
#include "solidBodyMotionFunction.H"
#include "PtrList.H"
#include "pointFields.H"
#include "boolList.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "volPointInterpolation.H"
#include "fvMesh.H"
#include "autoPtr.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
        Class multiInterpolatingSolidBodyMotionSolver Declaration
\*---------------------------------------------------------------------------*/

class multiInterpolatingSolidBodyMotionSolver
:
    public points0MotionSolver
{
    // Per-zone rigid body data

        //- Motion functions, one per cell zone
        PtrList<solidBodyMotionFunction> SBMFs_;

        //- Cell zone IDs
        labelList zoneIDs_;

        //- Centre of gravity (pivot point) per zone
        List<vector> zoneCofGs_;

        //- Surface patch ID per zone (for scale field computation)
        labelList zonePatchIDs_;


    // SLERP interpolation parameters

        //- Global inner distance fallback (overridden per zone)
        scalar di_;

        //- Global outer distance fallback (overridden per zone)
        scalar do_;

        //- Per-zone inner distances
        scalarList zoneInnerDist_;

        //- Per-zone outer distances
        scalarList zoneOuterDist_;

        //- Per-zone SLERP scale fields (1 near patch, 0 far away)
        //  Computed once in constructor from patch distances
        PtrList<pointScalarField> scales_;

        //- Per-zone: true → binary exact rigid (rigidOnly, tutorial-style)
        //            false → distance-based SLERP (default)
        List<bool> rigidOnly_;

        //- Per-zone point indices for rigidOnly zones (MPI-synced via syncTools)
        //  Only populated for zones where rigidOnly_[zi] == true
        List<labelList> rigidPointIDs_;

        //- Enable Laplacian background diffusion after SLERP/overlay/rigid passes
        bool enableBgLaplacian_;

        //- Face diffusivity = 1/wallDist for weighted Laplacian (computed once in constructor)
        autoPtr<surfaceScalarField> bgDiffusivityPtr_;


public:

    //- Runtime type information
    TypeName("multiInterpolatingSolidBodyMotionSolver");


    // Constructors

        //- Construct from mesh and dictionary
        multiInterpolatingSolidBodyMotionSolver
        (
            const polyMesh&,
            const dictionary&
        );

        //- Disallow copy construction
        multiInterpolatingSolidBodyMotionSolver
        (
            const multiInterpolatingSolidBodyMotionSolver&
        ) = delete;


    //- Destructor
    ~multiInterpolatingSolidBodyMotionSolver();


    // Member Functions

        //- Return point location obtained from the current motion field
        virtual tmp<pointField> curPoints() const;

        //- Solve for motion (no-op: scale fields computed in constructor)
        virtual void solve()
        {}


    // Member Operators

        //- Disallow assignment
        void operator=(const multiInterpolatingSolidBodyMotionSolver&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
