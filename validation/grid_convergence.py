"""Grid convergence of the steady RANS solution at trim.

Four meshes are generated from fom/mesh by changing the refinement of snappyHexMesh,
and simpleFoam is run on each, as in fom/Allrun. The lift, drag and moment coefficients
are averaged over the last 50 samples, and the Grid Convergence Index of Celik et al.
(2008) is computed on M3, the mesh of fom/mesh used by every co-simulation, from the
coefficients on M2, M3 and M4.

    python grid_convergence.py run/grid

Needs OpenFOAM v7 in the environment; simpleFoam runs on 16 processes.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

FOM = Path(__file__).resolve().parent.parent / 'fom'
AVERAGE = 50

# surface and feature level, level of the refinement box, level of the wing-flap gap,
# prism layers and thickness of the last one, relative to the local cell size
MESHES = {
    'M1': (3, 1, 2, 1, 0.40),
    'M2': (4, 2, 3, 2, 0.30),
    'M3': (5, 3, 5, 5, 0.20),
    'M4': (5, 4, 5, 5, 0.20),
}


def run(folder, application, *options):
    """OpenFOAM application in folder, simpleFoam on 16 processes."""
    command = [application, *options]
    if application == 'simpleFoam':
        command = ['mpirun', '--oversubscribe', '-np', '16', *command, '-parallel']
    with open(folder / ('log.' + application), 'w') as log:
        subprocess.run(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT, check=True)


def refine(path, surface, box, gap, layers, thickness):
    """Refinement of one mesh of the study in snappyHexMeshDict."""
    text = path.read_text()
    for pattern, value in [
            (r'(level\s*\(\s*)\d+\s+\d+', r'\g<1>%d %d' % (surface, surface)),
            (r'(\.eMesh"\s*;\s*level\s+)\d+', r'\g<1>%d' % surface),
            (r'(refinementBox\s*\{\s*mode\s+inside;\s*levels\s*\(\s*\(\s*1e\+?15\s+)\d+', r'\g<1>%d' % box),
            (r'(gapRegion\s*\{\s*mode\s+inside;\s*levels\s*\(\s*\(\s*1e\+?15\s+)\d+', r'\g<1>%d' % gap),
            (r'(nSurfaceLayers\s+)\d+', r'\g<1>%d' % layers),
            (r'(finalLayerThickness\s+)[\d.]+', r'\g<1>%g' % thickness),
            (r'(minThickness\s+)[\d.]+', r'\g<1>%g' % (0.1 * thickness))]:
        text, n = re.subn(pattern, value, text)
        assert n > 0, 'no match for %s' % pattern
    path.write_text(text)


def solve(folder, level):
    """Mesh and steady solution of one level; returns cells and C_L, C_D, C_M."""
    mesh, rans = folder / 'mesh', folder / 'rans'
    log = rans / 'log.simpleFoam'
    if not (log.exists() and 'End' in log.read_text()):
        shutil.rmtree(folder, ignore_errors=True)
        shutil.copytree(FOM / 'mesh', mesh)
        shutil.copytree(FOM / 'rans', rans)
        shutil.copy(FOM / 'case' / 'system' / 'decomposeParDict', rans / 'system')
        refine(mesh / 'system' / 'snappyHexMeshDict', *MESHES[level])
        for command in [['blockMesh'], ['surfaceFeatureExtract'], ['snappyHexMesh', '-overwrite'], ['checkMesh']]:
            run(mesh, *command)
        run(rans, 'extrudeMesh')
        run(rans, 'createPatch', '-overwrite')
        shutil.copytree(rans / '0.orig', rans / '0')
        run(rans, 'decomposePar')
        run(rans, 'simpleFoam')

    cells = int(re.search(r'cells:\s+(\d+)', (mesh / 'log.checkMesh').read_text()).group(1))
    # columns of forceCoeffs.dat: time, C_M, C_D, C_L
    coefficients = np.loadtxt(next((rans / 'postProcessing').glob('forceCoeffs/*/forceCoeffs.dat')), usecols=(1, 2, 3))
    C_M, C_D, C_L = coefficients[-AVERAGE:].mean(axis=0)
    return cells, C_L, C_D, C_M


def gci(f, n):
    """GCI on the medium mesh, from the values f and the cells n on coarse, medium and fine meshes.

    The refinement ratio of two meshes is the square root of the ratio of their cells,
    for a mesh one cell thick; the apparent order p is ln|e32 / e21| / ln r21.
    """
    e21, e32 = f[1] - f[2], f[0] - f[1]
    r21, r32 = np.sqrt(n[2] / n[1]), np.sqrt(n[1] / n[0])
    p = abs(np.log(abs(e32 / e21)) / np.log(r21))
    return 1.25 * abs(e32) / (r32**p - 1) / abs(f[1]) * 100


def report(results):
    print('%-4s %10s %10s %10s %10s' % ('mesh', 'cells', 'C_L', 'C_D', 'C_M'))
    for level, (cells, C_L, C_D, C_M) in results.items():
        print('%-4s %10d %10.6f %10.6f %10.6f' % (level, cells, C_L, C_D, C_M))
    cells = [results[level][0] for level in ('M2', 'M3', 'M4')]
    for i, name in [(1, 'C_L'), (2, 'C_D')]:
        f = [results[level][i] for level in ('M2', 'M3', 'M4')]
        print('%s: change M3 -> M4 %.2f %%, GCI on M3 %.2f %%' % (name, abs(f[1] - f[2]) / abs(f[2]) * 100, gci(f, cells)))


if __name__ == '__main__':
    study = Path(sys.argv[1]).resolve()
    report({level: solve(study / level, level) for level in MESHES})
