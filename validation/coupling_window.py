"""Refinement of the co-simulation window.

The flap is deflected to 15 deg over [0, 0.2] s and held, without gust, with the fluid
step of 7e-5 s and windows of N_win = 100, 50, 25 and 10 steps. For each window the
script reports the steady lift, mean of C_L over t >= 0.6 s, its change from the next
finer window, and the NRMSE of C_L(t) against the finest window over t >= 0.25 s,
relative to the mean of the latter.

    python coupling_window.py run/ [options of cosim_driver.py, as --np 16]

run/ is the folder written by fom/Allrun: every window runs in a copy of run/case,
started from the RANS solution in run/rans.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

DRIVER = Path(__file__).resolve().parent.parent / 'fom' / 'cosim_driver.py'
WINDOWS = [100, 50, 25, 10]


def lift(case):
    """C_L(t) of the forceCoeffs.dat files, one per window."""
    files = sorted((case / 'postProcessing').glob('forceCoeffs/*/forceCoeffs.dat'), key=lambda f: float(f.parent.name))
    t, C_L = np.concatenate([np.loadtxt(f, usecols=(0, 3), ndmin=2) for f in files]).T
    t, first = np.unique(t, return_index=True)
    return t, C_L[first]


def simulate(run, n):
    case = run / ('window_%d' % n)
    if not (case / 'structural_trajectory.csv').exists():
        shutil.rmtree(case, ignore_errors=True)
        # the case without the results of the co-simulation of fom/Allrun
        shutil.copytree(run / 'case', case, ignore=lambda folder, names: [
            name for name in names if Path(folder) == run / 'case' and
            (name.startswith(('processor', 'log.')) or name in ('checkpoint', 'postProcessing'))])
        subprocess.run([sys.executable, str(DRIVER), '--case', str(case), '--rans', str(run / 'rans'),
                        '--window', str(n), '--dt', '7e-5', '--t-end', '1.0',
                        '--flap-table', '0', '0', '0.2', '15', *sys.argv[2:]], check=True)
    return lift(case)


def report(histories):
    t_ref, C_L_ref = histories[WINDOWS[-1]]
    late = t_ref >= 0.25
    steady = {n: C_L[t >= 0.6].mean() for n, (t, C_L) in histories.items()}
    print('%6s %10s %12s %12s' % ('N_win', 'C_L steady', 'change [%]', 'NRMSE [%]'))
    for n, finer in zip(WINDOWS, WINDOWS[1:] + [None]):
        t, C_L = histories[n]
        error = np.interp(t_ref[late], t, C_L) - C_L_ref[late]
        nrmse = np.sqrt(np.mean(error**2)) / abs(C_L_ref[late].mean()) * 100
        change = abs(steady[finer] - steady[n]) / abs(steady[finer]) * 100 if finer else np.nan
        print('%6d %10.5f %12.3f %12.3f' % (n, steady[n], change, nrmse))


if __name__ == '__main__':
    run = Path(sys.argv[1]).resolve()
    report({n: simulate(run, n) for n in WINDOWS})
