"""Sensitivity of the gust response to the fluid time step.

A reference gust, W0 = 5 m/s from 0.1 to 0.9 s of the run, is simulated from the
pre-gust checkpoint with the fixed step of 7e-5 s and with the step adapted to a
maximum Courant number of 4, 2 and 1, over the same co-simulation windows of 3.5 ms.
For each run the script reports the maximum Courant number observed, the peak of the
lift increment over the lift before the gust, its time, and the deviation of both
from the run at fixed step.

    python time_step.py run/ [options of cosim_driver.py, as --np 16]

run/ is the folder written by fom/Allrun: every run starts from a copy of run/case
and of its checkpoint.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

DRIVER = Path(__file__).resolve().parent.parent / 'fom' / 'cosim_driver.py'
COURANT = [None, 4.0, 2.0, 1.0]      # None: fixed step
SKIP = 7e-4                          # restart transient of pimpleFoam at the start of each window [s]


def lift(case):
    """C_L(t) on the time of the run, without the first samples of each window."""
    t0 = json.load(open(case / 'checkpoint' / 'cosim_state.json'))['t_checkpoint_orig']
    files = sorted((case / 'postProcessing').glob('forceCoeffs/*/forceCoeffs.dat'), key=lambda f: float(f.parent.name))
    windows = [np.loadtxt(f, usecols=(0, 3), ndmin=2) for f in files]
    t, C_L = np.concatenate([w[w[:, 0] >= w[0, 0] + SKIP] for w in windows]).T
    t, first = np.unique(t, return_index=True)
    return t - t0, C_L[first]


def max_courant(case):
    """Largest Courant number in the log, after the first steps at the step of the checkpoint."""
    values = [float(v) for v in re.findall(r'Courant Number mean: \S+ max: (\S+)', (case / 'log.pimpleFoam').read_text())]
    return max(values[min(200, max(1, len(values) // 20)):])


def simulate(run, Co):
    case = run / ('courant_%s' % (Co or 'fixed'))
    if not (case / 'structural_trajectory.csv').exists():
        shutil.rmtree(case, ignore_errors=True)
        # the case and its checkpoint, without the results of the co-simulation of fom/Allrun
        shutil.copytree(run / 'case', case, ignore=lambda folder, names: [
            name for name in names if Path(folder) == run / 'case' and
            (name.startswith(('processor', 'log.')) or name == 'postProcessing')])
        options = ['--from-checkpoint', '--window', '50', '--dt', '7e-5', '--t-end', '1.2',
                   '--gust', '5', '0.8', '--gust-onset', '0.1']
        if Co:
            options += ['--max-courant', str(Co)]
        subprocess.run([sys.executable, str(DRIVER), '--case', str(case), *options, *sys.argv[2:]], check=True)
    t, C_L = lift(case)
    base = C_L[t < 0.1].mean()
    peak = np.argmax(np.abs(C_L - base))
    return max_courant(case), C_L[peak] - base, t[peak]


def report(results):
    _, dC_L_ref, t_ref = results[None]
    print('%8s %8s %10s %10s %10s %10s' % ('Co_max', 'observed', 'dC_L peak', 'deviation', 't peak', 'deviation'))
    for Co, (observed, dC_L, t_peak) in results.items():
        print('%8s %8.2f %10.4f %9.1f%% %10.4f %9.1f%%' % (Co or 'fixed', observed, dC_L, (dC_L / dC_L_ref - 1) * 100,
                                                          t_peak, (t_peak / t_ref - 1) * 100))


if __name__ == '__main__':
    run = Path(sys.argv[1]).resolve()
    report({Co: simulate(run, Co) for Co in COURANT})
