"""Interface energy balance of a co-simulation run.

The loads recorded by pimpleFoam over each window of a run are integrated again by
the structural integrator of the driver, from the checkpoint the run started from.
The work of the fluid on the structure,
    W_fluid = int (-F_y hd + M_z ad) dt,
is compared with the change of mechanical energy and with the energy dissipated by
the structural damping, over the whole run and window by window.

    python energy_balance.py run/sim_A_000_train
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'fom'))
from cosim_driver import K_H, D_H, K_ALPHA, M_HH, M_AA, M_HA, integrate_structure

D_ALPHA = 6.6      # pitch damping of the campaign [N m s/rad]


def loads(file):
    """F_y and M_z, pressure plus viscous, of one forces.dat."""
    rows = [re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line) for line in open(file) if not line.startswith('#')]
    t, Fy, Mz = np.array([(float(v[0]), float(v[2]) + float(v[5]), float(v[12]) + float(v[15])) for v in rows if len(v) >= 19]).T
    t, first = np.unique(t, return_index=True)
    return t, Fy[first], Mz[first]


def energy(x):
    h, hd, a, ad = x
    return 0.5 * (M_HH * hd**2 + 2.0 * M_HA * hd * ad + M_AA * ad**2) + 0.5 * (K_H * h**2 + K_ALPHA * a**2)


def balance(t, x, Fy, Mz):
    """Fluid work, change of mechanical energy and dissipated energy over t."""
    W_fluid = cumulative_trapezoid(-Fy * x[1] + Mz * x[3], t)[-1]
    W_damp = cumulative_trapezoid(D_H * x[1]**2 + D_ALPHA * x[3]**2, t)[-1]
    return W_fluid, energy(x[:, -1]) - energy(x[:, 0]), W_damp


if __name__ == '__main__':
    case = Path(sys.argv[1]).resolve()
    s = json.load(open(case / 'checkpoint' / 'cosim_state.json'))
    state = np.array([s['h'], s['hd'], s['a'], s['ad']])

    # one forces.dat per window, in the folder named after its start time
    files = sorted((case / 'postProcessing').glob('forces/*/forces.dat'), key=lambda f: float(f.parent.name))
    windows, imbalance = [], []
    for file in files:
        t, Fy, Mz = loads(file)
        if len(t) < 2:
            continue
        x = integrate_structure(state, t, Fy, Mz, D_ALPHA)
        state = x[:, -1]
        W_fluid, dE, W_damp = balance(t, x, Fy, Mz)
        imbalance.append(abs(W_fluid - dE - W_damp) / max(abs(W_fluid), 1e-30))
        windows.append((t, x, Fy, Mz))

    W_fluid, dE, W_damp = balance(*[np.concatenate(v, axis=-1) for v in zip(*windows)])
    print('%d windows from t = %.3f s to t = %.3f s' % (len(windows), windows[0][0][0], windows[-1][0][-1]))
    print('fluid work %.4e J, change of mechanical energy %.4e J, dissipated %.4e J'
          % (W_fluid, dE, W_damp))
    print('global imbalance %.2f %%' % (abs(W_fluid - dE - W_damp) / abs(W_fluid) * 100))
    print('imbalance per window: median %.1e, maximum %.1e' % (np.median(imbalance), np.max(imbalance)))
