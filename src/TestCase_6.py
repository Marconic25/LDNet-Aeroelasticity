#!/usr/bin/env python3
#%% Import modules
import time
import numpy as np

import aeroelastic as ae

#%% Set some hyperparameters
model_path = '../models/control_ldnet'

# Gust cases, each at its effort weight R*
cases = {
    'W0 = 20 m/s, Tg = 0.7 s': { 'W0': 20.0, 'Tg': 0.7, 'R': 1e-4 },
    'W0 = 10 m/s, Tg = 0.4 s': { 'W0': 10.0, 'Tg': 0.4, 'R': 3e-4 },
}

#%% Control LDNet
ldnet = ae.LDNet(model_path)
cl_trim = ae.trim_lift(ldnet)
print('C_L,trim = %1.3f' % cl_trim)

#%% Closed-loop simulations
t = ae.time_grid()
for name, c in cases.items():
    W = ae.gust(t, c['W0'], c['Tg'])
    opened = ae.simulate(ldnet, t, W)
    start = time.time()
    closed = ae.simulate(ldnet, t, W, ae.MPC(ldnet, cl_trim, c['R']))
    elapsed = time.time() - start

    m = ae.evaluate(closed, opened, c['Tg'], cl_trim)
    print('W0 = %g m/s  Tg = %g s  R = %g' % (c['W0'], c['Tg'], c['R']))
    print('    CLred = %.1f %%   |delta|max = %.1f deg   pitch ratio = %.2f   %s'
          % (m['CLred'], m['delta_max'], m['pitch_ratio'], 'stable' if m['stable'] else 'unstable'))
    print('    %.1f s of computation per simulated second' % (elapsed / t[-1]))
