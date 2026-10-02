#!/usr/bin/env python3
#%% Import modules
import numpy as np

import aeroelastic as ae

#%% Set some hyperparameters
model_path = '../models/control_ldnet'

# Grid of gust cases inspired by CS-25.341 and sweep of the effort weight
W0_list = [10.0, 20.0, 30.0]
Tg_list = [0.3, 0.4, 0.5, 0.7, 1.0, 1.2]
R_list = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

#%% Control LDNet
ldnet = ae.LDNet(model_path)
cl_trim = ae.trim_lift(ldnet)

#%% Closed-loop simulations
# For each case R* maximises CLred among the weights that pass the stability check.
t = ae.time_grid()
table = []
for W0 in W0_list:
    for Tg in Tg_list:
        W = ae.gust(t, W0, Tg)
        opened = ae.simulate(ldnet, t, W)
        metrics = dict()
        for R in R_list:
            closed = ae.simulate(ldnet, t, W, ae.MPC(ldnet, cl_trim, R))
            metrics[R] = ae.evaluate(closed, opened, Tg, cl_trim)
            print('W0 = %g  Tg = %g  R = %g:  CLred = %.1f %%  %s'
                  % (W0, Tg, R, metrics[R]['CLred'], 'stable' if metrics[R]['stable'] else 'unstable'))
        R_star = max([R for R in R_list if metrics[R]['stable']], key = lambda R: metrics[R]['CLred'])
        m = metrics[R_star]
        H = ae.U_INF * Tg / 2 / 0.3048
        k = np.pi * ae.CHORD / (ae.U_INF * Tg)
        table.append((W0, Tg, H, k, m['CLred'], R_star, m['delta_max'], m['pitch_ratio']))

#%% Results
print()
print('  W0 [m/s]   Tg [s]   H [ft]      k   CLred [%]       R*   |delta|max [deg]   pitch ratio')
for row in table:
    print('%10.0f %8.2f %8.0f %6.3f %11.1f %8.0e %18.1f %13.2f' % row)
