#!/usr/bin/env python3
#%% Import modules
import json
import numpy as np

import utils
import aeroelastic as ae

#%% Set some hyperparameters
model_path = '../models/control_ldnet'
linear_path = '../models/linear_model.json'

fit_stride = 10         # samples of the training runs entering the least-squares fit
N = 8                   # horizon of the controller [steps]
num_trajectories = 10   # test runs with the largest flap deflection
probe_every = 12        # one probe every 12 steps of dt_ref

#%% Dataset
dataset_train = utils.GLA_create_dataset('../data/GLA_train.h5')
dataset_tests = utils.GLA_create_dataset('../data/GLA_test.h5')

# The runs are recorded every 2.8e-4 s: the models step on the grid of dt_ref,
# every sub samples of the recorded trajectories.
dt_raw = dataset_tests['times'][1] - dataset_tests['times'][0]
sub = max(1, round(ae.DT / dt_raw))

#%% Fit of the linear model
# C_La is pinned to 2 pi and the apparent-mass scaling of C_L to 1, since the campaign
# determines them too weakly; the other gains are least-squares on the training runs.
structure_only = ae.LinearModel()

def regressors(dataset, stride):
    X, CL, CM = [], [], []
    for n in range(dataset['inp_signals'].shape[0]):
        sig = dataset['inp_signals'][n]
        eta = structure_only.signals(sig[:, :4], sig[:, 4], sig[:, 5])
        s = np.zeros((1, 4))
        lags = []
        for i in range(len(eta)):
            lags.append(s[0])
            s = structure_only.step(s, eta[i:i+1], dt_raw)
        w_m, w_g = structure_only.downwash(np.array(lags), eta)
        X.append(np.stack([w_m, w_g, eta[:, 2], eta[:, 3], np.ones(len(eta))], axis = 1)[::stride])
        CL.append(dataset['out_signals'][n, ::stride, 0, 0] / ae.Q_INF)
        CM.append(dataset['out_signals'][n, ::stride, 0, 1] / (ae.Q_INF * ae.CHORD))
    return np.concatenate(X), np.concatenate(CL), np.concatenate(CM)

X, CL, CM = regressors(dataset_train, fit_stride)
pinned = 2 * np.pi * X[:, 0] + X[:, 3]
bL, *_ = np.linalg.lstsq(X[:, [1, 2, 4]], CL - pinned, rcond = None)
bM, *_ = np.linalg.lstsq(X, CM, rcond = None)
coefficients = {
    'CL_a': 2 * np.pi, 'CL_g': bL[0], 'CL_d': bL[1], 'AM_L': 1.0, 'CL_0': bL[2],
    'CM_a': bM[0], 'CM_g': bM[1], 'CM_d': bM[2], 'AM_M': bM[3], 'CM_0': bM[4],
}
coefficients = {k: float(v) for k, v in coefficients.items()}
with open(linear_path, 'w') as f:
    json.dump(coefficients, f, indent = 2)
for k, v in coefficients.items():
    print('%-5s = %+.5f' % (k, v))

#%% Forecast of the controller's horizon against the full-order model
# Each model is kept on the true trajectory, driven by the recorded state, deflection
# and gust; at every probe it predicts C_L over the next N steps from there.
ldnet = ae.LDNet(model_path)
linear = ae.LinearModel(linear_path)

def forecast(model, n, every):
    sig = dataset_tests['inp_signals'][n]
    CL_true = dataset_tests['out_signals'][n, :, 0, 0] / ae.Q_INF
    x, delta, W = sig[:, :4], sig[:, 4], sig[:, 5]
    s = np.zeros((1, model.ds))
    probes = []
    for g, i in enumerate(range(0, len(sig) - (N + 1) * sub, sub)):
        if g % every == 0:
            z = s.copy()
            errors = []
            for k in range(N):
                j = i + k * sub
                eta = model.signals(x[j], delta[j], W[j])
                CL = model.loads(z, eta)[0][0] / ae.Q_INF
                errors.append(CL - CL_true[j])
                z = model.step(z, eta)
            window = slice(i, i + N * sub)
            probes.append((np.sqrt(np.mean(np.square(errors))), np.max(np.abs(delta[window])), np.max(W[window])))
        s = model.step(s, model.signals(x[i], delta[i], W[i]))
    return np.array(probes)

delta_max = np.max(np.abs(dataset_tests['inp_signals'][:, :, 4]), axis = 1)
chosen = np.argsort(-delta_max)[:num_trajectories]
E_lin = np.concatenate([forecast(linear, n, probe_every) for n in chosen])
E_ldn = np.concatenate([forecast(ldnet, n, probe_every) for n in chosen])

def rms(e):
    return np.sqrt(np.mean(e**2))

print('%d-step forecast of C_L on %d probes of %d test runs' % (N, len(E_lin), num_trajectories))
print('    RMS error: linear %.5f   LDNet %.5f   ratio %.1f' % (rms(E_lin[:, 0]), rms(E_ldn[:, 0]), rms(E_lin[:, 0]) / rms(E_ldn[:, 0])))
for name, col, edges in [('|delta|max [deg]', 1, [0, 2, 5, 8, 11, 100]), ('W max [m/s]', 2, [0, 5, 15, 25, 35, 100])]:
    print('    by %s' % name)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (E_lin[:, col] >= lo) & (E_lin[:, col] < hi)
        if m.sum() >= 10:
            print('        %3g-%-3g  %4d probes   linear %.5f   LDNet %.5f   ratio %.1f'
                  % (lo, hi, m.sum(), rms(E_lin[m, 0]), rms(E_ldn[m, 0]), rms(E_lin[m, 0]) / rms(E_ldn[m, 0])))
