#!/usr/bin/env python3
#%% Import modules
import numpy as np

import aeroelastic as ae

#%% Set some hyperparameters
model_path = '../models/control_ldnet'

# Reference gust at its effort weight R*
W0, Tg, R = 10.0, 0.4, 3e-4

num_realisations = 6                                    # noise realisations, seeds 100, 101, ...
noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]      # white noise sigma / W0
biases = [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]   # calibration bias b / W0
shifts = [-10, -5, -2, -1, 0, 1, 2, 5, 10]              # time shift tau / dt, positive for an early preview
noise_checks = 0.02                                     # sigma / W0 added to the checks below
biases_checked = [-0.05, 0.05]
shifts_checked = [-5, 5]

#%% Control LDNet and open loop
ldnet = ae.LDNet(model_path)
cl_trim = ae.trim_lift(ldnet)
t = ae.time_grid()
W = ae.gust(t, W0, Tg)
opened = ae.simulate(ldnet, t, W)
window = t <= Tg + 0.5

def run(field, sigma, seed, lam = 0.0):
    """Closed loop whose preview is fused from lidar scans of field; the plant sees the true gust."""
    lidar = ae.LidarPreview(field, sigma, np.random.default_rng(100 + seed), lam = lam)
    closed = ae.simulate(ldnet, t, W, ae.MPC(ldnet, cl_trim, R), lidar)
    m = ae.evaluate(closed, opened, Tg, cl_trim)
    # error delivered to the controller, first preview value against the true gust
    m['sigma_del'] = np.std(closed['W1'][window] - np.append(W[1:], W[-1])[window])
    return m

def shifted(W, d):
    """Gust profile measured d steps early (d > 0) or late (d < 0)."""
    if d > 0:
        return np.concatenate([W[d:], np.full(d, W[-1])])
    if d < 0:
        return np.concatenate([np.full(-d, W[0]), W[:d]])
    return W.copy()

def summary(ms):
    CLred = np.array([m['CLred'] for m in ms])
    return CLred.mean(), CLred.min(), CLred.max(), sum(not m['stable'] for m in ms)

exact = ae.constant_noise(0.0)
noisy = ae.constant_noise(noise_checks * W0)
seeds = range(num_realisations)

#%% White noise
white = dict()
for level in noise_levels:
    white[level] = [run(W, ae.constant_noise(level * W0), seed) for seed in seeds]
    print('white noise sigma/W0 = %4.2f:  CLred = %.1f %% [%.1f, %.1f]  flags %d   sigma_del = %.3f m/s'
          % ((level,) + summary(white[level]) + (np.mean([m['sigma_del'] for m in white[level]]),)))

#%% Range-proportional lidar noise
lidar = [run(W, ae.range_noise(), seed, lam = 1.0) for seed in seeds]
print('range-proportional noise:  CLred = %.1f %% [%.1f, %.1f]  flags %d' % summary(lidar))

#%% Calibration bias and time shift
bias = {b: run(W + b * W0, exact, 0) for b in biases}
bias_checks = {b: [run(W + b * W0, noisy, seed) for seed in seeds] for b in biases_checked}
shift = {d: run(shifted(W, d), exact, 0) for d in shifts}
shift_checks = {d: [run(shifted(W, d), noisy, seed) for seed in seeds] for d in shifts_checked}

for b in biases:
    print('bias b/W0 = %+5.2f:  CLred = %.1f %%  %s' % (b, bias[b]['CLred'], 'stable' if bias[b]['stable'] else 'unstable'))
for b in biases_checked:
    print('bias b/W0 = %+5.2f, sigma/W0 = %.2f:  CLred = %.1f %% [%.1f, %.1f]  flags %d' % ((b, noise_checks) + summary(bias_checks[b])))
for d in shifts:
    print('shift tau = %+3.0f ms:  CLred = %.1f %%  %s' % (d * ae.DT * 1e3, shift[d]['CLred'], 'stable' if shift[d]['stable'] else 'unstable'))
for d in shifts_checked:
    print('shift tau = %+3.0f ms, sigma/W0 = %.2f:  CLred = %.1f %% [%.1f, %.1f]  flags %d' % ((d * ae.DT * 1e3, noise_checks) + summary(shift_checks[d])))
