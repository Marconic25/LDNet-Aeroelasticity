"""Simulation campaign, drawn by Latin hypercube sampling.

    python campaign.py campaign.csv

Family A is the gust alone, family B the flap alone, driven by a ramp to the peak
delta_pk at the rate limit and a linear return to zero, family C the gust with the
flap commanded by the feed-forward law delta = -K_ff W(t - t_d). In family B half
of the peaks are negative. Each family is split into training, validation and test
runs. The table lists, for every run, its parameters and the options of
cosim_driver.py that simulate it from the pre-gust checkpoint.
"""
import csv
import sys

import numpy as np
from scipy.stats import qmc

U_INF = 80.0
T_END = 3.0
SEED = 42

# runs for training, validation and test, and Latin hypercube intervals
FAMILIES = {
    'A': ((20, 5, 5), {'r_g': (0.10, 0.60), 'T_g': (0.30, 1.20)}),
    'B': ((30, 8, 8), {'delta_pk': (2.0, 15.0), 'rate': (20.0, 200.0)}),
    'C': ((50, 10, 10), {'r_g': (0.10, 0.60), 'T_g': (0.30, 1.20), 'K_ff': (0.05, 0.20),
                         't_d': (0.00, 0.30), 'rate': (20.0, 200.0)}),
}


def driver_options(family, p):
    common = '--from-checkpoint --np 16 --window 50 --dt 7e-5 --t-end %g' % T_END
    if family == 'A':
        return '%s --gust %.4f %.6f' % (common, p['r_g'] * U_INF, p['T_g'])
    if family == 'B':
        # ramp to the peak at the rate limit, then linear return to zero at the end of the run
        t_rise = min(max(abs(p['delta_pk']) / p['rate'], 0.01), T_END / 2.0)
        return '%s --flap-table 0 0 %.6f %.4f %g 0' % (common, t_rise, p['delta_pk'], T_END)
    return '%s --gust %.4f %.6f --feed-forward %.6f %.6f %.2f' % (
        common, p['r_g'] * U_INF, p['T_g'], p['K_ff'], p['t_d'], p['rate'])


def main(path):
    rows = []
    for k, (family, (split, intervals)) in enumerate(FAMILIES.items()):
        n = sum(split)
        lo, hi = zip(*intervals.values())
        samples = qmc.scale(qmc.LatinHypercube(d=len(intervals), seed=SEED + k).random(n), lo, hi)
        if family == 'B':
            signs = np.ones(n)
            signs[:n // 2] = -1.0
            np.random.default_rng(SEED + k).shuffle(signs)
            samples[:, 0] *= signs
        names = ['train'] * split[0] + ['val'] * split[1] + ['test'] * split[2]
        for i, (sample, name) in enumerate(zip(samples, names)):
            p = dict(zip(intervals, sample))
            rows.append({'run': 'sim_%s_%03d_%s' % (family, i, name), 'family': family, 'split': name,
                         **{key: '%.6g' % v for key, v in p.items()},
                         'options': driver_options(family, p)})

    keys = ['run', 'family', 'split', 'r_g', 'T_g', 'delta_pk', 'rate', 'K_ff', 't_d', 'options']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print('%d runs written to %s' % (len(rows), path))


if __name__ == '__main__':
    main(sys.argv[1])
