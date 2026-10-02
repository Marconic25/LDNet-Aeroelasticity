"""Training data of the LDNet from the runs of the campaign.

Every run folder, named sim_<family>_<index>_<split> by campaign.py, holds the
structural_trajectory.csv written by cosim_driver.py. The runs of each split are
stacked into GLA_train.h5, GLA_valid.h5 and GLA_test.h5, the files read by
src/utils.py (GLA_create_dataset).

    python build_dataset.py runs/ ../data/
"""
import sys
from pathlib import Path

import h5py
import numpy as np

U_INF = 80.0
SPLITS = {'train': 'train', 'val': 'valid', 'test': 'test'}

# columns of structural_trajectory.csv
T, H, HD, ALPHA, AD, FY, MZ, W, DELTA = range(9)


def main(runs, output):
    output.mkdir(parents=True, exist_ok=True)
    datasets = {name: [] for name in SPLITS.values()}
    for run in sorted(runs.glob('sim_*_*_*')):
        _, family, _, split = run.name.split('_')
        data = np.loadtxt(run / 'structural_trajectory.csv', delimiter=',', skiprows=1)
        datasets[SPLITS[split]].append((family, data))

    for name, runs_of_split in datasets.items():
        if not runs_of_split:
            continue
        # sorted by family, as the runs were named
        runs_of_split.sort(key=lambda r: r[0])
        families = [r[0] for r in runs_of_split]
        data = np.stack([r[1] for r in runs_of_split])
        with h5py.File(output / ('GLA_%s.h5' % name), 'w') as f:
            f.create_dataset('points', data=np.array([[0.0, 0.0]]))
            f.create_dataset('times', data=data[0, :, T])
            f.create_dataset('input_parameters', data=np.full((len(data), 1), U_INF))
            f.create_dataset('input_signals', data=data[:, :, [H, HD, ALPHA, AD, DELTA, W]])
            f.create_dataset('output_signals', data=data[:, :, [FY, MZ]][:, :, None, :])
            f.create_dataset('sim_families', data=np.array([fam.encode() for fam in families]))
        print('%s: %d runs of %d samples' % (name, data.shape[0], data.shape[1]))


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
