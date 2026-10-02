#!/usr/bin/env python3
#%% Import modules
import os
import json
import time
import numpy as np
import tensorflow as tf
# We configure TensorFlow to work in double precision
tf.keras.backend.set_floatx('float64')

import utils
import optimization

#%% Set some hyperparameters
num_latent_states = 1
num_points_subsample = 1024   # query points drawn at every time step of every training trajectory
num_epochs_Adam = 300
num_epochs_BFGS = 2000
num_restarts = 2              # Adam from independent initialisations, the best one continues with BFGS
omega0 = 10.0                 # frequency of the sine layers of the reconstruction network
output_path = '../models/TestCase_4'

#%% Dataset
# Velocity and pressure of the full-order model on the nodes of a mesh around the airfoil
dataset_train = utils.FIELDS_create_dataset('../data/FIELDS_train.h5')
dataset_valid = utils.FIELDS_create_dataset('../data/FIELDS_valid.h5')
dataset_tests = utils.FIELDS_create_dataset('../data/FIELDS_test.h5')

dt = dataset_train['times'][1] - dataset_train['times'][0]
dt_base = dt

# The mean field of the training trajectories is subtracted, and the LDNet learns the fluctuation.
mean_field = dataset_train['out_fields'].mean(axis = (0, 1))
for dataset in (dataset_train, dataset_valid, dataset_tests):
    dataset['out_fields'] = dataset['out_fields'] - mean_field

#%% Define problem
problem = {
    "space": {
        "dimension" : 2
    },
    "input_parameters": [
        { "name": "U_inf" }
    ],
    "input_signals": [
        { "name": "h" },
        { "name": "hd" },
        { "name": "a" },
        { "name": "ad" },
        { "name": "delta" },
        { "name": "W_gust" }
    ],
    "output_fields": [
        { "name": "vx" },
        { "name": "vy" },
        { "name": "p" }
    ]
}

# min-max bounds of the training data; a constant channel gets a unit range
def bounds(v):
    lo, hi = float(np.min(v)), float(np.max(v))
    return { 'min': lo - 0.5, 'max': hi + 0.5 } if hi - lo < 1e-9 else { 'min': lo, 'max': hi }

normalization = {
    'space': {
        'min' : list(dataset_train['points'].min(axis = 0)),
        'max' : list(dataset_train['points'].max(axis = 0)),
    },
    'time': {
        'time_constant' : dt_base
    },
    'input_parameters': {
        'U_inf': { 'min': 0, 'max': 120 }
    },
    'input_signals': { v['name']: bounds(dataset_train['inp_signals'][:, :, i]) for i, v in enumerate(problem['input_signals']) },
    'output_fields': { v['name']: bounds(dataset_train['out_fields'][..., i]) for i, v in enumerate(problem['output_fields']) },
}

# For reproducibility (delete if you want to test other random initializations)
np.random.seed(0)
tf.random.set_seed(0)

# We rescale each variable between -1 and 1, drawing the query points of the training and validation trajectories.
utils.process_dataset(dataset_train, problem, normalization, num_points_subsample = num_points_subsample)
utils.process_dataset(dataset_valid, problem, normalization, num_points_subsample = num_points_subsample)
utils.process_dataset(dataset_tests, problem, normalization)

#%% Define LDNet model

class ModulatedSiren(tf.keras.Model):
    """Reconstruction network: sine layers on the coordinates, shifted by a small
    network of the latent state and of the input signals, and a linear output."""

    def __init__(self, num_code, num_coords, width, depth, num_out):
        super().__init__()
        self.num_code = num_code
        self.width = width
        self.depth = depth
        # initialisation of Sitzmann et al. (2020)
        self.sine = [tf.keras.layers.Dense(width, kernel_initializer = tf.keras.initializers.RandomUniform(-lim, lim))
                     for lim in [1.0 / num_coords] + [np.sqrt(6.0 / width) / omega0] * (depth - 1)]
        self.head = tf.keras.layers.Dense(num_out)
        self.modulation = tf.keras.Sequential([
                tf.keras.layers.Dense(width, activation = tf.nn.tanh),
                tf.keras.layers.Dense(width, activation = tf.nn.tanh),
                tf.keras.layers.Dense(depth * width)
            ])

    def call(self, x):
        shift = self.modulation(x[..., :self.num_code])
        h = x[..., self.num_code:]
        for l in range(self.depth):
            h = tf.sin(omega0 * self.sine[l](h) + shift[..., l * self.width:(l + 1) * self.width])
        return self.head(h)

def build_LDNet():
    # dynamics network
    input_shape = (num_latent_states + len(problem['input_parameters']) + len(problem['input_signals']),)
    NNdyn = tf.keras.Sequential([
                tf.keras.Input(input_shape),
                tf.keras.layers.Dense(7, activation = tf.nn.tanh),
                tf.keras.layers.Dense(7, activation = tf.nn.tanh),
                tf.keras.layers.Dense(num_latent_states)
            ])

    # reconstruction network
    num_code = num_latent_states + len(problem['input_signals'])
    NNrec = ModulatedSiren(num_code, problem['space']['dimension'], 24, 4, len(problem['output_fields']))
    NNrec(tf.zeros((1, 1, 1, num_code + problem['space']['dimension']), dtype = tf.float64))

    def evolve_dynamics(dataset):
        # initial condition
        state = tf.zeros((dataset['num_samples'], num_latent_states), dtype = tf.float64)
        state_history = tf.TensorArray(tf.float64, size = dataset['num_times'])
        state_history = state_history.write(0, state)
        dt_ref = normalization['time']['time_constant']

        # time integration
        for i in tf.range(dataset['num_times'] - 1):
            state = state + dt/dt_ref * NNdyn(tf.concat([state, dataset['inp_parameters'], dataset['inp_signals'][:,i,:]], axis = -1))
            state_history = state_history.write(i + 1, state)

        return tf.transpose(state_history.stack(), perm=(1,0,2))

    def reconstruct_output(dataset, states):
        states_expanded = tf.broadcast_to(tf.expand_dims(states, axis = 2),
            [dataset['num_samples'], dataset['num_times'], dataset['num_points'], num_latent_states])
        inp_signals_expanded = tf.broadcast_to(tf.expand_dims(dataset['inp_signals'], axis = 2),
            [dataset['num_samples'], dataset['num_times'], dataset['num_points'], len(problem['input_signals'])])
        return NNrec(tf.concat([states_expanded, inp_signals_expanded, dataset['points_full']], axis = 3))

    def LDNet(dataset):
        states = evolve_dynamics(dataset)
        return reconstruct_output(dataset, states)

    return NNdyn, NNrec, LDNet

#%% Loss function

def loss(LDNet, dataset):
    return tf.reduce_mean(tf.square(LDNet(dataset) - dataset['out_fields']))

#%% Training
best_valid = np.inf
for run in range(num_restarts):
    np.random.seed(run)
    tf.random.set_seed(run)
    NNdyn, NNrec, LDNet = build_LDNet()
    loss_train = lambda: loss(LDNet, dataset_train)
    loss_valid = lambda: loss(LDNet, dataset_valid)
    opt = optimization.OptimizationProblem(NNdyn.variables + NNrec.variables, loss_train, loss_valid)
    print('training (Adam), initialisation %d...' % run)
    opt.optimize_keras(num_epochs_Adam, tf.keras.optimizers.Adam(learning_rate=1e-2))
    if opt.loss_valid_history[-1] < best_valid:
        best_valid = opt.loss_valid_history[-1]
        best_weights = (NNdyn.get_weights(), NNrec.get_weights())

NNdyn, NNrec, LDNet = build_LDNet()
NNdyn.set_weights(best_weights[0])
NNrec.set_weights(best_weights[1])
loss_train = lambda: loss(LDNet, dataset_train)
loss_valid = lambda: loss(LDNet, dataset_valid)
opt = optimization.OptimizationProblem(NNdyn.variables + NNrec.variables, loss_train, loss_valid)
print('training (BFGS)...')
opt.optimize_BFGS(num_epochs_BFGS)

os.makedirs(output_path, exist_ok = True)
NNdyn.save_weights(output_path + '/NNdyn_weights.weights.h5')
NNrec.save_weights(output_path + '/NNrec_weights.weights.h5')
np.save(output_path + '/mean_field.npy', mean_field)
with open(output_path + '/config.json', 'w') as f:
    json.dump({'problem': problem, 'normalization': normalization, 'num_latent_states': num_latent_states}, f, indent = 2)

#%% Testing
# Reconstruction of the test trajectories on all the nodes, timed
start = time.time()
out_fields = LDNet(dataset_tests)
elapsed = time.time() - start

# the LDNet works with normalised fluctuations: we map them back and add the mean field
out_fields_FOM = utils.denormalize_output(dataset_tests['out_fields'], problem, normalization).numpy() + mean_field
out_fields_ROM = utils.denormalize_output(out_fields                 , problem, normalization).numpy() + mean_field

for i, name in enumerate(['v_x', 'v_y', 'p']):
    FOM, ROM = out_fields_FOM[..., i], out_fields_ROM[..., i]
    print('NRMSE %s: %1.3e' % (name, np.sqrt(np.mean(np.square(ROM - FOM))) / (np.max(FOM) - np.min(FOM))))
print('reconstruction of %d trajectories of %d time steps on %d nodes: %.1f s'
      % (dataset_tests['num_samples'], dataset_tests['num_times'], dataset_tests['num_points'], elapsed))
