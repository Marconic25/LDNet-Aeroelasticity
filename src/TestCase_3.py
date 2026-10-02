#!/usr/bin/env python3
#%% Import modules
import numpy as np
import scipy.stats
import tensorflow as tf
# We configure TensorFlow to work in double precision
tf.keras.backend.set_floatx('float64')

import utils
import optimization

#%% Set some hyperparameters
dt = 0.002
dt_base = 0.002
num_latent_states = 10
lambda_damp = 0.003           # decay coefficient of the latent update
alpha_reg = 3e-4              # weight of the Tikhonov penalty of both networks
num_epochs_Adam = 200
num_epochs_BFGS = 20000
patience = 200                # iterations without improvement of the validation loss

# Total number of hidden layers L, split 1:2 between the dynamics network
# (7 neurons per layer) and the reconstruction network (24 neurons per layer)
depths = [6, 12, 24]
num_runs = 2

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
    "output_signals": [
        { "name": "F_y" },
        { "name": "M_z" }
    ]
}

normalization = {
    'space': {
        'min' : [0, 0],
        'max' : [1, 1],
    },
    'time': {
        'time_constant' : dt_base
    },
    'input_parameters': {
        'U_inf': { 'min': 0, 'max': 120 }
    },
    'input_signals': {
        'h':      { 'min': -0.029, 'max':  0.009 },
        'hd':     { 'min': -0.54,  'max':  0.62  },
        'a':      { 'min': -0.014, 'max':  0.011 },
        'ad':     { 'min': -0.93,  'max':  0.93  },
        'delta':  { 'min': -14.09, 'max': 14.55  },
        'W_gust': { 'min':   0.0,  'max': 47.81  },
    },
    'output_signals': {
        'F_y': { 'min': -18.65, 'max': 547.29 },
        'M_z': { 'min': -44.99, 'max':  42.24 },
    }
}

#%% Dataset
dataset_train = utils.GLA_create_dataset('../data/GLA_train.h5')
dataset_valid = utils.GLA_create_dataset('../data/GLA_valid.h5')
dataset_tests = utils.GLA_create_dataset('../data/GLA_test.h5')

# We re-sample the time transients with timestep dt and we rescale each variable between -1 and 1.
utils.process_dataset(dataset_train, problem, normalization, dt = dt)
utils.process_dataset(dataset_valid, problem, normalization, dt = dt)
utils.process_dataset(dataset_tests, problem, normalization, dt = dt)

#%% Define LDNet model

def build_LDNet(depth):
    # dynamics network
    input_shape = (num_latent_states + len(problem['input_parameters']) + len(problem['input_signals']),)
    NNdyn = tf.keras.Sequential([tf.keras.Input(input_shape)]
            + [tf.keras.layers.Dense(7, activation = tf.nn.tanh) for _ in range(depth // 3)]
            + [tf.keras.layers.Dense(num_latent_states)])

    # reconstruction network
    input_shape = (None, None, num_latent_states + len(problem['input_signals']) + problem['space']['dimension'])
    NNrec = tf.keras.Sequential([tf.keras.Input(input_shape)]
            + [tf.keras.layers.Dense(24, activation = tf.nn.tanh) for _ in range(2 * depth // 3)]
            + [tf.keras.layers.Dense(len(problem['output_signals']))])

    def evolve_dynamics(dataset):
        # initial condition
        state = tf.zeros((dataset['num_samples'], num_latent_states), dtype = tf.float64)
        state_history = tf.TensorArray(tf.float64, size = dataset['num_times'])
        state_history = state_history.write(0, state)
        dt_ref = normalization['time']['time_constant']

        # time integration, damped forward Euler
        for i in tf.range(dataset['num_times'] - 1):
            state = (1 - lambda_damp) * state + dt/dt_ref * NNdyn(tf.concat([state, dataset['inp_parameters'], dataset['inp_signals'][:,i,:]], axis = -1))
            state_history = state_history.write(i + 1, state)

        return tf.transpose(state_history.stack(), perm=(1,0,2))

    def reconstruct_output(dataset, states):
        states_expanded = tf.broadcast_to(tf.expand_dims(states, axis = 2),
            [dataset['num_samples'], dataset['num_times'], dataset['num_points'], num_latent_states])
        inp_signals_expanded = tf.broadcast_to(tf.expand_dims(dataset['inp_signals'], axis = 2),
            [dataset['num_samples'], dataset['num_times'], dataset['num_points'], len(problem['input_signals'])])
        output = NNrec(tf.concat([states_expanded, inp_signals_expanded, dataset['points_full']], axis = 3))
        # nonlinear transformation to compress long tails
        beta = 0.05
        output = (output**3 + beta*output)/(1+beta)
        return output

    def LDNet(dataset):
        states = evolve_dynamics(dataset)
        return reconstruct_output(dataset, states)

    return NNdyn, NNrec, LDNet

#%% Loss function

def loss(LDNet, dataset):
    # squared error of the normalised loads, averaged over trajectories and time steps
    return tf.reduce_mean(tf.reduce_sum(tf.square(LDNet(dataset) - dataset['out_signals']), axis = 3))

def regularization(NN):
    # mean of the squared entries of the weight vector
    weights = tf.concat([tf.reshape(v, (-1,)) for v in NN.variables], axis = 0)
    return tf.reduce_mean(tf.square(weights))

def combined_NRMSE(LDNet, dataset):
    # F_y and M_z stacked and normalised by a single range, used to rank the configurations
    FOM = utils.denormalize_signals(dataset['out_signals'], problem, normalization).numpy()
    ROM = utils.denormalize_signals(LDNet(dataset), problem, normalization).numpy()
    return np.sqrt(np.mean(np.square(ROM - FOM))) / (np.max(FOM) - np.min(FOM))

#%% Training and testing
# Each depth is trained from two initialisations; both outcomes are reported.
results = dict()
for depth in depths:
    results[depth] = []
    for run in range(num_runs):
        print('depth %d, run %d' % (depth, run))
        np.random.seed(run)
        tf.random.set_seed(run)
        NNdyn, NNrec, LDNet = build_LDNet(depth)

        loss_train = lambda: loss(LDNet, dataset_train) + alpha_reg * (regularization(NNdyn) + regularization(NNrec))
        loss_valid = lambda: loss(LDNet, dataset_valid)
        opt = optimization.OptimizationProblem(NNdyn.variables + NNrec.variables, loss_train, loss_valid, patience)

        print('training (Adam)...')
        opt.optimize_keras(num_epochs_Adam, tf.keras.optimizers.Adam(learning_rate=1e-2))
        print('training (BFGS)...')
        opt.optimize_BFGS(num_epochs_BFGS)
        opt.restore_best()

        num_weights = sum(int(np.prod(v.shape)) for v in NNdyn.variables + NNrec.variables)
        results[depth].append(combined_NRMSE(LDNet, dataset_tests))
        print('L = %d  (%d weights):  combined NRMSE %1.2e' % (depth, num_weights, results[depth][-1]))

for depth, errors in results.items():
    print('L = %2d   best %1.2e   worst %1.2e' % (depth, min(errors), max(errors)))
