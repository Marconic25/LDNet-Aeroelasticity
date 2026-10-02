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
lambda_damp = 0.003           # decay coefficient of the latent update
alpha_reg = 3e-4              # weight of the Tikhonov penalty of both networks
num_epochs_Adam = 200
num_epochs_BFGS = 20000
patience = 200                # iterations without improvement of the validation loss

# Input signals, at each latent dimension, each trained from two initialisations
input_sets = [
    ['delta', 'W_gust'],
    ['h', 'a', 'delta', 'W_gust'],
    ['h', 'hd', 'a', 'ad', 'delta', 'W_gust'],
]
latent_dims = [1, 3, 5, 10]
num_runs = 2

#%% Define problem
problem = {
    "space": {
        "dimension" : 2
    },
    "input_parameters": [
        { "name": "U_inf" }
    ],
    "input_signals": [],   # set by each input set below
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

#%% Define LDNet model

def build_LDNet(num_latent_states):
    # dynamics network
    input_shape = (num_latent_states + len(problem['input_parameters']) + len(problem['input_signals']),)
    NNdyn = tf.keras.Sequential([
                tf.keras.Input(input_shape),
                tf.keras.layers.Dense(7, activation = tf.nn.tanh),
                tf.keras.layers.Dense(7, activation = tf.nn.tanh),
                tf.keras.layers.Dense(num_latent_states)
            ])

    # reconstruction network
    input_shape = (None, None, num_latent_states + len(problem['input_signals']) + problem['space']['dimension'])
    NNrec = tf.keras.Sequential([
                tf.keras.Input(input_shape),
                tf.keras.layers.Dense(24, activation = tf.nn.tanh),
                tf.keras.layers.Dense(24, activation = tf.nn.tanh),
                tf.keras.layers.Dense(24, activation = tf.nn.tanh),
                tf.keras.layers.Dense(24, activation = tf.nn.tanh),
                tf.keras.layers.Dense(len(problem['output_signals']))
            ])

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

def test_errors(LDNet, dataset):
    # NRMSE and Pearson dissimilarity of F_y, M_z and of the two stacked (combined)
    out_signals_FOM = utils.denormalize_signals(dataset['out_signals'], problem, normalization).numpy()[:, :, 0, :]
    out_signals_ROM = utils.denormalize_signals(LDNet(dataset), problem, normalization).numpy()[:, :, 0, :]
    errors = dict()
    for name, idx in [('F_y', [0]), ('M_z', [1]), ('combined', [0, 1])]:
        FOM = out_signals_FOM[:, :, idx].reshape(-1)
        ROM = out_signals_ROM[:, :, idx].reshape(-1)
        errors['NRMSE ' + name] = np.sqrt(np.mean(np.square(ROM - FOM))) / (np.max(FOM) - np.min(FOM))
        errors['1-rho ' + name] = 1 - scipy.stats.pearsonr(ROM, FOM)[0]
    return errors

#%% Training
# Each configuration is trained from independent initialisations; the run with the
# lowest validation loss is kept.
results = dict()
for inputs in input_sets:
    problem['input_signals'] = [{ "name": name } for name in inputs]

    # the datasets are processed again for every input set
    dataset_train = utils.GLA_create_dataset('../data/GLA_train.h5')
    dataset_valid = utils.GLA_create_dataset('../data/GLA_valid.h5')
    dataset_tests = utils.GLA_create_dataset('../data/GLA_test.h5')
    for dataset in (dataset_train, dataset_valid, dataset_tests):
        utils.select_signals(dataset, problem)
        utils.process_dataset(dataset, problem, normalization, dt = dt)

    for num_latent_states in latent_dims:
        best_valid = np.inf
        for run in range(num_runs):
            print('inputs %s, latent dimension %d, run %d' % (inputs, num_latent_states, run))
            np.random.seed(run)
            tf.random.set_seed(run)
            NNdyn, NNrec, LDNet = build_LDNet(num_latent_states)

            loss_train = lambda: loss(LDNet, dataset_train) + alpha_reg * (regularization(NNdyn) + regularization(NNrec))
            loss_valid = lambda: loss(LDNet, dataset_valid)
            opt = optimization.OptimizationProblem(NNdyn.variables + NNrec.variables, loss_train, loss_valid, patience)

            print('training (Adam)...')
            opt.optimize_keras(num_epochs_Adam, tf.keras.optimizers.Adam(learning_rate=1e-2))
            print('training (BFGS)...')
            opt.optimize_BFGS(num_epochs_BFGS)
            opt.restore_best()

            if opt.best_valid < best_valid:
                best_valid = opt.best_valid
                results[(len(inputs), num_latent_states)] = test_errors(LDNet, dataset_tests)

#%% Testing
for (num_inputs, num_latent_states), errors in results.items():
    print('%d inputs   d_s = %2d   NRMSE combined: %1.2e   1-rho combined: %1.2e'
          % (num_inputs, num_latent_states, errors['NRMSE combined'], errors['1-rho combined']))

