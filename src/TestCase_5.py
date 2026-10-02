#!/usr/bin/env python3
#%% Import modules
import os
import json
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
num_epochs_Adam = 400         # teacher forcing
num_epochs_BFGS = 4000        # teacher forcing
num_epochs_rollout = 500      # BFGS on the rollout loss
num_steps_rollout = 800       # N_rollout, 1.6 s

# Total number of hidden layers: L = 6 is the control LDNet, L = 12 is compared with it
depths = [6, 12]
output_path = '../models/TestCase_5'

# Structural parameters of the co-simulation campaign that generated the dataset: those
# of the closed loop except the pitch damping, 6.6 instead of 19.8 N m s/rad. As in the
# co-simulation driver, the plunge is forced by -F_y and the pitch by +M_z.
m_w, I_w, m_f, I_f = 19.24, 1.155, 1.19, 0.006
k_h, d_h, k_a, d_a = 25460.0, 88.0, 9530.0, 6.6
d_x = 0.779 - 0.40
M_hh = m_w + m_f
M_aa = I_w + (I_f + m_f * d_x**2)
M_ha = m_f * d_x
det_M = M_hh * M_aa - M_ha**2

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

# bounds of the structural state and of the loads, to move between physical and normalised values
state_min = np.array([normalization['input_signals'][k]['min'] for k in ['h', 'hd', 'a', 'ad']])
state_max = np.array([normalization['input_signals'][k]['max'] for k in ['h', 'hd', 'a', 'ad']])
loads_min = np.array([normalization['output_signals'][k]['min'] for k in ['F_y', 'M_z']])
loads_max = np.array([normalization['output_signals'][k]['max'] for k in ['F_y', 'M_z']])

#%% Structural model and its fixed Dormand-Prince step, in TensorFlow

def structural_derivative(x, Fy, Mz):
    h, hd, a, ad = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    fh = -Fy - d_h * hd - k_h * h
    fa = Mz - d_a * ad - k_a * a
    return tf.stack([hd, (M_aa * fh - M_ha * fa) / det_M, ad, (M_hh * fa - M_ha * fh) / det_M], axis = 1)

def structural_step(x, Fy, Mz):
    f = lambda y: structural_derivative(y, Fy, Mz)
    k1 = f(x)
    k2 = f(x + dt * (1 / 5 * k1))
    k3 = f(x + dt * (3 / 40 * k1 + 9 / 40 * k2))
    k4 = f(x + dt * (44 / 45 * k1 - 56 / 15 * k2 + 32 / 9 * k3))
    k5 = f(x + dt * (19372 / 6561 * k1 - 25360 / 2187 * k2 + 64448 / 6561 * k3 - 212 / 729 * k4))
    k6 = f(x + dt * (9017 / 3168 * k1 - 355 / 33 * k2 + 46732 / 5247 * k3 + 49 / 176 * k4 - 5103 / 18656 * k5))
    return x + dt * (35 / 384 * k1 + 500 / 1113 * k3 + 125 / 192 * k4 - 2187 / 6784 * k5 + 11 / 84 * k6)

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

    beta = 0.05   # nonlinear transformation to compress long tails

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
        return (output**3 + beta*output)/(1+beta)

    def LDNet(dataset):
        # teacher forcing: the structural state is read from the dataset at every step
        states = evolve_dynamics(dataset)
        return reconstruct_output(dataset, states)

    def rollout_error(dataset):
        # rollout: the predicted loads drive the structural model, whose state is fed
        # back to the networks; returns the sum of the squared errors of the
        # normalised structural state and loads over the steps 0..N_rollout
        dt_ref = normalization['time']['time_constant']
        state = tf.zeros((dataset['num_samples'], num_latent_states), dtype = tf.float64)
        x = utils.normalize_back(dataset['inp_signals'][:, 0, :4], state_min, state_max, axis = 1)
        point = dataset['points_full'][:, 0, :, :]
        error = tf.constant(0.0, tf.float64)
        for n in tf.range(num_steps_rollout + 1):
            eta = tf.concat([utils.normalize_forw(x, state_min, state_max, axis = 1), dataset['inp_signals'][:, n, 4:]], axis = 1)
            output = NNrec(tf.concat([state[:, None, :], eta[:, None, :], point], axis = 2)[:, None, :, :])[:, 0, 0, :]
            output = (output**3 + beta*output)/(1+beta)
            error += tf.reduce_sum(tf.square(eta[:, :4] - dataset['inp_signals'][:, n, :4]))
            error += tf.reduce_sum(tf.square(output - dataset['out_signals'][:, n, 0, :]))
            state = (1 - lambda_damp) * state + dt/dt_ref * NNdyn(tf.concat([state, dataset['inp_parameters'], eta], axis = -1))
            loads = utils.normalize_back(output, loads_min, loads_max, axis = 1)
            x = structural_step(x, loads[:, 0], loads[:, 1])
        return error

    return NNdyn, NNrec, LDNet, rollout_error

#%% Loss functions

def loss_tf(LDNet, dataset):
    # teacher-forcing loss: squared error of the normalised loads, averaged over trajectories and time steps
    return tf.reduce_mean(tf.reduce_sum(tf.square(LDNet(dataset) - dataset['out_signals']), axis = 3))

def loss_rollout(rollout_error, dataset):
    # rollout loss, without the Tikhonov penalty
    return rollout_error(dataset) / (dataset['num_samples'] * num_steps_rollout)

def regularization(NN):
    # mean of the squared entries of the weight vector
    weights = tf.concat([tf.reshape(v, (-1,)) for v in NN.variables], axis = 0)
    return tf.reduce_mean(tf.square(weights))

def test_errors(LDNet, dataset):
    # NRMSE of F_y, M_z and of the two stacked (combined), teacher forced
    out_signals_FOM = utils.denormalize_signals(dataset['out_signals'], problem, normalization).numpy()[:, :, 0, :]
    out_signals_ROM = utils.denormalize_signals(LDNet(dataset), problem, normalization).numpy()[:, :, 0, :]
    errors = dict()
    for name, idx in [('F_y', [0]), ('M_z', [1]), ('combined', [0, 1])]:
        FOM = out_signals_FOM[:, :, idx].reshape(-1)
        ROM = out_signals_ROM[:, :, idx].reshape(-1)
        errors[name] = np.sqrt(np.mean(np.square(ROM - FOM))) / (np.max(FOM) - np.min(FOM))
    return errors

#%% Training
# Teacher forcing first, then BFGS on the rollout loss warm-started from those weights;
# each stage keeps the weights at the minimum of its validation loss.
np.random.seed(0)
tf.random.set_seed(0)
for depth in depths:
    NNdyn, NNrec, LDNet, rollout_error = build_LDNet(depth)
    penalty = lambda: alpha_reg * (regularization(NNdyn) + regularization(NNrec))
    variables = NNdyn.variables + NNrec.variables

    print('L = %d: teacher forcing' % depth)
    loss_train = lambda: loss_tf(LDNet, dataset_train) + penalty()
    loss_valid = lambda: loss_tf(LDNet, dataset_valid)
    opt_tf = optimization.OptimizationProblem(variables, loss_train, loss_valid)
    print('training (Adam)...')
    opt_tf.optimize_keras(num_epochs_Adam, tf.keras.optimizers.Adam(learning_rate=1e-2))
    print('training (BFGS)...')
    opt_tf.optimize_BFGS(num_epochs_BFGS)
    opt_tf.restore_best()

    print('L = %d: rollout' % depth)
    loss_train = lambda: loss_rollout(rollout_error, dataset_train) + penalty()
    loss_valid = lambda: loss_rollout(rollout_error, dataset_valid)
    opt_ro = optimization.OptimizationProblem(variables, loss_train, loss_valid)
    print('training (BFGS)...')
    opt_ro.optimize_BFGS(num_epochs_rollout)
    opt_ro.restore_best()

    print('L = %d: validation rollout loss %1.3e at the warm start, %1.3e after fine-tuning'
          % (depth, opt_ro.loss_valid_history[0], opt_ro.best_valid))

    if depth == 6:
        # the control LDNet
        os.makedirs(output_path, exist_ok = True)
        NNdyn.save_weights(output_path + '/NNdyn_weights.weights.h5')
        NNrec.save_weights(output_path + '/NNrec_weights.weights.h5')
        config = {'problem': problem, 'normalization': normalization,
                  'num_latent_states': num_latent_states, 'lambda_damp': lambda_damp}
        with open(output_path + '/config.json', 'w') as f:
            json.dump(config, f, indent = 2)

        #%% Testing
        errors = test_errors(LDNet, dataset_tests)
        print('control LDNet, teacher forced on the test runs:   ' + '   '.join('NRMSE %s: %1.2e' % item for item in errors.items()))
