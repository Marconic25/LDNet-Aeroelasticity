import numpy as np
import tensorflow as tf
from scipy import interpolate
import h5py

def normalize_forw(v, v_min, v_max, axis = None):
    v_min, v_max = reshape_min_max(len(v.shape), v_min, v_max, axis)
    return (2.0*v - v_min - v_max) / (v_max - v_min)

def normalize_back(v, v_min, v_max, axis = None):
    v_min, v_max = reshape_min_max(len(v.shape), v_min, v_max, axis)
    return 0.5*(v_min + v_max + (v_max - v_min) * v)

def reshape_min_max(n, v_min, v_max, axis = None):
    if axis is not None:
        shape_min = [1] * n
        shape_max = [1] * n
        shape_min[axis] = len(v_min)
        shape_max[axis] = len(v_max)
        v_min = np.reshape(v_min, shape_min)
        v_max = np.reshape(v_max, shape_max)
    return v_min, v_max

def analyze_normalization(problem, normalization_definition):
    normalization = dict()
    normalization['dt_base'] = normalization_definition['time']['time_constant']
    normalization['x_min'] = np.array(normalization_definition['space']['min'])
    normalization['x_max'] = np.array(normalization_definition['space']['max'])
    for key, short in [('input_parameters', 'inp_parameters'), ('input_signals', 'inp_signals'),
                       ('output_fields', 'out_fields'), ('output_signals', 'out_signals')]:
        if len(problem.get(key, [])) > 0:
            normalization[short + '_min'] = np.array([normalization_definition[key][v['name']]['min'] for v in problem[key]])
            normalization[short + '_max'] = np.array([normalization_definition[key][v['name']]['max'] for v in problem[key]])
    return normalization

def dataset_normalize(dataset, problem, normalization_definition):
    normalization = analyze_normalization(problem, normalization_definition)
    dataset['times']              = dataset['times'] / normalization['dt_base']
    dataset['points']             = normalize_forw(dataset['points']        , normalization['x_min']             , normalization['x_max']             , axis = 1)
    dataset['points_full']        = normalize_forw(dataset['points_full']   , normalization['x_min']             , normalization['x_max']             , axis = 3)
    if dataset['inp_parameters'] is not None:
        dataset['inp_parameters'] = normalize_forw(dataset['inp_parameters'], normalization['inp_parameters_min'], normalization['inp_parameters_max'], axis = 1)
    if dataset['inp_signals'] is not None:
        dataset['inp_signals']    = normalize_forw(dataset['inp_signals']   , normalization['inp_signals_min']   , normalization['inp_signals_max']   , axis = 2)
    if dataset['out_fields'] is not None:
        dataset['out_fields']     = normalize_forw(dataset['out_fields']    , normalization['out_fields_min']    , normalization['out_fields_max']    , axis = 3)
    if dataset['out_signals'] is not None:
        dataset['out_signals']    = normalize_forw(dataset['out_signals']   , normalization['out_signals_min']   , normalization['out_signals_max']   , axis = 3)

def denormalize_output(out_fields, problem, normalization_definition):
    normalization = analyze_normalization(problem, normalization_definition)
    return normalize_back(out_fields , normalization['out_fields_min'], normalization['out_fields_max'], axis = 3)

def denormalize_signals(out_signals, problem, normalization_definition):
    normalization = analyze_normalization(problem, normalization_definition)
    return normalize_back(out_signals, normalization['out_signals_min'], normalization['out_signals_max'], axis = 3)

def process_dataset(dataset, problem, normalization_definition, dt = None, num_points_subsample = None):
    outputs = [k for k in ('out_fields', 'out_signals') if dataset[k] is not None]

    if dt is not None:
        times = np.arange(dataset['times'][0], dataset['times'][-1] + dt * 1e-10, step = dt)
        if dataset['inp_signals'] is not None:
            dataset['inp_signals'] = interpolate.interp1d(dataset['times'], dataset['inp_signals'], axis = 1)(times)
        for k in outputs:
            dataset[k] = interpolate.interp1d(dataset['times'], dataset[k], axis = 1)(times)
        dataset['times'] = times

    num_samples = dataset[outputs[0]].shape[0]
    num_times = dataset['times'].shape[0]
    num_points = dataset['points'].shape[0]
    num_x = dataset['points'].shape[1]

    points_full = np.broadcast_to(dataset['points'][None,None,:,:], [num_samples, num_times, num_points, num_x])

    if num_points_subsample is None or num_points_subsample >= num_points:
        dataset['points_full'] = points_full
    else:
        idxs = np.random.randint(0, num_points, size = (num_samples, num_times, num_points_subsample))
        i_s = np.arange(num_samples)[:, None, None]
        i_t = np.arange(num_times)[None, :, None]
        dataset['points_full'] = points_full[i_s, i_t, idxs, :]
        dataset['out_fields']  = dataset['out_fields'][i_s, i_t, idxs, :]

    dataset['num_points'] = dataset['points_full'].shape[2]
    dataset['num_times'] = num_times
    dataset['num_samples'] = num_samples

    dataset_normalize(dataset, problem, normalization_definition)

    for k in ['inp_parameters', 'inp_signals'] + outputs:
        if dataset[k] is not None:
            dataset[k] = tf.convert_to_tensor(dataset[k], tf.float64)

# Data conversion functions for specific Test Cases

def GLA_create_dataset(dataset_path):
    """Sectional loads of the co-simulation campaign, written by fom/build_dataset.py."""
    print('loading dataset %s' % dataset_path)
    with h5py.File(dataset_path, 'r') as f:
        return {
            'points' : f['points'][:], # [num_points x num_coordinates], the single query point
            'times' : f['times'][:], # [num_times]
            'inp_parameters' : f['input_parameters'][:], # [num_samples x num_par], U_inf
            'inp_signals' : f['input_signals'][:], # [num_samples x num_times x num_signals], h, hd, a, ad, delta, W
            'out_fields' : None,
            'out_signals' : f['output_signals'][:], # [num_samples x num_times x 1 x num_signals], F_y, M_z
            'families' : f['sim_families'][:].astype(str), # [num_samples], A, B or Cc
        }

def FIELDS_create_dataset(dataset_path):
    """Velocity and pressure of the co-simulation campaign on the nodes of a mesh around the airfoil."""
    print('loading dataset %s' % dataset_path)
    with h5py.File(dataset_path, 'r') as f:
        return {
            'points' : f['points'][:], # [num_points x num_coordinates]
            'times' : f['times'][:], # [num_times]
            'inp_parameters' : f['input_parameters'][:], # [num_samples x num_par], U_inf
            'inp_signals' : f['input_signals'][:], # [num_samples x num_times x num_signals], h, hd, a, ad, delta, W
            'out_fields' : f['output_fields'][:], # [num_samples x num_times x num_points x num_fields], vx, vy, p
            'out_signals' : None,
        }

def select_signals(dataset, problem, all_signals = ('h', 'hd', 'a', 'ad', 'delta', 'W_gust')):
    """Keep only the input signals listed in problem, in the order of the dataset."""
    idxs = [all_signals.index(v['name']) for v in problem['input_signals']]
    dataset['inp_signals'] = dataset['inp_signals'][:, :, idxs]
