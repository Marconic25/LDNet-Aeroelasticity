"""Closed loop of the gust load alleviation study.

    parameters          structure, air and flap, shared by every simulation
    structural model    two degrees of freedom, with its integrators
    LDNet               latent update and reconstruction of the loads
    linear model        linear unsteady aerodynamics, as alternative internal model
    MPC                 preview controller, solved by enumeration of the flap commands
    lidar               measurement model and fusion of the scans into the preview
    simulation          gust, closed-loop run and metrics
"""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.integrate import solve_ivp

tf.config.set_visible_devices([], 'GPU')
tf.keras.backend.set_floatx('float64')

#%% Parameters shared by every closed-loop simulation

U_INF = 80.0          # freestream velocity [m/s]
RHO = 1.225           # air density [kg/m^3]
CHORD = 1.0           # reference chord [m]
S_REF = 0.05          # reference surface [m^2]
Q_INF = 0.5 * RHO * U_INF**2 * S_REF   # dynamic pressure times S_ref [N]

M_WING = 19.24        # [kg]
I_WING = 1.155        # about the elastic axis [kg m^2]
K_H = 25460.0         # plunge stiffness [N/m]
D_H = 88.0            # plunge damping [N s/m]
K_ALPHA = 9530.0      # pitch stiffness [N m/rad]
D_ALPHA = 19.8        # pitch damping [N m s/rad]
X_EA = 0.40           # elastic axis from the leading edge [m]

M_FLAP = 1.19         # [kg]
I_FLAP_CG = 0.006     # about its centre of gravity [kg m^2]
X_HINGE = 0.779       # hinge from the leading edge [m]

DELTA_MAX = 14.0          # [deg]
DELTA_RATE_MAX = 300.0    # [deg/s]

DT = 0.002            # control step, equal to the reference time of the latent update [s]

# Structural state (h, hdot, alpha, alphadot) of the pre-gust checkpoint
# from which every co-simulation of the dataset starts
TRIM_STATE = (-6.49179e-3, 0.0, -8.76338e-4, 0.0)

#%% Structural model and its integrator
#
# The state is ordered as the input signals of the LDNet, x = (h, hdot, alpha, alphadot).
# The loads force the structure as in the co-simulation driver that generated the
# training data: the plunge equation carries -F_y and the pitch equation +M_z.
# The flap rotation is prescribed, and its inertial forcing f_delta is not included.

DX = X_HINGE - X_EA
I_FLAP_EA = I_FLAP_CG + M_FLAP * DX**2      # parallel-axis theorem
M_HH = M_WING + M_FLAP
M_AA = I_WING + I_FLAP_EA
M_HA = M_FLAP * DX
DET = M_HH * M_AA - M_HA**2


def acceleration(x, Fy, Mz):
    """Plunge and pitch accelerations of the state x under the loads (Fy, Mz)."""
    h, hd, a, ad = np.moveaxis(x, -1, 0)
    fh = -Fy - D_H * hd - K_H * h
    fa = Mz - D_ALPHA * ad - K_ALPHA * a
    return (M_AA * fh - M_HA * fa) / DET, (M_HH * fa - M_HA * fh) / DET


def derivative(x, Fy, Mz):
    """Time derivative of the state, the right-hand side of the structural equation."""
    hdd, add = acceleration(x, Fy, Mz)
    return np.stack([x[..., 1], hdd, x[..., 3], add], axis=-1)


def step(x, Fy, Mz, dt):
    """Advance the plant by dt with the loads held constant.

    Adaptive Dormand-Prince RK45 with the tolerances of the co-simulation driver.
    """
    sol = solve_ivp(lambda t, y: derivative(y, Fy, Mz), (0.0, dt), x,
                    method='RK45', rtol=1e-8, atol=1e-10, max_step=2 * dt)
    return sol.y[:, -1]


def step_fixed(x, Fy, Mz, dt):
    """One fixed step of the fifth-order Dormand-Prince update, on a batch of states.

    This is the integrator of the controller's prediction.
    """
    k1 = derivative(x, Fy, Mz)
    k2 = derivative(x + dt * (1 / 5 * k1), Fy, Mz)
    k3 = derivative(x + dt * (3 / 40 * k1 + 9 / 40 * k2), Fy, Mz)
    k4 = derivative(x + dt * (44 / 45 * k1 - 56 / 15 * k2 + 32 / 9 * k3), Fy, Mz)
    k5 = derivative(x + dt * (19372 / 6561 * k1 - 25360 / 2187 * k2
                              + 64448 / 6561 * k3 - 212 / 729 * k4), Fy, Mz)
    k6 = derivative(x + dt * (9017 / 3168 * k1 - 355 / 33 * k2 + 46732 / 5247 * k3
                              + 49 / 176 * k4 - 5103 / 18656 * k5), Fy, Mz)
    return x + dt * (35 / 384 * k1 + 500 / 1113 * k3 + 125 / 192 * k4
                     - 2187 / 6784 * k5 + 11 / 84 * k6)

#%% LDNet: latent update and reconstruction of the loads
#
# A trained model is a folder with the weights of the two networks and the
# config.json that stores the latent dimension, the decay coefficient and the
# normalisation bounds of every input and output. All methods act on a batch of
# states: the plant evaluates one, the controller the whole candidate grid.

SIGNALS = ('h', 'hd', 'a', 'ad', 'delta', 'W_gust')   # eta = (h, hdot, alpha, alphadot, delta, W)
BETA = 0.05                                           # output layer g(a) = (a^3 + beta a)/(1 + beta)


def network(n_in, width, depth, n_out):
    """Fully connected network with tanh hidden layers and a linear output."""
    net = tf.keras.Sequential([tf.keras.Input((n_in,))])
    for _ in range(depth):
        net.add(tf.keras.layers.Dense(width, activation='tanh'))
    net.add(tf.keras.layers.Dense(n_out))
    return net


def scale(v, bounds):
    """Affine map of [min, max] onto [-1, 1]."""
    return (2.0 * v - bounds['min'] - bounds['max']) / (bounds['max'] - bounds['min'])


class LDNet:

    def __init__(self, folder):
        folder = Path(folder)
        config = json.loads((folder / 'config.json').read_text())
        self.norm = config['normalization']
        self.ds = config['num_latent_states']
        self.decay = config['lambda_damp']
        self.dt_ref = self.norm['time']['time_constant']

        # 2 x 7 and 4 x 24 hidden neurons, the architecture of the control LDNet
        self.dyn = network(self.ds + 1 + len(SIGNALS), 7, 2, self.ds)
        self.rec = network(self.ds + len(SIGNALS) + 2, 24, 4, 2)
        self.dyn.load_weights(folder / 'NNdyn_weights.weights.h5')
        self.rec.load_weights(folder / 'NNrec_weights.weights.h5')

        # U_inf and the query point are fixed in every run
        self.u = scale(U_INF, self.norm['input_parameters']['U_inf'])
        space = self.norm['space']
        self.x_query = [scale(0.0, {'min': lo, 'max': hi})
                        for lo, hi in zip(space['min'], space['max'])]

        out = self.norm['output_signals']
        lo = np.array([out['F_y']['min'], out['M_z']['min']])
        hi = np.array([out['F_y']['max'], out['M_z']['max']])
        self.y0 = 0.5 * (hi + lo)
        self.yw = 0.5 * (hi - lo)

    def signals(self, x, delta, W):
        """Normalised input signals, one row for each state of the batch x."""
        x = np.atleast_2d(x)
        values = (x[:, 0], x[:, 1], x[:, 2], x[:, 3], delta, W)
        bounds = self.norm['input_signals']
        return np.stack([scale(np.broadcast_to(v, len(x)), bounds[k])
                         for v, k in zip(values, SIGNALS)], axis=1)

    def loads(self, s, eta):
        """Loads F_y [N] and M_z [N m] from the latent state s and the signals eta."""
        query = np.tile(self.x_query, (len(s), 1))
        a = self.rec(np.hstack([s, eta, query]), training=False).numpy()
        g = (a**3 + BETA * a) / (1.0 + BETA)
        y = self.y0 + self.yw * g
        return y[:, 0], y[:, 1]

    def step(self, s, eta):
        """Damped forward-Euler update of the latent state over one control step."""
        u = np.full((len(s), 1), self.u)
        rate = self.dyn(np.hstack([s, u, eta]), training=False).numpy()
        return (1.0 - self.decay) * s + (DT / self.dt_ref) * rate

#%% Linear unsteady aerodynamics
#
# Same interface as the LDNet, so that it replaces it inside the MPC: signals()
# returns the quasi-steady downwash of motion and gust, the deflection in radians
# and the apparent-mass term; the state s holds the four lag states z_1..z_4.

WAGNER = ((0.165, 0.0455), (0.335, 0.300))    # two-pole approximation of Jones, (A_i, b_i)
KUSSNER = ((0.5, 0.130), (0.5, 1.000))
X_34 = 0.75                                    # three-quarter chord, fraction of the chord


class LinearModel:

    def __init__(self, path=None):
        # gains of the loads; without them only signals, downwash and step are usable
        self.k = json.loads(Path(path).read_text()) if path else None
        self.ds = 4
        self.b = np.array([WAGNER[0][1], WAGNER[1][1], KUSSNER[0][1], KUSSNER[1][1]])

    def signals(self, x, delta, W):
        """Downwash of the motion and of the gust, deflection [rad], apparent-mass term."""
        x = np.atleast_2d(x)
        hd, a, ad = x[:, 1], x[:, 2], x[:, 3]
        w_m = a + hd / U_INF + (CHORD / U_INF) * (X_34 - X_EA / CHORD) * ad
        w_g = np.broadcast_to(W / U_INF, len(x))
        am = (np.pi * CHORD / (2.0 * U_INF)) * ad
        return np.stack([w_m, w_g, np.deg2rad(np.broadcast_to(delta, len(x))), am], axis=1)

    def downwash(self, s, eta):
        """Effective downwash of motion and gust after the wake lags."""
        (A1, B1), (A2, B2) = WAGNER
        (G1, C1), (G2, C2) = KUSSNER
        w_m = eta[:, 0] - A1 * B1 * s[:, 0] - A2 * B2 * s[:, 1]
        w_g = eta[:, 1] - G1 * C1 * s[:, 2] - G2 * C2 * s[:, 3]
        return w_m, w_g

    def loads(self, s, eta):
        """Loads F_y [N] and M_z [N m]."""
        k = self.k
        w_m, w_g = self.downwash(s, eta)
        d, am = eta[:, 2], eta[:, 3]
        CL = k['CL_a'] * w_m + k['CL_g'] * w_g + k['CL_d'] * d + k['AM_L'] * am + k['CL_0']
        CM = k['CM_a'] * w_m + k['CM_g'] * w_g + k['CM_d'] * d + k['AM_M'] * am + k['CM_0']
        return Q_INF * CL, Q_INF * CM * CHORD

    def step(self, s, eta, dt=DT):
        """Advance the lag states over dt, RK4 on dz_i/dt = (2 U / c) (w - b_i z_i)."""
        w = eta[:, [0, 0, 1, 1]]

        def f(z):
            return 2.0 * U_INF / CHORD * (-self.b * z + w)

        k1 = f(s)
        k2 = f(s + 0.5 * dt * k1)
        k3 = f(s + 0.5 * dt * k2)
        k4 = f(s + dt * k3)
        return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

#%% Preview MPC, solved by enumeration of the flap commands


class MPC:
    """Receding-horizon controller with the LDNet, or the linear model, as internal model.

    The deflection is held constant over the horizon of N steps and sought on a
    uniform grid of G candidates, all predicted at once. The controller keeps its
    own latent state, advanced at every step with the applied deflection and its
    estimate of the current gust: with an exact preview it coincides with the
    latent state of the plant.
    """

    def __init__(self, model, cl_trim, R, N=8, G=161):
        self.model = model
        self.cl_trim = cl_trim
        self.R = R
        self.N = N
        self.candidates = np.linspace(-DELTA_MAX, DELTA_MAX, G)
        self.reset()

    def reset(self):
        self.s = np.zeros((1, self.model.ds))
        self.delta = 0.0

    def __call__(self, x, preview, w_now):
        """Deflection [deg] for the state x, given the preview W_1..W_N and W at t_n."""
        net, dg = self.model, self.candidates

        s = np.repeat(self.s, len(dg), axis=0)
        xk = np.repeat(np.atleast_2d(x), len(dg), axis=0)
        J = self.R * dg**2
        for k in range(self.N):
            eta = net.signals(xk, dg, preview[k])
            Fy, Mz = net.loads(s, eta)
            J = J + (Fy / Q_INF - self.cl_trim)**2
            s = net.step(s, eta)
            xk = step_fixed(xk, Fy, Mz, DT)

        reach = DELTA_RATE_MAX * DT
        J[np.abs(dg - self.delta) > reach + 1e-9] = np.inf
        d = float(dg[np.argmin(J)])
        d = float(np.clip(d, max(-DELTA_MAX, self.delta - reach),
                          min(DELTA_MAX, self.delta + reach)))

        self.delta = d
        self.s = net.step(self.s, net.signals(x, d, w_now))
        return d

#%% Lidar measurement model and fusion of the scans into the preview


def constant_noise(sigma, J=50):
    """sigma_j = sigma at every node."""
    return np.full(J, float(sigma))


def range_noise(J=50):
    """Range-proportional noise: 1 m/s at the nearest node, 3 m/s at the farthest."""
    j = np.arange(1, J + 1)
    return 1.0 + 2.0 * (j - 1) / (J - 1)


class LidarPreview:
    """Gust preview fused from the repeated scans of a forward-looking lidar.

    field is the gust the lidar measures on the step grid: the true gust, or a
    biased or time-shifted copy of it for the study of systematic errors. At step
    n node j = 1..J returns field[n + j] plus an error of standard deviation
    sigma[j - 1]; with sigma = 0 the lidar is noise-free and the preview is the
    field itself. The scans of the J - 1 steps before t = 0 pre-fill the sums.
    """

    def __init__(self, field, sigma, rng=None, N=8, lam=0.0):
        self.J = len(sigma)
        self.N = N
        self.lam = lam
        self.rng = rng or np.random.default_rng()
        self.sigma = np.asarray(sigma, float)
        # inverse-variance weights; a noise-free lidar weighs its nodes equally
        self.weight = 1.0 / self.sigma**2 if np.all(self.sigma > 0) else np.ones(self.J)
        self.field = np.concatenate([field, np.zeros(self.J)])
        self.a = np.zeros(len(self.field))     # inverse-variance weighted sums a_m
        self.b = np.zeros(len(self.field))     # accumulated weights b_m
        for n in range(1 - self.J, 0):
            self.scan(n)

    def scan(self, n):
        """Add the measurements of the scan taken at step n to the sums of the inverse-variance fusion."""
        m = n + np.arange(1, self.J + 1)
        seen = m >= 0
        m = m[seen]
        y = self.field[m] + self.rng.normal(0.0, self.sigma[seen])
        self.a[m] += y * self.weight[seen]
        self.b[m] += self.weight[seen]

    def __call__(self, n):
        """Scan at step n; return the preview W_1..W_N and the fused estimate of W at t_n."""
        self.scan(n)
        now = max(0.0, self.a[n] / self.b[n]) if self.b[n] > 0 else 0.0

        a = self.a[n + 1:n + 1 + self.J]
        b = self.b[n + 1:n + 1 + self.J]
        y = np.divide(a, b, out=np.zeros_like(a), where=b > 0)
        if self.lam > 0:
            # weighted Whittaker smoother
            D = np.diff(np.eye(len(b)), 2, axis=0)
            y = np.linalg.solve(np.diag(b) + self.lam * b.mean() * (D.T @ D), b * y)
        return np.maximum(0.0, y[:self.N]), now

#%% Closed-loop simulation on the reduced model and metrics


def time_grid(t_end=3.0):
    return np.arange(int(round(t_end / DT)) + 1) * DT


def gust(t, W0, Tg):
    """Discrete '1 - cos' gust with onset at t = 0."""
    return np.where((t >= 0) & (t <= Tg), W0 / 2 * (1 - np.cos(2 * np.pi * t / Tg)), 0.0)


def trim_lift(model):
    """C_L,trim: lift coefficient of the model at the pre-gust state."""
    s = np.zeros((1, model.ds))
    Fy, _ = model.loads(s, model.signals(TRIM_STATE, 0.0, 0.0))
    return float(Fy[0]) / Q_INF


def simulate(ldnet, t, W, controller=None, lidar=None):
    """Run the reduced model under the gust W(t), with the flap commanded by controller.

    With no controller the flap stays at zero (open loop). The controller receives
    the preview fused from the scans of lidar, by default a noise-free lidar that
    measures the true gust. Each step n records the deflection, lift, accelerations and first preview
    value W_1 at t_n, and the structural state that the step reaches.
    """
    if controller is not None:
        controller.reset()
        if lidar is None:
            lidar = LidarPreview(W, constant_noise(0.0), N=controller.N)

    x = np.array(TRIM_STATE)
    s = np.zeros((1, ldnet.ds))
    keys = ('h', 'hd', 'alpha', 'alphad', 'hdd', 'alphadd', 'delta', 'CL', 'W1')
    out = {k: np.zeros(len(t)) for k in keys}

    for n in range(len(t)):
        delta, preview = 0.0, [np.nan]
        if controller is not None:
            preview, w_now = lidar(n)
            delta = controller(x, preview, w_now)

        eta = ldnet.signals(x, delta, W[n])
        Fy, Mz = ldnet.loads(s, eta)
        Fy, Mz = Fy[0], Mz[0]
        hdd, add = acceleration(x, Fy, Mz)
        s = ldnet.step(s, eta)
        x = step(x, Fy, Mz, DT)

        for k, v in zip(keys, (*x, hdd, add, delta, Fy / Q_INF, preview[0])):
            out[k][n] = v

    out['t'] = t
    out['W'] = W
    return out


def evaluate(closed, opened, Tg, cl_trim):
    """Peak lift-excursion reduction, peak deflection, pitch ratio and stability check.

    Everything is measured over the window t <= Tg + 0.5 s. A run is stable if
    alphadot, alphaddot and hddot stay within three times their open-loop peak.
    """
    w = closed['t'] <= Tg + 0.5

    def peak(run, key):
        return np.max(np.abs(run[key][w]))

    exc_open = np.max(np.abs(opened['CL'][w] - cl_trim))
    exc_closed = np.max(np.abs(closed['CL'][w] - cl_trim))
    return {
        'CLred': (exc_open - exc_closed) / exc_open * 100.0,
        'delta_max': peak(closed, 'delta'),
        'pitch_ratio': peak(closed, 'alpha') / peak(opened, 'alpha'),
        'stable': all(peak(closed, k) <= 3.0 * peak(opened, k) + 1e-9
                      for k in ('alphad', 'alphadd', 'hdd')),
    }
