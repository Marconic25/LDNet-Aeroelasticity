"""Partitioned co-simulation of the full-order model .

The fluid is solved by pimpleFoam on the case of case/, the structure by this
driver. Over each co-simulation window the driver predicts the structural motion
by the first-order hold, writes it to the motion tables read by the mesh-motion
library, runs pimpleFoam over the window, reads back F_y and M_z and integrates the
structural equation with the adaptive RK45 pair to the end of the window.

The flap deflection is zero, a table of knots (the ramp of family B of the campaign),
the feed-forward law of family C, or the command of the MPC of mpc_server.py, planned
again at every window on the structural state of the co-simulation (closed loop on the full-order model).

    python cosim_driver.py --case run/ --np 16 --from-checkpoint --gust 20 0.7
    python cosim_driver.py --case run/ --np 16 --from-checkpoint --flap-table 0 0 0.055 -8.2 3 0
    python cosim_driver.py --case run/ --np 16 --from-checkpoint --gust 20 0.7 --feed-forward 0.1 0.05 120
    python cosim_driver.py --case run/ --np 16 --from-checkpoint --gust 20 0.7 --mpc 1e-4 --d-alpha 19.8

Each run writes structural_trajectory.csv: time from the start of the run, structural
state, loads F_y and M_z, gust and deflection, every four fluid steps.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# Structural parameters. The pitch damping of the dataset campaign is
# 6.6 N m s/rad; the closed loops use 19.8 (--d-alpha).
M_WING, I_WING = 19.24, 1.155
K_H, D_H = 25460.0, 88.0
K_ALPHA = 9530.0
M_FLAP, I_FLAP_CG = 1.19, 0.006
EA_X, EA_Y = 0.40, 0.0            # elastic axis in the undeformed mesh
HINGE_X, HINGE_Y = 0.779, 0.0     # flap hinge in the undeformed mesh

U_INF = 80.0
DELTA_REF_MAX = 20.0              # saturation of the feed-forward law [deg]
MPC_DT = 0.002                    # step of the controller's preview [s]

# Mass matrix, with the flap carried rigidly by the wing
DX, DY = HINGE_X - EA_X, HINGE_Y - EA_Y
M_HH = M_WING + M_FLAP
M_AA = I_WING + (I_FLAP_CG + M_FLAP * (DX**2 + DY**2))    # flap inertia by the parallel-axis theorem
M_HA = M_FLAP * DX
DET = M_HH * M_AA - M_HA * M_HA


#%% Gust and flap deflection, on the time of the run

def gust(t, W0, Tg):
    """Discrete '1 - cos' gust, from t = 0 to t = Tg."""
    if 0.0 <= t <= Tg:
        return W0 / 2.0 * (1.0 - np.cos(2.0 * np.pi * t / Tg))
    return 0.0


def flap_table(knots):
    """Deflection interpolated linearly between the knots t0 delta0 t1 delta1 ..., held after the last."""
    return lambda t: float(np.interp(t, knots[0::2], knots[1::2]))


def flap_feed_forward(K_ff, t_d, rate, W, dt, t_end):
    """Family C: delta = -K_ff W(t - t_d), saturated and then rate limited step by step."""
    t = np.arange(0.0, t_end + dt, dt)
    delta = np.zeros(len(t))
    for k in range(len(t)):
        ref = np.clip(-K_ff * W(max(0.0, t[k] - t_d)), -DELTA_REF_MAX, DELTA_REF_MAX)
        delta[k] = ref if k == 0 else delta[k - 1] + np.clip(ref - delta[k - 1], -rate * dt, rate * dt)
    return lambda s: float(np.interp(s, t, delta))


#%% Structural model
#
# The loads force the structure with -F_y on the plunge and +M_z on the pitch, the
# convention of the whole campaign. The inertial forcing f_delta of the prescribed
# flap rotation is not included.

def integrate_structure(state, t_win, Fy, Mz, d_alpha):
    """Adaptive RK45 over the window, with the loads interpolated linearly in time."""
    Fy_t = interp1d(t_win, Fy, kind='linear', fill_value='extrapolate')
    Mz_t = interp1d(t_win, Mz, kind='linear', fill_value='extrapolate')

    def rhs(t, x):
        h, hd, a, ad = x
        fh = -float(Fy_t(t)) - D_H * hd - K_H * h
        fa = float(Mz_t(t)) - d_alpha * ad - K_ALPHA * a
        return [hd, (M_AA * fh - M_HA * fa) / DET, ad, (M_HH * fa - M_HA * fh) / DET]

    sol = solve_ivp(rhs, [t_win[0], t_win[-1]], state, t_eval=t_win,
                    max_step=2 * (t_win[1] - t_win[0]), rtol=1e-8, atol=1e-10)
    return sol.y


#%% Motion tables of the mesh-motion library

def write_motion_table(path, t, tx, ty, rz_deg):
    """tabulated6DoFMotion table: translation and rotation about z of a body."""
    with open(path, 'w') as f:
        f.write('(\n')
        for row in zip(t, tx, ty, rz_deg):
            f.write('  (%.10e  ((%.10e %.10e 0) (0 0 %.10e)))\n' % row)
        f.write(')\n')


def motion_tables(h, alpha, delta_deg):
    """Wing and flap motion from the undeformed mesh.

    The wing translates by h and rotates by alpha about the elastic axis; the flap
    follows the hinge carried by the wing and rotates by alpha + delta about it.
    A positive delta deflects the trailing edge downwards, a negative rotation of the
    flap zone in the mesh.
    """
    d = -np.radians(delta_deg)
    hinge_x = np.cos(alpha) * DX - np.sin(alpha) * DY + EA_X
    hinge_y = np.sin(alpha) * DX + np.cos(alpha) * DY + EA_Y + h
    return (np.degrees(alpha), hinge_x - HINGE_X, hinge_y - HINGE_Y, np.degrees(alpha + d))


#%% OpenFOAM case

class Case:

    def __init__(self, path, container, bind):
        self.path = Path(path).resolve()
        self.container = container
        self.bind = bind
        self.checkpoint = self.path / 'checkpoint'

    def run(self, command, log=None):
        """Run an OpenFOAM command in the case, inside the container if one is given."""
        if self.container:
            cmd = ['apptainer', 'exec', '--bind', self.bind, self.container, '/bin/bash', '-c',
                   'source /opt/openfoam7/etc/bashrc && cd %s && %s' % (self.path, command)]
        else:
            cmd = ['/bin/bash', '-c', 'cd %s && %s' % (self.path, command)]
        if log is None:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(result.stdout[-2000:], result.stderr[-2000:])
        else:
            with open(self.path / log, 'a') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise RuntimeError('%s failed' % command.split()[0])

    def processors(self):
        return sorted(self.path.glob('processor*'))

    def control_dict(self, start, end, latest_time, write_interval=None):
        """Set the time interval of the next pimpleFoam run in system/controlDict."""
        path = self.path / 'system' / 'controlDict'
        text = path.read_text()
        text = re.sub(r'startFrom[^\n;]*;', 'startFrom       %s;' % ('latestTime' if latest_time else 'startTime'), text)
        text = re.sub(r'(startTime\s+)[\d.eE+\-]+\s*;', r'\g<1>%.10e;' % start, text)
        text = re.sub(r'stopAt[^\n;]*;', 'stopAt          endTime;', text)
        text = re.sub(r'(endTime\s+)[\d.eE+\-]+\s*;', r'\g<1>%.10e;' % end, text)
        if write_interval is not None:
            # only the writeInterval of the run, not those of the function objects
            pre, sep, post = text.partition('functions')
            text = re.sub(r'(writeInterval\s+)[\d.eE+\-]+\s*;', r'\g<1>%s;' % write_interval, pre, count=1) + sep + post
        path.write_text(text)

    def adjustable_time_step(self, max_co, max_dt=5e-4):
        """Fluid step adapted to the Courant number max_co, capped at max_dt.

        The fields are written at the end of each window by run time, and the loads
        at every fluid step, since a window of long adapted steps holds few samples.
        """
        path = self.path / 'system' / 'controlDict'
        pre, sep, post = path.read_text().partition('functions')
        pre = re.sub(r'adjustTimeStep\s+\w+\s*;', 'adjustTimeStep  yes;', pre)
        pre = re.sub(r'maxDeltaT\s+[^;]*;\n?', '', pre)
        pre = re.sub(r'maxCo\s+[\d.eE+\-]+\s*;', 'maxCo           %g;\nmaxDeltaT       %g;' % (max_co, max_dt), pre)
        pre = re.sub(r'writeControl\s+\w+\s*;', 'writeControl    adjustableRunTime;', pre, count=1)
        post = re.sub(r'(writeInterval\s+)\d+\s*;', r'\g<1>1;', post)
        path.write_text(pre + sep + post)

    def write_gust_inlet(self, W0, t0, Tg):
        """Inlet velocity U_inf + W(t) as a codedFixedValue condition."""
        (self.path / '0.orig' / 'include' / 'fixedInletU').write_text(
            'inlet\n{\n'
            '    type        codedFixedValue;\n'
            '    value       uniform (%.4f 0 0);\n' % U_INF +
            '    name        gustInlet;\n'
            '    code\n    #{\n'
            '        const scalar t    = this->db().time().value();\n'
            '        const scalar U0   = %.4f;\n' % U_INF +
            '        const scalar Wg0  = %.4f;\n' % W0 +
            '        const scalar t0   = %.6f;\n' % t0 +
            '        const scalar Tg   = %.6f;\n' % Tg +
            '        scalar wy = 0.0;\n'
            '        if (t >= t0 && t <= t0 + Tg)\n        {\n'
            '            wy = 0.5*Wg0*(1.0 - Foam::cos(2.0*Foam::constant::mathematical::pi*(t - t0)/Tg));\n'
            '        }\n'
            '        operator==(vector(U0, wy, 0.0));\n'
            '    #};\n}\n')

    def patch_gust(self, W0, t0, Tg):
        """Set the gust of a restored checkpoint: the coded inlet is stored in the U fields."""
        for proc in self.processors():
            for time_dir in proc.iterdir():
                u = time_dir / 'U'
                if not u.exists():
                    continue
                text = u.read_bytes().decode('latin-1')
                text = re.sub(r'const scalar Wg0\s*=\s*[\d.eE+\-]+\s*;', 'const scalar Wg0  = %.4f;' % W0, text)
                text = re.sub(r'const scalar t0\s*=\s*[\d.eE+\-]+\s*;', 'const scalar t0   = %.6f;' % t0, text)
                text = re.sub(r'const scalar Tg\s*=\s*[\d.eE+\-]+\s*;', 'const scalar Tg   = %.6f;' % Tg, text)
                u.write_bytes(text.encode('latin-1'))

    def reset(self):
        """Remove the results of previous runs and copy 0.orig to 0."""
        for path in [self.path / 'log.pimpleFoam', self.path / 'postProcessing', *self.processors()]:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        for d in self.path.iterdir():
            if d.is_dir() and re.match(r'^\d+\.?\d*$', d.name) and d.name != '0':
                shutil.rmtree(d)
        shutil.rmtree(self.path / '0', ignore_errors=True)
        shutil.copytree(self.path / '0.orig', self.path / '0')

    def relax_match_tolerance(self):
        # the flap zone straddles processor boundaries: after its rigid motion the
        # face-area check between processors needs a looser tolerance
        for proc in self.processors():
            boundary = proc / 'constant' / 'polyMesh' / 'boundary'
            if boundary.exists():
                boundary.write_text(re.sub(r'matchTolerance\s+[\d.eE+\-]+\s*;', 'matchTolerance  1e-3;', boundary.read_text()))

    def decompose(self, dt):
        self.control_dict(0.0, dt, latest_time=False)
        self.run('decomposePar -force')
        self.relax_match_tolerance()

    def read_forces(self, t_start, t_end):
        """F_y and M_z, pressure plus viscous, for t in (t_start, t_end].

        pimpleFoam writes a forces.dat per run, in a folder named after its start time.
        """
        files = sorted((self.path / 'postProcessing' / 'forces').glob('*/forces.dat'))
        starts = sorted(float(f.parent.name) for f in files)
        t, Fy, Mz = [], [], []
        for i, file in enumerate(files):
            next_start = starts[i + 1] if i + 1 < len(starts) else t_end + 1.0
            for line in open(file):
                if line.startswith('#'):
                    continue
                v = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
                if len(v) < 19:
                    continue
                tk = float(v[0])
                if starts[i] - 1e-12 <= tk < next_start - 1e-12 and t_start < tk <= t_end + 1e-12:
                    t.append(tk)
                    Fy.append(float(v[2]) + float(v[5]))
                    Mz.append(float(v[12]) + float(v[15]))
        order = np.argsort(t)
        return np.array(t)[order], np.array(Fy)[order], np.array(Mz)[order]

    def seed_point_displacement(self, t):
        # pimpleFoam does not write pointDisplacement at the end of a window:
        # copy it into the time folder the next window starts from
        for proc in self.processors():
            times = [(abs(float(d.name) - t), d) for d in proc.iterdir() if d.is_dir() and is_number(d.name)]
            target = min(times)[1] / 'pointDisplacement'
            source = proc / '0' / 'pointDisplacement'
            if source.exists() and source.resolve() != target.resolve():
                shutil.copy2(source, target)

    def save_checkpoint(self, t, state, dt):
        """Store the fluid fields and the structural state at t, as the start of later runs."""
        self.checkpoint.mkdir(exist_ok=True)
        for proc in self.processors():
            latest = max((float(d.name), d) for d in proc.iterdir() if d.is_dir() and is_number(d.name))
            shutil.rmtree(self.checkpoint / proc.name, ignore_errors=True)
            shutil.copytree(proc / 'constant', self.checkpoint / proc.name / 'constant')
            shutil.copytree(latest[1], self.checkpoint / proc.name / ('%.6g' % latest[0]))
        h, hd, a, ad = state
        json.dump({'h': h, 'hd': hd, 'a': a, 'ad': ad, 'dt': dt, 't_checkpoint_orig': t},
                  open(self.checkpoint / 'cosim_state.json', 'w'), indent=2)

    def load_checkpoint(self):
        """Restore the fluid fields of the checkpoint; return its time and structural state."""
        for proc in self.processors():
            shutil.rmtree(proc)
        for proc in sorted(self.checkpoint.glob('processor*')):
            shutil.copytree(proc, self.path / proc.name)
        self.relax_match_tolerance()
        s = json.load(open(self.checkpoint / 'cosim_state.json'))
        return s['t_checkpoint_orig'], np.array([s['h'], s['hd'], s['a'], s['ad']])


def is_number(name):
    try:
        float(name)
        return True
    except ValueError:
        return False


def static_equilibrium(rans):
    """Plunge and pitch under the steady loads of the RANS solution."""
    files = sorted((rans / 'postProcessing' / 'forces').glob('*/forces.dat'))
    last = [re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line) for line in open(files[-1]) if not line.startswith('#')][-1]
    Fy, Mz = float(last[2]) + float(last[5]), float(last[12]) + float(last[15])
    return -Fy / K_H, Mz / K_ALPHA


#%% Controller of the closed loop

class Controller:
    """mpc_server.py run as a child process, one JSON line per co-simulation window."""

    def __init__(self, R, model, linear, container, bind):
        server = Path(__file__).resolve().parent / 'mpc_server.py'
        args = '--R %g %s' % (R, '--linear %s' % linear if linear else '--model %s' % model if model else '')
        # one thread each for TensorFlow, which shares the cores with pimpleFoam
        command = 'OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 python3 -u %s %s' % (server, args)
        cmd = ['/bin/bash', '-c', command]
        if container:
            cmd = ['apptainer', 'exec', '--bind', bind, container] + cmd
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        if not json.loads(self.process.stdout.readline()).get('ready'):
            raise RuntimeError('the MPC server did not start')

    def __call__(self, state, preview, w_now):
        request = {'state': [float(v) for v in state], 'wseq': preview, 'wnow': w_now}
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        return float(json.loads(self.process.stdout.readline())['delta'])

    def close(self):
        self.process.stdin.write(json.dumps({'cmd': 'quit'}) + '\n')
        self.process.stdin.flush()
        self.process.wait(timeout=30)


#%% Co-simulation

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--case', default='.', help='OpenFOAM case, a copy of case/ with its mesh')
    parser.add_argument('--rans', help='steady RANS solution of the trim state (fresh start only)')
    parser.add_argument('--np', type=int, default=16, help='MPI processes of pimpleFoam')
    parser.add_argument('--window', type=int, default=50, help='fluid steps per co-simulation window')
    parser.add_argument('--dt', type=float, default=7e-5, help='fluid time step [s]')
    parser.add_argument('--t-end', type=float, default=3.0, help='duration of the run [s]')
    parser.add_argument('--from-checkpoint', action='store_true', help='start from the pre-gust checkpoint')
    parser.add_argument('--save-checkpoint', type=float, help='store a checkpoint at this time [s]')
    parser.add_argument('--max-courant', type=float, help='fluid step adapted to this Courant number; the window stays WINDOW x DT long')
    parser.add_argument('--gust', type=float, nargs=2, default=(0.0, 1.0), metavar=('W0', 'TG'), help='gust peak [m/s] and duration [s]')
    parser.add_argument('--gust-onset', type=float, default=0.0, help='start of the gust from the start of the run [s]')
    parser.add_argument('--flap-table', type=float, nargs='+', metavar='T DELTA', help='knots of the deflection [s, deg], as in family B')
    parser.add_argument('--feed-forward', type=float, nargs=3, metavar=('K_FF', 'T_D', 'RATE'), help='family C [deg s/m, s, deg/s]')
    parser.add_argument('--mpc', type=float, metavar='R', help='closed loop with the MPC at this effort weight [deg^-2]')
    parser.add_argument('--model', help='LDNet of the MPC (default models/control_ldnet)')
    parser.add_argument('--linear', help='coefficients of the linear model, as internal model of the MPC')
    parser.add_argument('--d-alpha', type=float, default=6.6, help='pitch damping [N m s/rad]')
    parser.add_argument('--container', help='OpenFOAM v7 Apptainer image, if OpenFOAM is not installed')
    parser.add_argument('--tf-container', help='TensorFlow Apptainer image for the MPC, if needed')
    parser.add_argument('--bind', default='/work', help='path bound inside the containers')
    args = parser.parse_args()

    case = Case(args.case, args.container, args.bind)
    W0, Tg = args.gust
    onset = args.gust_onset
    dt, t_end = args.dt, args.t_end
    window_dt = args.window * dt

    # gust and flap deflection on the time of the run
    W = lambda t: gust(t - onset, W0, Tg)
    controller = None
    command = [0.0]
    if args.mpc is not None:
        controller = Controller(args.mpc, args.model, args.linear, args.tf_container, args.bind)
        flap = lambda t: command[0]
    elif args.flap_table:
        flap = flap_table(args.flap_table)
    elif args.feed_forward:
        flap = flap_feed_forward(*args.feed_forward, W, dt, t_end)
    else:
        flap = lambda t: 0.0

    # t0 is the time of the fluid solution at the start of the run
    if args.from_checkpoint:
        t0 = json.load(open(case.checkpoint / 'cosim_state.json'))['t_checkpoint_orig']
        case.write_gust_inlet(W0, t0 + onset, Tg)
        case.reset()
        t0, state = case.load_checkpoint()
        case.control_dict(t0, dt, latest_time=True)
        case.patch_gust(W0, t0 + onset, Tg)
        case.run('topoSet')
    else:
        t0 = 0.0
        case.write_gust_inlet(W0, onset, Tg)
        case.reset()
        case.run('mapFields %s -sourceTime latestTime -consistent' % Path(args.rans).resolve())
        unmapped = case.path / '0' / 'pointDisplacement.unmapped'
        if unmapped.exists():
            unmapped.rename(case.path / '0' / 'pointDisplacement')
        case.run('topoSet')
        h, a = static_equilibrium(Path(args.rans))
        state = np.array([h, 0.0, a, 0.0])

    # the mesh starts at the initial position
    for body in ['wingMotion.dat', 'flapMotion.dat']:
        write_motion_table(case.path / 'constant' / body, [t0, t0 + t_end + window_dt],
                           [0, 0], [state[0]] * 2, [np.degrees(state[2])] * 2)
    if not args.from_checkpoint:
        case.decompose(dt)
    if args.max_courant:
        case.adjustable_time_step(args.max_courant)

    t_cur, n_window, record = t0, 0, []
    while t_cur < t0 + t_end - 1e-12:
        t_next = min(t_cur + window_dt, t0 + t_end)
        t_win = np.arange(t_cur, t_next + dt * 0.5, dt)
        if len(t_win) < 2:
            break
        s = t_cur - t0

        if controller is not None:
            # one decision per window, held over it
            preview = [W(s + (k + 1) * MPC_DT) for k in range(8)]
            command[0] = controller(state, preview, W(s))

        # first-order hold of the structural motion over the window
        h_hold = state[0] + state[1] * (t_win - t_cur)
        a_hold = state[2] + state[3] * (t_win - t_cur)
        wing_rz, flap_tx, flap_ty, flap_rz = motion_tables(h_hold, a_hold, np.array([flap(t - t0) for t in t_win]))
        # the tables cover the time before the window and after the end of the run
        t_table = np.concatenate([[0.0] if t_cur > 0 else [], t_win, [t0 + t_end + window_dt]])
        pad = lambda v: np.concatenate([[v[0]] if t_cur > 0 else [], v, [v[-1]]])
        write_motion_table(case.path / 'constant' / 'wingMotion.dat', t_table, np.zeros(len(t_table)), pad(h_hold), pad(wing_rz))
        write_motion_table(case.path / 'constant' / 'flapMotion.dat', t_table, pad(flap_tx), pad(flap_ty), pad(flap_rz))

        # fields written at the end of the window, by steps or, with an adapted step, by run time
        write_interval = '%.10e' % (t_next - t_cur) if args.max_courant else '%d' % (len(t_win) - 1)
        case.control_dict(t_cur, t_next, latest_time=n_window > 0 or args.from_checkpoint, write_interval=write_interval)
        case.run('mpirun --oversubscribe -np %d pimpleFoam -parallel' % args.np, log='log.pimpleFoam')
        case.seed_point_displacement(t_next)

        # loads over the window, without the first samples of the restart of pimpleFoam
        t_f, Fy_f, Mz_f = case.read_forces(t_cur, t_next)
        skip = 10 if n_window == 0 else 2
        t_f, Fy_f, Mz_f = t_f[skip:], Fy_f[skip:], Mz_f[skip:]
        Fy = np.interp(t_win, t_f, Fy_f)
        Mz = np.interp(t_win, t_f, Mz_f)

        trajectory = integrate_structure(state, t_win, Fy, Mz, args.d_alpha)
        state = trajectory[:, -1]
        print('window %4d  t = %.4f s  h = %.3f mm  alpha = %.3f deg  delta = %.2f deg'
              % (n_window, t_next - t0, 1e3 * state[0], np.degrees(state[2]), flap(t_next - t0)))

        # every fourth fluid step, after the first samples of the window
        first = 10 if n_window == 0 and not args.from_checkpoint else 2
        for i in range(first, len(t_win), 4):
            record.append((t_win[i] - t0, *trajectory[:, i], Fy[i], Mz[i], W(t_win[i] - t0), flap(t_win[i] - t0)))

        t_cur = t_next
        n_window += 1
        if args.save_checkpoint is not None and t_cur >= args.save_checkpoint and not (case.checkpoint / 'cosim_state.json').exists():
            case.save_checkpoint(t_cur, state, dt)

    if controller is not None:
        controller.close()

    # the recorded loads are smoothed by a Gaussian kernel of 15 samples, 0.021 s
    record = np.array(record)
    record[:, 5] = gaussian_filter1d(record[:, 5], sigma=15)
    record[:, 6] = gaussian_filter1d(record[:, 6], sigma=15)
    np.savetxt(case.path / 'structural_trajectory.csv', record, delimiter=',', comments='',
               header='t,h,hd,alpha,ad,Fy,Mz,W_gust,delta', fmt=['%.8e'] * 5 + ['%.6f'] * 4)


if __name__ == '__main__':
    main()
