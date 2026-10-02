"""Controller of the full-order closed loop.

The co-simulation driver starts this script once and exchanges one JSON line per
coupling window: it sends {"state": [h, hdot, alpha, alphadot], "wseq": [W_1..W_N],
"wnow": W} and receives {"delta": deflection in degrees}. The controller is the
MPC of the reduced closed loop, src/aeroelastic.py, with the LDNet as internal
model, or the linear unsteady model with --linear.

    python mpc_server.py --R 1e-4
    python mpc_server.py --R 1e-2 --linear ../models/linear_model.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import aeroelastic as ae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=ROOT / 'models' / 'control_ldnet')
    parser.add_argument('--linear', help='coefficients of the linear model, instead of the LDNet')
    parser.add_argument('--R', type=float, required=True, help='effort weight [deg^-2]')
    parser.add_argument('--N', type=int, default=8, help='horizon [steps]')
    args = parser.parse_args()

    model = ae.LinearModel(args.linear) if args.linear else ae.LDNet(args.model)
    cl_trim = ae.trim_lift(model)
    mpc = ae.MPC(model, cl_trim, args.R, args.N)
    print(json.dumps({'ready': True, 'CLTRIM': cl_trim}), flush=True)

    for line in sys.stdin:
        request = json.loads(line)
        if request.get('cmd') == 'quit':
            break
        if request.get('cmd') == 'reset':
            mpc.reset()
            reply = {'ok': True}
        else:
            reply = {'delta': mpc(request['state'], request['wseq'], request['wnow'])}
        print(json.dumps(reply), flush=True)


if __name__ == '__main__':
    main()
