import sys
import subprocess
import os.path as osp


if __name__ == '__main__':
    algos = ['ddpg', 'ppo', 'trpo', 'vpg']
    utils = ['plot', 'test_policy']

    assert len(sys.argv) > 1, 'Invalid command'
    cmd = sys.argv[1]
    runner = sys.executable if sys.executable else 'python'

    n_runs = None
    if cmd in utils:
        runfile = osp.join(osp.abspath(osp.dirname(__file__)), 'common', cmd +'.py')
    else:
        runfile = osp.join(osp.abspath(osp.dirname(__file__)), 'zoo/pg', cmd +'.py')
        if len(sys.argv) > 2 and sys.argv[2] == '-n':
            n_runs = int(sys.argv[3])

    if n_runs:
        args = sys.argv[4:] if len(sys.argv) > 4 else []
        for seed in range(n_runs):
            subprocess.check_call([runner, runfile] + args + ['--seed', str(seed)])
    else:
        args = sys.argv[2:] if len(sys.argv) > 2 else []
        subprocess.check_call([runner, runfile] + args)
