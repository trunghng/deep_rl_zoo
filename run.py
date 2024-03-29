import sys
import subprocess
import os.path as osp


if __name__ == '__main__':
    sa_algos = ['ddpg', 'ppo', 'sac', 'trpo', 'vpg', 'dqn']
    ma_algos = ['maddpg']
    utils = ['plot', 'test_policy']

    assert len(sys.argv) > 1, 'Invalid command'
    cmd = sys.argv[1]
    runner = sys.executable if sys.executable else 'python'

    n_runs = None
    if cmd in utils:
        runfile = osp.join(osp.abspath(osp.dirname(__file__)), 'common', cmd +'.py')
    else:
        if cmd in sa_algos:
            runfile = osp.join(osp.abspath(osp.dirname(__file__)), 'zoo/single', cmd +'.py')
        elif cmd in ma_algos:
            runfile = osp.join(osp.abspath(osp.dirname(__file__)), 'zoo/multi', cmd +'.py')
        if len(sys.argv) > 2 and sys.argv[2] == '-n':
            n_runs = int(sys.argv[3])

    if n_runs:
        args = sys.argv[4:] if len(sys.argv) > 4 else []
        for seed in range(n_runs):
            subprocess.check_call([runner, runfile] + args + ['--seed', str(seed)])
    else:
        args = sys.argv[2:] if len(sys.argv) > 2 else []
        subprocess.check_call([runner, runfile] + args)