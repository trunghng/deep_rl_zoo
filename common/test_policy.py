import time, argparse, json
import os.path as osp
from types import SimpleNamespace

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch

from common.mpi_utils import mpi_get_statistics
from common.utils import make_atari_env
from zoo.single.ddpg import DDPG
from zoo.single.dqn import DQN
from zoo.single.ppo import PPO
from zoo.single.sac import SAC
from zoo.single.trpo import TRPO
from zoo.single.vpg import VPG


def test(args) -> None:
    log_dir, eps, max_ep_len, render = args.log_dir, args.eps, args.max_ep_len, args.render

    with open(osp.join(log_dir, 'config.json')) as f:
        config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    algos = {
        'ddpg': DDPG, 'dqn': DQN,
        'ppo': PPO,'sac': SAC,
        'trpo': TRPO, 'vpg': VPG
    }
    model = algos[config.algo](config)
    model.load(osp.join(log_dir, 'model.pt'))
    render_mode = 'human' if render else None

    if hasattr(config, 'atari') and config.atari:
        env = make_atari_env(config.env, render_mode=render_mode)
    else:
        env = gym.make(config.env, render_mode=render_mode)
    returns, eps_len = [], []

    for ep in range(1, eps + 1):
        obs, _ = env.reset()
        rewards, step = [], 0

        while True:
            action = model.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            step += 1

            if terminated or truncated or step == max_ep_len:
                return_, ep_len = sum(rewards), len(rewards)
                returns.append(return_)
                eps_len.append(ep_len)
                print('Ep: %d\tReturn: %.4f\tEpLen: %.4f' % (ep, return_, ep_len))
                break
    env.close()
    avg_return, std_return = mpi_get_statistics(returns)
    avg_eplen, std_eplen = mpi_get_statistics(eps_len)
    print('AvgReturn: %.4f\tStdReturn: %.4f\nAvgEpLen: %.4f\tStdEpLen: %.4f'%
        (avg_return, std_return, avg_eplen, std_eplen))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Policy testing')
    parser.add_argument('--log-dir', type=str, required=True,
                        help='Path to the log directory, which stores model file, config file, etc')
    parser.add_argument('--eps', type=int, default=50,
                        help='Number of episodes')
    parser.add_argument('--max-ep-len', type=int, default=1000,
                        help='Maximum length of an episode')
    parser.add_argument('--render', action='store_true',
                        help='Whether to render the experiment')
    args = parser.parse_args()
    test(args)
