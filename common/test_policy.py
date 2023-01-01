import gym
import torch
import time, argparse
from common.mpi_utils import mpi_get_statistics


def load_policy(path):
    policy = torch.load(path)

    def get_action(obs):
        return policy.act(obs)
    return policy, get_action


def run_policy(get_action, args):
    env_id, n_eps, max_ep_len, render = args.env, args.n_eps, args.max_ep_len, args.render
    print(f'Testing policy on {env_id} in {n_eps} episodes with maximum length of {max_ep_len}')
    env = gym.make(env_id)
    returns, eps_len = [], []

    for ep in range(1, n_eps + 1):
        obs = env.reset()
        rewards, step = [], 0

        while True:
            if render:
                env.render()
                time.sleep(1e-3)

            action = get_action(obs)
            obs, reward, terminated, _ = env.step(action)
            rewards.append(reward)
            step += 1

            if terminated or step == max_ep_len:
                return_, ep_len = sum(rewards), len(rewards)
                returns.append(return_)
                eps_len.append(ep_len)
                print('Ep: %d\tReturn: %.4f\tEpLen: %.4f' % (ep, return_, ep_len))
                break
    avg_return, std_return = mpi_get_statistics(returns)
    avg_eplen, std_eplen = mpi_get_statistics(eps_len)
    print('AvgReturn: %.4f\tStdReturn: %.4f\nAvgEpLen: %.4f\tStdEpLen: %.4f'%
        (avg_return, std_return, avg_eplen, std_eplen))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Policy Testing')
    parser.add_argument('-e', '--env', type=str, choices=['CartPole-v0', 'HalfCheetah-v2'],
                        help='Environment ID')
    parser.add_argument('-p', '--path', type=str,
                        help='Model path')
    parser.add_argument('-n', '--n-eps', type=int, default=50,
                        help='Number of episodes')
    parser.add_argument('-l', '--max-ep-len', type=int, default=1000,
                        help='Maximum length of an episode')
    parser.add_argument('-r', '--render', action='store_true',
                        help='Whether to render the experiment')
    args = parser.parse_args()

    policy, get_action = load_policy(args.path)
    run_policy(get_action, args)
