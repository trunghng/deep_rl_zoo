import sys
from os.path import dirname, join, realpath
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, '..'))
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import gym
import cmds
from agent import DQN


def main():
    args = cmds.get_args()

    if args.atari:
        env = make_atari(args.env_name)
        env = wrap_deepmind(env)
        env = wrap_pytorch(env)
        state_dim =  env.observation_space.shape
        n_actions = env.action_space.n
    else:
        env = gym.make(args.env_name)
        state_dim =  env.observation_space.shape[0]
        n_actions = env.action_space.n

    agent = DQN(env, state_dim, n_actions, args)
    if args.eval:
        agent.test(args.model_path)
    else:
        agent.train()


if __name__ == '__main__':
    main()
