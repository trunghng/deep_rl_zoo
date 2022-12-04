from vpg import VPG
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vanilla Policy Gradient')
    parser.add_argument('--env', type=str, choices=['CartPole-v0', 'Pendulum-v0'],
                        help='OpenAI enviroment')
    parser.add_argument('--eval', action='store_true',
                        help='Whether to enable evaluation')
    parser.add_argument('--model-path', type=str,
                        help='Model path to load')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--pi-lr', type=float,
                        help='Learning rate for policy optimization')
    parser.add_argument('--v-lr', type=float,
                        help='Learning rate for value function optimization')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs')
    parser.add_argument('--steps-per-epoch', type=int,
                        help='Maximum number of epoch for each epoch')
    parser.add_argument('--train-v-iters', type=int,
                        help='Value network update frequency')
    parser.add_argument('--max-ep-len', type=int,
                        help='Maximum episode/trajectory length')
    parser.add_argument('--gamma', type=float,
                        help='Discount factor')
    parser.add_argument('--lamb', type=float,
                        help='Eligibility trace')
    parser.add_argument('--goal', type=int,
                        help='Goal total reward to end training')
    parser.add_argument('--save', action='store_true',
                        help='Whether to save training model')
    parser.add_argument('--render', action='store_true',
                        help='Whether to save training result as video')
    parser.add_argument('--plot', action='store_true',
                        help='Whether to plot training statistics and save as image')
    args = parser.parse_args()
    agent = VPG(args)
    if args.eval:
        agent.load(args.model_path)
        agent.test()
    else:
        agent.train()
