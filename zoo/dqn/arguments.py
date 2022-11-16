import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(description='Deep Q-Learning', 
                                    formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-a', '--agent', type=str, default='DQN',
                        help='Agent name')
    parser.add_argument('-e', '--env-name', type=str, default='PongNoFrameskip-v4',
                        help='Enviroment name')
    parser.add_argument('-ei', '--epsilon-init', type=float, default=1.0,
                        help='Initial value of epsilon')
    parser.add_argument('-ef', '--epsilon-final', type=float, default=0.01,
                        help='Final value of epsilon')
    parser.add_argument('-ed', '--epsilon-decay', type=float, default=100,
                        help='Final value of epsilon')
    parser.add_argument('-g', '--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('-lr', '--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('-bs', '--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('-rs', '--buffer-size', type=int, default=100000,
                        help='Maximum memory buffer size')
    parser.add_argument('-t', '--train-freq', type=int, default=1,
                        help='Number of steps between optimization steps')
    parser.add_argument('-u', '--update-target', type=int, default=1000,
                        help='Interval of target network update')
    parser.add_argument('--tau', type=float, default=0.001,
                        help = 'Soft update parameter')
    parser.add_argument('-eps', '--num-episodes', type=int, default=600,
                        help = 'Maximum number of episodes')
    parser.add_argument('-l', '--logging-interval', type=int, default=10,
                        help = 'Interval of score tracking')
    parser.add_argument('-te', '--termination', type=int, default=17,
                        help = 'Terminal score')
    parser.add_argument('-v', '--print-freq', type=int, default=10000,
                        help='Result display interval')
    parser.add_argument('-s', '--save-model', action='store_true',
                        help='To save the model after training')
    parser.add_argument('-r', '--render-video', action='store_true',
                        help='To render video output after training')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='To plot the result and save')
    parser.add_argument('--atari', action='store_true',
                        help='To use atari environment')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('-c', '--use-cuda', action='store_true',
                        help='To enable CUDA training')
    parser.add_argument('--eval', action='store_true',
                        help='To enable evaluation')
    parser.add_argument('-m', '--model-path', type=str,
                        help='Model path to load')
    parser.add_argument('-d', '--default-env', type=int, choices=[0, 1, 2],
        help='Default experiments with default settings:\n 0: None\n 1: LunarLander-v2\n 2: PongNoFrameskip-v4')

    args = parser.parse_args()

    if args.default_env == 1:
        args.agent = 'DQN'
        args.env_name = 'LunarLander-v2'
        args.epsilon_init = 1
        args.epsilon_final = 0.01
        args.epsilon_decay = 100
        args.gamma = 0.99
        args.lr = 0.0001
        args.buffer_size = 100000
        args.batch_size = 64
        args.train_freq = 4
        args.update_target = 1000
        args.tau = 1e-3
        args.num_episodes = 1500
        args.logging_interval = 100
        args.termination = 200
        args.print_freq = 100
        args.save_model = True
        args.render_video = True
        args.plot = True
        args.atari = False
        args.seed = 1
    elif args.default_env == 2:
        args.agent = 'DQN'
        args.env_name = 'PongNoFrameskip-v4'
        args.epsilon_init = 1
        args.epsilon_final = 0.01
        args.epsilon_decay = 100
        args.gamma = 0.99
        args.lr = 0.0001
        args.buffer_size = 100000
        args.batch_size = 64
        args.train_freq = 4
        args.update_target = 1000
        args.tau = 1e-3
        args.num_episodes = 600
        args.logging_interval = 10
        args.termination = 17
        args.print_freq = 5
        args.save_model = True
        args.render_video = True
        args.plot = True
        args.atari = True
        args.seed = 1

    args.cuda = not args.use_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda:0' if args.cuda else 'cpu')
    
    del args.default_env
    del args.cuda
    del args.use_cuda

    return args