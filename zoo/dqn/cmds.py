import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(description='Deep Q-Learning')

    parser.add_argument('--env-name', type=str,
                        help='Enviroment name')
    parser.add_argument('--atari', action='store_true',
                        help='Whether to use atari environment')
    parser.add_argument('--eval', action='store_true',
                        help='Whether to enable evaluation')
    parser.add_argument('--model-path', type=str,
                        help='Model path to load')
    parser.add_argument('--double', action='store_true',
                        help='Whether to use double Q-network')
    parser.add_argument('--dueling', action='store_true',
                        help='Whether to use dueling Q-network')
    parser.add_argument('--prioritized-replay', action='store_true',
                        help='Whether to use prioritized replay')
    parser.add_argument('--alpha', type=float,
                        help='Alpha value for prioritized replay')
    
    parser.add_argument('--epsilon-init', type=float, default=1.0,
                        help='Initial value of epsilon')
    parser.add_argument('--epsilon-final', type=float, default=0.01,
                        help='Final value of epsilon')
    parser.add_argument('--epsilon-decay', type=float, default=100,
                        help='Final value of epsilon')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--buffer-size', type=int, default=100000,
                        help='Maximum memory buffer size')
    parser.add_argument('--train-freq', type=int, default=1,
                        help='Number of steps between optimization steps')
    parser.add_argument('--update-target', type=int, default=1000,
                        help='Target network update frequency')
    parser.add_argument('--tau', type=float, default=0.001,
                        help = 'Smoothness parameter, used for target network soft update')
    parser.add_argument('--num-episodes', type=int, default=600,
                        help = 'Maximum number of episodes')
    parser.add_argument('--logging-interval', type=int, default=10,
                        help = 'Score tracking interval')
    parser.add_argument('--termination', type=int, default=17,
                        help = 'Terminal score')
    parser.add_argument('--print-freq', type=int, default=10000,
                        help='Result display interval')
    parser.add_argument('--save-model', action='store_true',
                        help='Whether to save the model after training')
    parser.add_argument('--render-video', action='store_true',
                        help='Whether to render video output after training')
    parser.add_argument('--plot', action='store_true',
                        help='Whether to plot the result and save')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--use-cuda', action='store_true',
                        help='Whether to enable CUDA training')
    parser.add_argument('--default-exp', type=int, choices=[0, 1, 2],
                        help='Default experiments with default settings')
    args = parser.parse_args()

    if args.default_exp == 1:
        # LunarLander-v2 default experiment
        args.env_name = 'LunarLander-v2'
        args.atari = False
        args.epsilon_init = 1
        args.epsilon_final = 0.01
        args.epsilon_decay = 200
        args.gamma = 0.99
        args.lr = 5e-4
        args.buffer_size = 100000
        args.batch_size = 64
        args.train_freq = 4
        args.update_target = 4
        args.tau = 1e-3
        args.num_episodes = 1000
        args.logging_interval = 100
        args.termination = 200
        args.print_freq = 100
        args.save_model = True
        args.render_video = True
        args.plot = True
        args.seed = 1
    elif args.default_exp == 2:
        # PongNoFrameskip-v4 default experiment
        args.env_name = 'PongNoFrameskip-v4'
        args.atari = True
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
        args.print_freq = 10
        args.save_model = True
        args.render_video = True
        args.plot = True
        args.seed = 1

    args.cuda = not args.use_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda:0' if args.cuda else 'cpu')
    
    del args.default_exp
    del args.cuda
    del args.use_cuda
    return args
