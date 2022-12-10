# import subprocess
# from vpg import VPG
# import argparse
 
# subprocess.run(["./bash_scripts/test.sh"], shell=True)

# parser = argparse.ArgumentParser(description='Vanilla Policy Gradient')
# parser.add_argument('--env', type=str, default='Pendulum-v1',
#                     help='OpenAI enviroment')
# parser.add_argument('--eval', action='store_true',
#                     help='Whether to enable evaluation')
# parser.add_argument('--model-path', type=str,
#                     help='Model path to load')
# parser.add_argument('--seed', type=int, default=1,
#                     help='Random seed')
# parser.add_argument('--pi-lr', type=float,
#                     help='Learning rate for policy optimization')
# parser.add_argument('--v-lr', type=float,
#                     help='Learning rate for value function optimization')
# parser.add_argument('--epochs', type=int, default=500,
#                     help='Number of epochs')
# parser.add_argument('--steps-per-epoch', type=int,
#                     help='Maximum number of epoch for each epoch')
# parser.add_argument('--train-v-iters', type=int,
#                     help='Value network update frequency')
# parser.add_argument('--max-ep-len', type=int, default=1000,
#                     help='Maximum episode/trajectory length')
# parser.add_argument('--gamma', type=float, default=0.98,
#                     help='Discount factor')
# parser.add_argument('--lamb', type=float, default=0.96,
#                     help='Eligibility trace')
# parser.add_argument('--goal', type=int, default=195,
#                     help='Goal total reward to end training')
# parser.add_argument('--save', action='store_true',
#                     help='Whether to save training model')
# parser.add_argument('--render', action='store_true',
#                     help='Whether to save training result as video')
# parser.add_argument('--plot', action='store_true',
#                     help='Whether to plot training statistics and save as image')
# args = parser.parse_args()

# steps_per_epoch = [2500, 5000]
# train_v_iters = [10, 20, 40]
# pi_lrs = [1e-2, 8e-3, 5e-3, 1e-3, 7e-4]
# v_lrs = [1e-2, 7e-3, 3e-3, 1e-3]

# for steps in steps_per_epoch:
#     for v_iters in train_v_iters:
#         for pi_lr in pi_lrs:
#             for v_lr in v_lrs:
#                 print('----------------------------------')
#                 print(f'steps_per_epoch: {steps}, v_iters: {v_iters}, pi_lr: {pi_lr}, v_lr: {v_lr}')
#                 args.steps_per_epoch = steps
#                 args.train_v_iters = v_iters
#                 args.v_lr = v_lr
#                 args.pi_lr = pi_lr
#                 agent = VPG(args)
#                 agent.train()

import multiprocessing as mp
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(rank)







