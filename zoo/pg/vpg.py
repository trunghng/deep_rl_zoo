import gym
import torch
from torch.optim import Adam
import numpy as np
import argparse, random, os
import common.mpi_utils as mpi
from common.logger import Logger
from zoo.pg.network import MLPStochasticActorCritic
from zoo.pg.utils import Buffer, set_seed


class VPG:


    def __init__(self, args):
        '''
        Vanilla Policy Gradient w/ Generalized Advantage Estimators & 
            using rewards-to-go as target for the value function, which is chosen as the baseline

        :param env: (str) Environment name
        :param exp_name: (str) Experiment name
        :param seed: (int) Seed for RNG
        :param hidden_layers: (List[int]) Hidden layers size of policy & value function networks
        :param pi_lr: (float) Learning rate for policy optimizer
        :param v_lr: (float) Learning rate for value function optimizer
        :param epochs: (int) Number of epochs
        :param steps_per_epoch: (int) Maximum number of steps per epoch
        :param train_v_iters: (int) Number of GD-steps to take on value function per epoch
        :param max_ep_len: (int) Maximum episode/trajectory length
        :param gamma: (float) Discount factor
        :param lamb: (float) Lambda for GAE
        :param save: (bool) Whether to save the final model
        :param save_freq: (int) Model saving frequency
        :param render: (bool) Whether to render the training result in video
        :param plot: (bool) Whether to plot the statistics and save as image
        '''
        self.env = gym.make(args.env)
        set_seed(args.seed + 10 * mpi.proc_rank())
        observation_space = self.env.observation_space
        action_space = self.env.action_space
        self.ac = MLPStochasticActorCritic(observation_space, action_space, args.hidden_layers)
        mpi.sync_params(self.ac)
        self.actor_opt = Adam(self.ac.actor.parameters(), lr=args.pi_lr)
        self.critic_opt = Adam(self.ac.critic.parameters(), lr=args.v_lr)
        self.epochs = args.epochs
        self.proc_steps_per_epoch = int(args.steps_per_epoch / mpi.n_procs())
        self.steps_per_epoch = args.steps_per_epoch
        self.train_v_iters = args.train_v_iters
        self.max_ep_len = args.max_ep_len
        self.buffer = Buffer(self.proc_steps_per_epoch, args.gamma, args.lamb)
        self.save = args.save
        self.save_freq = args.save_freq
        self.render = args.render
        self.plot = args.plot
        if args.exp_name:
            exp_name = args.exp_name
            log_dir = os.path.join(os.getcwd(), 'data', exp_name, f'{exp_name}_s{args.seed}')
        else:
            exp_name = None
            log_dir = None
        self.logger = Logger(log_dir=log_dir, exp_name=exp_name)
        config_dict = vars(args)
        config_dict['algo'] = 'vpg'
        self.logger.save_config(config_dict)
        self.logger.set_saver(self.ac)


    def update_params(self):
        '''
        Update policy and value networks' parameters
        '''
        def compute_pi_loss(observations, actions, advs):
            _, log_prob = self.ac.actor(observations, actions)
            loss = -(log_prob * advs).mean()
            return loss

        def compute_v_loss(observations, rewards_to_go):
            values = self.ac.critic(observations)
            loss = ((values - rewards_to_go) ** 2).mean()
            return loss, values

        observations, actions, log_probs, advs, rewards_to_go = self.buffer.get()

        self.actor_opt.zero_grad()
        pi_loss = compute_pi_loss(observations, actions, advs)
        pi_loss.backward()
        mpi.mpi_avg_grads(self.ac.actor)
        self.actor_opt.step()

        for _ in range(self.train_v_iters):
            self.critic_opt.zero_grad()
            v_loss, v_values = compute_v_loss(observations, rewards_to_go)
            v_loss.backward()
            mpi.mpi_avg_grads(self.ac.critic)
            self.critic_opt.step()

        self.logger.add({
            'pi-loss': pi_loss.item(),
            'v-loss': v_loss.item(),
            'v-values': v_values.detach().numpy()
        })


    def train_one_epoch(self):
        '''
        One epoch training
        '''
        step = 0

        while step < self.proc_steps_per_epoch:
            observation = self.env.reset()
            rewards = []

            while True:
                action, log_prob, value = self.ac.step(observation)
                next_observation, reward, terminated, _ = self.env.step(action)
                self.buffer.add(observation, action, reward, float(value), float(log_prob))
                observation = next_observation
                rewards.append(reward)
                step += 1

                if terminated or len(rewards) == self.max_ep_len\
                        or step == self.proc_steps_per_epoch:
                    if terminated:
                        value = 0
                        self.logger.add({
                            'episode-return': sum(rewards),
                            'episode-length': len(rewards)
                        })
                    else:
                        _, _, value = self.ac.step(observation)
                    self.buffer.finish_rollout(value)
                    break
        self.update_params()


    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.train_one_epoch()
            self.logger.log_epoch('epoch', epoch)
            self.logger.log_epoch('pi-loss', average_only=True)
            self.logger.log_epoch('v-loss', average_only=True)
            self.logger.log_epoch('v-values', need_optima=True)
            self.logger.log_epoch('episode-return', need_optima=True)
            self.logger.log_epoch('episode-length', average_only=True)
            self.logger.log_epoch('total-env-interacts', epoch * self.steps_per_epoch)
            self.logger.dump_epoch()

            if self.save and epoch % self.save_freq == 0:
                self.logger.save_state()
        self.env.close()
        if self.render:
            self.logger.render(self.env)
        if self.plot:
            self.logger.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vanilla Policy Gradient')
    parser.add_argument('--env', type=str, default='HalfCheetah-v2',
                        help='Environment ID')
    parser.add_argument('--exp-name', type=str, default='vpg',
                        help='Experiment name')
    parser.add_argument('--cpu', type=int, default=4,
                        help='Number of CPUs for parallel computing')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for RNG')
    parser.add_argument('--hidden-layers', nargs='+', type=int, default=[64, 32],
                        help='Hidden layers size of policy & value function networks')
    parser.add_argument('--pi-lr', type=float, default=3e-4,
                        help='Learning rate for policy optimizer')
    parser.add_argument('--v-lr', type=float, default=1e-3,
                        help='Learning rate for value function optimizer')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--steps-per-epoch', type=int, default=4000,
                        help='Maximum number of steps for each epoch')
    parser.add_argument('--train-v-iters', type=int, default=80,
                        help='Number of gradient descent steps to take on value function per epoch')
    parser.add_argument('--max-ep-len', type=int, default=1000,
                        help='Maximum episode/trajectory length')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--lamb', type=float, default=0.97,
                        help='Lambda for Generalized Advantage Estimation')
    parser.add_argument('--save', action='store_true',
                        help='Whether to save training model')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Model saving frequency')
    parser.add_argument('--render', action='store_true',
                        help='Whether to save training result as video')
    parser.add_argument('--plot', action='store_true',
                        help='Whether to plot training statistics and save as image')
    args = parser.parse_args()
    mpi.mpi_fork(args.cpu)
    mpi.setup_pytorch_for_mpi()

    agent = VPG(args)
    agent.train()
