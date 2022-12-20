import gym
import torch
from torch.optim import Adam
import numpy as np
import argparse, random
import common.mpi_utils as mpi
from common.logger import Logger
from zoo.pg.network import MLPActorCritic
from zoo.pg.utils import Buffer


class VPG:


    def __init__(self, args):
        '''
        Vanilla Policy Gradient with Actor-Critic approach & Generalized Advantage Estimators & 
            using rewards-to-go as target for the value function, which is chosen as the baseline

        :param env: (str) OpenAI environment name
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
        :param goal: (float) Total reward threshold for early stopping
        :param save: (bool) Whether to save the final model
        :param save_freq: (int) Model saving frequency
        :param render: (bool) Whether to render the training result in video
        :param plot: (bool) Whether to plot the statistics and save as image
        :param model_dir: (str) Model directory
        :param video_dir: (str) Video directory
        :param figure_dir: (str) Figure directory
        '''
        self.env = gym.make(args.env)
        self._seed(args.seed)
        observation_space = self.env.observation_space
        action_space = self.env.action_space
        self.ac = MLPActorCritic(observation_space, action_space, args.hidden_layers)
        mpi.sync_params(self.ac)
        self.actor_opt = Adam(self.ac.actor.parameters(), lr=args.pi_lr)
        self.critic_opt = Adam(self.ac.critic.parameters(), lr=args.v_lr)
        self.epochs = args.epochs
        self.steps_per_epoch = int(args.steps_per_epoch / mpi.n_procs())
        self.train_v_iters = args.train_v_iters
        self.max_ep_len = args.max_ep_len
        self.buffer = Buffer(self.steps_per_epoch, args.gamma, args.lamb)
        self.save = args.save
        self.save_freq = args.save_freq
        self.render = args.render
        self.plot = args.plot
        self.logger = Logger(args.env, args.model_dir, args.video_dir, args.figure_dir)
        self.logger.log(f'Algorithm: VPG\nEnvironment: {args.env}\nSeed: {args.seed}')
        self.logger.set_saver(self.ac)


    def _seed(self, seed: int):
        '''
        Set global seed
        '''
        seed += 10 * mpi.proc_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.env.seed(seed)


    def _compute_pi_loss(self, observations, actions, advs):
        '''
        :param observations:
        :param actions:
        :param advs:
        '''
        _, log_prob = self.ac.actor(observations, actions)
        loss = -(log_prob * advs).mean()
        return loss


    def _compute_v_loss(self, observations, rewards_to_go):
        '''
        :param observations:
        :param rewards_to_go:
        '''
        values = self.ac.critic(observations)
        loss = ((values - rewards_to_go) ** 2).mean()
        return loss


    def _update_params(self):
        '''
        Update pi's & v's parameters
        '''
        observations, actions, log_probs, advs, rewards_to_go = self.buffer.get()

        self.actor_opt.zero_grad()
        pi_loss = self._compute_pi_loss(observations, actions, advs)
        pi_loss.backward()
        mpi.mpi_avg_grads(self.ac.actor)
        self.actor_opt.step()

        for _ in range(self.train_v_iters):
            self.critic_opt.zero_grad()
            v_loss = self._compute_v_loss(observations, rewards_to_go)
            v_loss.backward()
            mpi.mpi_avg_grads(self.ac.critic)
            self.critic_opt.step()

        self.logger.add(PiLoss=pi_loss.item(), VLoss=v_loss.item())


    def _train_one_epoch(self):
        '''
        One epoch training
        '''
        returns, eps_len, step = [], [], 0

        while step < self.steps_per_epoch:
            observation = self.env.reset()
            rewards = []

            while True:
                action, log_prob, value = self.ac.step(observation)
                next_observation, reward, terminated, _ = self.env.step(action)
                self.buffer.add(observation, action, reward, float(value), float(log_prob))
                observation = next_observation
                rewards.append(reward)
                step += 1

                if terminated or len(rewards) == self.max_ep_len or step == self.steps_per_epoch:
                    if terminated:
                        value = 0
                        return_, ep_len = sum(rewards), len(rewards)
                        returns.append(return_)
                        eps_len.append(ep_len)
                        self.logger.add(Return=returns, EpLen=eps_len)
                    else:
                        _, _, value = self.ac.step(observation)
                    self.buffer.finish_rollout(value)
                    break
        self._update_params()


    def train(self):
        self.logger.log('---Training---')
        for epoch in range(1, self.epochs + 1):
            self._train_one_epoch()
            self.logger.log_tabular('Epoch', epoch)
            self.logger.log_tabular('PiLoss')
            self.logger.log_tabular('VLoss')
            self.logger.log_tabular('Return')
            self.logger.log_tabular('EpLen')
            self.logger.log()

            if self.save and epoch % self.save_freq == 0:
                self.logger.save_state()
        self.env.close()
        if self.render:
            self.logger.render(self.env)
        if self.plot:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vanilla Policy Gradient')
    parser.add_argument('--env', type=str, choices=['CartPole-v0', 'HalfCheetah-v2'],
                        help='Environment ID')
    parser.add_argument('--eval', action='store_true',
                        help='Whether to enable evaluation')
    parser.add_argument('--model-path', type=str,
                        help='Model path to load')
    parser.add_argument('--cpu', type=int, default=4,
                        help='Number of CPUs for parallel computing')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for RNG')
    parser.add_argument('--hidden-layers', nargs='+', type=int,
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
    parser.add_argument('--goal', type=int,
                        help='Total reward threshold for early stopping')
    parser.add_argument('--save', action='store_true',
                        help='Whether to save training model')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Model saving frequency')
    parser.add_argument('--render', action='store_true',
                        help='Whether to save training result as video')
    parser.add_argument('--plot', action='store_true',
                        help='Whether to plot training statistics and save as image')
    parser.add_argument('--model-dir', type=str, default='./zoo/pg/output/models/vpg',
                        help='Where to save the model')
    parser.add_argument('--video-dir', type=str, default='./zoo/pg/output/videos/vpg',
                        help='Where to save the video output')
    parser.add_argument('--figure-dir', type=str, default='./zoo/pg/output/figures/vpg',
                        help='Where to save the plots')
    args = parser.parse_args()
    mpi.mpi_fork(args.cpu)
    mpi.setup_pytorch_for_mpi()

    agent = VPG(args)
    agent.train()
