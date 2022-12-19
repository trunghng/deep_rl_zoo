import gym
from gym.wrappers.monitoring import video_recorder
import numpy as np
import torch
from torch.optim import Adam
from torch.autograd import grad
from torch.distributions import kl_divergence
from network import MLPActorCritic
from utils import *
import argparse
from copy import deepcopy
import random
from os.path import dirname, join, realpath
import sys
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, '..'))
import common.mpi_utils as mpi
from common.logger import Logger


class TRPO:


    def __init__(self, args):
        '''
        TRPO & Natural Policy Gradient w/ Actor-Critic approach & 
            Generalized Advantage Estimators & 
            using rewards-to-go as target for the value function

        :param env: (str) OpenAI environment name
        :param seed: (int) Seed for RNG
        :param hidden_layers: (List[int]) Hidden layers size of policy & value function networks
        :param v_lr: (float) Learning rate for value function optimizer
        :param epochs: (int) Number of epochs
        :param steps_per_epoch: (int) Maximum number of steps per epoch
        :param train_v_iters: (int) Number of GD-steps to take on value function per epoch
        :param max_ep_len: (int) Maximum episode/trajectory length
        :param gamma: (float) Discount factor
        :param lamb: (float) Lambda for GAE
        :param goal: (float) Total reward threshold for early stopping
        :param delta: (float) KL divergence threshold
        :param damping_coeff: (float) Damping coefficient
        :param cg_iters: (int) Number of iterations of CG to perform
        :param linesearch: (bool) Whether to use backtracking line search (if not, TRPO -> NPG)
        :param backtrack_iters: (int) Maximum number of steps of line search
        :param backtrack_coeff: (float) How far back to step during backtracking line search
        :param save: (bool) Whether to save the final model
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
        if not args.eval:
            self.critic_opt = Adam(self.ac.critic.parameters(), lr=args.v_lr)
            self.epochs = args.epochs
            self.steps_per_epoch = int(args.steps_per_epoch / mpi.n_procs())
            self.train_v_iters = args.train_v_iters
            self.max_ep_len = args.max_ep_len
            self.buffer = MPIBuffer(self.steps_per_epoch, args.gamma, args.lamb)
            self.goal = args.goal
            self.delta = args.delta
            self.damping_coeff = args.damping_coeff
            self.cg_iters = args.cg_iters
            if args.linesearch:
                self.linesearch = True
                self.backtrack_iters = args.backtrack_iters
                self.backtrack_coeff = args.backtrack_coeff
                algo = 'TRPO'
            else:
                self.linesearch = False
                algo = 'NPG'
            self.goal = args.goal
            self.save = args.save
            self.save_freq = args.save_freq
            self.render = args.render
            self.plot = args.plot
            self.logger = Logger(args.env, algo, args.model_dir, args.video_dir, args.figure_dir)
            self.logger.log(f'Algorithm: {algo}\nEnvironment: {args.env}\nSeed: {args.seed}')
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


    def _update_params(self):
        '''
        Update pi's & v's parameters
        '''
        observations, actions, logps_old, advs, rewards_to_go = self.buffer.get()
        with torch.no_grad():
            pi_old, _ = self.ac.actor(observations, actions)

        def pi_loss_kl(need_loss: str, need_kl: str):
            '''
            Compute surrogate loss and average KL divergence
            '''
            pi, logps = self.ac.actor(observations, actions)
            loss_kl = dict()

            if need_loss:
                # ratio = pi(a|s) / pi_old(a|s)
                ratio = torch.exp(logps - logps_old)
                loss_kl['loss'] = -(ratio * advs).mean()
            if need_kl:
                loss_kl['kl'] = kl_divergence(pi, pi_old).mean()
            return loss_kl

        def compute_pi_loss():
            return pi_loss_kl(True, False)['loss']

        def compute_kl():
            return pi_loss_kl(False, True)['kl']

        # @torch.no_grad()
        def compute_pi_loss_kl():
            loss_kl = pi_loss_kl(True, True)
            return loss_kl['loss'], loss_kl['kl']

        def compute_v_loss():
            '''
            Compute value function loss
            '''
            values = self.ac.critic(observations)
            loss = ((values - rewards_to_go) ** 2).mean()
            return loss

        pi_loss = compute_pi_loss()
        pi_loss_old = pi_loss.item()
        # Compute policy gradient vector
        g = flatten(grad(pi_loss, self.ac.actor.parameters()))

        def Fx(x):
            '''
            Compute the product Fx of Fisher Information Matrix (FIM) w/ vector :param x:
                FIM: F = grad**2 kl
            '''
            kl = compute_kl()
            grad_kl = flatten(grad(kl, self.ac.actor.parameters(), create_graph=True))
            grad_kl_x = (grad_kl * x).sum()
            Fx_ = flatten(grad(grad_kl_x, self.ac.actor.parameters()))
            return Fx_ + self.damping_coeff * x

        '''
        Compute natural gradient: x = (F^-1)g
            => (g^T)(F^-1)g = (x^T)Fx
            => step_size = sqrt(2*delta / (g^T)(F^-1)g)
                         = sqrt(2*delta / (x^T)Fx)
        '''
        x = conjugate_gradient(Fx, g, self.cg_iters)
        step_size = torch.sqrt(2 * self.delta / (torch.dot(x, Fx(x)) + 1e-8))

        '''
        Update pi's parameters (theta):
            - w/ linesearch:
                theta := theta + alpha^j * step_size * (F^-1)g
                       = theta + alpha^j * step_size * x
            - w/o linesearch:
                theta := theta + step_size * (F^-1)g
                       = theta + step_size * x
        '''
        old_params = []
        for param in deepcopy(self.ac.actor).parameters():
            old_params.append(param.data.view(-1))
        old_params = torch.cat(old_params)

        def linesearch(scale):
            params = old_params - scale * step_size * x
            prev_idx = 0
            for param in self.ac.actor.parameters():
                size = int(np.prod(list(param.size())))
                param.data.copy_(params[prev_idx:prev_idx + size].view(param.size()))
                prev_idx += size

            pi_loss, kl = compute_pi_loss_kl()
            return pi_loss, kl.item()

        if self.linesearch:
            for j in range(self.backtrack_iters):
                pi_loss, kl = linesearch(self.backtrack_coeff ** j)
                if kl <= self.delta and pi_loss <= pi_loss_old:
                    self.logger.log('Accepting new params at step %d of line search'%j)
                    break
                if j == self.backtrack_iters - 1:
                    self.logger.log('Line search failed! Keeping old params')
                    pi_loss, kl = linesearch(0)
            pi_loss.backward()
            mpi.mpi_avg_grads(self.ac.actor)
        else:
            pi_loss, kl = linesearch(1.0)
        
        # Update v's parameters
        for _ in range(self.train_v_iters):
            self.critic_opt.zero_grad()
            v_loss = compute_v_loss()
            v_loss.backward()
            mpi.mpi_avg_grads(self.ac.critic)
            self.critic_opt.step()

        self.logger.add(PiLoss=pi_loss.item(), VLoss=v_loss.item(), KL=kl)


    def _train_one_epoch(self):
        '''
        Perform one training epoch
        '''
        returns = []
        eps_len = []
        step = 0

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
            self.logger.log_tabular('KL')
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


    def test(self, vid_path: str=None, model_path: str=None):
        if mpi.proc_rank() == 0:
            print('---Evaluating---')
            if model_path:
                self.ac.actor.load_state_dict(torch.load(model_path))
            if vid_path:
                vr = video_recorder.VideoRecorder(self.env, path=vid_path)
            obs = self.env.reset()
            step = total_reward = 0
            while True:
                self.env.render()
                if vid_path:
                    vr.capture_frame()
                action, _, _ = self.ac.step(obs)
                obs, reward, terminated, _ = self.env.step(action)
                step += 1
                total_reward += reward
                if terminated:
                    print(f'Episode finished after {step} steps\nTotal reward: {total_reward}')
                    break
            self.env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trust Region Policy Optimization')
    parser.add_argument('--env', type=str, choices=['CartPole-v0', 'HalfCheetah-v2'],
                        help='OpenAI enviroment name')
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
    parser.add_argument('--delta', type=float, default=0.01,
                        help='KL divergence threshold')
    parser.add_argument('--damping-coeff', type=float, default=0.1,
                        help='Damping coefficient')
    parser.add_argument('--cg-iters', type=int, default=10,
                        help='Number of iterations of Conjugate gradient to perform')
    parser.add_argument('--linesearch', action='store_false', 
                        help='Whether to use backtracking line-search')
    parser.add_argument('--backtrack-iters', type=int, default=10,
                        help='Maximum number of steps in the backtracking line search')
    parser.add_argument('--backtrack-coeff', type=float, default=0.8,
                        help='how far back to step during backtracking line search')
    parser.add_argument('--save', action='store_true',
                        help='Whether to save training model')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Model saving frequency')
    parser.add_argument('--render', action='store_true',
                        help='Whether to save training result as video')
    parser.add_argument('--plot', action='store_true',
                        help='Whether to plot training statistics and save as image')
    parser.add_argument('--model-dir', type=str, default='./output/models',
                        help='Where to save the model')
    parser.add_argument('--video-dir', type=str, default='./output/videos',
                        help='Where to save the video output')
    parser.add_argument('--figure-dir', type=str, default='./output/figures',
                        help='Where to save the plots')
    args = parser.parse_args()
    
    if not args.eval and args.model_path or (not args.model_path and args.eval):
        parser.error('Arguments --eval & --model-path must be specified at the same time.')
    if args.linesearch and not (args.backtrack_coeff and args.backtrack_iters):
        parser.error('Arguments --backtrack-iters & --backtrack-coeff are required when enabling --linesearch.')
    mpi.mpi_fork(args.cpu)
    mpi.setup_pytorch_for_mpi()

    agent = TRPO(args)
    if args.eval:
        agent.test(model_path=args.model_path)
    else:
        agent.train()
