import argparse, random, os
from typing import List, Tuple
from copy import deepcopy

import gymnasium as gym
from gymnasium.spaces import Space, Box, Discrete
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import grad
from torch.distributions import kl_divergence

from common.policy import CategoricalPolicy, DiagonalGaussianPolicy
from common.vf import StateValueFunction
from common.utils import set_seed, to_tensor, flatten, conjugate_gradient, dim
from common.buffer import RolloutBuffer
import common.mpi_utils as mpi
from common.logger import Logger


class ActorCritic(nn.Module):
    
    def __init__(self,
                obs_space: Space,
                action_space: Space,
                hidden_sizes: List[int]=[64, 64],
                activation: nn.Module=nn.Tanh,
                device: str='cpu') -> None:
        super().__init__()
        obs_dim = dim(obs_space)
        action_dim = dim(action_space)

        # continous action space
        if isinstance(action_space, Box):
            self.actor = DiagonalGaussianPolicy(obs_dim, action_dim, hidden_sizes, activation).to(device)
        # discrete action space
        elif isinstance(action_space, Discrete):
            self.actor = CategoricalPolicy(obs_dim, action_dim, hidden_sizes, activation).to(device)

        self.critic = StateValueFunction(obs_dim, hidden_sizes, activation).to(device)
        self.device = device


    def step(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        observation = to_tensor(observation, device=self.device)
        with torch.no_grad():
            pi = self.actor._distribution(observation)
            action = pi.sample()
            log_prob = self.actor._log_prob(pi, action)
            value = self.critic(observation)
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()


class TRPO:
    """
    TRPO & Natural Policy Gradient w/ Generalized Advantage Estimators & 
        using rewards-to-go as target for the value function

    :param env: (str) Environment ID.
    :param exp_name: (str) Experiment name.
    :param cpu: (int) Number of CPUs for parallel computing.
    :param seed: (int) Seed for RNG.
    :param hidden_sizes: (List[int]) Sizes of policy & value function networks' hidden layers.
    :param v_lr: (float) Learning rate for value function optimizer
    :param epochs: (int) Number of epochs to train agent.
    :param steps_per_epoch: (int) Maximum number of steps per epoch
    :param train_v_iters: (int) Number of GD-steps to take on value function per epoch
    :param max_ep_len: (int) Maximum episode/trajectory length
    :param gamma: (float) Discount factor
    :param lamb: (float) Lambda for GAE-lambda
    :param delta: (float) KL divergence threshold
    :param damping_coeff: (float) Damping coefficient
    :param cg_iters: (int) Number of iterations of CG to perform
    :param linesearch: (bool) Whether to use backtracking line search (if not, TRPO -> NPG)
    :param backtrack_iters: (int) Maximum number of steps of line search
    :param backtrack_coeff: (float) How far back to step during backtracking line search
    :param save: (bool) Whether to save the final model.
    :param save_every: (int) Model saving frequency.
    :param render: (bool) Whether to render the training result.
    :param plot: (bool) Whether to plot the statistics.
    """

    def __init__(self, args):
        self.env = gym.make(args.env)
        set_seed(args.seed + 10 * mpi.proc_rank())
        observation_space = self.env.observation_space
        action_space = self.env.action_space
        self.device = 'cuda:0' if torch.cuda.is_available() and args.cpu == 1 else 'cpu'
        self.ac = ActorCritic(observation_space, action_space, args.hidden_sizes, device=self.device)
        mpi.sync_params(self.ac)
        self.critic_opt = Adam(self.ac.critic.parameters(), lr=args.v_lr)
        self.epochs = args.epochs
        self.proc_steps_per_epoch = int(args.steps_per_epoch / mpi.n_procs())
        self.steps_per_epoch = args.steps_per_epoch
        self.train_v_iters = args.train_v_iters
        self.max_ep_len = args.max_ep_len
        self.buffer = RolloutBuffer(self.proc_steps_per_epoch, args.gamma, args.lamb)
        self.delta = args.delta
        self.damping_coeff = args.damping_coeff
        self.cg_iters = args.cg_iters
        if args.linesearch:
            self.backtrack_iters = args.backtrack_iters
            self.backtrack_coeff = args.backtrack_coeff
            algo = 'trpo'
        else:
            algo = 'npg'
        self.save = args.save
        self.save_every = args.save_every
        self.render = args.render
        self.plot = args.plot
        if args.exp_name:
            exp_name = args.exp_name
            log_dir = os.path.join(os.getcwd(), 'data', exp_name, f'{exp_name}_s{args.seed}')
        else:
            log_dir = None
        config_dict = vars(args)
        config_dict['algo'] = algo
        self.logger = Logger(log_dir=log_dir)
        self.logger.save_config(config_dict)
        self.logger.set_saver(self.ac)


    def update_params(self) -> None:
        """Update policy and value network's parameters"""

        observations, actions, logps_old, advs, rewards_to_go \
            = map(lambda x: x.to(self.device), self.buffer.get())
        with torch.no_grad():
            pi_old, _ = self.ac.actor(observations, actions)

        def pi_loss_kl(need_loss: str, need_kl: str):
            """Compute surrogate loss and average KL divergence"""
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

        def compute_pi_loss_kl():
            loss_kl = pi_loss_kl(True, True)
            return loss_kl['loss'], loss_kl['kl']

        def compute_v_loss():
            values = self.ac.critic(observations)
            loss = ((values - rewards_to_go) ** 2).mean()
            return loss, values

        pi_loss = compute_pi_loss()
        pi_loss_old = pi_loss.item()
        # Compute policy gradient vector
        g = flatten(grad(pi_loss, self.ac.actor.parameters()))
        g = torch.as_tensor(mpi.mpi_avg(g), dtype=torch.float32, device=self.device)

        def Fx(x):
            """Compute the product Fx of Fisher Information Matrix (FIM) w/ vector :param x:
                FIM: F = grad**2 kl
            """

            kl = compute_kl()
            grad_kl = flatten(grad(kl, self.ac.actor.parameters(), create_graph=True))
            grad_kl_x = (grad_kl * x).sum()
            Fx_ = flatten(grad(grad_kl_x, self.ac.actor.parameters()))
            return torch.as_tensor(mpi.mpi_avg(Fx_), dtype=torch.float32, device=self.device) + self.damping_coeff * x

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

        if hasattr(self, 'backtrack_coeff'):
            for j in range(self.backtrack_iters):
                pi_loss, kl = linesearch(self.backtrack_coeff ** j)
                if kl <= self.delta and pi_loss <= pi_loss_old:
                    self.logger.log('Accepting new params at step %d of line search'%j)
                    break
            else:
                self.logger.log('Line search failed! Keeping old params')
                pi_loss, kl = linesearch(0)
            pi_loss.backward()
            mpi.mpi_avg_grads(self.ac.actor)
        else:
            pi_loss, kl = linesearch(1.0)

        for _ in range(self.train_v_iters):
            self.critic_opt.zero_grad()
            v_loss, v_values = compute_v_loss()
            v_loss.backward()
            mpi.mpi_avg_grads(self.ac.critic)
            self.critic_opt.step()

        self.logger.add({
            'pi-loss': pi_loss.item(),
            'v-loss': v_loss.item(),
            'v-values': v_values.detach().cpu().numpy(),
            'kl': kl
        })


    def act(self, observation: np.ndarray) -> np.ndarray:
        return self.ac.step(observation)[0]


    def load(self, model_path: str) -> None:
        """Model loading"""
        self.ac.load_state_dict(torch.load(model_path))


    def train(self) -> None:
        for epoch in range(1, self.epochs + 1):
            step = 0
            while step < self.proc_steps_per_epoch:
                observation, _ = self.env.reset()
                rewards = []

                while True:
                    action, log_prob, value = self.ac.step(observation)
                    next_observation, reward, terminated, truncated, _ = self.env.step(action)
                    self.buffer.add(observation, action, reward, float(value), float(log_prob))
                    observation = next_observation
                    rewards.append(reward)
                    step += 1

                    if terminated or truncated or len(rewards) == self.max_ep_len\
                            or step == self.proc_steps_per_epoch:
                        if terminated or truncated:
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

            self.logger.log_epoch('epoch', epoch)
            self.logger.log_epoch('pi-loss', average_only=True)
            self.logger.log_epoch('v-loss', average_only=True)
            self.logger.log_epoch('v-values', need_optima=True)
            self.logger.log_epoch('kl', average_only=True)
            self.logger.log_epoch('episode-return', need_optima=True)
            self.logger.log_epoch('episode-length', average_only=True)
            self.logger.log_epoch('total-env-interacts', epoch * self.steps_per_epoch)
            self.logger.dump_epoch()

            if self.save and epoch % self.save_every == 0:
                self.logger.save_state()
        self.env.close()
        if self.render:
            self.logger.render(self.act)
        if self.plot:
            self.logger.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trust Region Policy Optimization')
    parser.add_argument('--env', type=str, default='HalfCheetah-v4',
                        help='Environment ID')
    parser.add_argument('--exp-name', type=str, default='trpo',
                        help='Experiment name')
    parser.add_argument('--cpu', type=int, default=4,
                        help='Number of CPUs for parallel computing')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for RNG')
    parser.add_argument('--hidden-sizes', nargs='+', type=int, default=[64, 32],
                        help="Sizes of policy & value function networks' hidden layers")
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
                        help='Whether to save the final model')
    parser.add_argument('--save-every', type=int, default=1,
                        help='Model saving frequency')
    parser.add_argument('--render', action='store_true',
                        help='Whether to render the training result')
    parser.add_argument('--plot', action='store_true',
                        help='Whether to plot the training statistics')
    args = parser.parse_args()

    if args.linesearch and not (args.backtrack_coeff and args.backtrack_iters):
        parser.error('Arguments --backtrack-iters & --backtrack-coeff are required when enabling --linesearch.')
    mpi.mpi_fork(args.cpu)
    mpi.setup_pytorch_for_mpi()

    agent = TRPO(args)
    agent.train()
