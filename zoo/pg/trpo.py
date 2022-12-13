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
from os.path import join
from copy import deepcopy
import random


class TRPO:


    def __init__(self, args,
                model_dir='./output/models',
                video_dir='./output/videos',
                figure_dir='./output/figures'):
        '''
        TRPO & Natural Policy Gradient w/ Actor-Critic approach & 
            Generalized Advantage Estimators & 
            using rewards-to-go as target for the value function

        :param env: (str) OpenAI's environment name
        :param seed: (int) Seed for RNG
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
        self._env = gym.make(args.env)
        self._seed(args.seed)
        observation_space = self._env.observation_space
        action_space = self._env.action_space
        self._ac = MLPActorCritic(observation_space, action_space)
        if not args.eval:
            self._critic_opt = Adam(self._ac.critic.parameters(), lr=args.v_lr)
            self._epochs = args.epochs
            self._steps_per_epoch = args.steps_per_epoch
            self._train_v_iters = args.train_v_iters
            self._max_ep_len = args.max_ep_len
            self._buffer = Buffer(args.max_ep_len, args.gamma, args.lamb)
            self._goal = args.goal
            self._delta = args.delta
            self._damping_coeff = args.damping_coeff
            self._cg_iters = args.cg_iters
            if args.linesearch:
                self._linesearch = True
                self._backtrack_iters = args.backtrack_iters
                self._backtrack_coeff = args.backtrack_coeff
                algo = 'TRPO'
            else:
                self._linesearch = False
                algo = 'NPG'
            print(f'Algorithm: {algo}\nEnvironment: {args.env}')
            basename = f'{algo}-{args.env}'
            self._model_path = join(model_dir, f'{basename}.pth') if args.save else None
            self._vid_path = join(video_dir, f'{basename}.mp4') if args.render else None
            self._plot_path = join(figure_dir, f'{basename}.png') if args.plot else None


    def _seed(self, seed: int):
        '''
        Set global seed
        '''
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self._env.seed(seed)


    def _update_params(self):
        '''
        Update pi's & v's parameters
        '''
        observations, actions, logps_old, advs, rewards_to_go = self._buffer.get()
        with torch.no_grad():
            # TODO: logps_old
            pi_old, _ = self._ac.actor(observations, actions)

        def pi_loss_kl(need_loss: str, need_kl: str):
            '''
            Compute surrogate loss and average KL divergence
            '''
            pi, logps = self._ac.actor(observations, actions)
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

        @torch.no_grad()
        def compute_pi_loss_kl():
            loss_kl = pi_loss_kl(True, True)
            return loss_kl['loss'], loss_kl['kl']

        def compute_v_loss():
            '''
            Compute value function loss
            '''
            values = self._ac.critic(observations)
            loss = ((values - rewards_to_go) ** 2).mean()
            return loss

        pi_loss = compute_pi_loss()
        pi_loss_old = pi_loss.item()
        v_loss_old = compute_v_loss().item()
        # Compute policy gradient vector
        g = flatten(grad(pi_loss, self._ac.actor.parameters()))

        def Fx(x):
            '''
            Compute the product of Fisher Information Matrix (FIM) w/ vector :param x:
                FIM: F = grad**2 kl
            '''
            kl = compute_kl()
            grad_kl = grad(kl, self._ac.actor.parameters(), create_graph=True)
            grad_kl = flatten(grad_kl)
            grad_kl_x = (grad_kl * x).sum()
            grad2_kl_x = grad(grad_kl_x, self._ac.actor.parameters())
            Fx_ = flatten(grad2_kl_x)
            return Fx_ + self._damping_coeff * x

        '''
        Compute natural gradient: x = (F^-1)g
            => (g^T)(F^-1)g = (x^T)Fx
            => step_size = sqrt(2*delta / (g^T)(F^-1)g)
                         = sqrt(2*delta / (x^T)Fx)
        '''
        x = conjugate_gradient(Fx, g, self._cg_iters)
        step_size = torch.sqrt(2 * self._delta / (torch.dot(x, Fx(x)) + 1e-8))

        '''
        Update pi's parameters (theta):
            - w/ linesearch:
                theta := theta + alpha^j * step_size * (F^-1)g
                       = theta + alpha^j * step_size * x
            - w/o linesearch:
                theta := theta + step_size * (F^-1)g
                       = theta + step_size * x
        '''
        actor_old = deepcopy(self._ac.actor)
        old_params = []
        for param in actor_old.parameters():
            old_params.append(param.data.view(-1))
        old_params = torch.cat(old_params)

        @torch.no_grad()
        def linesearch(scale):
            params = old_params - scale * step_size * x
            prev_idx = 0
            for param in self._ac.actor.parameters():
                size = int(np.prod(list(param.size())))
                param.data.copy_(params[prev_idx:prev_idx + size].view(param.size()))
                prev_idx += size

            pi_loss, kl = compute_pi_loss_kl()
            return pi_loss.item(), kl.item()

        if self._linesearch:
            for j in range(self._backtrack_iters):
                pi_loss, kl = linesearch(self._backtrack_coeff ** j)
                if kl <= self._delta and pi_loss <= pi_loss_old:
                    print('Accepting new params at step %d of line search.' % j)
                    break
                if j == self._backtrack_iters - 1:
                    print('Line search failed! Keeping old params.')
                    pi_loss, kl = linesearch(0)
        else:
            pi_loss, kl = linesearch(1.0)
        
        # Update v's parameters
        for _ in range(self._train_v_iters):
            self._critic_opt.zero_grad()
            v_loss = compute_v_loss()
            v_loss.backward()
            self._critic_opt.step()

        update_info = dict()
        # update_info['pi_loss_old'] = pi_loss_old
        # update_info['v_loss_old'] = v_loss_old
        update_info['pi_loss'] = pi_loss
        update_info['v_loss'] = v_loss.item()
        update_info['kl'] = kl
        return update_info


    def _train_one_epoch(self):
        '''
        Perform one training epoch
        '''
        returns = []
        eps_len = []
        step = 0

        while step <= self._steps_per_epoch:
            observation = self._env.reset()
            rewards = []

            while True:
                action, log_prob, value = self._ac.step(observation)
                next_observation, reward, terminated, _ = self._env.step(action)
                self._buffer.add(observation, float(action), reward, float(value), float(log_prob), terminated)
                observation = next_observation
                rewards.append(reward)

                if terminated or (len(rewards) == self._max_ep_len):
                    return_, ep_len = sum(rewards), len(rewards)
                    step += ep_len
                    eps_len.append(ep_len)
                    returns.append(return_)
                    break
        epoch_info = self._update_params()
        epoch_info['returns'] = returns
        epoch_info['eps_len'] = eps_len
        return epoch_info


    def train(self):
        print('---Training---')
        for epoch in range(1, self._epochs + 1):
            info = self._train_one_epoch()
            avg_return = np.mean(info['returns'])
            print('Epoch: %3d \tpi_loss: %.3f \tv_loss: %.3f \tavg_kl: %.4f \treturn: %.3f \tep_len: %.3f'%
                (epoch, info['pi_loss'], info['v_loss'], info['kl'], avg_return, np.mean(info['eps_len'])))
            if self._goal and avg_return >= self._goal:
                print(f'Environment solved at epoch {epoch}!')
                break
        self._env.close()
        if self._model_path:
            torch.save(self._ac.actor.state_dict(), self._model_path)
            print(f'Model is saved successfully at {self._model_path}')
        if self._vid_path:
            self.test(vid_path=self._vid_path)
            print(f'Video is renderred successfully at {self._vid_path}')
        if self._plot_path:
            pass


    def test(self, vid_path: str=None, model_path: str=None):
        print('---Evaluating---')
        if model_path:
            self._ac.actor.load_state_dict(torch.load(model_path))
        if vid_path:
            vr = video_recorder.VideoRecorder(self._env, path=vid_path)
        obs = self._env.reset()
        step = total_reward = 0
        while True:
            self._env.render()
            if vid_path:
                vr.capture_frame()
            action, _, _ = self._ac.step(obs)
            obs, reward, terminated, _ = self._env.step(action)
            step += 1
            total_reward += reward
            if terminated:
                print(f'Episode finished after {step} steps\nTotal reward: {total_reward}')
                break
        self._env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trust Region Policy Optimization')
    parser.add_argument('--env', type=str, choices=['CartPole-v0', 'Pendulum-v1'],
                        help='OpenAI enviroment name')
    parser.add_argument('--eval', action='store_true',
                        help='Whether to enable evaluation')
    parser.add_argument('--model-path', type=str,
                        help='Model path to load')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--v-lr', type=float,
                        help='Learning rate for value function optimizer')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs')
    parser.add_argument('--steps-per-epoch', type=int,
                        help='Maximum number of steps for each epoch')
    parser.add_argument('--train-v-iters', type=int,
                        help='Number of gradient descent steps to take on value function per epoch')
    parser.add_argument('--max-ep-len', type=int,
                        help='Maximum episode/trajectory length')
    parser.add_argument('--gamma', type=float,
                        help='Discount factor')
    parser.add_argument('--lamb', type=float,
                        help='Lambda for Generalized Advantage Estimation')
    parser.add_argument('--goal', type=int,
                        help='Total reward threshold for early stopping')
    parser.add_argument('--delta', type=float,
                        help='KL divergence threshold')
    parser.add_argument('--damping-coeff', type=float,
                        help='Damping coefficient')
    parser.add_argument('--cg-iters', type=int,
                        help='Number of iterations of Conjugate gradient to perform')
    parser.add_argument('--linesearch', action='store_true', 
                        help='Whether to use backtracking line-search')
    parser.add_argument('--backtrack-iters', type=int,
                        help='Maximum number of steps in the backtracking line search')
    parser.add_argument('--backtrack-coeff', type=float,
                        help='how far back to step during backtracking line search')
    parser.add_argument('--save', action='store_true',
                        help='Whether to save training model')
    parser.add_argument('--render', action='store_true',
                        help='Whether to save training result as video')
    parser.add_argument('--plot', action='store_true',
                        help='Whether to plot training statistics and save as image')
    args = parser.parse_args()
    
    if not args.eval and args.model_path or (not args.model_path and args.eval):
        parser.error('Arguments --eval & --model-path must be specified at the same time.')
    if args.linesearch and not (args.backtrack_coeff and args.backtrack_iters):
        parser.error('Arguments --backtrack-iters & --backtrack-coeff are required when enabling --linesearch.')

    agent = TRPO(args)
    if args.eval:
        agent.test(model_path=args.model_path)
    else:
        agent.train()
