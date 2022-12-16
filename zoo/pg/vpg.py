from os.path import dirname, join, realpath
import sys
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, '..'))
import argparse
import random
import gym
from gym.wrappers.monitoring import video_recorder
import torch
from torch.optim import Adam
import numpy as np
from network import MLPActorCritic
from utils import Buffer, MPIBuffer
import common.mpi_utils as mpi
from common.logger import Logger


class VPG:


    def __init__(self, args,
                model_dir='./output/models',
                video_dir='./output/videos',
                figure_dir='./output/figures'):
        '''
        Vanilla Policy Gradient with Actor-Critic approach & Generalized Advantage Estimators & 
            using rewards-to-go as target for the value function, which is chosen as the baseline

        :param env: (str) OpenAI environment name
        :param seed: (int) Seed for RNG
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
        :param render: (bool) Whether to render the training result in video
        :param plot: (bool) Whether to plot the statistics and save as image
        :param model_dir: (str) Model directory
        :param video_dir: (str) Video directory
        :param figure_dir: (str) Figure directory
        '''
        mpi.print_one(args.seed)
        self._env = gym.make(args.env)
        self._seed(args.seed)
        observation_space = self._env.observation_space
        action_space = self._env.action_space
        self._ac = MLPActorCritic(observation_space, action_space, args.hidden_layers)
        mpi.sync_params(self._ac)
        if not args.eval:
            self._actor_opt = Adam(self._ac.actor.parameters(), lr=args.pi_lr)
            self._critic_opt = Adam(self._ac.critic.parameters(), lr=args.v_lr)
            self._epochs = args.epochs
            self._steps_per_epoch = int(args.steps_per_epoch / mpi.n_procs())
            self._train_v_iters = args.train_v_iters
            self._max_ep_len = args.max_ep_len
            self._buffer = MPIBuffer(self._steps_per_epoch, args.gamma, args.lamb)
            self._goal = args.goal
            basename = f'VPG-{args.env}'
            self._model_path = join(model_dir, f'{basename}.pth') if args.save else None
            self._vid_path = join(video_dir, f'{basename}.mp4') if args.render else None
            self._plot_path = join(figure_dir, f'{basename}.png') if args.plot else None
            self._logger = Logger(args.env, 'VPG')


    def _seed(self, seed: int):
        '''
        Set global seed
        '''
        seed += 10000 * mpi.proc_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self._env.seed(seed)


    def _compute_pi_loss(self, observations, actions, advs):
        '''
        :param observations:
        :param actions:
        :param advs:
        '''
        _, log_prob = self._ac.actor(observations, actions)
        loss = -(log_prob * advs).mean()
        return loss


    def _compute_v_loss(self, observations, rewards_to_go):
        '''
        :param observations:
        :param rewards_to_go:
        '''
        values = self._ac.critic(observations)
        loss = ((values - rewards_to_go) ** 2).mean()
        return loss


    def _update_params(self):
        '''
        Update parameters
        '''
        observations, actions, log_probs, advs, rewards_to_go = self._buffer.get()

        self._actor_opt.zero_grad()
        pi_loss = self._compute_pi_loss(observations, actions, advs)
        pi_loss.backward()
        mpi.mpi_avg_grads(self._ac.actor)
        self._actor_opt.step()

        for _ in range(self._train_v_iters):
            self._critic_opt.zero_grad()
            v_loss = self._compute_v_loss(observations, rewards_to_go)
            v_loss.backward()
            mpi.mpi_avg_grads(self._ac.critic)
            self._critic_opt.step()

        self._logger.add(PiLoss=pi_loss.item(), VLoss=v_loss.item())


    def _train_one_epoch(self):
        '''
        One epoch training
        '''
        step = 0
        returns = []
        eps_len = []

        while step < self._steps_per_epoch:
            observation = self._env.reset()
            rewards = []

            while True:
                action, log_prob, value = self._ac.step(observation)
                next_observation, reward, terminated, _ = self._env.step(action)
                self._buffer.add(observation, action, reward, float(value), float(log_prob))
                observation = next_observation
                rewards.append(reward)
                step += 1

                if terminated or (len(rewards) == self._max_ep_len) or (step == self._steps_per_epoch - 1):
                    if terminated:
                        value = 0
                        return_, ep_len = sum(rewards), len(rewards)
                        returns.append(return_)
                        eps_len.append(ep_len)
                        self._logger.add(Return=returns, EpLen=eps_len)
                    else:
                        _, _, value = self._ac.step(observation)
                    self._buffer.finish_rollout(value)
                    break
        self._update_params()


    def train(self):
        mpi.print_one('---Training---')
        for epoch in range(1, self._epochs + 1):
            self._train_one_epoch()
            # print('epoch: %3d \t pi_loss: %.3f \t v_loss: %.3f \t return: %.3f \t ep_len: %.3f'%
            #     (epoch, pi_loss, v_loss, np.mean(returns), np.mean(eps_len)))
            # if self._goal and np.mean(returns) >= self._goal:
            #     print(f'Environment solved at epoch {epoch}!')
            #     break
            self._logger.log_tabular('Epoch', epoch)
            self._logger.log_tabular('PiLoss')
            self._logger.log_tabular('VLoss')
            self._logger.log_tabular('Return')
            self._logger.log_tabular('EpLen')
            self._logger.log()
        self._env.close()
        if self._model_path:
            if mpi.proc_rank() == 0:
                torch.save(self._ac.actor.state_dict(), self._model_path)
                print(f'Model is saved successfully at {self._model_path}')
        if self._vid_path:
            if mpi.proc_rank() == 0:
                self.test(vid_path=self._vid_path)
                print(f'Video is renderred successfully at {self._vid_path}')
        if self._plot_path:
            pass


    def test(self, vid_path: str=None, model_path: str=None):
        if mpi.proc_rank() == 0:
            print('---Evaluating---')
            if model_path:
                self._ac.actor.load_state_dict(torch.load(model_path))
            if vid_path:
                vr = video_recorder.VideoRecorder(self._env, path=vid_path)
            obs = self._env.reset()
            step = total_reward =0
            while True:
                self._env.render()
                if vid_path:
                    vr.capture_frame()
                action, _, _ = self._ac.step(obs)
                obs, reward, terminated, _ = self._env.step(action)
                step += 1
                total_reward += reward
                if terminated:
                    print(f'Episode finished after {step} steps\nTotal reward {total_reward}')
                    break
            self._env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vanilla Policy Gradient')
    parser.add_argument('--env', type=str, choices=['CartPole-v0', 'HalfCheetah-v2'],
                        help='OpenAI enviroment name')
    parser.add_argument('--eval', action='store_true',
                        help='Whether to enable evaluation')
    parser.add_argument('--model-path', type=str,
                        help='Model path to load')
    parser.add_argument('--cpu', type=int,
                        help='Number of CPUs for parallel computing')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--hidden-layers', nargs='+', type=int,
                        help='Hidden layers size of policy & value function networks')
    parser.add_argument('--pi-lr', type=float,
                        help='Learning rate for policy optimizer')
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
    parser.add_argument('--save', action='store_true',
                        help='Whether to save training model')
    parser.add_argument('--render', action='store_true',
                        help='Whether to save training result as video')
    parser.add_argument('--plot', action='store_true',
                        help='Whether to plot training statistics and save as image')
    args = parser.parse_args()
    mpi.mpi_fork(args.cpu)
    mpi.setup_pytorch_for_mpi()

    if not args.eval and args.model_path or (not args.model_path and args.eval):
        parser.error('Arguments --eval & --model-path must be specified together.')

    agent = VPG(args)
    if args.eval:
        agent.test(model_path=args.model_path)
    else:
        agent.train()
