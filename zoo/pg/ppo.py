import torch
from torch.optim import Adam
import numpy as np
import gym
from gym.wrappers.monitoring import video_recorder
import argparse
from os.path import join
import random
from utils import Buffer
from network import MLPActorCritic


class PPO:


    def __init__(self, args,
                model_dir='./output/models',
                video_dir='./output/videos',
                figure_dir='./output/figures'):
        '''
        PPO w/ Actor-Critic approach & 
            Generalized Advantage Estimators & 
            using rewards-to-go as target for the value function

        :param env: (str) OpenAI's environment name
        :param seed: (int) Seed for RNG
        :param pi_lr: (float )Learning rate for policy opitimizer
        :param v_lr: (float) Learning rate for value function optimizer
        :param epochs: (int) Number of epochs
        :param steps_per_epoch: (int) Maximum number of steps per epoch
        :param train_pi_iters: (int) Number of GD-steps to take on policy loss per epoch
        :param train_v_iters: (int) Number of GD-steps to take on value function per epoch
        :param max_ep_len: (int) Maximum episode/trajectory length
        :param gamma: (float) Discount factor
        :param lamb: (float) Lambda for GAE
        :param kl_target: (float) KL divergence threshold
        :param clip: (bool) Whether to use clipping, enable penalty otherwise
        :param clip_ratio: (float) Hyperparamter for clipping the policy objective
        :param goal: (float) Total reward threshold for early stopping
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
            self._actor_opt = Adam(self._ac.actor.parameters(), lr=args.pi_lr)
            self._critic_opt = Adam(self._ac.critic.parameters(), lr=args.v_lr)
            self._epochs = args.epochs
            self._steps_per_epoch = args.steps_per_epoch
            self._train_pi_iters = args.train_pi_iters
            self._train_v_iters = args.train_v_iters
            self._max_ep_len = args.max_ep_len
            self._buffer = Buffer(args.max_ep_len, args.gamma, args.lamb)
            self._kl_target = args.kl_target
            if args.clip:
                self._clip_ratio = args.clip_ratio
            else:
                pass
            self._goal = args.goal
            basename = f'PPO-{args.env}'
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
        observations, actions, logps_old, advs, rewards_to_go = self._buffer.get()

        def compute_pi_loss():
            pi, logp = self._ac.actor(observations, actions)
            ratio = torch.exp(logp - logps_old)
            loss_cpi = ratio * advs
            clip_advs = ((1 + self._clip_ratio) * (advs > 0) + (1 - self._clip_ratio) * (advs < 0)) * advs
            pi_loss = -torch.min(loss_cpi, clip_advs).mean()

            # approximated KL
            approx_kl = (logps_old - logp).mean()
            return pi_loss, approx_kl.item()


        def compute_v_loss():
            '''
            Compute value function loss
            '''
            values = self._ac.critic(observations)
            v_loss = ((values - rewards_to_go) ** 2).mean()
            return v_loss

        for step in range(1, self._train_pi_iters + 1):
            self._actor_opt.zero_grad()
            pi_loss, kl = compute_pi_loss()
            if kl > 1.5 * self._kl_target:
                print(f'Early stopping at step {step} due to exceed KL target')
                break
            pi_loss.backward()
            self._actor_opt.step()

        for _ in range(self._train_v_iters):
            self._critic_opt.zero_grad()
            v_loss = compute_v_loss()
            v_loss.backward()
            self._critic_opt.step()

        update_info = dict()
        update_info['pi_loss'] = pi_loss.item()
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
        info = self._update_params()
        info['returns'] = returns
        info['eps_len'] = eps_len
        return info


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
    parser = argparse.ArgumentParser(description='Proximal Policy Optimization')
    parser.add_argument('--env', type=str, choices=['CartPole-v0', 'Pendulum-v1'],
                        help='OpenAI enviroment name')
    parser.add_argument('--eval', action='store_true',
                        help='Whether to enable evaluation')
    parser.add_argument('--model-path', type=str,
                        help='Model path to load')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--pi-lr', type=float,
                        help='Learning rate for policy optimizer')
    parser.add_argument('--v-lr', type=float,
                        help='Learning rate for value function optimizer')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs')
    parser.add_argument('--steps-per-epoch', type=int,
                        help='Maximum number of steps for each epoch')
    parser.add_argument('--train-pi-iters', type=int,
                        help='Number of gradient descent steps to take on policy loss per epoch')
    parser.add_argument('--train-v-iters', type=int,
                        help='Number of gradient descent steps to take on value function per epoch')
    parser.add_argument('--max-ep-len', type=int,
                        help='Maximum episode/trajectory length')
    parser.add_argument('--gamma', type=float,
                        help='Discount factor')
    parser.add_argument('--lamb', type=float,
                        help='Lambda for Generalized Advantage Estimation')
    parser.add_argument('--kl-target', type=float,
                        help='KL divergence threshold')
    parser.add_argument('--clip', action='store_true',
                        help='Whether to use PPO-Clip, use PPO-Penalty otherwise')
    parser.add_argument('--clip-ratio', type=float,
                        help='Hyperparameter for clipping in the policy objective')
    parser.add_argument('--goal', type=int,
                        help='Total reward threshold for early stopping')
    parser.add_argument('--save', action='store_true',
                        help='Whether to save training model')
    parser.add_argument('--render', action='store_true',
                        help='Whether to save training result as video')
    parser.add_argument('--plot', action='store_true',
                        help='Whether to plot training statistics and save as image')
    args = parser.parse_args()
    
    if not args.eval and args.model_path or (not args.model_path and args.eval):
        parser.error('Arguments --eval & --model-path must be specified at the same time.')
    if args.clip and not args.clip_ratio:
        parser.error('Argument --clip-ratio is required when --clip is enabled.')

    agent = PPO(args)
    if args.eval:
        agent.test(model_path=args.model_path)
    else:
        agent.train()
