from os.path import join
import gym
from gym.wrappers.monitoring import video_recorder
import torch
from torch.optim import Adam
import numpy as np
from policies import MLPActorCritic
from utils import Buffer


class VPG:


    def __init__(self, args,
                models_dir: str='./output/models',
                vids_dir: str='./output/videos',
                imgs_dir: str='./output/images'):
        '''
        Vanilla Policy Gradient with Actor-Critic approach & 
            Generalized Advantage Estimators

        :param env: environment name
        :param seed: random seed
        :param pi_lr: learning rate for policy network (actor) optimization
        :param v_lr: learning rate for value network (critic) optimization
        :param epochs: number of epochs
        :param step_per_epoch: maximum number of steps per epoch
        :param train_v_iters: frequency of gradient descent on value network
        :param max_ep_len: maximum episode/trajectory length
        :param gamma: discount factor
        :param lamb: eligibility trace
        :param goal: ideal total reward to finish learning
        :param save: whether to save the final model
        :param render: whether to render the training result in video
        :param plot: whether to plot the statistics and save as image
        '''
        self._env = gym.make(args.env)
        self._env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        observation_space = self._env.observation_space
        action_space = self._env.action_space
        self._ac = MLPActorCritic(observation_space, action_space)
        self._actor_opt = Adam(self._ac.actor.parameters(), lr=args.pi_lr)
        self._critic_opt = Adam(self._ac.critic.parameters(), lr=args.v_lr)
        self._epochs = args.epochs
        self._steps_per_epoch = args.steps_per_epoch
        self._train_v_iters = args.train_v_iters
        self._buffer = Buffer(args.max_ep_len, args.gamma, args.lamb)
        self._goal = args.goal
        basename = f'VPG-{args.env}'
        self._model_path = join(models_dir, f'{basename}.pth') if args.save else None
        self._vid_path = join(vids_dir, f'{basename}.mp4') if args.render else None
        self._plot_path = join(imgs_dir, f'{basename}.png') if args.plot else None


    def _compute_pi_loss(self, observations, actions, advs):
        '''
        :param observations:
        :param actions:
        :param advs:
        '''
        _, eligibilities = self._ac.actor(torch.as_tensor(observations, dtype=torch.float32),
                                        torch.as_tensor(actions, dtype=torch.int32))
        loss = -(eligibilities * advs).mean()
        return loss


    def _compute_v_loss(self, observations, reward_to_go_list):
        '''
        :param observations:
        :param reward_to_go_list:
        '''
        values = self._ac.critic(torch.as_tensor(observations, dtype=torch.float32))
        loss = ((values - reward_to_go_list) ** 2).mean()
        return loss


    def _update_params(self, trajectory_data):
        '''
        Update parameters
        '''
        observations, actions, advs, reward_to_go_list = trajectory_data

        self._actor_opt.zero_grad()
        pi_loss = self._compute_pi_loss(torch.as_tensor(observations, dtype=torch.float32),
                                         torch.as_tensor(actions, dtype=torch.int32),
                                         torch.as_tensor(advs, dtype=torch.float32))
        pi_loss.backward()
        self._actor_opt.step()

        for _ in range(self._train_v_iters):
            self._critic_opt.zero_grad()
            v_loss = self._compute_v_loss(torch.as_tensor(observations, dtype=torch.float32),
                                torch.as_tensor(reward_to_go_list, dtype=torch.float32))
            v_loss.backward()
            self._critic_opt.step()

        return pi_loss, v_loss


    def _train_one_epoch(self):
        '''
        One epoch training
        '''
        epoch_step = 0
        returns = []
        eps_len = []

        while epoch_step < self._steps_per_epoch:
            observation = self._env.reset()
            rewards = []

            while True:
                action, log_prob, value = self._ac.step(observation)
                next_observation, reward, terminated, _ = self._env.step(action)

                self._buffer.add(observation, int(action), reward, terminated, float(value))
                observation = next_observation
                rewards.append(reward)

                if terminated:
                    return_, ep_len = sum(rewards), len(rewards)
                    epoch_step += ep_len
                    returns.append(return_)
                    eps_len.append(ep_len)
                    break

        trajectory_data = self._buffer.get()
        pi_loss, v_loss = self._update_params(trajectory_data)
        return pi_loss, v_loss, returns, eps_len


    def train(self):
        for epoch in range(self._epochs):
            pi_loss, v_loss, returns, eps_len = self._train_one_epoch()
            print('epoch: %3d \t pi_loss: %.3f \t v_loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (epoch, pi_loss, v_loss, np.mean(returns), np.mean(eps_len)))
            if np.mean(returns) >= self._goal:
                print(f'Environment solved at epoch {epoch}!')
                break
        self._env.close()
        if self._model_path:
            torch.save(self._ac.actor.state_dict(), self._model_path)
        if self._vid_path:
            self.test(vid_path=self._vid_path)
        if self._plot_path:
            pass


    def load(self, model_path: str):
        self._ac.actor.load_state_dict(torch.load(model_path))
        self._ac.actor.eval()


    def test(self, vid_path: str=None, model_path: str=None):
        if model_path:
            self.load(model_path)
        if vid_path:
            vr = video_recorder.VideoRecorder(self._env, path=vid_path)

        obs = self._env.reset()
        step = 0
        while True:
            self._env.render()
            if vid_path:
                vr.capture_frame()

            action, _, _ = self._ac.step(obs)
            obs, reward, terminated, _ = self._env.step(action)
            step += 1

            if terminated:
                print(f'Episode finished after {step} steps')
                break
        del vr
