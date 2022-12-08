from os.path import join
import argparse
import gym
from gym.wrappers.monitoring import video_recorder
import torch
from torch.optim import Adam
import numpy as np
from network import MLPActorCritic
from utils import Buffer


class VPG:


    def __init__(self, args,
                model_dir: str='./output/models',
                video_dir: str='./output/videos',
                figure_dir: str='./output/images'):
        '''
        Vanilla Policy Gradient with Actor-Critic approach & Generalized Advantage Estimators & 
            using rewards-to-go as target for the value function, which is chosen as the baseline

        :param env: (str)
            OpenAI's environment name
        :param seed: (int)
            Random seed
        :param pi_lr: (float)
            Learning rate for policy network (actor) optimization
        :param v_lr: (float)
            Learning rate for value network (critic) optimization
        :param epochs: (int)
            Number of epochs
        :param steps_per_epoch: (int)
            Maximum number of steps per epoch
        :param train_v_iters: (int)
            Number of GD-steps to take on value func per epoch
        :param max_ep_len: (int)
            Maximum episode/trajectory length
        :param gamma: (float)
            Discount factor
        :param lamb: (float)
            Lambda for GAE
        :param goal: (float)
            Total reward threshold for early stopping
        :param save: (bool)
            Whether to save the final model
        :param render: (bool)
            Whether to render the training result in video
        :param plot: (bool)
            Whether to plot the statistics and save as image
        :param model_dir: (str)
            Model directory
        :param video_dir: (str)
            Video directory
        :param figure_dir: (str)
            Figure directory
        '''
        self._env = gym.make(args.env)
        self._env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        observation_space = self._env.observation_space
        action_space = self._env.action_space
        self._ac = MLPActorCritic(observation_space, action_space)
        if not args.eval:
            self._actor_opt = Adam(self._ac.actor.parameters(), lr=args.pi_lr)
            self._critic_opt = Adam(self._ac.critic.parameters(), lr=args.v_lr)
            self._epochs = args.epochs
            self._steps_per_epoch = args.steps_per_epoch
            self._train_v_iters = args.train_v_iters
            self._max_ep_len = args.max_ep_len
            self._buffer = Buffer(args.max_ep_len, args.gamma, args.lamb)
            self._goal = args.goal
            basename = f'VPG-{args.env}'
            self._model_path = join(model_dir, f'{basename}.pth') if args.save else None
            self._vid_path = join(video_dir, f'{basename}.mp4') if args.render else None
            self._plot_path = join(figure_dir, f'{basename}.png') if args.plot else None


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


    def _compute_v_loss(self, observations, rewards_to_go):
        '''
        :param observations:
        :param rewards_to_go:
        '''
        values = self._ac.critic(torch.as_tensor(observations, dtype=torch.float32))
        loss = ((values - rewards_to_go) ** 2).mean()
        return loss


    def _update_params(self, trajectory_data):
        '''
        Update parameters
        '''
        observations, actions, advs, rewards_to_go = trajectory_data
        self._ac.train()

        self._actor_opt.zero_grad()
        pi_loss = self._compute_pi_loss(torch.as_tensor(observations, dtype=torch.float32),
                                        torch.as_tensor(actions, dtype=torch.int32),
                                        torch.as_tensor(advs, dtype=torch.float32))
        pi_loss.backward()
        self._actor_opt.step()

        for _ in range(self._train_v_iters):
            self._critic_opt.zero_grad()
            v_loss = self._compute_v_loss(torch.as_tensor(observations, dtype=torch.float32),
                                        torch.as_tensor(rewards_to_go, dtype=torch.float32))
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
                self._ac.eval()
                action, log_prob, value = self._ac.step(observation)
                next_observation, reward, terminated, _ = self._env.step(action)

                self._buffer.add(observation, int(action), reward, terminated, float(value))
                observation = next_observation
                rewards.append(reward)

                if terminated or (len(rewards) == self._max_ep_len):
                    return_, ep_len = sum(rewards), len(rewards)
                    epoch_step += ep_len
                    returns.append(return_)
                    eps_len.append(ep_len)
                    break

        trajectory_data = self._buffer.get()
        pi_loss, v_loss = self._update_params(trajectory_data)
        return pi_loss, v_loss, returns, eps_len


    def load(self, model_path: str):
        self._ac.actor.load_state_dict(torch.load(model_path))
        self._ac.actor.eval()


    def train(self):
        print('---Training---')
        for epoch in range(1, self._epochs + 1):
            pi_loss, v_loss, returns, eps_len = self._train_one_epoch()
            print('epoch: %3d \t pi_loss: %.3f \t v_loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (epoch, pi_loss, v_loss, np.mean(returns), np.mean(eps_len)))
            if self._goal and np.mean(returns) >= self._goal:
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
        self._env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vanilla Policy Gradient')
    parser.add_argument('--env', type=str, choices=['CartPole-v0', 'Pendulum-v1'],
                        help='OpenAI enviroment')
    parser.add_argument('--eval', action='store_true',
                        help='Whether to enable evaluation')
    parser.add_argument('--model-path', type=str,
                        help='Model path to load')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--pi-lr', type=float,
                        help='Learning rate for policy optimization')
    parser.add_argument('--v-lr', type=float,
                        help='Learning rate for value function optimization')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs')
    parser.add_argument('--steps-per-epoch', type=int,
                        help='Maximum number of epoch for each epoch')
    parser.add_argument('--train-v-iters', type=int,
                        help='Value network update frequency')
    parser.add_argument('--max-ep-len', type=int,
                        help='Maximum episode/trajectory length')
    parser.add_argument('--gamma', type=float,
                        help='Discount factor')
    parser.add_argument('--lamb', type=float,
                        help='Eligibility trace')
    parser.add_argument('--goal', type=int,
                        help='Goal total reward to end training')
    parser.add_argument('--save', action='store_true',
                        help='Whether to save training model')
    parser.add_argument('--render', action='store_true',
                        help='Whether to save training result as video')
    parser.add_argument('--plot', action='store_true',
                        help='Whether to plot training statistics and save as image')
    args = parser.parse_args()

    if not args.eval and args.model_path or (not args.model_path and args.eval):
        parser.error('Arguments --eval & --model-path must be specified together.')

    agent = VPG(args)
    if args.eval:
        agent.test(model_path=args.model_path)
    else:
        agent.train()
