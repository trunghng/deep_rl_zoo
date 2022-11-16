import sys
from os.path import dirname, join, realpath
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, '..'))

import gym
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as opt
import math

from typing import List, Tuple
from collections import deque
from network import DeepQNet, CNNDeepQNet
from replay_buffer import ReplayBuffer
from utils import plot, save_video
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch


class DQN:
    '''
    DQN agent class
    '''

    def __init__(self,
                env_name: str,
                epsilon_init: float,
                epsilon_final: float,
                epsilon_decay: float,
                gamma: float,
                lr: float,
                buffer_size: int,
                batch_size: int,
                train_freq: int,
                update_target: int,
                tau: float,
                num_episodes: int,
                logging_interval: int,
                termination: float,
                print_freq: int,
                save_model: bool,
                render_video: bool,
                plot: bool,
                atari: bool,
                seed: int,
                device):
        '''
        Parameters
        ----------
        env_name: OpenAI's environment name
        epsilon_init: initial value for exploration param, epsilon, linearly annealing
        epsilon_final: final value for epsilon linearly annealing
        gamma: discount factor
        lr: learning rate
        buffer_size: replay buffer size
        batch_size: mini batch size
        train_freq: number of actions selected by the agent between successive SGD updates
        update_target: 
        tau: for Q network parameters' soft update
        num_episodes: number of episodes
        logging_interval: number of episodes to be tracked
        termination: allows algorithm ends when total reward >= @termination
        print_freq: result printing frequency
        save_model: whether to save the model
        render_video: whether to render the final simulation and save it
        plot: whether to plot the result
        atari: whether to use atari environment
        seed: random seed
        device: device type
        '''
        self.device = device
        if atari:
            env = make_atari(env_name)
            env = wrap_deepmind(env)
            self.env = wrap_pytorch(env)
            state_dim =  self.env.observation_space.shape
            n_actions = self.env.action_space.n
            self.Qnet = CNNDeepQNet(state_dim, n_actions, seed).to(self.device)
            self.Qnet_target = CNNDeepQNet(state_dim, n_actions, seed).to(self.device)
        else:
            self.env = gym.make(env_name)
            state_dim =  self.env.observation_space.shape[0]
            n_actions = self.env.action_space.n
            self.Qnet = DeepQNet(state_dim, n_actions, seed).to(self.device)
            self.Qnet_target = DeepQNet(state_dim, n_actions, seed).to(self.device)

        self.optimizer = opt.Adam(self.Qnet.parameters(), lr=lr)
        self.action_space = range(n_actions)
        self.epsilon_init = epsilon_init
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size, seed)
        self.train_freq = train_freq
        self.update_target = update_target
        self.tau = tau
        self.n_eps = num_episodes
        self.logging_interval = logging_interval
        self.termination = termination
        self.print_freq = print_freq
        self.model_path = f'/models/{self.name()}-{env_name}.pth' \
            if save_model else None
        self.plot = plot
        random.seed(seed)
        self.env.seed(seed)
        self.step = 0


    def name(self) -> str:
        return type(self).__name__


    def anneal_epsilon(self, ep: int):
        '''
        Epsilon linearly annealing
        '''
        return self.epsilon_final + (self.epsilon_init - self.epsilon_final) * math.exp(-1. * ep /self.epsilon_decay)


    def store_transition(self,
                        state: np.ndarray,
                        action: int,
                        reward: float,
                        next_state: np.ndarray,
                        terminated: bool) -> None:
        '''
        Store transition into the replay buffer

        Parameters
        ----------
        state: current state of the agent
        action: action taken at @state
        reward: reward corresponding
        next_state: next state of the agent
        terminated: whether terminated
        '''
        self.buffer.add(state, action, reward, next_state, terminated)


    def behave(self, state: np.ndarray, epsilon: float=0) -> int:
        '''
        Select action according to the behavior policy
            here we are using epsilon-greedy as behavior policy
        '''
        if random.random() <= epsilon:
            action = random.choice(self.action_space)
        else:
            state = torch.tensor(np.array(state))[None, ...].to(self.device)
            self.Qnet.eval()
            with torch.no_grad():
                action_values = self.Qnet(state.float())
            action = torch.argmax(action_values).item()
            self.Qnet.train()
        return action


    def compute_td_loss(self, states, actions, rewards, next_states, terminated):
        q_target_next = torch.max(self.Qnet_target(next_states), dim=1)[0].unsqueeze(1)
        q_target = rewards + self.gamma * q_target_next * (1 - terminated)
        q_expected = self.Qnet(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_target)
        return loss


    def learn(self):
        if self.step % self.train_freq == 0 \
                and len(self.buffer) >= self.batch_size:
            states, actions, rewards, next_states, terminated = self.buffer.sample(self.batch_size)
            states = torch.from_numpy(states).float().to(self.device)
            actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
            rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
            next_states = torch.from_numpy(next_states).float().to(self.device)
            terminated = torch.from_numpy(np.vstack(terminated).astype(np.uint8)).float().to(self.device)

            loss = self.compute_td_loss(states, actions, rewards, next_states, terminated)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.step % self.update_target == 0:
            for target_param, param in zip(self.Qnet_target.parameters(), self.Qnet.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        self.step += 1


    def load(self, model_path: str) -> None:
        '''
        Load checkpoint
        '''
        self.Qnet.load_state_dict(torch.load(model_path))
        self.Qnet.eval()


    def save(self) -> None:
        '''
        Save model
        '''
        torch.save(self.Qnet.state_dict(), self.model_path)


    def train(self) -> Tuple[List[float], List[float]]:
        scores = []
        avg_scores = []

        for ep in range(1, self.n_eps):
            state = self.env.reset()
            total_reward = 0
            epsilon = self.anneal_epsilon(ep)

            while True:
                action = self.behave(state, epsilon)
                next_state, reward, terminated, _ = self.env.step(action)
                total_reward += reward
                if terminated:
                    break
                self.store_transition(state, action, reward, next_state, terminated)
                self.learn()
                state = next_state

            scores.append(total_reward)
            avg_score = np.mean(scores[-self.logging_interval:])
            avg_scores.append(avg_score)

            if self.print_freq and ep % self.print_freq == 0:
                print(f'Ep {ep}, average score {avg_score:.2f}')
            if avg_score >= self.termination:
                print(f'Environment solved in {ep} episodes!')
                if self.model_path:
                    self.save()
                if self.render_video:
                    try:
                        save_video(self.env, self, self.model_path)
                    except Exception:
                        print('Model path needed!')
                break

        if self.plot:
            plot(scores, 'Episodes', 'Score', 'Score per episode')
            plot(avg_scores, 'Episodes', 'Average score', 'Average score per episode')
        self.env.close()

        return scores, avg_scores


    def test(self, model_path: str):
        self.load(f'./models/{model_path}')
        state = self.env.reset()
        while True:
            self.env.render()
            action = self.behave(state)
            state, _, terminated, _ = self.env.step(action)
            if terminated:
                break
        self.env.close()


class DoubleDQN(DQN):
    '''
    Double DQN agent class
    '''

    def __init__(self,
                env_name: str,
                epsilon_init: float,
                epsilon_final: float,
                epsilon_decay: float,
                gamma: float,
                lr: float,
                buffer_size: int,
                batch_size: int,
                train_freq: int,
                target_update: int,
                tau: float,
                n_eps: int,
                logging_interval: int,
                termination: float,
                print_freq: int,
                save_model: bool,
                render_video: bool,
                plot: bool,
                atari: bool,
                seed: int,
                device: str):
        super().__init__(env_name, epsilon_init, epsilon_final, epsilon_decay, gamma,
                        lr, buffer_size, batch_size, train_freq, target_update, 
                        tau, n_eps, logging_interval, termination, print_freq, save_model, 
                        render_video, plot, atari, seed, device)


    def compute_td_loss(self, states, actions, rewards, next_states, terminated):
        greedy_actions = torch.argmax(self.Qnet(next_states), dim=1).unsqueeze(1)
        q_target_greedy = self.Qnet_target(next_states).gather(1, greedy_actions)
        q_target = rewards + self.gamma * q_target_greedy * (1 - terminated)
        q_expected = self.Qnet(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_target)
        return loss
