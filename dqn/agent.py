import gym
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as opt

from abc import ABC, abstractmethod
from typing import List
from collections import deque
import sys
from os.path import dirname, join, realpath
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, 'utils'))
import render
import plot
from network import DeepQNetwork
from replay_buffer import ReplayBuffer


class Agent(ABC):
    '''
    Agent abstract class
    '''

    @abstractmethod
    def behave(self, state):
        pass


    @abstractmethod
    def learn(self):
        pass


    @abstractmethod
    def run(self):
        pass



class DQN(Agent):
    '''
    DQN agent class
    '''

    def __init__(self, env_name: str,
                epsilon_init: float,
                epsilon_final: float,
                gamma: float,
                lr: float,
                buffer_size: int,
                batch_size: int,
                update_freq: int,
                tau: float,
                n_eps: int,
                horizon: int,
                logging_window: int,
                termination: float,
                verbose: bool=False,
                verbose_freq: int=50,
                save_model: bool=False,
                render_video: bool=False,
                save_plot: bool=False,
                plot: bool=False,
                seed: int=0):
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
        update_freq: number of actions seleced by the agent between successive SGD updates
        tau: for Q network parameters' soft update
        n_eps: number of episodes
        horizon: number of steps allowed
        logging_window: number of episodes to be tracked
        termination: allows algorithm ends when total reward >= @termination
        verbose: whether to display result
        verbose_freq: display every @verbose_freq
        save_model: whether to save the model
        render_video: whether to render the final simulation and save it
        save_plot: whether to save the plot
        plot: whether to plot the result
        seed: random seed
        '''
        self.env = gym.make(env_name)
        state_size =  self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        self.action_space = range(action_size)
        self.epsilon = epsilon_init
        self.epsilon_final = epsilon_final
        self.gamma = gamma
        self.lr = lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size, batch_size, seed)
        self.update_freq = update_freq
        self.tau = tau
        self.n_eps = n_eps
        self.horizon = horizon
        self.logging_window = logging_window
        self.termination = termination
        self.verbose = verbose
        self.verbose_freq = verbose_freq
        self.model_path = f'../models/{self.name()}-{env_name}.pth' \
            if save_model else None
        self.video_path = f'../outputs/videos/{self.name()}-{env_name}.mp4' \
            if render_video else None
        self.image_path = f'../outputs/images/{self.name()}-{env_name}.png' \
            if save_plot else None
        self.plot = plot
        random.seed(seed)
        self.env.seed(seed)
        self.current_step = 0

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.Qnet = DeepQNetwork(state_size, action_size, [64, 64], seed).to(self.device)
        self.Qnet_target = DeepQNetwork(state_size, action_size, [64, 64], seed).to(self.device)
        self.optimizer = opt.Adam(self.Qnet.parameters(), lr=lr)


    def name(self) -> str:
        return type(self).__name__


    def anneal_epsilon(self, epsilon_decay: float) -> None:
        '''
        Epsilon linearly annealing

        Parameters
        ----------
        epsilon_decay: decrease amount
        '''
        if self.epsilon > self.epsilon_final:
            self.epsilon -= epsilon_decay


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


    def behave(self, state: np.ndarray) -> int:
        '''
        Select action according to the behavior policy
            here we are using epsilon-greedy as behavior policy
        '''
        if random.random() <= self.epsilon:
            action = random.choice(self.action_space)
        else:
            state = torch.tensor(np.array(state)).to(self.device)
            self.Qnet.eval()
            with torch.no_grad():
                action_values = self.Qnet(state)
            action = torch.argmax(action_values).item()
            self.Qnet.train()
        return action


    def get_target(self, next_states, rewards, terminated):
        q_targets_next = torch.max(self.Qnet_target(next_states), dim=1)[0].unsqueeze(1)
        q_targets = rewards + self.gamma * q_targets_next * (1 - terminated)

        return q_targets


    def learn(self) -> None:
        if self.current_step % self.update_freq == 0 \
                and len(self.buffer) >= self.batch_size:
            states, actions, rewards, next_states, terminated = self.buffer.sample()
            states = torch.from_numpy(np.vstack(states)).float().to(self.device)
            actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
            rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
            next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
            terminated = torch.from_numpy(np.vstack(terminated).astype(np.uint8)).float().to(self.device)

            q_targets = self.get_target(next_states, rewards, terminated)
            q_expected = self.Qnet(states).gather(1, actions)
            
            loss = F.mse_loss(q_expected, q_targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            for target_param, param in zip(self.Qnet_target.parameters(), self.Qnet.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        self.current_step += 1


    def load(self, model_path: str) -> None:
        '''
        Load checkpoint
        '''
        self.Qnet.load_state_dict(torch.load(model_path))
        self.Qnet.eval()


    def save(self) -> None:
        '''
        Save checkpoint
        '''
        torch.save(self.Qnet.state_dict(), self.model_path)


    def run(self) -> List[float]:
        total_reward_list = []
        total_reward_history = deque(maxlen=self.logging_window)
        epsilon_decay = (self.epsilon - self.epsilon_final) / (self.n_eps / 2)

        for ep in range(1, self.n_eps + 1):
            state = self.env.reset()
            total_reward = 0
            step = 0

            while step < self.horizon:
                action = self.behave(state)
                next_state, reward, terminated, _ = self.env.step(action)
                total_reward += reward

                if terminated:
                    break

                self.store_transition(state, action, reward, next_state, terminated)
                self.learn()
                state = next_state
                step += 1

            total_reward_list.append(total_reward)
            total_reward_history.append(total_reward)
            self.anneal_epsilon(epsilon_decay)

            if self.verbose and ep % self.verbose_freq == 0:
                print(f'Ep {ep}, average total reward {np.mean(total_reward_history):.2f}')
            if np.mean(total_reward_history) >= self.termination:
                print(f'Environment solved in {ep} episodes!')
                if (self.model_path):
                    self.save()
                if (self.video_path):
                    assert self.model_path, 'Model path needed!'
                    render.save_video(self.env, self, self.model_path, self.video_path)
                break

        if self.plot:
            plot.total_reward(total_reward_list, self.image_path)

        return total_reward_list



class DoubleDQN(DQN):
    '''
    Double DQN agent class
    '''

    def __init__(self, env_name: str,
                epsilon_init: float,
                epsilon_final: float,
                gamma: float,
                lr: float,
                buffer_size: int,
                batch_size: int,
                update_freq: int,
                tau: float,
                n_eps: int,
                horizon: int,
                logging_window: int,
                termination: float,
                verbose: bool=False,
                verbose_freq: int=50,
                save_model: bool=False,
                render_video: bool=False,
                save_plot: bool=False,
                plot: bool=False,
                seed: int=0):
        super().__init__(env_name, epsilon_init, epsilon_final, gamma, lr,
                        buffer_size, batch_size, update_freq, tau,
                        n_eps, horizon, logging_window, termination,
                        verbose, verbose_freq, save_model, render_video,
                        save_plot, plot, seed)


    def get_target(self, next_states, rewards, terminated):
        max_actions = torch.argmax(self.Qnet(next_states), dim=1).unsqueeze(1)
        q_targets_max_actions = self.Qnet_target(next_states).gather(1, max_actions)
        q_targets = rewards + self.gamma * q_targets_max_actions * (1 - terminated)

        return q_targets
