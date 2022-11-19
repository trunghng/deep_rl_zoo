import gym
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as opt
import math

from typing import List, Tuple
from collections import deque
from os.path import basename, splitext
from network import DeepQNet, CNNDeepQNet, DuelingQNet
from replay_buffer import ReplayBuffer
from utils import plot
from gym.wrappers.monitoring import video_recorder


class DQN:
    '''
    DQN agent class
    '''

    def __init__(self,
                env,
                state_dim: List[int],
                n_actions: int,
                args):
        '''
        Parameters
        ----------
        env_name: OpenAI's environment name
        state_dim: state dimensionality
        n_actions: number of actions
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
        self.env = env
        self.device = args.device
        # Select network architecture
        if args.dueling:
            architecture_txt = 'Dueling Q-network'
            network = DuelingQNet
        elif args.atari:
            architecture_txt = 'CNN Q-network'
            network = CNNDeepQNet
        else:
            architecture_txt = 'Q-network'
            network = DeepQNet
        self.Qnet = network(state_dim, n_actions).to(self.device)
        self.Qnet_target = network(state_dim, n_actions).to(self.device)

        # Select Q-learning update type
        if args.double:
            td_error_cal_txt = 'Double Q-learning'
            self._compute_td_loss_ = self._compute_td_loss_double_q
        else:
            td_error_cal_txt = 'Q-learning'
            self._compute_td_loss_ = self._compute_td_loss

        # Select replay buffer type
        if args.prioritized_replay:
            replay_txt = 'prioritized replay'
            pass
        else:
            replay_txt = 'experience replay'
            self.buffer = ReplayBuffer(args.buffer_size)
        print(f'Using DQN with:\n\t- Network architecture: {architecture_txt}\
                \n\t- TD error computation: {td_error_cal_txt}\
                \n\t- Replay buffer type: {replay_txt}')

        self.optimizer = opt.Adam(self.Qnet.parameters(), lr=args.lr)
        self.action_space = range(n_actions)
        self.epsilon_init = args.epsilon_init
        self.epsilon_final = args.epsilon_final
        self.epsilon_decay = args.epsilon_decay
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.train_freq = args.train_freq
        self.update_target = args.update_target
        self.tau = args.tau
        self.n_eps = args.num_episodes
        self.logging_interval = args.logging_interval
        self.termination = args.termination
        self.print_freq = args.print_freq
        self.model_path = f'/models/{self._name()}-{args.env_name}.pth' \
            if args.save_model else None
        self.plot = args.plot
        self._seed(args.seed)
        self.step = 0


    def _seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        self.env.seed(seed)


    def _name(self) -> str:
        return type(self).__name__


    def _anneal_epsilon(self, ep: int):
        '''
        Epsilon linearly annealing
        '''
        return self.epsilon_final + (self.epsilon_init - self.epsilon_final) * math.exp(-1. * ep /self.epsilon_decay)


    def _store_transition(self,
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


    def _act(self, state: np.ndarray, epsilon: float=0) -> int:
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


    def _compute_td_loss(self, states, actions, rewards, next_states, terminated):
        '''
        Compute TD error error according to Q-learning
        '''
        q_target_next = torch.max(self.Qnet_target(next_states), dim=1)[0].unsqueeze(1)
        q_target = rewards + self.gamma * q_target_next * (1 - terminated)
        q_expected = self.Qnet(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_target)
        return loss

    def _compute_td_loss_double_q(self, states, actions, rewards, next_states, terminated):
        '''
        Compute TD error according to double Q-learning
        '''
        greedy_actions = torch.argmax(self.Qnet(next_states), dim=1).unsqueeze(1)
        q_target_greedy = self.Qnet_target(next_states).gather(1, greedy_actions)
        q_target = rewards + self.gamma * q_target_greedy * (1 - terminated)
        q_expected = self.Qnet(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_target)
        return loss


    def _learn(self):
        if self.step % self.train_freq == 0 \
                and len(self.buffer) >= self.batch_size:
            states, actions, rewards, next_states, terminated = self.buffer.sample(self.batch_size)
            states = torch.from_numpy(states).float().to(self.device)
            actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
            rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
            next_states = torch.from_numpy(next_states).float().to(self.device)
            terminated = torch.from_numpy(np.vstack(terminated).astype(np.uint8)).float().to(self.device)

            loss = self._compute_td_loss_(states, actions, rewards, next_states, terminated)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.step % self.update_target == 0:
            self.Qnet_target.load_state_dict(self.Qnet.state_dict())
        self.step += 1


    def _load(self, model_path: str) -> None:
        '''
        Load checkpoint
        '''
        self.Qnet.load_state_dict(torch.load(model_path))
        self.Qnet.eval()


    def _save(self) -> None:
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
            epsilon = self._anneal_epsilon(ep)

            while True:
                action = self._act(state, epsilon)
                next_state, reward, terminated, _ = self.env.step(action)
                total_reward += reward
                if terminated:
                    break
                self._store_transition(state, action, reward, next_state, terminated)
                self._learn()
                state = next_state

            scores.append(total_reward)
            avg_score = np.mean(scores[-self.logging_interval:])
            avg_scores.append(avg_score)

            if self.print_freq and ep % self.print_freq == 0:
                print(f'Ep {ep}, average score {avg_score:.2f}')
            if avg_score >= self.termination:
                print(f'Environment solved in {ep} episodes!')
                if self.model_path:
                    self._save()
                    self.test(self.model_path, self.render_video)
                break

        if self.plot:
            plot(scores, 'Episodes', 'Score', 'Score per episode')
            plot(avg_scores, 'Episodes', 'Average score', 'Average score per episode')
        self.env.close()

        return scores, avg_scores


    def test(self, model_path: str, render_video: bool=False):
        '''
        Parameters
        ----------
        model_path: model path to load
        render_video: whether to render output video
        '''
        self._load(f'/models/{model_path}')
        if render_video:
            video_path = f'/outputs/videos/{splitext(basename(model_path))[0]}.mp4'
            video = video_recorder.VideoRecorder(self.env, path=video_path)
            __render = lambda:\
                self.env.render(mode='rgb_array');\
                video.capture_frame()
        else:
            __render = lambda: self.env.render()
        state = self.env.reset()
        while True:
            __render()
            action = self._act(state)
            state, _, terminated, _ = self.env.step(action)
            if terminated:
                break
        self.env.close()
