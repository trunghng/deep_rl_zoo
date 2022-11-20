import gym
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as opt
from gym.wrappers.monitoring import video_recorder

from typing import List, Tuple
import sys
from os.path import dirname, join, realpath, basename, splitext
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, '..'))
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from network import DeepQNet, CNNDeepQNet, DuelingQNet
from replay_buffer import ReplayBuffer, PrioritizedReplay
from utils import plot


class DQN:
    '''
    DQN agent class
    '''

    def __init__(self, args):
        '''
        Parameters
        ----------
        env_name: OpenAI's environment name
        atari: whether to use atari environment
        eval: whether to enable evaluation
        dueling: whether to use dueling network
        double: whether to use double Q-learning
        tau: smoothness parameter, used for target network soft update
        prioritized_replay: whether to use prioritized replay buffer
        epsilon_init: initial value for exploration param, epsilon, linearly annealing
        epsilon_final: final value for epsilon linearly annealing
        gamma: discount factor
        lr: learning rate
        buffer_size: replay buffer size
        batch_size: mini batch size
        train_freq: number of actions selected by the agent between successive SGD updates
        update_target: target network update frequency
        num_episodes: number of episodes
        logging_interval: number of episodes to be tracked
        termination: allows algorithm ends when total reward >= @termination
        print_freq: result printing frequency
        save_model: whether to save the model
        render_video: whether to render the final simulation and save it
        plot: whether to plot the result
        seed: random seed
        device: device type
        '''
        self._env, state_dim, n_actions = self._create_env(args.env_name, args.atari, args.eval)
        self._device = args.device
        seed = args.seed
        random.seed(seed)
        self._env.seed(seed)

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
        self._Qnet = network(state_dim, n_actions, seed).to(self._device)
        self._Qnet_target = network(state_dim, n_actions, seed).to(self._device)

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
            self._buffer = PrioritizedReplay(args.buffer_size, seed)
        else:
            replay_txt = 'experience replay'
            self._buffer = ReplayBuffer(args.buffer_size, seed)

        # Select target network update type
        if args.tau:
            update_target_txt = 'enabled'
            self._tau = args.tau
            self._update_target_network = self._soft_update_target
        else:
            update_target_txt = 'disabled'
            self._update_target_network = self._update_target

        print(f'Using DQN with:\n- Network architecture: {architecture_txt}\
                \n- TD error computation: {td_error_cal_txt}\
                \n- Replay buffer type: {replay_txt}\
                \n- Soft update target: {update_target_txt}')

        self._optimizer = opt.Adam(self._Qnet.parameters(), lr=args.lr)
        self._action_space = range(n_actions)
        self._epsilon_init = args.epsilon_init
        self._epsilon_final = args.epsilon_final
        self._epsilon_decay = args.epsilon_decay
        self._gamma = args.gamma
        self._batch_size = args.batch_size
        self._train_freq = args.train_freq
        self._update_target = args.update_target
        self._n_eps = args.num_episodes
        self._logging_interval = args.logging_interval
        self._termination = args.termination
        self._print_freq = args.print_freq
        self._model_path = f'./models/{args.env_name}.pth' if args.save_model else None
        self._image_paths = [f'./outputs/images/{args.env_name}-score-per-ep.png',\
            f'./outputs/images/{args.env_name}-avg-score-per-ep.png'] if args.plot else None
        self._render_video = args.render_video
        self._step = 0


    def _create_env(self, env_name: str, atari: bool, evaluate: bool):
        if evaluate:
            mode = 'human'
        else:
            mode = 'rgb_array'

        if atari:
            env = make_atari(env_name, mode)
            env = wrap_deepmind(env)
            env = wrap_pytorch(env)
            state_dim =  env.observation_space.shape
            n_actions = env.action_space.n
        else:
            env = gym.make(env_name)
            state_dim = env.observation_space.shape[0]
            n_actions = env.action_space.n
        return env, state_dim, n_actions


    def _anneal_epsilon(self, ep: int):
        '''
        Epsilon linearly annealing
        '''
        return self._epsilon_final + (self._epsilon_init - self._epsilon_final) \
                * np.exp(-1. * ep / self._epsilon_decay)


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
        self._buffer.add(state, action, reward, next_state, terminated)


    def _act(self, state: np.ndarray, epsilon: float=0.0) -> int:
        '''
        Select action according to the behavior policy
            here we use epsilon-greedy as behavior policy
        '''
        if random.random() <= epsilon:
            action = random.choice(self._action_space)
        else:
            state = torch.tensor(np.array(state))[None, ...].to(self._device)
            self._Qnet.eval()
            with torch.no_grad():
                action_values = self._Qnet(state.float())
            action = torch.argmax(action_values).item()
            self._Qnet.train()
        return action


    def _compute_td_loss(self, states, actions, rewards, next_states, terminated):
        '''
        Compute TD error error according to Q-learning
        '''
        q_target_next = torch.max(self._Qnet_target(next_states), dim=1)[0].unsqueeze(1)
        q_target = rewards + self._gamma * q_target_next * (1 - terminated)
        q_expected = self._Qnet(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_target)
        return loss

    def _compute_td_loss_double_q(self, states, actions, rewards, next_states, terminated):
        '''
        Compute TD error according to double Q-learning
        '''
        greedy_actions = torch.argmax(self._Qnet(next_states), dim=1).unsqueeze(1)
        q_target_greedy = self._Qnet_target(next_states).gather(1, greedy_actions)
        q_target = rewards + self._gamma * q_target_greedy * (1 - terminated)
        q_expected = self._Qnet(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_target)
        return loss


    def _update_target(self):
        '''
        Update target network's parameters
        '''
        self._Qnet_target.load_state_dict(self._Qnet.state_dict())


    def _soft_update_target(self):
        '''
        Soft update for target network's parameters
        '''
        try:
            for target_param, param in zip(self._Qnet_target.parameters(), self._Qnet.parameters()):
                target_param.data.copy_(self._tau * param.data + (1.0 - self._tau) * target_param.data)
        except TypeError:
            print('Smothness parameter tau is required!')


    def _learn(self):
        if self._step % self._train_freq == 0 and len(self._buffer) >= self._batch_size:
            states, actions, rewards, next_states, terminated = self._buffer.sample(self._batch_size)
            states = torch.from_numpy(states).float().to(self._device)
            actions = torch.from_numpy(np.vstack(actions)).long().to(self._device)
            rewards = torch.from_numpy(np.vstack(rewards)).float().to(self._device)
            next_states = torch.from_numpy(next_states).float().to(self._device)
            terminated = torch.from_numpy(np.vstack(terminated).astype(np.uint8)).float().to(self._device)

            loss = self._compute_td_loss_(states, actions, rewards, next_states, terminated)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        if self._step % self._update_target == 0:
            self._update_target_network()

        self._step += 1


    def _load(self, model_path: str) -> None:
        '''
        Load model
        '''
        self._Qnet.load_state_dict(torch.load(model_path, map_location=self._device))
        self._Qnet.eval()


    def _save(self) -> None:
        '''
        Save model
        '''
        torch.save(self._Qnet.state_dict(), self._model_path)


    def train(self) -> Tuple[List[float], List[float]]:
        scores = []
        avg_scores = []

        for ep in range(1, self._n_eps):
            state = self._env.reset()
            total_reward = 0
            epsilon = self._anneal_epsilon(ep)

            while True:
                action = self._act(state, epsilon)
                next_state, reward, terminated, _ = self._env.step(action)
                total_reward += reward
                if terminated:
                    break
                self._store_transition(state, action, reward, next_state, terminated)
                self._learn()
                state = next_state

            scores.append(total_reward)
            avg_score = np.mean(scores[-self._logging_interval:])
            avg_scores.append(avg_score)

            if self._print_freq and ep % self._print_freq == 0:
                print(f'Ep {ep}, average score {avg_score:.2f}')
            if avg_score >= self._termination:
                print(f'Environment solved in {ep} episodes!')
                if self._model_path:
                    self._save()
                    self.test(self._model_path, self._render_video)
                break

        if self._image_paths:
            plot(scores, 'Episodes', 'Score', 'Score per episode', self._image_paths[0])
            plot(avg_scores, 'Episodes', 'Average score', 
                'Average score per episode', self._image_paths[1])
        self._env.close()

        return scores, avg_scores


    def test(self, model_path: str, render_video: bool=False):
        '''
        Parameters
        ----------
        model_path: model path to load
        render_video: whether to render output video
        '''
        self._load(model_path)
        video_path = f'./outputs/videos/{splitext(basename(model_path))[0]}.mp4'
        video = video_recorder.VideoRecorder(self._env, path=video_path)
        state = self._env.reset()

        while True:
            self._env.render(mode='rgb_array');
            if render_video:
                video.capture_frame()
            action = self._act(state)
            state, _, terminated, _ = self._env.step(action)
            if terminated:
                break
        self._env.close()

