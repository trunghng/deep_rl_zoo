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
import argparse


class DQN:
    '''
    DQN agent class
    '''

    def __init__(self, args,
                model_dir: str='./output/models',
                video_dir: str='./output/videos',
                figure_dir: str='./output/figures'):
        '''
        Parameters
        ----------
        :param env: (str)
            OpenAI's environment name
        :param atari: (bool)
            Whether to use atari environment
        :param eval: (bool)
            Whether to enable evaluation
        :param dueling: (bool)
            Whether to use dueling network
        :param double: (bool)
            Whether to use double Q-learning
        :param tau: (float)
            Smoothness parameter, used for target network soft update
        :param prioritized_replay: (bool)
            Whether to use prioritized replay buffer
        :param epsilon_init: (float)
            Initial value for epsilon, linearly annealing
        :param epsilon_final: (float)
            Final value for epsilon linearly annealing
        :param epsilon_decay: (float)
            Decay value for epsilon linearly annealing
        :param gamma: (float)
            Discount factor
        :param lr: (float)
            Learning rate
        :param buffer_size: (int)
            Replay buffer size
        :param batch_size: (int)
            Mini batch size
        :param train_freq: (int)
            Number of actions selected by the agent between successive SGD updates
        :param update_target: (int)
            Target network update frequency
        :param n_eps: (int)
            Number of episodes
        :param log_int: (int)
            Logging interval, i.e. number of episodes to be tracked
        :param goal: (float)
            Total reward threshold for early stopping
        :param print_freq: (int)
            Result printing frequency
        :param save: (bool)
            Whether to save the model
        :param render: (bool)
            Whether to render the final simulation and save it as video
        :param plot: (bool)
            Whether to plot the result and save it
        :param seed: (int)
            Random seed
        :param device:
            Device type
        :param model_dir: (str)
            Model directory
        :param video_dir: (str)
            Video directory
        :param figure_dir: (str)
            Figure directory
        '''
        self._env, state_dim, n_actions = self._create_env(args.env, args.atari, args.eval)
        self._device = args.device
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        self._env.seed(seed)
        # Select network architecture
        if args.dueling:
            architecture_txt = 'Dueling Q-network'
            arch_savename = 'Duel'
            network = DuelingQNet
        elif args.atari:
            architecture_txt = 'CNN Q-network'
            arch_savename = 'Conv'
            network = CNNDeepQNet
        else:
            architecture_txt = 'MLP Q-network'
            arch_savename = 'MLP'
            network = DeepQNet
        self._Qnet = network(state_dim, n_actions, seed).to(self._device)
        self._Qnet_target = network(state_dim, n_actions, seed).to(self._device)
        # Select Q-learning update type
        if args.double:
            td_error_cal_txt = 'Double Q-learning'
            learn_savename = 'DoubleDeepQ'
            self._compute_td_loss_ = self._compute_td_loss_double_q
        else:
            td_error_cal_txt = 'Q-learning'
            learn_savename = 'DeepQ'
            self._compute_td_loss_ = self._compute_td_loss
        # Select replay buffer type
        if args.prioritized_replay:
            replay_txt = 'prioritized replay'
            replay_savename = 'PrioRep'
            self._buffer = PrioritizedReplay(args.buffer_size)
        else:
            replay_txt = 'experience replay'
            replay_savename = 'ExpRep'
            self._buffer = ReplayBuffer(args.buffer_size)
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
        if not args.eval:
            self._optimizer = opt.Adam(self._Qnet.parameters(), lr=args.lr)
            self._action_space = range(n_actions)
            self._epsilon_init = args.epsilon_init
            self._epsilon_final = args.epsilon_final
            self._epsilon_decay = args.epsilon_decay
            self._gamma = args.gamma
            self._batch_size = args.batch_size
            self._train_freq = args.train_freq
            self._update_target = args.update_target
            self._n_eps = args.n_eps
            self._log_int = args.log_int
            self._goal = args.goal
            self._print_freq = args.print_freq
            basename = '-'.join([arch_savename, learn_savename, replay_savename, args.env])
            self._model_path = join(model_dir, f'{basename}.pth') if args.save else None
            self._video_path = join(video_dir, f'{basename}.mp4') if args.render else None
            self._figure_paths = [join(figure_dir, f'{basename}-score-per-ep.png'),\
                join(figure_dir, f'{basename}-avgscore-per-ep.png')] if args.plot else None
            self._step = 0


    def _create_env(self, env: str, atari: bool, evaluate: bool):
        if atari:
            if evaluate:
                mode = 'human'
            else:
                mode = 'rgb_array'

            env = make_atari(env, mode)
            env = wrap_deepmind(env)
            env = wrap_pytorch(env)
            state_dim =  env.observation_space.shape
            n_actions = env.action_space.n
        else:
            env = gym.make(env)
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


    def load(self, model_path: str) -> None:
        '''
        Load model
        '''
        self._Qnet.load_state_dict(torch.load(model_path, map_location=self._device))
        self._Qnet.eval()


    def train(self) -> Tuple[List[float], List[float]]:
        print('---Training---')
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
            avg_score = np.mean(scores[-self._log_int:])
            avg_scores.append(avg_score)

            if self._print_freq and ep % self._print_freq == 0:
                print(f'Ep {ep}, average score {avg_score:.2f}')
            if self._goal and avg_score >= self._goal:
                print(f'Environment solved in {ep} episodes!')
                break

        self._env.close()
        if self._model_path:
            torch.save(self._Qnet.state_dict(), self._model_path)
            print(f'Model is saved successfully at {self._model_path}')
        if self._video_path:
            self.test(video_path=self._video_path)
            print(f'Video is renderred successfully at {self._video_path}')
        if self._figure_paths:
            plot(scores, 'Episodes', 'Score', 'Score per episode', self._figure_paths[0])
            plot(avg_scores, 'Episodes', 'Average score', 
                'Average score per episode', self._figure_paths[1])

        return scores, avg_scores


    def test(self, model_path: str=None, video_path: str=None):
        '''
        Parameters
        ----------
        model_path: model path to load
        render_video: whether to render output video
        '''
        print('---Evaluating---')
        if model_path:
            self.load(model_path)
        if video_path:
            video = video_recorder.VideoRecorder(self._env, path=video_path)

        state = self._env.reset()
        step = 0
        total_reward = 0
        while True:
            self._env.render('rgb_array')
            if video_path:
                video.capture_frame()
            action = self._act(state)
            state, reward, terminated, _ = self._env.step(action)
            step += 1
            total_reward += reward
            if terminated:
                print(f'Episode finished after {step} steps.\nTotal reward: {total_reward}')
                break
        self._env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Q-Learning')

    parser.add_argument('--env', type=str, choices=['LunarLander-v2', 'PongNoFrameskip-v4'],
                        help='OpenAI enviroment name')
    parser.add_argument('--atari', action='store_true',
                        help='Whether to use atari environment')
    parser.add_argument('--eval', action='store_true',
                        help='Whether to enable evaluation')
    parser.add_argument('--model-path', type=str,
                        help='Model path to load')
    parser.add_argument('--double', action='store_true',
                        help='Whether to use double Q-network')
    parser.add_argument('--dueling', action='store_true',
                        help='Whether to use dueling Q-network')
    parser.add_argument('--prioritized-replay', action='store_true',
                        help='Whether to use prioritized replay')
    parser.add_argument('--alpha', type=float,
                        help='Alpha value for prioritized replay')
    parser.add_argument('--epsilon-init', type=float,
                        help='Initial value of epsilon')
    parser.add_argument('--epsilon-final', type=float,
                        help='Final value of epsilon')
    parser.add_argument('--epsilon-decay', type=float,
                        help='Final value of epsilon')
    parser.add_argument('--gamma', type=float,
                        help='Discount factor')
    parser.add_argument('--lr', type=float,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int,
                        help='Batch size')
    parser.add_argument('--buffer-size', type=int,
                        help='Maximum memory buffer size')
    parser.add_argument('--train-freq', type=int,
                        help='Number of steps between optimization steps')
    parser.add_argument('--update-target', type=int,
                        help='Target network update frequency')
    parser.add_argument('--tau', type=float,
                        help = 'Smoothness parameter, used for target network soft update')
    parser.add_argument('--n-eps', type=int,
                        help = 'Maximum number of episodes')
    parser.add_argument('--log-int', type=int,
                        help = 'Score tracking interval')
    parser.add_argument('--goal', type=int,
                        help = 'Terminal score')
    parser.add_argument('--print-freq', type=int,
                        help='Result display interval')
    parser.add_argument('--save', action='store_true',
                        help='Whether to save the model after training')
    parser.add_argument('--render', action='store_true',
                        help='Whether to render video output after training')
    parser.add_argument('--plot', action='store_true',
                        help='Whether to plot the result and save')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--use-cuda', action='store_true',
                        help='Whether to enable CUDA training')
    args = parser.parse_args()

    args.cuda = args.use_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda:0' if args.cuda else 'cpu')

    if not args.eval and args.model_path or (not args.model_path and args.eval):
        parser.error('Arguments --eval & --model-path must be specified together.')

    agent = DQN(args)
    if args.eval:
        agent.test(args.model_path)
    else:
        agent.train()
