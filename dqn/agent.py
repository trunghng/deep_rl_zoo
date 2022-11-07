import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as opt

from network import DeepQNetwork
from replay_buffer import ReplayBuffer


class Agent:
    '''
    Agent class
    '''

    def __init__(self, state_size: int,
                action_size: int,
                epsilon_init: float,
                epsilon_final: float,
                gamma: float,
                lr: float,
                buffer_size: int,
                batch_size: int,
                update_freq: int,
                tau: float,
                seed: int):
        '''
        Parameters
        ----------
        state_size: state size
        action_size: action size
        epsilon_init: initial value for exploration param, epsilon, linearly annealing
        epsilon_final: final value for epsilon linearly annealing
        gamma: discount factor
        lr: learning rate
        buffer_size: replay buffer size
        batch_size: mini batch size
        update_freq: number of actions seleced by the agent between successive SGD updates
        tau: for Q network parameters' soft update
        seed: random seed
        '''
        random.seed(seed)
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
        self.current_step = 0

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.Qnet = DeepQNetwork(state_size, action_size, lr, seed).to(self.device)
        self.Qnet_target = DeepQNetwork(state_size, action_size, lr, seed).to(self.device)
        self.optimizer = opt.Adam(self.Qnet.parameters(), lr=lr)


    def epsilon_annealing(self, epsilon_decay: float) -> None:
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


    def learn(self) -> None:
        if self.current_step % self.update_freq == 0 \
                and len(self.buffer) >= self.batch_size:
            states, actions, rewards, next_states, terminated = self.buffer.sample()
            states = torch.from_numpy(np.vstack(states)).float().to(self.device)
            actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
            rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
            next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
            terminated = torch.from_numpy(np.vstack(terminated).astype(np.uint8)).float().to(self.device)

            q_targets_next = self.Qnet_target(next_states).detach().max(1)[0].unsqueeze(1)
            q_targets = rewards + self.gamma * q_targets_next * (1 - terminated)
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


    def save(self, model_path: str) -> None:
        '''
        Save checkpoint
        '''
        torch.save(self.Qnet.state_dict(), model_path)
