import numpy as np
import gym
import torch

from collections import deque
from typing import List
import sys
from agent import Agent
from os.path import dirname, join, realpath
dir_path = dirname(dirname(realpath(__file__)))
sys.path.insert(1, join(dir_path, 'utils'))
import render
import plot


def dqn(env,
        epsilon_init: float,
        epsilon_final: float,
        gamma: float,
        lr: float,
        buffer_size: int,
        batch_size: int,
        update_freq: int,
        tau: float,
        model_path: str,
        video_path: str,
        n_eps: int,
        termination: float,
        verbose: bool=False,
        verbose_freq: int=50,
        save_model: bool=False,
        render_video: bool=False,
        seed: int=0) -> List[float]:
    '''
    Parameters
    ----------
    n_steps: number of time steps
    epsilon_init: initial value of epsilon
    epsilon_final: stopping value of epislon
    gamma: discount factor
    lr: learning rate
    buffer_size: replay buffer size
    batch_size: mini batch size
    update_freq: number of actions seleced by the agent between successive SGD updates
    tau: for Q network parameters' soft update
    model_path: model path, for model saving
    video_path: video path, for saving output as video
    n_eps: number of episodes
    termination: allows algorithm ends when total reward >= @termination
    verbose: whether to display result
    verbose_freq: display every @verbose_freq
    save_model: whether to save model
    render_video: whether to render output as video
    seed: random seed
    '''
    env.seed(seed)
    state_size =  env.observation_space.shape[0]
    action_size = env.action_space.n
    epsilon_decay = (epsilon_init - epsilon_final) / (n_eps / 2)
    agent = Agent(state_size, 
                action_size,
                epsilon_init,
                epsilon_final,
                gamma, lr,
                buffer_size,
                batch_size,
                update_freq,
                tau,
                seed)
    total_reward_list = []
    total_reward_history = deque(maxlen=verbose_freq)

    for ep in range(1, n_eps + 1):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.behave(state)
            next_state, reward, terminated, _ = env.step(action)
            total_reward += reward

            if terminated:
                break

            agent.store_transition(state, action, reward, next_state, terminated)
            agent.learn()
            state = next_state

        total_reward_list.append(total_reward)
        total_reward_history.append(total_reward)
        agent.epsilon_annealing(epsilon_decay)

        if verbose and ep % verbose_freq == 0:
            print(f'Ep {ep}, average total reward {np.mean(total_reward_history):.2f}')
        if np.mean(total_reward_history) >= termination:
            print(f'Environment solved in {ep} episodes!')
            if (save_model):
                assert model_path, 'Model path needed!'
                agent.save(model_path)
            if (render_video):
                assert model_path, 'Model path needed!'
                assert video_path, 'video path needed!'
                render.save_video(env, agent, model_path, video_path)
            break
    return total_reward_list


if __name__ == '__main__':
    env_name = 'LunarLander-v2'
    env = gym.make(env_name)
    epsilon_init = 1
    epsilon_final = 0.01
    gamma = 0.99
    lr = 5e-4
    buffer_size = 100000
    batch_size = 64
    update_freq = 4
    tau = 1e-3
    model_path = f'../models/dqn-{env_name}.pth'
    video_path = f'../outputs/videos/dqn-{env_name}.mp4'
    n_eps = 2000
    termination = 200
    verbose = True
    verbose_freq = 100
    save_model = True
    render_video = True
    seed = 1

    total_reward_list = dqn(env,
                            epsilon_init,
                            epsilon_final,
                            gamma, lr,
                            buffer_size,
                            batch_size,
                            update_freq,
                            tau,
                            model_path,
                            video_path,
                            n_eps,
                            termination,
                            verbose,
                            verbose_freq,
                            save_model,
                            render_video,
                            seed)

    image_path = f'../outputs/images/dqn-{env_name}.png'
    plot.total_reward(total_reward_list, image_path)
