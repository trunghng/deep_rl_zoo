import gym
import numpy as np
from dqn import Agent
# from utils import plot_learning_curve


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    n_actions = env.action_space.n
    epsilon = 0.99
    eps_end = 0.01
    state_dims =  env.observation_space.shape[0]
    
    total_reward_list, avg_reward_list, eps_history = [], [], []
    n_episodes = 1200
    epsilon_dec = (epsilon - eps_end) / n_episodes

    agent = Agent(gamma=0.99, epsilon=epsilon, batch_size=64,
            n_actions=n_actions, eps_end=eps_end,
            input_dims=[state_dims], lr=0.001)

    for ep in range(n_episodes):
        total_reward = 0
        state = env.reset()

        if agent.epsilon > agent.eps_min:
            agent.set_epsilon(epsilon_dec)

        while True:
            action = agent.epsilon_greedy(state)
            next_state, reward, terminated, _ = env.step(action)
            total_reward += reward

            if terminated:
                break

            agent.store_transition(state, action, 
                reward, next_state, terminated)
            agent.learn()
            state = next_state

        total_reward_list.append(total_reward)
        eps_history.append(agent.epsilon)

        avg_reward = np.mean(total_reward_list[-100:])
        # avg_reward_list.append(avg_reward)

        if (ep + 1) % 100 == 0:
            # avg_reward = np.mean(total_reward_list)
            # avg_reward_list.append(avg_reward)
            # total_reward_list = []

            print('Episode {}; Reward {:.2f}; Average Reward: {:.2f}; epsilon {:.2f}'
                .format(ep + 1, total_reward, avg_reward, agent.epsilon))

    env.close()
    # plt.plot(100 * (np.arange(len(ave_reward_list)) + 1), ave_reward_list)
    # plt.xlabel('Episodes')
    # plt.ylabel('Average Reward')
    # plt.title('Average Reward vs Episodes')
    # plt.savefig('./mountain_car.png')
    # plt.close()



