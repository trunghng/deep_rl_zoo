import gym
from agent import DQN, DoubleDQN


if __name__ == '__main__':
    env_name = 'LunarLander-v2'
    epsilon_init = 0.99
    epsilon_final = 0.01
    gamma = 0.99
    lr = 5e-4
    buffer_size = 100000
    batch_size = 64
    update_freq = 4
    tau = 1e-3
    save_model = True
    render_video = True
    save_plot = True
    n_eps = 2000
    horizon = 1000
    logging_window = 100
    termination = 200
    verbose = True
    verbose_freq = 50
    plot = True
    seed = 1
    agents = [DQN, DoubleDQN]

    for i in range(len(agents)):
        agent = agents[i](env_name, epsilon_init, epsilon_final, gamma,
                        lr, buffer_size, batch_size, update_freq, tau,
                        n_eps, horizon, logging_window, termination, 
                        verbose, verbose_freq, save_model, render_video,
                        save_plot, plot, seed)
        print(f'{agent.name()} algorithm')
        agent.run()
