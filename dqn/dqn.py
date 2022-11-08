import gym
from agent import DQN


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
    image_path = f'../outputs/images/dqn-{env_name}.png'
    n_eps = 2000
    logging_window = 100
    termination = 200
    verbose = True
    verbose_freq = 100
    plot = True
    seed = 1

    agent = DQN(env,
                epsilon_init,
                epsilon_final,
                gamma, lr,
                buffer_size,
                batch_size,
                update_freq,
                tau,
                n_eps,
                logging_window,
                termination,
                verbose,
                verbose_freq,
                model_path,
                video_path,
                image_path,
                plot,
                seed)
    agent.run()
