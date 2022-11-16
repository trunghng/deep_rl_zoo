from typing import List
from os.path import basename, splitext
import numpy as np
import matplotlib.pyplot as plt
from gym.wrappers.monitoring import video_recorder


def plot(list_: List[float], xlabel: str, ylabel: str, plot_name: str):
    '''
    Plotting function
    '''
    fig = plt.figure(figsize=(10, 10))
    plt.plot(np.arange(len(list_)), list_)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plot_name)
    plot_img = plot_name.lower().replace(' ', '_')
    plot_path = f'/outputs/images/{plot_img}.png'
    plt.savefig(plot_path)
    plt.close()
    print(f'Plot saved successful at {plot_path}!')

        
def save_video(env, agent, model_path: str):
    '''
    Play video with a saved model
    '''
    video_path = f'/outputs/videos/{splitext(basename(model_path))[0]}.mp4'
    video = video_recorder.VideoRecorder(env, path=video_path)
    agent.load(model_path)
    state = env.reset()
    while True:
        frame = env.render(mode='rgb_array')
        video.capture_frame()
        action = agent.behave(state)
        next_state, reward, terminated, _ = env.step(action)
        if terminated:
            break
        state = next_state
    env.close()
    print(f'Video saved successful at {video_path}!')