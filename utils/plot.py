from typing import List
import numpy as np
import matplotlib.pyplot as plt


def total_reward(total_reward_list: List[float], image_path: str=None):
    '''
    Plot total reward through episodes

    Parameters
    ----------
    total_reward_list: list of total reward
    image_path: path to save
    '''
    fig = plt.figure(figsize=(10, 10))
    plt.plot(np.arange(len(total_reward_list)), total_reward_list)
    plt.ylabel('Total reward')
    plt.xlabel('Episodes')
    if image_path:
        plt.savefig(image_path)
    plt.close()