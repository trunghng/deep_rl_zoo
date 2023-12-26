from typing import List

import torch
import torch.nn as nn


def MLP(sizes: List[int],
        activation: nn.Module,
        output_activation: nn.Module=nn.Identity):
    """Create an MLP

    :param sizes: list of layers' size
    :param activation: activation layer type
    :param output_activation: output layer type
    """
    layers = []
    for i in range(len(sizes) - 1):
        activation_ = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), activation_()]
    return nn.Sequential(*layers)


def CNN():
    pass