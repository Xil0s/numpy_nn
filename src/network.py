from __future__ import annotations

import numpy as np

from .layers import Layer
from .losses import Loss


def nn_forward(data: np.ndarray, layers : list[Layer]):
    res = [data]
    for layer in layers:
        res.append(layer.forward(res[-1]))
    return res

def nn_backward(Y: np.ndarray, layers: list[Layer], layers_data: list[np.ndarray], loss_func: Loss):

    if len(layers_data) != len(layers) + 1:
        raise ValueError("Expected the input plus one output per layer")
    Y_hat = layers_data[-1]

    upstream_gradient = loss_func.backward(Y_hat, Y)
    n = len(layers)
    gradients = [None] * n

    for i in reversed(range(n)):
        upstream_gradient, gradient = layers[i].backward(layers_data[i], upstream_gradient)
        gradients[i] = gradient
    return gradients
