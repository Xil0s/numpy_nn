from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    from .layers import Layer

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    expz = np.exp(z)
    return expz / np.sum(expz, axis=1, keepdims=True)

def softmax_backward(z, upstream_gradient):
    s = softmax(z)
    return s * (upstream_gradient - np.sum(upstream_gradient * s, axis=1, keepdims=True))

class Activation(Enum):
    RELU = (
        lambda z: np.maximum(z, 0),
        lambda z, upstream_gradient: upstream_gradient * (z > 0),
    )
    SOFTMAX = (softmax, softmax_backward)

    def __init__(self, forward, backward):
        self.forward = forward
        self.backward = backward

class Loss(Enum):
    MSE = (
        lambda y_hat, y: np.mean((y_hat - y) ** 2),
        lambda y_hat, y: 2 * (y_hat - y) / y.size,
    )
    CROSS_ENTROPY = (
        lambda y_hat, y: -np.mean(np.sum(y * np.log(y_hat + 1e-15), axis=1)), # mean over all sample of sum over outputs of loss
        lambda y_hat, y: -(y / (y_hat + 1e-15)) / y.shape[0],
    )
    def __init__(self, forward, backward):
            self.forward = forward
            self.backward = backward

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
        upstream_gradient, dw, db = layers[i].backward(layers_data[i], upstream_gradient)
        gradients[i] = (dw, db)
    return gradients
