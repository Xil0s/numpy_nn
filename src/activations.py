from enum import Enum

import numpy as np


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
