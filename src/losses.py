from enum import Enum

import numpy as np


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
