import numpy as np
from abc import ABC, abstractmethod
from .activations import Activation

class Layer(ABC):
    @abstractmethod
    def forward(self, x : np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, x : np.ndarray, upstream_gradient : np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        pass

class ActivationLayer(Layer):
    def __init__(self, activation : Activation):
        self.activation = activation

    def forward(self, x : np.ndarray):
        return self.activation.forward(x)

    def backward(self, x : np.ndarray, upstream_gradient : np.ndarray):
        return self.activation.backward(x, upstream_gradient), None, None

class DenseLayer(Layer):
    def __init__(self, input_n : int, output_n : int):
        self.W = np.random.randn(input_n, output_n) * np.sqrt(2 / input_n)
        self.b = np.zeros((1, output_n))

    def forward(self, x : np.ndarray):
        return x @ self.W + self.b

    def backward(self, x : np.ndarray, upstream_gradient : np.ndarray):
        dW = x.T @ upstream_gradient
        db = np.sum(upstream_gradient, axis=0, keepdims=True)
        dx = upstream_gradient @ self.W.T
        return dx, dW, db


class ConvolutionalLayer(Layer):
    def __init__(self, input_n : int, output_n : int, kernel_size : int):
        self.kernel_size = kernel_size
        self.W = np.random.randn(input_n, output_n) * np.sqrt(2 / input_n)
        self.b = np.zeros((1, output_n))
        self.kernel = np.random.randn(kernel_size,kernel_size, output_n)

    def forward(self, x : np.ndarray):
        pass

    def backward(self, x : np.ndarray, upstream_gradient : np.ndarray):
        pass
