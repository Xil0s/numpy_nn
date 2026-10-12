import numpy as np
from abc import ABC, abstractmethod
from .activations import Activation
from scipy.signal import fftconvolve

class Layer(ABC):
    @abstractmethod
    def forward(self, x : np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, x : np.ndarray, upstream_gradient : np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        pass

class ActivationLayer(Layer):
    def __init__(self, activation : Activation):
        self.activation = activation

    def forward(self, x : np.ndarray):
        return self.activation.forward(x)

    def backward(self, x : np.ndarray, upstream_gradient : np.ndarray):
        return self.activation.backward(x, upstream_gradient), {}
    
class DenseLayer(Layer):
    def __init__(self, input_n : int, output_n : int):
        W = np.random.randn(input_n, output_n) * np.sqrt(2 / input_n)
        b = np.zeros((1, output_n))
        self.weights = {"W": W, "b": b}

    def forward(self, x : np.ndarray):
        return x @ self.weights["W"] + self.weights["b"]
    
    def backward(self, x : np.ndarray, upstream_gradient : np.ndarray):
        dW = x.T @ upstream_gradient
        db = np.sum(upstream_gradient, axis=0, keepdims=True)
        dx = upstream_gradient @ self.weights["W"].T
        return dx, {"W": dW, "b": db}


class ConvolutionalLayer(Layer):
    def __init__(self, output_n : int, kernel_size : int):
        self.kernel_size = kernel_size
        
        kernel = np.random.randn(output_n, kernel_size, kernel_size)
        b = np.zeros((1, output_n))
        self.weights = {"kernel": kernel, "b": b}

    def forward(self, x : np.ndarray):
        return fftconvolve(
                x[:, None, :, :],
                self.weights["kernel"][None, :, :, :],
                mode="valid",
                axes=(-2, -1)
                ) + self.weights["b"].reshape(1, -1, 1, 1)
    
    def backward(self, x : np.ndarray, upstream_gradient : np.ndarray):
        dkernel = fftconvolve(
                x[:, None, :, :],
                upstream_gradient[:, :, ::-1, ::-1],
                mode="valid",
                axes=(-2, -1)
                ).sum(axis = 0)[:,::-1, ::-1]
        
        db = np.sum(upstream_gradient, axis=(0, 2, 3)).reshape(1, -1)
        
        dx = fftconvolve(
                upstream_gradient,
                self.weights["kernel"][None, :, ::-1, ::-1],
                mode="full",
                axes=(-2, -1)
                ).sum(axis=1)
        
        return dx, {"kernel": dkernel, "b": db}

class PoolingLayer(Layer):
    pass

class MaxPoolingLayer(PoolingLayer):
    pass

class AvgPoolingLayer(PoolingLayer):
    pass

class ReshapeLayer(Layer):
    def __init__(self, dims1, dims2):
        if np.prod(dims1) != np.prod(dims2):
            raise ValueError("Input and output dimensions must match")
        self.dims1 = dims1
        self.dims2 = dims2

    def forward(self, x : np.ndarray):
        return x.reshape(x.shape[0], *self.dims2)
    
    def backward(self, x : np.ndarray, upstream_gradient : np.ndarray):
        return upstream_gradient.reshape(upstream_gradient.shape[0], *self.dims1), {}

class FlattenLayer(ReshapeLayer):
    def __init__(self, input_shape):
        super().__init__(input_shape, (np.prod(input_shape),))