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
    """Stride-1 valid convolution on (batch, channels, height, width)."""
    def __init__(self, input_n: int, output_n: int, kernel_size: int):
   
        
        self.kernel_size = kernel_size
        self.input_n = input_n
        kernel = np.random.randn(output_n, input_n, kernel_size, kernel_size)
        kernel *= np.sqrt(2 / (input_n * kernel_size ** 2))
        self.weights = {"kernel": kernel, "b": np.zeros((1, output_n))}

    def _image_input(self, x):
        # Preserve support for the original single-channel (batch, H, W) input.
        images = x[:, None, :, :] if x.ndim == 3 else x
        if images.ndim != 4 or images.shape[1] != self.input_n:
            raise ValueError(f"Expected (batch, {self.input_n}, height, width)")
        if min(images.shape[-2:]) < self.kernel_size:
            raise ValueError("Kernel cannot be larger than the input image")
        return images

    def forward(self, x: np.ndarray):
        images = self._image_input(x)
        kernel = self.weights["kernel"]
        output = None
        # Sum each input channel's contribution to every output filter.
        for c in range(self.input_n):
            contribution = fftconvolve(
                images[:, None, c, :, :], kernel[None, :, c, :, :],
                mode="valid", axes=(-2, -1),
            )
            if output is None:
                output = contribution
            else:
                output += contribution
        return output + self.weights["b"].reshape(1, -1, 1, 1)

    def backward(self, x: np.ndarray, upstream_gradient: np.ndarray):
        images = self._image_input(x)
        kernel = self.weights["kernel"]
        expected = (images.shape[0], kernel.shape[0],
                    images.shape[2] - self.kernel_size + 1,
                    images.shape[3] - self.kernel_size + 1)
        if upstream_gradient.shape != expected:
            raise ValueError(f"Expected upstream gradient shape {expected}")
        dtype = np.result_type(images, kernel, upstream_gradient)
        dkernel = np.zeros(kernel.shape, dtype=dtype)
        dx = np.zeros(images.shape, dtype=dtype)
        for c in range(self.input_n):
            dkernel[:, c] = fftconvolve(
                images[:, None, c, :, :], upstream_gradient[:, :, ::-1, ::-1],
                mode="valid", axes=(-2, -1),
            ).sum(axis=0)[:, ::-1, ::-1]
            dx[:, c] = fftconvolve(
                upstream_gradient, kernel[None, :, c, ::-1, ::-1],
                mode="full", axes=(-2, -1),
            ).sum(axis=1)
        db = upstream_gradient.sum(axis=(0, 2, 3)).reshape(1, -1)
        return (dx[:, 0] if x.ndim == 3 else dx), {"kernel": dkernel, "b": db}

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