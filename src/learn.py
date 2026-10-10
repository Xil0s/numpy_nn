import numpy as np
from .functions import nn_backward, nn_forward, Loss
from .layers import Layer

def learn(epochs: int, X: np.ndarray,
          Y: np.ndarray, learning_rate: float, layers: list[Layer],
          X_val: np.ndarray, Y_val: np.ndarray, loss_f: Loss, print_rate: int = 10):
    if print_rate < 1:
        raise ValueError("print_rate must be positive")
    losses, val_losses = [], []

    for epoch in range(epochs):
        layers_data = nn_forward(X, layers)
        loss = loss_f.forward(layers_data[-1], Y)

        val_predictions = nn_forward(X_val, layers)[-1]
        val_loss = loss_f.forward(val_predictions, Y_val)
        losses.append(loss)
        val_losses.append(val_loss)

        if epoch % print_rate == 0:
            print(f"Epoch {epoch}: in-sample loss = {loss:.4f}, OOS loss = {val_loss:.4f}")


        gradients = nn_backward(Y, layers, layers_data, loss_f)
        for layer, grad in zip(layers, gradients):
            dw, db = grad
            if dw is not None and db is not None:
                layer.W -= learning_rate * dw
                layer.b -= learning_rate * db

    return losses, val_losses, layers
