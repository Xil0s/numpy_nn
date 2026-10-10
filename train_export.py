"""Train the notebook model and export its final weights for browser inference."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from src.layers import DenseLayer, ActivationLayer
from src.functions import Activation, Loss, nn_forward, nn_backward

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'digit-web' / 'dist'

def save_model(layers, epochs, learning_rate, metrics):
    # The browser format stores each dense layer with its following activation.
    pairs = []
    for i in range(0, len(layers), 2):
        if (i + 1 >= len(layers) or not isinstance(layers[i], DenseLayer)
                or not isinstance(layers[i + 1], ActivationLayer)):
            raise ValueError("Browser export requires alternating DenseLayer and ActivationLayer")
        pairs.append((layers[i], layers[i + 1].activation))
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = {f'{name}{i}': getattr(layer, name)
              for i, (layer, _) in enumerate(pairs) for name in ('W', 'b')}
    np.savez_compressed(ROOT / f'weights_lr{learning_rate:g}_epoch{epochs}.npz', **arrays,
                        activations=np.array([a.name for _, a in pairs]),
                        epochs=epochs, learning_rate=learning_rate)
    model = {
        'epochs': epochs, 'learning_rate': learning_rate,
        'input_size': 28, 'pixel_scale': 255, 'metrics': metrics,
        'layers': [{'W': layer.W.tolist(), 'b': layer.b.ravel().tolist(),
                    'activation': activation.name}
                   for layer, activation in pairs],
    }
    (OUT / 'model.json').write_text(json.dumps(model, separators=(',', ':')))

if __name__ == '__main__':
    dataset = Path.home() / '.cache/kagglehub/datasets/oddrationale/mnist-in-csv/versions/2/mnist_train.csv'
    if not dataset.is_file():
        import kagglehub
        dataset = Path(kagglehub.dataset_download("oddrationale/mnist-in-csv")) / "mnist_train.csv"
    df = pd.read_csv(dataset)
    y = df.iloc[:, 0].to_numpy()
    X = df.iloc[:, 1:].to_numpy() / 255
    Y = np.eye(10)[y]
    indices = np.random.default_rng(42).permutation(len(X))
    split = int(0.8 * len(X))
    train_indices, val_indices = indices[:split], indices[split:]
    X_train, Y_train = X[train_indices], Y[train_indices]
    X_val, Y_val = X[val_indices], Y[val_indices]
    np.random.seed(42)
    layers = [DenseLayer(784, 128), ActivationLayer(Activation.RELU),
              DenseLayer(128, 10), ActivationLayer(Activation.SOFTMAX)]
    history = []
    start = time.monotonic()
    for epoch in range(1000):
        data = nn_forward(X_train, layers)
        loss = float(Loss.CROSS_ENTROPY.forward(data[-1], Y_train))
        val_predictions = nn_forward(X_val, layers)[-1]
        val_loss = float(Loss.CROSS_ENTROPY.forward(val_predictions, Y_val))
        history.append({'epoch': epoch, 'train_loss': loss, 'val_loss': val_loss})
        grads = nn_backward(Y_train, layers, data, Loss.CROSS_ENTROPY)
        for layer, (dW, db) in zip(layers, grads):
            if dW is not None and db is not None:
                layer.W -= dW
                layer.b -= db
        if epoch % 10 == 0 or epoch == 999:
            print(f'Epoch {epoch+1}/1000: train={loss:.4f}, validation={val_loss:.4f}, elapsed={time.monotonic()-start:.1f}s', flush=True)
    predictions = nn_forward(X_val, layers)[-1]
    metrics = {'validation_loss': float(Loss.CROSS_ENTROPY.forward(predictions, Y_val)),
               'validation_accuracy': float(np.mean(predictions.argmax(axis=1) == Y_val.argmax(axis=1))),
               'train_samples': len(X_train), 'validation_samples': len(X_val)}
    save_model(layers, 1000, 1, metrics)
    (ROOT / 'training_history.json').write_text(json.dumps(history))
    np.savez_compressed(ROOT / 'inference_check.npz', X=X_val[:20],
                        predictions=predictions[:20], labels=Y_val[:20].argmax(axis=1))
    print('Saved final model:', metrics, flush=True)
