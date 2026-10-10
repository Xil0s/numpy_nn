# NumPy NN

A neural network built from scratch with NumPy, trained to recognize handwritten MNIST digits. Includes explicit backpropagation, training and validation loss plots, saved weights, and a browser drawing interface.

The current model is `784 → 128 → 10`, with separate dense and activation layers. ReLU, softmax, cross-entropy, and MSE are implemented. The convolution layer is a placeholder and is not ready to use.

## Setup

Requires Python 3.10 or newer. Browser inference verification also requires Node.js.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook neurnet.ipynb
```

Run the notebook from top to bottom to download MNIST, split it into training and validation sets, train, plot losses, and export the model. The default run uses learning rate 1 and 1000 full-batch epochs.

To train using the script instead:

```sh
python3 train_export.py
```

The script uses a cached MNIST CSV when available and downloads the dataset otherwise.

## Try the saved model

```sh
python3 -m http.server 8000 --directory digit-web/dist
```

Open http://localhost:8000 and draw one digit. Predictions run locally in your browser, with confidence for each digit.

The included model was trained at learning rate 1 for 1000 epochs and achieved 97.24% accuracy on the held-out 20% validation split.

```sh
python3 verify_inference.py
```

This checks browser/Python prediction agreement on saved validation samples and consistency between the NumPy and browser weight files.

## Files

- `src/layers.py`: dense, activation, and unfinished convolution layers.
- `src/functions.py`: activations, losses, and network forward/backward passes.
- `src/learn.py`: training loop with validation losses.
- `neurnet.ipynb`: MNIST training walkthrough.
- `train_export.py`: training script and model exporter.
- `weights_lr1_epoch1000.npz`: saved weights, biases, activations, and training settings.
- `digit-web/dist/`: drawing interface and browser model.
