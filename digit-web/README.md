# Digit reader

Draw a single digit with a mouse or finger. Predictions run entirely in the browser using the trained notebook weights. Confidence shows softmax probabilities.

From the parent project directory, train and export the exact final model:

```sh
VECLIB_MAXIMUM_THREADS=4 python3 train_export.py
```

This uses the notebook's network and backpropagation, learning rate 1, 1000 full-batch epochs, and the same seeded 80/20 training/validation split. Outputs:

- `weights_lr1_epoch1000.npz`: weights, biases, activation names, and training settings.
- `digit-web/dist/model.json`: browser inference model and validation metrics.
- `training_history.json`: training and validation loss history.

For a local interface:

```sh
python3 -m http.server 8000 --directory digit-web/dist
```

Open http://localhost:8000. Keep the saved model next to the HTML and scripts. The drawing is cropped, resized to fit within 20 × 20 pixels, centered by intensity, padded to 28 × 28, and normalized by 255. Input polarity is white ink on black, matching MNIST.

The final notebook cell also exports the model after training. Saved notebook outputs do not retain live Python objects.
