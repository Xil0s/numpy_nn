/* The same dense layers, ReLU, and stable softmax as the notebook. */
(function (root) {
  function infer(input, model) {
    if (input.length !== 784 || !Array.from(input).every(Number.isFinite)) throw new Error('Expected 784 finite pixels.');
    let current = Array.from(input);
    for (const layer of model.layers) {
      const output = layer.b.slice();
      for (let i = 0; i < current.length; i++) {
        for (let j = 0; j < output.length; j++) output[j] += current[i] * layer.W[i][j];
      }
      if (layer.activation === 'RELU') current = output.map(x => Math.max(0, x));
      else if (layer.activation === 'SOFTMAX') {
        const max = Math.max(...output);
        const exps = output.map(x => Math.exp(x - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        current = exps.map(x => x / sum);
      } else throw new Error('Unsupported activation: ' + layer.activation);
    }
    return current;
  }
  root.inferDigit = infer;
  if (typeof module !== 'undefined') module.exports = infer;
})(typeof window !== 'undefined' ? window : globalThis);
