"""Check exported browser predictions against Python on held-out MNIST samples."""
import json
import subprocess
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
with np.load(ROOT / 'inference_check.npz') as check:
    inputs = check['X'].tolist()
    expected = check['predictions']
script = '''const fs = require('fs');
const infer = require('./digit-web/dist/inference.js');
const model = JSON.parse(fs.readFileSync('./digit-web/dist/model.json', 'utf8'));
const inputs = JSON.parse(fs.readFileSync(0, 'utf8'));
console.log(JSON.stringify(inputs.map(x => infer(x, model))));'''
result = subprocess.run(['node', '-e', script], input=json.dumps(inputs), text=True,
                        capture_output=True, cwd=ROOT, check=True)
actual = np.array(json.loads(result.stdout))
np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-8)
assert np.array_equal(actual.argmax(axis=1), expected.argmax(axis=1))
with np.load(ROOT / 'weights_lr1_epoch1000.npz', allow_pickle=False) as weights:
    model = json.loads((ROOT / 'digit-web/dist/model.json').read_text())
    assert int(weights['epochs']) == model['epochs'] == 1000
    assert float(weights['learning_rate']) == model['learning_rate'] == 1
    for i, layer in enumerate(model['layers']):
        np.testing.assert_array_equal(weights[f'W{i}'], layer['W'])
        np.testing.assert_array_equal(weights[f'b{i}'].ravel(), layer['b'])
print('Browser predictions match Python on 20 held-out digits; saved weights match the browser model.')
