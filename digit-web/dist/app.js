const pad = document.getElementById('pad');
const ctx = pad.getContext('2d', {willReadFrequently: true});
const preview = document.getElementById('preview');
const previewCtx = preview.getContext('2d');
const status = document.getElementById('status');
const predict = document.getElementById('predict');
const bars = document.getElementById('bars');
let model, drawing = false, hasInk = false, previous, timer;
for (let digit = 0; digit < 10; digit++) {
  const row = document.createElement('div');
  row.className = 'bar-row';
  row.innerHTML = `<span>${digit}</span><div class="bar-track"><div class="bar-fill"></div></div><span class="bar-value">—</span>`;
  bars.append(row);
}
function clearPad() {
  clearTimeout(timer);
  drawing = false; hasInk = false;
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, 280, 280);
  previewCtx.fillStyle = '#000'; previewCtx.fillRect(0, 0, 28, 28);
  document.getElementById('digit').textContent = '—';
  document.getElementById('confidence').textContent = '—';
  for (const row of bars.children) {
    row.classList.remove('winner'); row.querySelector('.bar-fill').style.width = '0%';
    row.querySelector('.bar-value').textContent = '—';
  }
  if (model) status.textContent = 'Ready. Draw a digit to begin.';
}
function point(event) {
  const rect = pad.getBoundingClientRect();
  return {x: (event.clientX - rect.left) * 280 / rect.width, y: (event.clientY - rect.top) * 280 / rect.height};
}
function stroke(a, b) {
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 18; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); hasInk = true;
}
pad.addEventListener('pointerdown', event => {
  if (event.pointerType === 'mouse' && event.button !== 0) return;
  event.preventDefault(); clearTimeout(timer); drawing = true;
  pad.setPointerCapture(event.pointerId); previous = point(event); stroke(previous, {x: previous.x + 0.01, y: previous.y});
});
pad.addEventListener('pointermove', event => {
  if (!drawing) return;
  const next = point(event); stroke(previous, next); previous = next;
});
function finish() {
  if (!drawing) return;
  drawing = false;
  if (model) timer = setTimeout(readDigit, 200);
}
pad.addEventListener('pointerup', finish);
pad.addEventListener('pointercancel', finish);
pad.addEventListener('lostpointercapture', finish);
function preprocess() {
  const pixels = ctx.getImageData(0, 0, 280, 280).data;
  let minX = 280, minY = 280, maxX = -1, maxY = -1;
  for (let y = 0; y < 280; y++) for (let x = 0; x < 280; x++) {
    if (pixels[(y * 280 + x) * 4] > 20) {
      minX = Math.min(minX, x); minY = Math.min(minY, y);
      maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
    }
  }
  if (maxX < 0) return null;
  const small = document.createElement('canvas'); small.width = small.height = 28;
  const sctx = small.getContext('2d', {willReadFrequently: true});
  sctx.fillStyle = '#000'; sctx.fillRect(0, 0, 28, 28);
  const w = maxX - minX + 1, h = maxY - minY + 1, scale = 20 / Math.max(w, h);
  sctx.imageSmoothingEnabled = true; sctx.imageSmoothingQuality = 'high';
  sctx.drawImage(pad, minX, minY, w, h, (28 - w * scale) / 2, (28 - h * scale) / 2, w * scale, h * scale);
  const data = sctx.getImageData(0, 0, 28, 28).data;
  let mass = 0, cx = 0, cy = 0;
  for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
    const value = data[(y * 28 + x) * 4]; mass += value; cx += x * value; cy += y * value;
  }
  if (!mass) return null;
  previewCtx.fillStyle = '#000'; previewCtx.fillRect(0, 0, 28, 28);
  previewCtx.drawImage(small, Math.round(13.5 - cx / mass), Math.round(13.5 - cy / mass));
  const normalized = previewCtx.getImageData(0, 0, 28, 28).data;
  return Array.from({length: 784}, (_, i) => normalized[i * 4] / 255);
}
function readDigit() {
  if (!model) throw new Error('The model is still loading.');
  const input = hasInk ? preprocess() : null;
  if (!input) { status.textContent = 'Draw a digit first.'; return {prediction: null}; }
  const probabilities = inferDigit(input, model);
  const winner = probabilities.indexOf(Math.max(...probabilities));
  document.getElementById('digit').textContent = winner;
  document.getElementById('confidence').textContent = (probabilities[winner] * 100).toFixed(1) + '%';
  for (let i = 0; i < 10; i++) {
    const row = bars.children[i]; row.classList.toggle('winner', i === winner);
    row.querySelector('.bar-fill').style.width = probabilities[i] * 100 + '%';
    row.querySelector('.bar-value').textContent = (probabilities[i] * 100).toFixed(1) + '%';
  }
  status.textContent = 'Prediction ready. Clear to try another digit.';
  return {prediction: winner, confidence: probabilities[winner], probabilities};
}
document.getElementById('clear').addEventListener('click', clearPad);
predict.addEventListener('click', readDigit);
clearPad();
fetch('model.json').then(response => {
  if (!response.ok) throw new Error('Model file unavailable.'); return response.json();
}).then(loaded => {
  model = loaded; predict.disabled = false; status.textContent = 'Ready. Draw a digit to begin.';
}).catch(() => { status.textContent = 'Could not load the model. Refresh to try again.'; });
if (document.modelContext?.registerTool) {
  for (const tool of [
    {name: 'read_drawn_digit', description: 'Predict the current handwritten digit and show its confidence.', inputSchema: {type: 'object', properties: {}, additionalProperties: false}, annotations: {readOnlyHint: false}, execute(input) { if (!input || Object.keys(input).length) throw new Error('Expected an empty object.'); return readDigit(); }},
    {name: 'clear_drawing', description: 'Clear the drawing and displayed prediction.', inputSchema: {type: 'object', properties: {}, additionalProperties: false}, annotations: {readOnlyHint: false}, execute(input) { if (!input || Object.keys(input).length) throw new Error('Expected an empty object.'); clearPad(); return {cleared: true}; }}
  ]) { try { Promise.resolve(document.modelContext.registerTool(tool)).catch(() => {}); } catch {} }
}
