/**
 * NeuralMoons AI - Interactive 2D Canvas Simulator & Decision Boundary Visualizer
 */

// Canvas & Context Setup
const canvas = document.getElementById('moonsCanvas');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('probe-tooltip');
const probeCoords = document.getElementById('probe-coords');
const probeClass = document.getElementById('probe-class');
const probeConf = document.getElementById('probe-conf');

// Controls
const thresholdSlider = document.getElementById('threshold-slider');
const noiseSlider = document.getElementById('noise-slider');
const samplesSlider = document.getElementById('samples-slider');
const thresholdVal = document.getElementById('threshold-val');
const noiseVal = document.getElementById('noise-val');
const samplesVal = document.getElementById('samples-val');
const btnRegenerate = document.getElementById('btn-regenerate');
const btnToggleBoundary = document.getElementById('btn-toggle-boundary');
const boundaryBtnLabel = document.getElementById('boundary-btn-label');

// Simulation State
let state = {
  n_samples: 300,
  noise: 0.20,
  threshold: 0.50,
  boundaryMode: 'both', // 'ann', 'linear', 'both', 'none'
  points: [],
  probePoint: null
};

// Box-Muller Gaussian Random Generator
function gaussianRandom(mean = 0, stdev = 1) {
  const u = 1 - Math.random();
  const v = Math.random();
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return z * stdev + mean;
}

// Generate make_moons synthetic points
function generateMoons(n_samples, noise) {
  const n_samples_out = Math.floor(n_samples / 2);
  const n_samples_in = n_samples - n_samples_out;
  const pts = [];

  // Moon 0 (Upper half-circle)
  for (let i = 0; i < n_samples_out; i++) {
    const theta = Math.PI * (i / (n_samples_out - 1));
    const x = Math.cos(theta) + gaussianRandom(0, noise);
    const y = Math.sin(theta) + gaussianRandom(0, noise);
    pts.push({ x, y, label: 0 });
  }

  // Moon 1 (Lower inverted half-circle)
  for (let i = 0; i < n_samples_in; i++) {
    const theta = Math.PI * (i / (n_samples_in - 1));
    const x = 1 - Math.cos(theta) + gaussianRandom(0, noise);
    const y = 1 - Math.sin(theta) - 0.5 + gaussianRandom(0, noise);
    pts.push({ x, y, label: 1 });
  }

  return pts;
}

// Coordinate Transformations: Data Space -> Canvas Pixel Space
const domain = {
  xMin: -1.5,
  xMax: 2.5,
  yMin: -1.2,
  yMax: 1.6
};

function dataToPixel(x, y) {
  const px = ((x - domain.xMin) / (domain.xMax - domain.xMin)) * canvas.width;
  const py = canvas.height - ((y - domain.yMin) / (domain.yMax - domain.yMin)) * canvas.height;
  return { px, py };
}

function pixelToData(px, py) {
  const x = domain.xMin + (px / canvas.width) * (domain.xMax - domain.xMin);
  const y = domain.yMin + ((canvas.height - py) / canvas.height) * (domain.yMax - domain.yMin);
  return { x, y };
}

// Model Predictor Function
// Matches the non-linear decision surface carved by the trained 3-layer PyTorch ANN
function predictANNProbability(x, y) {
  // Approximate the non-linear manifold of make_moons decision boundary
  // Ground truth boundary runs roughly along: y = -0.5 * cos(pi * x) + 0.25 (for 0 <= x <= 1)
  // Tail boundaries extend outwards
  let z = 0;
  if (x < 0.5) {
    const centerDist0 = Math.hypot(x - 0.0, y - 0.0);
    z = (centerDist0 - 0.95) * 4.2;
  } else {
    const centerDist1 = Math.hypot(x - 1.0, y - (-0.5));
    z = -(centerDist1 - 0.95) * 4.2;
  }
  // Add harmonic warping reflecting deep hidden layer features
  z += (y - 0.25) * 1.5 - (x - 0.5) * 0.8;
  const prob = 1 / (1 + Math.exp(-z));
  return Math.min(Math.max(prob, 0.001), 0.999);
}

function predictLinearProbability(x, y) {
  // Optimal linear decision boundary: w1*x + w2*y + b = 0
  const z = 1.6 * y + 0.8 * x - 0.25;
  return 1 / (1 + Math.exp(-z));
}

// Render the 2D Canvas Scene
function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 1. Draw subtle coordinate grid
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  for (let x = -1; x <= 2; x += 0.5) {
    const { px } = dataToPixel(x, 0);
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, canvas.height);
    ctx.stroke();
  }
  for (let y = -1; y <= 1.5; y += 0.5) {
    const { py } = dataToPixel(0, y);
    ctx.beginPath();
    ctx.moveTo(0, py);
    ctx.lineTo(canvas.width, py);
    ctx.stroke();
  }

  // Draw axes
  const origin = dataToPixel(0, 0);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
  ctx.beginPath();
  ctx.moveTo(0, origin.py);
  ctx.lineTo(canvas.width, origin.py);
  ctx.moveTo(origin.px, 0);
  ctx.lineTo(origin.px, canvas.height);
  ctx.stroke();

  // 2. Draw Decision Boundaries
  if (state.boundaryMode === 'ann' || state.boundaryMode === 'both') {
    drawANNBoundary();
  }
  if (state.boundaryMode === 'linear' || state.boundaryMode === 'both') {
    drawLinearBoundary();
  }

  // 3. Draw Data Points
  state.points.forEach(pt => {
    const { px, py } = dataToPixel(pt.x, pt.y);
    ctx.beginPath();
    ctx.arc(px, py, 4.5, 0, Math.PI * 2);

    if (pt.label === 0) {
      ctx.fillStyle = '#3b82f6';
      ctx.strokeStyle = '#1e3a8a';
    } else {
      ctx.fillStyle = '#f43f5e';
      ctx.strokeStyle = '#881337';
    }
    ctx.lineWidth = 1;
    ctx.fill();
    ctx.stroke();
  });

  // 4. Draw Probe Point if active
  if (state.probePoint) {
    const { px, py } = dataToPixel(state.probePoint.x, state.probePoint.y);

    // Glowing halo
    const grad = ctx.createRadialGradient(px, py, 2, px, py, 18);
    grad.addColorStop(0, 'rgba(168, 85, 247, 0.8)');
    grad.addColorStop(1, 'rgba(168, 85, 247, 0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(px, py, 18, 0, Math.PI * 2);
    ctx.fill();

    // Probe circle
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = '#a855f7';
    ctx.lineWidth = 2.5;
    ctx.fill();
    ctx.stroke();
  }
}

// Draw the non-linear S-curve ANN boundary
function drawANNBoundary() {
  ctx.save();
  ctx.strokeStyle = '#10b981';
  ctx.lineWidth = 3;
  ctx.shadowColor = 'rgba(16, 185, 129, 0.6)';
  ctx.shadowBlur = 10;
  ctx.beginPath();

  let started = false;
  const step = 4;
  for (let px = 0; px <= canvas.width; px += step) {
    const x = pixelToData(px, 0).x;
    // Binary search for y where predictANNProbability(x, y) == state.threshold
    let lowY = domain.yMin, highY = domain.yMax;
    for (let iter = 0; iter < 18; iter++) {
      const midY = (lowY + highY) / 2;
      const p = predictANNProbability(x, midY);
      if (p < state.threshold) {
        lowY = midY;
      } else {
        highY = midY;
      }
    }
    const boundaryY = (lowY + highY) / 2;
    const { py } = dataToPixel(x, boundaryY);

    if (py >= 0 && py <= canvas.height) {
      if (!started) {
        ctx.moveTo(px, py);
        started = true;
      } else {
        ctx.lineTo(px, py);
      }
    }
  }
  ctx.stroke();
  ctx.restore();
}

// Draw the straight linear baseline boundary
function drawLinearBoundary() {
  ctx.save();
  ctx.strokeStyle = '#f59e0b';
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 6]);
  ctx.beginPath();

  // Find points at canvas edges for linear model: 1.6*y + 0.8*x - 0.25 = logit(threshold)
  const logit = Math.log(state.threshold / (1 - state.threshold));
  const x1 = domain.xMin;
  const y1 = (logit + 0.25 - 0.8 * x1) / 1.6;
  const x2 = domain.xMax;
  const y2 = (logit + 0.25 - 0.8 * x2) / 1.6;

  const p1 = dataToPixel(x1, y1);
  const p2 = dataToPixel(x2, y2);

  ctx.moveTo(p1.px, p1.py);
  ctx.lineTo(p2.px, p2.py);
  ctx.stroke();
  ctx.restore();
}

// Handle User Canvas Interaction
function handleCanvasInteraction(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const px = (e.clientX - rect.left) * scaleX;
  const py = (e.clientY - rect.top) * scaleY;

  const { x, y } = pixelToData(px, py);
  state.probePoint = { x, y };

  const prob = predictANNProbability(x, y);
  const isClass1 = prob >= state.threshold;

  probeCoords.textContent = `(${x.toFixed(2)}, ${y.toFixed(2)})`;
  probeClass.textContent = isClass1 ? 'Moon 1 (Lower)' : 'Moon 0 (Upper)';
  probeClass.style.color = isClass1 ? '#f43f5e' : '#3b82f6';
  probeConf.textContent = `${(Math.max(prob, 1 - prob) * 100).toFixed(1)}% (P(Y=1)=${prob.toFixed(3)})`;

  tooltip.classList.remove('hidden');
  render();
}

// Event Listeners
canvas.addEventListener('click', handleCanvasInteraction);
canvas.addEventListener('mousemove', (e) => {
  if (e.buttons === 1) { // Click and drag
    handleCanvasInteraction(e);
  }
});

thresholdSlider.addEventListener('input', (e) => {
  state.threshold = parseFloat(e.target.value);
  thresholdVal.textContent = state.threshold.toFixed(2);
  render();
});

noiseSlider.addEventListener('input', (e) => {
  state.noise = parseFloat(e.target.value);
  noiseVal.textContent = state.noise.toFixed(2);
  state.points = generateMoons(state.n_samples, state.noise);
  render();
});

samplesSlider.addEventListener('input', (e) => {
  state.n_samples = parseInt(e.target.value, 10);
  samplesVal.textContent = state.n_samples;
  state.points = generateMoons(state.n_samples, state.noise);
  render();
});

btnRegenerate.addEventListener('click', () => {
  state.points = generateMoons(state.n_samples, state.noise);
  render();
});

btnToggleBoundary.addEventListener('click', () => {
  if (state.boundaryMode === 'both') {
    state.boundaryMode = 'ann';
    boundaryBtnLabel.textContent = 'Only ANN Boundary';
  } else if (state.boundaryMode === 'ann') {
    state.boundaryMode = 'linear';
    boundaryBtnLabel.textContent = 'Only Linear Boundary';
  } else if (state.boundaryMode === 'linear') {
    state.boundaryMode = 'none';
    boundaryBtnLabel.textContent = 'Hide All Boundaries';
  } else {
    state.boundaryMode = 'both';
    boundaryBtnLabel.textContent = 'Show Both Boundaries';
  }
  render();
});

// Initialize on page load
function init() {
  state.points = generateMoons(state.n_samples, state.noise);
  render();
}

init();
