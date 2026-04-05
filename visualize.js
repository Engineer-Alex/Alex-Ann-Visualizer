/*
  Alex_A Neural Network Input Grid #13
  Reference: Eng. Alex Amaral (AB&C Engineering Systems, LLC)
  Prepared with Visual Studio Code + GitHub Copilot + GitHub
  Date: 2026-04-05

  Academic note:
  This file drives the complete educational demonstration. It captures freehand
  strokes, stores them as a 28×28 intensity matrix, normalizes the matrix,
  estimates the digit with a lightweight non-MNIST prototype matcher, and then
  projects the result onto an ANN-style visualization for teaching purposes.
*/

// DOM references for the input canvas, network canvas, and live metric labels.
const inputCanvas = document.getElementById('inputCanvas');
const networkCanvas = document.getElementById('networkCanvas');
const clearButton = document.getElementById('clearButton');
const inputEnergyLabel = document.getElementById('inputEnergy');
const topOutputLabel = document.getElementById('topOutput');

const inputCtx = inputCanvas.getContext('2d');
const networkCtx = networkCanvas.getContext('2d');

// Global interaction state: whether the pointer is drawing, the last analysis,
// and the current cursor cell used to render the crosshair.
let isDrawing = false;
let latestAnalysis = null;
let blinkAnimationId = null;
let cursorCell = null;

// Grid and brush configuration.
// `pixelGrid` stores the grayscale intensity of the user's stroke.
// `BRUSH_KERNEL` softly spreads each paint action to mimic handwriting.
const GRID_SIZE = 28;
const CELL_SIZE = inputCanvas.width / GRID_SIZE;
const pixelGrid = Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(0));
const BRUSH_KERNEL = [
  [0.00, 0.05, 0.10, 0.05, 0.00],
  [0.05, 0.24, 0.48, 0.24, 0.05],
  [0.10, 0.48, 0.82, 0.48, 0.10],
  [0.05, 0.24, 0.48, 0.24, 0.05],
  [0.00, 0.05, 0.10, 0.05, 0.00],
];

// -----------------------------------------------------------------------------
// Canvas rendering helpers
// -----------------------------------------------------------------------------

// Draw the visible 28×28 lattice so each matrix cell is clear to the learner.
function drawGridLines() {
  inputCtx.save();
  inputCtx.strokeStyle = 'rgba(120, 180, 255, 0.18)';
  inputCtx.lineWidth = 1;

  for (let i = 0; i <= GRID_SIZE; i += 1) {
    const x = i * CELL_SIZE;
    inputCtx.beginPath();
    inputCtx.moveTo(x, 0);
    inputCtx.lineTo(x, inputCanvas.height);
    inputCtx.stroke();
  }

  for (let i = 0; i <= GRID_SIZE; i += 1) {
    const y = i * CELL_SIZE;
    inputCtx.beginPath();
    inputCtx.moveTo(0, y);
    inputCtx.lineTo(inputCanvas.width, y);
    inputCtx.stroke();
  }

  inputCtx.restore();
}

// Draw a small cross at the current pointer location to show where input lands.
function drawCursorMarker() {
  if (!cursorCell) {
    return;
  }

  const centerX = (cursorCell.col + 0.5) * CELL_SIZE;
  const centerY = (cursorCell.row + 0.5) * CELL_SIZE;
  const arm = Math.max(3, CELL_SIZE * 0.35);

  inputCtx.save();
  inputCtx.strokeStyle = '#ffffff';
  inputCtx.lineWidth = 1.5;
  inputCtx.beginPath();
  inputCtx.moveTo(centerX - arm, centerY);
  inputCtx.lineTo(centerX + arm, centerY);
  inputCtx.moveTo(centerX, centerY - arm);
  inputCtx.lineTo(centerX, centerY + arm);
  inputCtx.stroke();
  inputCtx.restore();
}

// Repaint the entire input canvas from the numeric `pixelGrid` intensities.
function renderInputGrid() {
  inputCtx.fillStyle = '#0f1a2b';
  inputCtx.fillRect(0, 0, inputCanvas.width, inputCanvas.height);

  for (let row = 0; row < GRID_SIZE; row += 1) {
    for (let col = 0; col < GRID_SIZE; col += 1) {
      const activation = pixelGrid[row][col];
      const tone = Math.round(236 - activation * 220);
      const blueMix = Math.round(246 - activation * 180);
      inputCtx.fillStyle = `rgb(${tone - 8}, ${tone}, ${blueMix})`;
      inputCtx.fillRect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE);
    }
  }

  drawGridLines();
  drawCursorMarker();
}

function resetInputGrid() {
  for (let row = 0; row < GRID_SIZE; row += 1) {
    for (let col = 0; col < GRID_SIZE; col += 1) {
      pixelGrid[row][col] = 0;
    }
  }

  cursorCell = null;
  renderInputGrid();
}

function updateCursorPosition(x, y) {
  cursorCell = {
    col: clamp(Math.floor(x / CELL_SIZE), 0, GRID_SIZE - 1),
    row: clamp(Math.floor(y / CELL_SIZE), 0, GRID_SIZE - 1),
  };
}

function paintAt(x, y) {
  updateCursorPosition(x, y);

  const kernelRadius = Math.floor(BRUSH_KERNEL.length / 2);
  for (let ky = 0; ky < BRUSH_KERNEL.length; ky += 1) {
    for (let kx = 0; kx < BRUSH_KERNEL[ky].length; kx += 1) {
      const row = cursorCell.row + ky - kernelRadius;
      const col = cursorCell.col + kx - kernelRadius;

      if (row < 0 || row >= GRID_SIZE || col < 0 || col >= GRID_SIZE) {
        continue;
      }

      const strength = BRUSH_KERNEL[ky][kx];
      pixelGrid[row][col] = clamp(pixelGrid[row][col] + strength * 0.34);
    }
  }

  renderInputGrid();
}

function drawStroke(x, y) {
  paintAt(x, y);
}

// -----------------------------------------------------------------------------
// Input sampling and normalization helpers
// -----------------------------------------------------------------------------

// Convert the current grid into a numerical matrix and compute average energy.
function sampleInputMatrix() {
  const matrix = pixelGrid.map((row) => row.map((value) => Number(value.toFixed(4))));
  const totalEnergy = matrix.reduce(
    (sum, row) => sum + row.reduce((rowSum, value) => rowSum + value, 0),
    0,
  );

  return {
    matrix,
    energy: totalEnergy / (GRID_SIZE * GRID_SIZE),
  };
}

// Create a blank matrix, useful during resampling, shifting, and template building.
function createEmptyMatrix(size = GRID_SIZE) {
  return Array.from({ length: size }, () => Array(size).fill(0));
}

// Bilinear sampling estimates values between cells when the drawing is rescaled.
function sampleMatrixValue(matrix, row, col) {
  const row0 = clamp(Math.floor(row), 0, GRID_SIZE - 1);
  const col0 = clamp(Math.floor(col), 0, GRID_SIZE - 1);
  const row1 = clamp(row0 + 1, 0, GRID_SIZE - 1);
  const col1 = clamp(col0 + 1, 0, GRID_SIZE - 1);
  const rowMix = row - row0;
  const colMix = col - col0;

  const top = matrix[row0][col0] * (1 - colMix) + matrix[row0][col1] * colMix;
  const bottom = matrix[row1][col0] * (1 - colMix) + matrix[row1][col1] * colMix;
  return top * (1 - rowMix) + bottom * rowMix;
}

function shiftMatrix(matrix, rowShift, colShift) {
  const shifted = createEmptyMatrix();

  for (let row = 0; row < GRID_SIZE; row += 1) {
    for (let col = 0; col < GRID_SIZE; col += 1) {
      const targetRow = row + rowShift;
      const targetCol = col + colShift;
      if (targetRow >= 0 && targetRow < GRID_SIZE && targetCol >= 0 && targetCol < GRID_SIZE) {
        shifted[targetRow][targetCol] = matrix[row][col];
      }
    }
  }

  return shifted;
}

// Normalize the handwritten digit by cropping, resizing, and re-centering it.
// This reduces sensitivity to where the user happened to draw on the canvas.
function normalizeMatrix(matrix) {
  let minRow = GRID_SIZE;
  let minCol = GRID_SIZE;
  let maxRow = -1;
  let maxCol = -1;

  for (let row = 0; row < GRID_SIZE; row += 1) {
    for (let col = 0; col < GRID_SIZE; col += 1) {
      if (matrix[row][col] > 0.06) {
        minRow = Math.min(minRow, row);
        minCol = Math.min(minCol, col);
        maxRow = Math.max(maxRow, row);
        maxCol = Math.max(maxCol, col);
      }
    }
  }

  if (maxRow === -1 || maxCol === -1) {
    return createEmptyMatrix();
  }

  minRow = Math.max(0, minRow - 1);
  minCol = Math.max(0, minCol - 1);
  maxRow = Math.min(GRID_SIZE - 1, maxRow + 1);
  maxCol = Math.min(GRID_SIZE - 1, maxCol + 1);

  const cropHeight = maxRow - minRow + 1;
  const cropWidth = maxCol - minCol + 1;
  const targetSize = 20;
  const scale = Math.min(targetSize / cropWidth, targetSize / cropHeight);
  const scaledHeight = Math.max(1, Math.round(cropHeight * scale));
  const scaledWidth = Math.max(1, Math.round(cropWidth * scale));
  const normalized = createEmptyMatrix();
  const rowOffset = Math.floor((GRID_SIZE - scaledHeight) / 2);
  const colOffset = Math.floor((GRID_SIZE - scaledWidth) / 2);

  for (let row = 0; row < scaledHeight; row += 1) {
    for (let col = 0; col < scaledWidth; col += 1) {
      const sourceRow = minRow + ((row + 0.5) / scaledHeight) * cropHeight - 0.5;
      const sourceCol = minCol + ((col + 0.5) / scaledWidth) * cropWidth - 0.5;
      normalized[rowOffset + row][colOffset + col] = sampleMatrixValue(matrix, sourceRow, sourceCol);
    }
  }

  let mass = 0;
  let rowMoment = 0;
  let colMoment = 0;
  for (let row = 0; row < GRID_SIZE; row += 1) {
    for (let col = 0; col < GRID_SIZE; col += 1) {
      const value = normalized[row][col];
      mass += value;
      rowMoment += row * value;
      colMoment += col * value;
    }
  }

  if (mass < 0.0001) {
    return normalized;
  }

  const centerRow = rowMoment / mass;
  const centerCol = colMoment / mass;
  return shiftMatrix(normalized, Math.round(13.5 - centerRow), Math.round(13.5 - centerCol));
}

function stampTemplatePoint(matrix, row, col, scale = 0.34) {
  const kernelRadius = Math.floor(BRUSH_KERNEL.length / 2);

  for (let ky = 0; ky < BRUSH_KERNEL.length; ky += 1) {
    for (let kx = 0; kx < BRUSH_KERNEL[ky].length; kx += 1) {
      const targetRow = row + ky - kernelRadius;
      const targetCol = col + kx - kernelRadius;

      if (targetRow < 0 || targetRow >= GRID_SIZE || targetCol < 0 || targetCol >= GRID_SIZE) {
        continue;
      }

      matrix[targetRow][targetCol] = clamp(matrix[targetRow][targetCol] + BRUSH_KERNEL[ky][kx] * scale);
    }
  }
}

function drawTemplateStroke(matrix, points, scale = 0.34) {
  for (let index = 0; index < points.length - 1; index += 1) {
    const [x0, y0] = points[index];
    const [x1, y1] = points[index + 1];
    const steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0)) * 2 + 1;

    for (let step = 0; step <= steps; step += 1) {
      const t = step / steps;
      const x = x0 + (x1 - x0) * t;
      const y = y0 + (y1 - y0) * t;
      stampTemplatePoint(matrix, Math.round(y), Math.round(x), scale);
    }
  }
}

// -----------------------------------------------------------------------------
// Prototype generation for the non-MNIST classifier
// -----------------------------------------------------------------------------

// Cache the handcrafted digit prototypes so they are built only once.
let digitTemplateCache = null;
function getDigitTemplates() {
  if (digitTemplateCache) {
    return digitTemplateCache;
  }

  const templateLibrary = {
    0: [
      [
        [[8, 4], [5, 8], [5, 19], [8, 24], [14, 25], [20, 24], [23, 19], [23, 8], [20, 4], [14, 3], [8, 4]],
      ],
    ],
    1: [
      [
        [[10, 8], [13, 4], [13, 24]],
        [[9, 24], [18, 24]],
        [[10, 8], [13, 4], [16, 8]],
      ],
    ],
    2: [
      [
        [[6, 8], [10, 4], [18, 4], [22, 8], [19, 12], [14, 16], [9, 20], [6, 24], [22, 24]],
      ],
    ],
    3: [
      [
        [[7, 6], [11, 4], [18, 4], [22, 8], [18, 12], [12, 14], [18, 16], [22, 20], [18, 24], [10, 24], [7, 22]],
      ],
    ],
    4: [
      [
        [[18, 4], [18, 24]],
        [[7, 14], [22, 14]],
        [[8, 4], [8, 14]],
      ],
    ],
    5: [
      [
        [[21, 4], [8, 4], [8, 13], [18, 13], [22, 17], [18, 24], [8, 24]],
      ],
    ],
    6: [
      [
        [[20, 5], [12, 4], [8, 8], [7, 15], [10, 23], [18, 24], [22, 20], [18, 15], [10, 15]],
      ],
    ],
    7: [
      [
        [[6, 5], [22, 5], [15, 24]],
      ],
    ],
    8: [
      [
        [[10, 4], [18, 4], [22, 8], [18, 13], [10, 13], [6, 8], [10, 4]],
        [[10, 13], [18, 13], [22, 18], [18, 24], [10, 24], [6, 18], [10, 13]],
      ],
    ],
    9: [
      [
        [[18, 24], [21, 19], [21, 10], [18, 5], [10, 4], [6, 8], [10, 13], [18, 13], [21, 10]],
      ],
      [
        [[17, 23], [20, 19], [21, 12], [18, 6], [11, 5], [7, 8], [9, 13], [16, 13], [20, 11], [19, 18], [17, 24]],
      ],
      [
        [[17, 22], [20, 17], [20, 9], [17, 5], [11, 5], [8, 8], [10, 12], [16, 12], [20, 10]],
        [[20, 10], [20, 18], [16, 24]],
      ],
    ],
  };

  digitTemplateCache = Object.fromEntries(
    Object.entries(templateLibrary).map(([digit, variants]) => [
      digit,
      variants.map((strokes) => {
        const matrix = createEmptyMatrix();
        strokes.forEach((stroke) => drawTemplateStroke(matrix, stroke, 0.34));
        return normalizeMatrix(matrix);
      }),
    ]),
  );

  return digitTemplateCache;
}

// Compare a normalized input against a prototype using overlap and density.
function scoreMatrixAgainstTemplate(matrix, template) {
  let overlap = 0;
  let union = 0;
  let difference = 0;
  let templateMass = 0;
  let inputMass = 0;

  for (let row = 0; row < GRID_SIZE; row += 1) {
    for (let col = 0; col < GRID_SIZE; col += 1) {
      const inputValue = matrix[row][col];
      const templateValue = template[row][col];
      overlap += Math.min(inputValue, templateValue);
      union += Math.max(inputValue, templateValue);
      difference += Math.abs(inputValue - templateValue);
      inputMass += inputValue;
      templateMass += templateValue;
    }
  }

  const overlapRatio = overlap / Math.max(templateMass, 0.0001);
  const jaccard = overlap / Math.max(union, 0.0001);
  const densityPenalty = Math.abs(inputMass - templateMass) / Math.max(templateMass, 0.0001);
  return overlapRatio * 1.55 + jaccard * 1.35 - difference / (GRID_SIZE * GRID_SIZE) * 0.55 - densityPenalty * 0.25;
}

function measureRegion(matrix, rowStart, rowEnd, colStart, colEnd) {
  let sum = 0;
  let count = 0;

  for (let row = rowStart; row <= rowEnd; row += 1) {
    for (let col = colStart; col <= colEnd; col += 1) {
      sum += matrix[row][col];
      count += 1;
    }
  }

  return count > 0 ? sum / count : 0;
}

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

// -----------------------------------------------------------------------------
// Feature extraction and educational classification
// -----------------------------------------------------------------------------

// Analyze the input matrix, estimate the best digit, and synthesize layer
// activations that can be visualized as a simplified ANN.
function analyzeMatrix(matrix, energy) {
  const normalizedMatrix = normalizeMatrix(matrix);
  const segments = {
    top: measureRegion(normalizedMatrix, 2, 5, 6, 21),
    upperLeft: measureRegion(normalizedMatrix, 5, 11, 3, 7),
    upperRight: measureRegion(normalizedMatrix, 5, 11, 20, 24),
    middle: measureRegion(normalizedMatrix, 11, 15, 6, 21),
    lowerLeft: measureRegion(normalizedMatrix, 16, 23, 3, 7),
    lowerRight: measureRegion(normalizedMatrix, 16, 23, 20, 24),
    bottom: measureRegion(normalizedMatrix, 22, 25, 6, 21),
    center: measureRegion(normalizedMatrix, 6, 23, 11, 16),
  };

  const leftMass = measureRegion(normalizedMatrix, 0, 27, 0, 13);
  const rightMass = measureRegion(normalizedMatrix, 0, 27, 14, 27);
  const upperMass = measureRegion(normalizedMatrix, 0, 13, 0, 27);
  const lowerMass = measureRegion(normalizedMatrix, 14, 27, 0, 27);
  const centerColumn = measureRegion(normalizedMatrix, 0, 27, 11, 16);
  const centerRow = measureRegion(normalizedMatrix, 11, 16, 0, 27);

  const inputActivations = [];
  for (let row = 0; row < 4; row += 1) {
    for (let col = 0; col < 4; col += 1) {
      inputActivations.push(
        measureRegion(
          normalizedMatrix,
          row * 7,
          Math.min(27, row * 7 + 6),
          col * 7,
          Math.min(27, col * 7 + 6),
        ),
      );
    }
  }

  if (energy < 0.01) {
    return {
      bestDigit: null,
      confidence: 0,
      secondaryDigit: null,
      secondaryConfidence: 0,
      inputActivations,
      hidden1Activations: Array(14).fill(0),
      hidden2Activations: Array(14).fill(0),
      outputActivations: Array(10).fill(0),
    };
  }

  const rawScores = Array(10).fill(0);
  const digitTemplates = getDigitTemplates();

  Object.entries(digitTemplates).forEach(([digit, templates]) => {
    const bestTemplateScore = Math.max(
      ...templates.map((template) => scoreMatrixAgainstTemplate(normalizedMatrix, template)),
    );
    rawScores[Number(digit)] = bestTemplateScore * 2.25;
  });

  rawScores[0] += segments.top * 0.85 + segments.bottom * 0.85 + segments.upperLeft * 0.7 + segments.upperRight * 0.7 + (1 - segments.middle) * 0.28;
  rawScores[1] += centerColumn * 0.95 + segments.upperRight * 0.25 + segments.lowerRight * 0.25;
  rawScores[2] += segments.top * 0.75 + segments.upperRight * 0.65 + segments.middle * 0.65 + segments.lowerLeft * 0.55 + segments.bottom * 0.65;
  rawScores[3] += segments.top * 0.78 + segments.upperRight * 0.72 + segments.middle * 0.64 + segments.lowerRight * 0.72 + segments.bottom * 0.68 + rightMass * 0.14 - segments.lowerLeft * 0.12;
  rawScores[4] += segments.upperLeft * 0.62 + segments.upperRight * 0.58 + segments.middle * 0.92 + centerColumn * 0.32 + segments.lowerRight * 0.42 - segments.bottom * 0.12;
  rawScores[5] += segments.top * 0.62 + segments.upperLeft * 0.68 + segments.middle * 0.6 + segments.lowerRight * 0.58 + segments.bottom * 0.62;
  rawScores[6] += segments.top * 0.52 + segments.upperLeft * 0.8 + segments.middle * 0.62 + segments.lowerLeft * 0.46 + segments.lowerRight * 0.56 + segments.bottom * 0.6 - rightMass * 0.12;
  rawScores[7] += segments.top * 0.92 + segments.upperRight * 0.55 + segments.lowerRight * 0.52 - segments.lowerLeft * 0.18;
  rawScores[8] += segments.top * 0.55 + segments.upperLeft * 0.5 + segments.upperRight * 0.5 + segments.middle * 0.72 + segments.lowerLeft * 0.46 + segments.lowerRight * 0.46 + segments.bottom * 0.52;
  rawScores[9] += segments.top * 0.62 + segments.upperLeft * 0.34 + segments.upperRight * 0.6 + segments.middle * 0.62 + segments.lowerRight * 0.62 + upperMass * 0.12 - segments.lowerLeft * 0.14;

  let bestDigit = 0;
  let bestScore = -Infinity;
  rawScores.forEach((score, digit) => {
    if (score > bestScore) {
      bestScore = score;
      bestDigit = digit;
    }
  });

  const positiveScores = rawScores.map((score) => Math.max(score, 0));
  const rankedScores = positiveScores
    .map((score, digit) => ({ digit, score }))
    .sort((a, b) => b.score - a.score);

  const secondaryDigit = rankedScores[1]?.digit ?? null;
  const secondaryScore = rankedScores[1]?.score ?? 0;
  const maxScore = Math.max(...positiveScores, 0.0001);
  const scoreTotal = positiveScores.reduce((sum, score) => sum + score, 0);
  const outputActivations = positiveScores.map((score) => clamp(score / maxScore));
  const confidence = scoreTotal > 0 ? (positiveScores[bestDigit] / scoreTotal) * 100 : 0;
  const secondaryConfidence = scoreTotal > 0 ? (secondaryScore / scoreTotal) * 100 : 0;

  const hidden1Activations = [
    segments.top,
    segments.upperLeft,
    segments.upperRight,
    segments.middle,
    segments.lowerLeft,
    segments.lowerRight,
    segments.bottom,
    segments.center,
    leftMass,
    rightMass,
    upperMass,
    lowerMass,
    centerColumn,
    clamp(energy * 2.2),
  ];

  const hidden2Activations = [
    outputActivations[0],
    outputActivations[1],
    outputActivations[2],
    outputActivations[3],
    outputActivations[4],
    outputActivations[6],
    outputActivations[7],
    outputActivations[9],
    centerRow,
    centerColumn,
    segments.upperRight,
    segments.lowerLeft,
    clamp((leftMass + rightMass) / 2),
    outputActivations[bestDigit],
  ];

  return {
    bestDigit,
    confidence,
    secondaryDigit,
    secondaryConfidence,
    inputActivations,
    hidden1Activations,
    hidden2Activations,
    outputActivations,
  };
}

function formatTopOutput(analysis) {
  return analysis.bestDigit === null
    ? 'None'
    : `${analysis.bestDigit} (${analysis.confidence.toFixed(1)}%)`;
}

function updateReadouts() {
  const { matrix, energy } = sampleInputMatrix();
  const analysis = { ...analyzeMatrix(matrix, energy), energy, blinkStrength: 0 };
  latestAnalysis = analysis;
  inputEnergyLabel.textContent = energy.toFixed(3);
  topOutputLabel.textContent = formatTopOutput(analysis);
  initNetworkCanvas(analysis);
}

function stopOutputBlinking() {
  if (blinkAnimationId !== null) {
    cancelAnimationFrame(blinkAnimationId);
    blinkAnimationId = null;
  }
}

function startOutputBlinking() {
  stopOutputBlinking();

  if (!latestAnalysis || latestAnalysis.bestDigit === null) {
    return;
  }

  const animate = (timestamp) => {
    const blinkStrength = 0.5 + 0.5 * Math.sin(timestamp / 170);
    initNetworkCanvas({ ...latestAnalysis, blinkStrength });
    blinkAnimationId = requestAnimationFrame(animate);
  };

  blinkAnimationId = requestAnimationFrame(animate);
}

function clearCanvas() {
  stopOutputBlinking();
  cursorCell = null;
  resetInputGrid();
  updateReadouts();
}

// -----------------------------------------------------------------------------
// Interaction wiring: pointer events update the matrix and refresh the display.
// -----------------------------------------------------------------------------
inputCanvas.style.touchAction = 'none';

inputCanvas.addEventListener('pointerdown', (event) => {
  event.preventDefault();
  stopOutputBlinking();
  isDrawing = true;
  const rect = inputCanvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * (inputCanvas.width / rect.width);
  const y = (event.clientY - rect.top) * (inputCanvas.height / rect.height);
  paintAt(x, y);
  updateReadouts();
});

inputCanvas.addEventListener('pointermove', (event) => {
  const rect = inputCanvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * (inputCanvas.width / rect.width);
  const y = (event.clientY - rect.top) * (inputCanvas.height / rect.height);

  updateCursorPosition(x, y);

  if (!isDrawing) {
    renderInputGrid();
    return;
  }

  drawStroke(x, y);
  updateReadouts();
});

['pointerup', 'pointerleave', 'pointercancel'].forEach((eventName) => {
  inputCanvas.addEventListener(eventName, () => {
    if (isDrawing) {
      isDrawing = false;
      updateReadouts();
      startOutputBlinking();
    }

    cursorCell = null;
    renderInputGrid();
  });
});

clearButton.addEventListener('click', clearCanvas);

// -----------------------------------------------------------------------------
// Network visualization helpers
// -----------------------------------------------------------------------------

// Create one horizontal layer of square nodes for the network diagram.
function createRowLayer({ centerY, activations, color, title, subtitle, maxBoxSize = 48, sidePadding = 72, gap = 8 }) {
  const count = activations.length;
  const usableWidth = networkCanvas.width - sidePadding * 2;
  const boxSize = Math.min(maxBoxSize, (usableWidth - gap * (count - 1)) / count);
  const rowWidth = count * boxSize + (count - 1) * gap;
  const startX = (networkCanvas.width - rowWidth) / 2 + boxSize / 2;

  return {
    centerX: networkCanvas.width / 2,
    centerY,
    title,
    subtitle,
    color,
    activations,
    nodes: Array.from({ length: count }, (_, index) => ({
      x: startX + index * (boxSize + gap),
      y: centerY,
      size: boxSize,
    })),
  };
}

// Provide a safe empty analysis object before the user has drawn anything.
function createEmptyAnalysis() {
  return {
    energy: 0,
    bestDigit: null,
    confidence: 0,
    inputActivations: Array(16).fill(0),
    hidden1Activations: Array(14).fill(0),
    hidden2Activations: Array(14).fill(0),
    outputActivations: Array(10).fill(0),
  };
}

// Render the textual summary above the network canvas.
function drawNetworkSummary(analysis) {
  networkCtx.textAlign = 'left';
  networkCtx.fillStyle = '#e8eef7';
  networkCtx.font = '600 20px Segoe UI';

  const summaryText = analysis.bestDigit === null
    ? `Live guess: —   |   Input energy: ${analysis.energy.toFixed(3)}   |   Filled boxes = stronger activation`
    : `Live guess: ${analysis.bestDigit} (${analysis.confidence.toFixed(1)}%)   |   Input energy: ${analysis.energy.toFixed(3)}   |   Filled boxes = stronger activation`;

  networkCtx.fillText(summaryText, 28, 38);

  networkCtx.fillStyle = '#b8cbe3';
  networkCtx.font = '15px Segoe UI';
  const secondaryText = analysis.secondaryDigit === null
    ? 'Secondary prediction: —'
    : `Secondary prediction: ${analysis.secondaryDigit} (${analysis.secondaryConfidence.toFixed(1)}%)`;
  networkCtx.fillText(secondaryText, 28, 64);
}

function drawLayerLabels(layer) {
  const labelY = layer.centerY - (layer.nodes[0]?.size ?? 0) / 2 - 18;
  networkCtx.textAlign = 'center';
  networkCtx.fillStyle = '#e8eef7';
  networkCtx.font = '600 18px Segoe UI';
  networkCtx.fillText(layer.title, layer.centerX, labelY);

  networkCtx.fillStyle = '#8aa0bf';
  networkCtx.font = '14px Segoe UI';
  networkCtx.fillText(layer.subtitle, layer.centerX, labelY + 20);
}

// Draw weighted-looking connections between consecutive network layers.
function drawConnections(layers) {
  for (let layerIndex = 0; layerIndex < layers.length - 1; layerIndex += 1) {
    const sourceLayer = layers[layerIndex];
    const targetLayer = layers[layerIndex + 1];

    sourceLayer.nodes.forEach((sourceNode, sourceIndex) => {
      targetLayer.nodes.forEach((targetNode, targetIndex) => {
        const sourceStrength = sourceLayer.activations[sourceIndex] ?? 0;
        const targetStrength = targetLayer.activations[targetIndex] ?? 0;
        const strength = clamp((sourceStrength + targetStrength) / 2);

        if (strength < 0.04 && sourceStrength < 0.05 && targetStrength < 0.05) {
          return;
        }

        const midY = (sourceNode.y + targetNode.y) / 2;
        networkCtx.beginPath();
        networkCtx.moveTo(sourceNode.x, sourceNode.y + sourceNode.size / 2);
        networkCtx.bezierCurveTo(sourceNode.x, midY, targetNode.x, midY, targetNode.x, targetNode.y - targetNode.size / 2);
        networkCtx.lineWidth = 0.8 + strength * 2.2;
        networkCtx.strokeStyle = `rgba(79, 209, 255, ${0.02 + strength * 0.34})`;
        networkCtx.stroke();
      });
    });
  }
}

// Draw square neurons whose fill level represents activation strength.
function drawNodes(layer, options = {}) {
  const { isOutput = false, highlightIndex = -1, blinkStrength = 0 } = options;

  layer.nodes.forEach((node, index) => {
    const activation = clamp(layer.activations[index] ?? 0);
    const isHighlighted = index === highlightIndex;
    const pulse = isHighlighted ? blinkStrength : 0;
    const size = node.size + pulse * 4;
    const left = node.x - size / 2;
    const top = node.y - size / 2;
    const innerPadding = 4;
    const innerSize = size - innerPadding * 2;
    const fillHeight = innerSize * activation;
    const fillTop = top + innerPadding + innerSize - fillHeight;

    networkCtx.fillStyle = 'rgba(8, 16, 29, 0.98)';
    networkCtx.fillRect(left, top, size, size);

    networkCtx.fillStyle = 'rgba(14, 24, 38, 0.95)';
    networkCtx.fillRect(left + innerPadding, top + innerPadding, innerSize, innerSize);

    if (fillHeight > 0) {
      const fillGradient = networkCtx.createLinearGradient(0, top + innerPadding, 0, top + innerPadding + innerSize);
      fillGradient.addColorStop(0, `rgba(${layer.color}, ${0.28 + activation * 0.22})`);
      fillGradient.addColorStop(1, `rgba(${layer.color}, ${0.78 + activation * 0.18 + pulse * 0.08})`);
      networkCtx.fillStyle = fillGradient;
      networkCtx.shadowColor = `rgba(${layer.color}, 0.95)`;
      networkCtx.shadowBlur = 4 + activation * 12 + pulse * 10;
      networkCtx.fillRect(left + innerPadding, fillTop, innerSize, fillHeight);
      networkCtx.shadowBlur = 0;
    }

    networkCtx.lineWidth = isHighlighted ? 3 + pulse : 2;
    networkCtx.strokeStyle = isHighlighted
      ? `rgba(255, 255, 255, ${0.6 + pulse * 0.35})`
      : 'rgba(232, 238, 247, 0.38)';
    networkCtx.strokeRect(left, top, size, size);

    if (isOutput) {
      networkCtx.fillStyle = isHighlighted ? '#ffffff' : '#f4f8ff';
      networkCtx.font = '600 16px Segoe UI';
      networkCtx.textAlign = 'center';
      networkCtx.fillText(String(index), node.x, node.y + size / 2 + 22);
    }
  });
}

// Master renderer for the network panel: clear, summarize, then draw all layers.
function initNetworkCanvas(analysis = createEmptyAnalysis()) {
  networkCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);
  networkCtx.fillStyle = '#0a121e';
  networkCtx.fillRect(0, 0, networkCanvas.width, networkCanvas.height);

  networkCtx.fillStyle = 'rgba(79, 209, 255, 0.08)';
  networkCtx.fillRect(0, 0, networkCanvas.width, 92);

  drawNetworkSummary(analysis);

  const layers = [
    createRowLayer({
      centerY: 170,
      activations: analysis.inputActivations,
      color: '102, 217, 255',
      title: 'Input',
      subtitle: '784 neurons',
      maxBoxSize: 48,
      gap: 8,
    }),
    createRowLayer({
      centerY: 320,
      activations: analysis.hidden1Activations,
      color: '126, 240, 165',
      title: 'Hidden 1',
      subtitle: '50 neurons',
      maxBoxSize: 52,
      gap: 10,
    }),
    createRowLayer({
      centerY: 480,
      activations: analysis.hidden2Activations,
      color: '255, 222, 89',
      title: 'Hidden 2',
      subtitle: '50 neurons',
      maxBoxSize: 52,
      gap: 10,
    }),
    createRowLayer({
      centerY: 650,
      activations: analysis.outputActivations,
      color: '245, 248, 255',
      title: 'Output',
      subtitle: '10 neurons',
      maxBoxSize: 60,
      gap: 14,
    }),
  ];

  drawConnections(layers);
  layers.forEach((layer, index) => {
    drawNodes(layer, {
      isOutput: index === layers.length - 1,
      highlightIndex: index === layers.length - 1 && analysis.bestDigit !== null ? analysis.bestDigit : -1,
      blinkStrength: index === layers.length - 1 ? (analysis.blinkStrength ?? 0) : 0,
    });
    drawLayerLabels(layer);
  });
}

// Final startup sequence: clear the grid, draw the empty network, and compute the initial labels.
resetInputGrid();
initNetworkCanvas();
updateReadouts();