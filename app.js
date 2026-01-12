/**
 * Neural Network Pattern Recognition
 * Single-screen demo with train/test mode
 */

// Network: 25 inputs (5x5 grid) -> 16 hidden -> 8 hidden -> 4 outputs
const network = new NeuralNetwork([25, 16, 8, 4]);
let visualizer;

// State
let isTraining = false;
let trainInterval = null;
let pixelGrid = new Array(25).fill(0);

// Test mode state
let currentTestIndex = 0;
let testResults = [];

// Accuracy history for chart
let accuracyHistory = {
    train: [],
    test: [],
    maxPoints: 50  // Keep last 50 data points
};

/**
 * Initialize the application
 */
function init() {
    visualizer = new NetworkVisualizer('networkCanvas', network);

    buildPixelGrid();
    buildPatternSamples();
    buildPredictionBars();
    buildTestGrid();

    // Event listeners
    document.getElementById('clearGridBtn').addEventListener('click', clearPixelGrid);
    document.getElementById('trainBtn').addEventListener('click', toggleTraining);
    document.getElementById('testBtn').addEventListener('click', openTestMode);
    document.getElementById('resetBtn').addEventListener('click', resetNetwork);

    document.getElementById('learningRate').addEventListener('input', (e) => {
        document.getElementById('lrValue').textContent = e.target.value;
        network.setLearningRate(parseFloat(e.target.value));
    });

    // Test mode controls
    document.getElementById('testPrevBtn').addEventListener('click', showPrevTest);
    document.getElementById('testNextBtn').addEventListener('click', showNextTest);
    document.getElementById('testCloseBtn').addEventListener('click', closeTestMode);

    // Initial draw
    updatePrediction();
    updateStats();
    drawAccuracyChart();
    visualizer.draw();
}

/**
 * Draw the accuracy history chart
 */
function drawAccuracyChart() {
    const canvas = document.getElementById('accuracyChart');
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const padding = { left: 35, right: 10, top: 10, bottom: 20 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Clear canvas
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, width, height);

    // Draw Y-axis labels
    ctx.fillStyle = '#6a6a8a';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 4; i++) {
        const y = padding.top + (chartHeight / 4) * (4 - i);
        const label = (i * 25) + '%';
        ctx.fillText(label, padding.left - 5, y);
    }

    // Draw grid lines
    ctx.strokeStyle = '#2a2a4a';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = padding.top + (chartHeight / 4) * i;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();
    }

    // Draw 25% baseline (random guess for 4 classes)
    ctx.strokeStyle = '#ff6b6b';
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    const baselineY = padding.top + chartHeight * 0.75;
    ctx.moveTo(padding.left, baselineY);
    ctx.lineTo(width - padding.right, baselineY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Label for baseline
    ctx.fillStyle = '#ff6b6b';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('random', padding.left + 5, baselineY - 5);

    // If no data, show placeholder text
    if (accuracyHistory.train.length === 0) {
        ctx.fillStyle = '#6a6a8a';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Click "Train" to watch accuracy improve!', width / 2, height / 2);
        return;
    }

    const points = accuracyHistory.train.length;
    const xStep = chartWidth / Math.max(points - 1, 1);

    // Draw train accuracy line (cyan) with glow
    ctx.shadowColor = '#00d9ff';
    ctx.shadowBlur = 6;
    ctx.strokeStyle = '#00d9ff';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    accuracyHistory.train.forEach((val, i) => {
        const x = padding.left + i * xStep;
        const y = padding.top + chartHeight - (val * chartHeight);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Draw test accuracy line (purple) with glow
    ctx.shadowColor = '#9d4edd';
    ctx.shadowBlur = 6;
    ctx.strokeStyle = '#9d4edd';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    accuracyHistory.test.forEach((val, i) => {
        const x = padding.left + i * xStep;
        const y = padding.top + chartHeight - (val * chartHeight);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Draw current values as dots
    if (points > 0) {
        const lastX = padding.left + (points - 1) * xStep;

        // Train dot
        ctx.fillStyle = '#00d9ff';
        ctx.beginPath();
        ctx.arc(lastX, padding.top + chartHeight - (accuracyHistory.train[points - 1] * chartHeight), 5, 0, Math.PI * 2);
        ctx.fill();

        // Test dot
        ctx.fillStyle = '#9d4edd';
        ctx.beginPath();
        ctx.arc(lastX, padding.top + chartHeight - (accuracyHistory.test[points - 1] * chartHeight), 5, 0, Math.PI * 2);
        ctx.fill();
    }
}

/**
 * Build the 5x5 pixel grid for drawing
 */
function buildPixelGrid() {
    const container = document.getElementById('pixelGrid');
    for (let i = 0; i < 25; i++) {
        const pixel = document.createElement('div');
        pixel.className = 'pixel';
        pixel.dataset.index = i;
        pixel.addEventListener('click', () => togglePixel(i));
        container.appendChild(pixel);
    }
}

/**
 * Build the test grid (non-interactive display)
 */
function buildTestGrid() {
    const container = document.getElementById('testGrid');
    for (let i = 0; i < 25; i++) {
        const pixel = document.createElement('div');
        pixel.className = 'pixel';
        container.appendChild(pixel);
    }
}

/**
 * Toggle a pixel on/off
 */
function togglePixel(index) {
    pixelGrid[index] = pixelGrid[index] === 0 ? 1 : 0;
    updatePixelDisplay();
    updatePrediction();
    visualizer.update();
}

/**
 * Update pixel grid display
 */
function updatePixelDisplay() {
    const pixels = document.querySelectorAll('#pixelGrid .pixel');
    pixels.forEach((pixel, i) => {
        pixel.classList.toggle('active', pixelGrid[i] === 1);
    });
}

/**
 * Clear the pixel grid
 */
function clearPixelGrid() {
    pixelGrid = new Array(25).fill(0);
    updatePixelDisplay();
    updatePrediction();
    visualizer.update();
}

/**
 * Load a pattern into the grid
 */
function loadPattern(pattern) {
    pixelGrid = [...pattern];
    updatePixelDisplay();
    updatePrediction();
    visualizer.update();
}

/**
 * Build sample pattern buttons
 */
function buildPatternSamples() {
    const container = document.getElementById('patternSamples');

    // Find one sample of each type dynamically
    const samples = PATTERN_NAMES.map((name, typeIndex) => {
        // Find first pattern where targets[typeIndex] === 1
        return IMAGE_DATASET.find(p => p.targets[typeIndex] === 1);
    });

    samples.forEach((sample, typeIndex) => {
        const sampleDiv = document.createElement('div');
        sampleDiv.className = 'pattern-sample';
        sampleDiv.title = sample.name;

        const miniGrid = document.createElement('div');
        miniGrid.className = 'mini-grid';

        sample.inputs.forEach(val => {
            const pixel = document.createElement('div');
            pixel.className = 'mini-pixel' + (val === 1 ? ' active' : '');
            miniGrid.appendChild(pixel);
        });

        const label = document.createElement('span');
        label.className = 'sample-label';
        label.textContent = PATTERN_NAMES[typeIndex];

        sampleDiv.appendChild(miniGrid);
        sampleDiv.appendChild(label);
        sampleDiv.addEventListener('click', () => loadPattern(sample.inputs));

        container.appendChild(sampleDiv);
    });
}

/**
 * Build prediction bars
 */
function buildPredictionBars() {
    const container = document.getElementById('predictionBars');

    PATTERN_NAMES.forEach((name, i) => {
        const barContainer = document.createElement('div');
        barContainer.className = 'prediction-bar-container';

        const label = document.createElement('span');
        label.className = 'bar-label';
        label.textContent = name;

        const barBg = document.createElement('div');
        barBg.className = 'bar-bg';

        const bar = document.createElement('div');
        bar.className = 'bar-fill';
        bar.id = 'predBar' + i;

        const value = document.createElement('span');
        value.className = 'bar-value';
        value.id = 'predValue' + i;
        value.textContent = '0%';

        barBg.appendChild(bar);
        barContainer.appendChild(label);
        barContainer.appendChild(barBg);
        barContainer.appendChild(value);
        container.appendChild(barContainer);
    });
}

/**
 * Get current pixel count
 */
function getPixelCount() {
    return pixelGrid.filter(p => p === 1).length;
}

/**
 * Update pixel warning visibility
 */
function updatePixelWarning() {
    const pixelCount = getPixelCount();
    const warning = document.getElementById('pixelWarning');
    const minPixels = 4;

    if (pixelCount < minPixels) {
        warning.classList.remove('hidden');
        warning.querySelector('.warning-text').textContent =
            'Draw at least 4 pixels (' + pixelCount + '/4)';
    } else {
        warning.classList.add('hidden');
    }
}

/**
 * Update prediction display
 */
function updatePrediction() {
    const pixelCount = getPixelCount();
    const minPixels = 4;

    // Update warning
    updatePixelWarning();

    // Check minimum pixel requirement
    if (pixelCount < minPixels) {
        // Reset all bars and show message
        PATTERN_NAMES.forEach((name, i) => {
            const bar = document.getElementById('predBar' + i);
            bar.style.width = '0%';
            bar.classList.remove('winner');
            document.getElementById('predValue' + i).textContent = '-';
        });
        document.getElementById('predictedClass').textContent = 'Need 4+ pixels';
        return;
    }

    const outputs = network.predict(pixelGrid);
    let maxIndex = 0;
    let maxValue = outputs[0];

    // Update bars and find winner
    outputs.forEach((val, i) => {
        const percent = Math.round(val * 100);
        const bar = document.getElementById('predBar' + i);
        bar.style.width = percent + '%';
        bar.classList.remove('winner');
        document.getElementById('predValue' + i).textContent = percent + '%';

        if (val > maxValue) {
            maxValue = val;
            maxIndex = i;
        }
    });

    // Highlight winner
    document.getElementById('predBar' + maxIndex).classList.add('winner');
    document.getElementById('predictedClass').textContent = PATTERN_NAMES[maxIndex];
}

/**
 * Update training stats and accuracy chart
 */
function updateStats() {
    document.getElementById('epochCount').textContent = Math.floor(network.epoch / IMAGE_DATASET.length);

    // Training accuracy
    let trainCorrect = 0;
    IMAGE_DATASET.forEach(ex => {
        const predicted = network.classify(ex.inputs);
        const expected = ex.targets.indexOf(1);
        if (predicted === expected) trainCorrect++;
    });
    const trainAcc = trainCorrect / IMAGE_DATASET.length;
    document.getElementById('trainAccuracy').textContent = Math.round(trainAcc * 100) + '%';

    // Test accuracy
    let testCorrect = 0;
    TEST_DATASET.forEach(ex => {
        const predicted = network.classify(ex.inputs);
        const expected = ex.targets.indexOf(1);
        if (predicted === expected) testCorrect++;
    });
    const testAcc = testCorrect / TEST_DATASET.length;
    document.getElementById('testAccuracy').textContent = Math.round(testAcc * 100) + '%';

    // Record accuracy history (only during training)
    if (isTraining) {
        accuracyHistory.train.push(trainAcc);
        accuracyHistory.test.push(testAcc);

        // Keep only last maxPoints
        if (accuracyHistory.train.length > accuracyHistory.maxPoints) {
            accuracyHistory.train.shift();
            accuracyHistory.test.shift();
        }

        drawAccuracyChart();
    }
}

/**
 * Toggle training on/off
 */
function toggleTraining() {
    if (isTraining) {
        stopTraining();
    } else {
        startTraining();
    }
}

/**
 * Start training
 */
function startTraining() {
    isTraining = true;
    const btn = document.getElementById('trainBtn');
    btn.textContent = 'Stop';
    btn.classList.add('btn-training');

    trainInterval = setInterval(() => {
        // Train on 5 epochs per frame for visible progress
        for (let i = 0; i < 5; i++) {
            IMAGE_DATASET.forEach(ex => network.train(ex.inputs, ex.targets));
        }

        updatePrediction();
        updateStats();
        visualizer.update();

        // Check if fully trained
        let correct = 0;
        IMAGE_DATASET.forEach(ex => {
            if (network.classify(ex.inputs) === ex.targets.indexOf(1)) correct++;
        });

        // Cap at 1000 epochs (epoch count = network.epoch / dataset size)
        const maxEpochs = 1000 * IMAGE_DATASET.length;
        if (correct === IMAGE_DATASET.length || network.epoch > maxEpochs) {
            stopTraining();
        }
    }, 50);
}

/**
 * Stop training
 */
function stopTraining() {
    isTraining = false;
    if (trainInterval) {
        clearInterval(trainInterval);
        trainInterval = null;
    }

    const btn = document.getElementById('trainBtn');
    btn.textContent = 'Train';
    btn.classList.remove('btn-training');
}

/**
 * Reset the network
 */
function resetNetwork() {
    stopTraining();
    network.reset();
    visualizer.previousWeights = null;
    visualizer.weightChanges = null;

    // Clear accuracy history
    accuracyHistory.train = [];
    accuracyHistory.test = [];

    updatePrediction();
    updateStats();
    drawAccuracyChart();
    visualizer.update();
}

// =============================================
// TEST MODE
// =============================================

/**
 * Open test mode overlay
 */
function openTestMode() {
    stopTraining();

    // Calculate results for all test patterns
    testResults = TEST_DATASET.map(ex => {
        const predicted = network.classify(ex.inputs);
        const expected = ex.targets.indexOf(1);
        return {
            pattern: ex,
            predicted: predicted,
            expected: expected,
            correct: predicted === expected
        };
    });

    currentTestIndex = 0;

    // Update counts
    document.getElementById('testCount').textContent = TEST_DATASET.length;
    document.getElementById('testTotal').textContent = TEST_DATASET.length;

    // Show overlay
    document.getElementById('testOverlay').classList.remove('hidden');

    // Display first test
    displayCurrentTest();
}

/**
 * Close test mode overlay
 */
function closeTestMode() {
    document.getElementById('testOverlay').classList.add('hidden');
}

/**
 * Display current test pattern and result
 */
function displayCurrentTest() {
    const result = testResults[currentTestIndex];
    const pattern = result.pattern;

    // Update test grid
    const pixels = document.querySelectorAll('#testGrid .pixel');
    pixels.forEach((pixel, i) => {
        pixel.classList.toggle('active', pattern.inputs[i] === 1);
    });

    // Update pattern name
    document.getElementById('testPatternName').textContent = pattern.name;

    // Update expected and predicted
    document.getElementById('testExpected').textContent = PATTERN_NAMES[result.expected];
    document.getElementById('testPredicted').textContent = PATTERN_NAMES[result.predicted];

    // Update status
    const statusEl = document.getElementById('testStatus');
    if (result.correct) {
        statusEl.textContent = 'CORRECT';
        statusEl.className = 'test-status correct';
    } else {
        statusEl.textContent = 'INCORRECT';
        statusEl.className = 'test-status incorrect';
    }

    // Update progress
    document.getElementById('testProgress').textContent =
        (currentTestIndex + 1) + ' / ' + TEST_DATASET.length;

    // Update score
    const score = testResults.filter(r => r.correct).length;
    document.getElementById('testScore').textContent = score;

    // Load pattern into main grid for visualization
    loadPattern(pattern.inputs);
}

/**
 * Show previous test pattern
 */
function showPrevTest() {
    if (currentTestIndex > 0) {
        currentTestIndex--;
        displayCurrentTest();
    }
}

/**
 * Show next test pattern
 */
function showNextTest() {
    if (currentTestIndex < TEST_DATASET.length - 1) {
        currentTestIndex++;
        displayCurrentTest();
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', init);
