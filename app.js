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
    visualizer.draw();
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

    // Show 4 samples, one of each type
    const samples = [
        IMAGE_DATASET[0],  // Horizontal
        IMAGE_DATASET[4],  // Vertical
        IMAGE_DATASET[8],  // Diagonal
        IMAGE_DATASET[12]  // Cross
    ];

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
 * Update prediction display
 */
function updatePrediction() {
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
 * Update training stats
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
    document.getElementById('trainAccuracy').textContent =
        Math.round(trainCorrect / IMAGE_DATASET.length * 100) + '%';

    // Test accuracy
    let testCorrect = 0;
    TEST_DATASET.forEach(ex => {
        const predicted = network.classify(ex.inputs);
        const expected = ex.targets.indexOf(1);
        if (predicted === expected) testCorrect++;
    });
    document.getElementById('testAccuracy').textContent =
        Math.round(testCorrect / TEST_DATASET.length * 100) + '%';
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
    updatePrediction();
    updateStats();
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
