/**
 * Main Application
 * Supports both Logic Gates and Image Recognition modes
 */

// Networks for each mode
let logicNetwork = new NeuralNetwork([2, 4, 1]);
let imageNetwork = new NeuralNetwork([25, 16, 8, 4]);

let currentNetwork = logicNetwork;
let visualizer;
let currentMode = 'logic';

// Training state
let isTraining = false;
let trainInterval = null;
let currentLogicDataset = null;
let pixelGrid = new Array(25).fill(0);

/**
 * Initialize the application
 */
function init() {
    visualizer = new NetworkVisualizer('networkCanvas', currentNetwork);

    setupModeButtons();
    setupLogicMode();
    setupImageMode();

    visualizer.draw();
}

// =============================================
// MODE SWITCHING
// =============================================

function setupModeButtons() {
    document.getElementById('logicModeBtn').addEventListener('click', () => switchMode('logic'));
    document.getElementById('imageModeBtn').addEventListener('click', () => switchMode('image'));
}

function switchMode(mode) {
    stopTraining();
    currentMode = mode;

    // Update button styles
    document.getElementById('logicModeBtn').classList.toggle('selected', mode === 'logic');
    document.getElementById('imageModeBtn').classList.toggle('selected', mode === 'image');

    // Show/hide mode sections
    document.getElementById('logicMode').style.display = mode === 'logic' ? 'block' : 'none';
    document.getElementById('imageMode').style.display = mode === 'image' ? 'block' : 'none';

    // Switch network
    if (mode === 'logic') {
        currentNetwork = logicNetwork;
        updateLayerLabels(['Input (2)', 'Hidden (4)', 'Output (1)']);
    } else {
        currentNetwork = imageNetwork;
        updateLayerLabels(['Input (25)', 'Hidden (16)', 'Hidden (8)', 'Output (4)']);
        updateImagePrediction();
    }

    visualizer.setNetwork(currentNetwork);
    visualizer.draw();

    updateExplanation(mode);
}

function updateLayerLabels(labels) {
    const container = document.getElementById('layerLabels');
    while (container.firstChild) {
        container.removeChild(container.firstChild);
    }

    const colors = ['input-label', 'hidden-label', 'hidden-label', 'output-label'];
    labels.forEach((text, i) => {
        const span = document.createElement('span');
        span.className = 'label ' + (i === 0 ? colors[0] : (i === labels.length - 1 ? colors[3] : colors[1]));
        span.textContent = text;
        container.appendChild(span);
    });
}

function updateExplanation(mode) {
    const titleEl = document.getElementById('explanationTitle');
    const textEl = document.getElementById('explanationText');

    while (textEl.firstChild) {
        textEl.removeChild(textEl.firstChild);
    }

    if (mode === 'logic') {
        titleEl.textContent = 'Logic Gates Mode';
        const points = [
            'Simple 3-layer network: 2 inputs, 4 hidden neurons, 1 output',
            'Perfect for understanding basic neural network concepts',
            'XOR is the classic problem that requires a hidden layer to solve'
        ];
        points.forEach(text => {
            const p = document.createElement('p');
            p.textContent = text;
            textEl.appendChild(p);
        });
    } else {
        titleEl.textContent = 'Image Recognition Mode';
        const points = [
            'Deep 4-layer network: 25 inputs (5x5 pixels), 16 hidden, 8 hidden, 4 outputs',
            'Recognizes patterns: Horizontal lines, Vertical lines, Diagonals, and Crosses',
            'Draw your own pattern or click samples to test the network!'
        ];
        points.forEach(text => {
            const p = document.createElement('p');
            p.textContent = text;
            textEl.appendChild(p);
        });
    }
}

// =============================================
// LOGIC GATES MODE
// =============================================

function setupLogicMode() {
    document.getElementById('xorBtn').addEventListener('click', () => selectLogicProblem('XOR'));
    document.getElementById('andBtn').addEventListener('click', () => selectLogicProblem('AND'));
    document.getElementById('orBtn').addEventListener('click', () => selectLogicProblem('OR'));

    // Build training panel
    buildLogicTrainingPanel();
}

function buildLogicTrainingPanel() {
    const panel = document.getElementById('logicTrainingPanel');

    // Stats display
    const stats = document.createElement('div');
    stats.className = 'stats-display';
    stats.id = 'logicStats';

    const statItems = [
        { id: 'logicEpochCount', label: 'Epochs', value: '0' },
        { id: 'logicAccuracy', label: 'Accuracy', value: '0%' },
        { id: 'logicError', label: 'Avg Error', value: '-' }
    ];

    statItems.forEach(item => {
        const div = document.createElement('div');
        div.className = 'stat-item';
        const num = document.createElement('span');
        num.className = 'stat-number';
        num.id = item.id;
        num.textContent = item.value;
        const label = document.createElement('span');
        label.className = 'stat-label';
        label.textContent = item.label;
        div.appendChild(num);
        div.appendChild(label);
        stats.appendChild(div);
    });
    panel.appendChild(stats);

    // Training controls
    const controls = document.createElement('div');
    controls.className = 'training-controls';

    const trainBtn = document.createElement('button');
    trainBtn.id = 'logicTrainBtn';
    trainBtn.className = 'btn btn-large btn-success';
    trainBtn.textContent = 'Train Until Learned';
    trainBtn.addEventListener('click', trainLogicUntilLearned);

    const train100Btn = document.createElement('button');
    train100Btn.className = 'btn btn-primary';
    train100Btn.textContent = 'Train 100 Epochs';
    train100Btn.addEventListener('click', () => trainLogicEpochs(100));

    const resetBtn = document.createElement('button');
    resetBtn.className = 'btn btn-danger';
    resetBtn.textContent = 'Reset';
    resetBtn.addEventListener('click', resetLogicNetwork);

    controls.appendChild(trainBtn);
    controls.appendChild(train100Btn);
    controls.appendChild(resetBtn);
    panel.appendChild(controls);

    // Learning rate
    const speedDiv = document.createElement('div');
    speedDiv.className = 'speed-control';
    const lrLabel = document.createElement('label');
    lrLabel.textContent = 'Learning Rate: ';
    const lrSpan = document.createElement('span');
    lrSpan.id = 'logicLrValue';
    lrSpan.textContent = '0.5';
    lrLabel.appendChild(lrSpan);

    const lrInput = document.createElement('input');
    lrInput.type = 'range';
    lrInput.id = 'logicLearningRate';
    lrInput.min = '0.1';
    lrInput.max = '2';
    lrInput.step = '0.1';
    lrInput.value = '0.5';
    lrInput.addEventListener('input', () => {
        lrSpan.textContent = lrInput.value;
        logicNetwork.setLearningRate(parseFloat(lrInput.value));
    });

    speedDiv.appendChild(lrLabel);
    speedDiv.appendChild(lrInput);
    panel.appendChild(speedDiv);
}

function selectLogicProblem(name) {
    stopTraining();
    logicNetwork.reset();
    currentLogicDataset = LOGIC_DATASETS[name];

    document.getElementById('logicTableSection').style.display = 'grid';
    document.getElementById('logicProblemName').textContent = name;

    document.querySelectorAll('.problem-btn').forEach(btn => btn.classList.remove('selected'));
    document.getElementById(name.toLowerCase() + 'Btn').classList.add('selected');

    updateLogicTable();
    updateLogicStats();
    visualizer.update();
}

function updateLogicTable() {
    if (!currentLogicDataset) return;

    const tbody = document.getElementById('logicTableBody');
    while (tbody.firstChild) {
        tbody.removeChild(tbody.firstChild);
    }

    currentLogicDataset.forEach(example => {
        const prediction = logicNetwork.predict(example.inputs)[0];
        const expected = example.targets[0];
        const isCorrect = Math.round(prediction) === expected;

        const row = document.createElement('tr');
        row.className = isCorrect ? 'correct' : 'incorrect';

        [example.inputs[0], example.inputs[1], expected, prediction.toFixed(3), isCorrect ? 'Correct' : 'Learning...'].forEach((val, i) => {
            const td = document.createElement('td');
            td.textContent = val;
            if (i === 3) td.className = 'prediction-cell';
            if (i === 4) td.className = 'status-cell';
            row.appendChild(td);
        });

        tbody.appendChild(row);
    });
}

function updateLogicStats() {
    document.getElementById('logicEpochCount').textContent = Math.floor(logicNetwork.epoch / 4);

    if (currentLogicDataset) {
        let correct = 0;
        let totalError = 0;
        currentLogicDataset.forEach(ex => {
            const pred = logicNetwork.predict(ex.inputs)[0];
            if (Math.round(pred) === ex.targets[0]) correct++;
            totalError += Math.pow(ex.targets[0] - pred, 2);
        });
        document.getElementById('logicAccuracy').textContent = Math.round(correct / 4 * 100) + '%';
        document.getElementById('logicError').textContent = (totalError / 4).toFixed(4);
    }
}

function trainLogicEpochs(count) {
    if (!currentLogicDataset) return;

    for (let i = 0; i < count; i++) {
        currentLogicDataset.forEach(ex => logicNetwork.train(ex.inputs, ex.targets));
    }

    updateLogicTable();
    updateLogicStats();
    visualizer.update();
}

function trainLogicUntilLearned() {
    if (!currentLogicDataset || isTraining) {
        stopTraining();
        return;
    }

    isTraining = true;
    const btn = document.getElementById('logicTrainBtn');
    btn.textContent = 'Stop Training';
    btn.classList.remove('btn-success');
    btn.classList.add('btn-warning');

    trainInterval = setInterval(() => {
        currentLogicDataset.forEach(ex => logicNetwork.train(ex.inputs, ex.targets));
        updateLogicTable();
        updateLogicStats();
        visualizer.update();

        let correct = 0;
        currentLogicDataset.forEach(ex => {
            if (Math.round(logicNetwork.predict(ex.inputs)[0]) === ex.targets[0]) correct++;
        });
        if (correct === 4 || logicNetwork.epoch > 50000) stopTraining();
    }, 20);
}

function resetLogicNetwork() {
    stopTraining();
    logicNetwork.reset();
    if (currentLogicDataset) {
        updateLogicTable();
        updateLogicStats();
        visualizer.update();
    }
}

// =============================================
// IMAGE RECOGNITION MODE
// =============================================

function setupImageMode() {
    buildPixelGrid();
    buildPatternSamples();
    buildPredictionBars();

    document.getElementById('clearGridBtn').addEventListener('click', clearPixelGrid);
    document.getElementById('imageTrainBtn').addEventListener('click', trainImageUntilLearned);
    document.getElementById('imageTrain100Btn').addEventListener('click', () => trainImageEpochs(100));
    document.getElementById('imageResetBtn').addEventListener('click', resetImageNetwork);

    document.getElementById('imageLearningRate').addEventListener('input', (e) => {
        document.getElementById('imageLrValue').textContent = e.target.value;
        imageNetwork.setLearningRate(parseFloat(e.target.value));
    });
}

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

function togglePixel(index) {
    pixelGrid[index] = pixelGrid[index] === 0 ? 1 : 0;
    updatePixelDisplay();
    updateImagePrediction();
    visualizer.update();
}

function updatePixelDisplay() {
    const pixels = document.querySelectorAll('.pixel');
    pixels.forEach((pixel, i) => {
        pixel.classList.toggle('active', pixelGrid[i] === 1);
    });
}

function clearPixelGrid() {
    pixelGrid = new Array(25).fill(0);
    updatePixelDisplay();
    updateImagePrediction();
    visualizer.update();
}

function loadPattern(pattern) {
    pixelGrid = [...pattern];
    updatePixelDisplay();
    updateImagePrediction();
    visualizer.update();
}

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

function updateImagePrediction() {
    const outputs = imageNetwork.predict(pixelGrid);
    let maxIndex = 0;
    let maxValue = outputs[0];

    outputs.forEach((val, i) => {
        const percent = Math.round(val * 100);
        document.getElementById('predBar' + i).style.width = percent + '%';
        document.getElementById('predValue' + i).textContent = percent + '%';

        if (val > maxValue) {
            maxValue = val;
            maxIndex = i;
        }
    });

    document.getElementById('predictedClass').textContent = PATTERN_NAMES[maxIndex];
}

function updateImageStats() {
    document.getElementById('imageEpochCount').textContent = Math.floor(imageNetwork.epoch / IMAGE_DATASET.length);

    let correct = 0;
    let totalError = 0;

    IMAGE_DATASET.forEach(ex => {
        const predicted = imageNetwork.classify(ex.inputs);
        const expected = ex.targets.indexOf(1);
        if (predicted === expected) correct++;

        const outputs = imageNetwork.predict(ex.inputs);
        ex.targets.forEach((t, i) => {
            totalError += Math.pow(t - outputs[i], 2);
        });
    });

    document.getElementById('imageAccuracy').textContent = Math.round(correct / IMAGE_DATASET.length * 100) + '%';
    document.getElementById('imageError').textContent = (totalError / IMAGE_DATASET.length / 4).toFixed(4);
}

function trainImageEpochs(count) {
    for (let i = 0; i < count; i++) {
        IMAGE_DATASET.forEach(ex => imageNetwork.train(ex.inputs, ex.targets));
    }

    updateImagePrediction();
    updateImageStats();
    visualizer.update();
}

function trainImageUntilLearned() {
    if (isTraining) {
        stopTraining();
        return;
    }

    isTraining = true;
    const btn = document.getElementById('imageTrainBtn');
    btn.textContent = 'Stop Training';
    btn.classList.remove('btn-success');
    btn.classList.add('btn-warning');

    trainInterval = setInterval(() => {
        for (let i = 0; i < 5; i++) {
            IMAGE_DATASET.forEach(ex => imageNetwork.train(ex.inputs, ex.targets));
        }

        updateImagePrediction();
        updateImageStats();
        visualizer.update();

        let correct = 0;
        IMAGE_DATASET.forEach(ex => {
            if (imageNetwork.classify(ex.inputs) === ex.targets.indexOf(1)) correct++;
        });

        if (correct === IMAGE_DATASET.length || imageNetwork.epoch > 100000) {
            stopTraining();
        }
    }, 30);
}

function resetImageNetwork() {
    stopTraining();
    imageNetwork.reset();
    updateImagePrediction();
    updateImageStats();
    visualizer.update();
}

// =============================================
// COMMON FUNCTIONS
// =============================================

function stopTraining() {
    isTraining = false;
    if (trainInterval) {
        clearInterval(trainInterval);
        trainInterval = null;
    }

    // Reset logic button
    const logicBtn = document.getElementById('logicTrainBtn');
    if (logicBtn) {
        logicBtn.textContent = 'Train Until Learned';
        logicBtn.classList.remove('btn-warning');
        logicBtn.classList.add('btn-success');
    }

    // Reset image button
    const imageBtn = document.getElementById('imageTrainBtn');
    if (imageBtn) {
        imageBtn.textContent = 'Train Until Learned';
        imageBtn.classList.remove('btn-warning');
        imageBtn.classList.add('btn-success');
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', init);
