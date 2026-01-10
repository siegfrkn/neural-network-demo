/**
 * Main Application
 * Connects the neural network with visualization and user controls
 */

// Initialize the neural network and visualizer
const network = new NeuralNetwork(2, 3, 1);
let visualizer;
let autoTraining = false;
let autoTrainInterval = null;
let currentDataset = null;
let datasetIndex = 0;

// DOM Elements
const elements = {
    input1: document.getElementById('input1'),
    input2: document.getElementById('input2'),
    expectedOutput: document.getElementById('expectedOutput'),
    input1Value: document.getElementById('input1Value'),
    input2Value: document.getElementById('input2Value'),
    expectedOutputValue: document.getElementById('expectedOutputValue'),
    learningRate: document.getElementById('learningRate'),
    lrValue: document.getElementById('lrValue'),
    epochCount: document.getElementById('epochCount'),
    currentError: document.getElementById('currentError'),
    prediction: document.getElementById('prediction'),
    weightsDisplay: document.getElementById('weightsDisplay'),
    explanationText: document.getElementById('explanationText'),
    stepBtn: document.getElementById('stepBtn'),
    trainBtn: document.getElementById('trainBtn'),
    autoTrainBtn: document.getElementById('autoTrainBtn'),
    resetBtn: document.getElementById('resetBtn'),
    xorBtn: document.getElementById('xorBtn'),
    andBtn: document.getElementById('andBtn'),
    orBtn: document.getElementById('orBtn')
};

/**
 * Initialize the application
 */
function init() {
    visualizer = new NetworkVisualizer('networkCanvas', network);

    // Set up event listeners
    setupInputListeners();
    setupButtonListeners();

    // Initial draw
    updateDisplay();
    visualizer.draw();
}

/**
 * Set up input slider listeners
 */
function setupInputListeners() {
    // Input sliders
    elements.input1.addEventListener('input', () => {
        elements.input1Value.textContent = elements.input1.value;
        runForwardPass();
    });

    elements.input2.addEventListener('input', () => {
        elements.input2Value.textContent = elements.input2.value;
        runForwardPass();
    });

    elements.expectedOutput.addEventListener('input', () => {
        elements.expectedOutputValue.textContent = elements.expectedOutput.value;
    });

    // Learning rate slider
    elements.learningRate.addEventListener('input', () => {
        const lr = parseFloat(elements.learningRate.value);
        elements.lrValue.textContent = lr.toFixed(2);
        network.setLearningRate(lr);
    });
}

/**
 * Set up button listeners
 */
function setupButtonListeners() {
    // Step forward button - single training step with animation
    elements.stepBtn.addEventListener('click', () => {
        if (autoTraining) return;

        const inputs = [
            parseFloat(elements.input1.value),
            parseFloat(elements.input2.value)
        ];
        const targets = [parseFloat(elements.expectedOutput.value)];

        // Animate forward propagation
        network.forward(inputs);
        visualizer.animateForward(() => {
            // Then animate backpropagation
            network.backward(targets);
            network.epoch++;
            network.lastError = Math.pow(targets[0] - network.outputActivations[0], 2);

            visualizer.animateBackward(() => {
                updateDisplay();
                visualizer.update();
            });
        });
    });

    // Train button - multiple epochs without animation
    elements.trainBtn.addEventListener('click', () => {
        if (autoTraining) return;

        const inputs = [
            parseFloat(elements.input1.value),
            parseFloat(elements.input2.value)
        ];
        const targets = [parseFloat(elements.expectedOutput.value)];

        // Train for 100 epochs
        for (let i = 0; i < 100; i++) {
            network.train(inputs, targets);
        }

        updateDisplay();
        visualizer.update();
    });

    // Auto train toggle
    elements.autoTrainBtn.addEventListener('click', () => {
        if (autoTraining) {
            stopAutoTrain();
        } else {
            startAutoTrain();
        }
    });

    // Reset button
    elements.resetBtn.addEventListener('click', () => {
        stopAutoTrain();
        network.reset();
        currentDataset = null;
        updateDisplay();
        visualizer.update();
    });

    // Preset problem buttons
    elements.xorBtn.addEventListener('click', () => loadDataset('XOR'));
    elements.andBtn.addEventListener('click', () => loadDataset('AND'));
    elements.orBtn.addEventListener('click', () => loadDataset('OR'));
}

/**
 * Run a forward pass and update display
 */
function runForwardPass() {
    const inputs = [
        parseFloat(elements.input1.value),
        parseFloat(elements.input2.value)
    ];

    network.forward(inputs);
    updateDisplay();
    visualizer.update();
}

/**
 * Start auto training
 */
function startAutoTrain() {
    autoTraining = true;
    elements.autoTrainBtn.textContent = 'Stop Training';
    elements.autoTrainBtn.classList.remove('btn-info');
    elements.autoTrainBtn.classList.add('btn-warning');

    autoTrainInterval = setInterval(() => {
        if (currentDataset) {
            // Cycle through dataset
            const example = currentDataset[datasetIndex];
            network.train(example.inputs, example.targets);
            datasetIndex = (datasetIndex + 1) % currentDataset.length;

            // Update sliders to show current training example
            elements.input1.value = example.inputs[0];
            elements.input2.value = example.inputs[1];
            elements.expectedOutput.value = example.targets[0];
            elements.input1Value.textContent = example.inputs[0];
            elements.input2Value.textContent = example.inputs[1];
            elements.expectedOutputValue.textContent = example.targets[0];
        } else {
            // Train on current input values
            const inputs = [
                parseFloat(elements.input1.value),
                parseFloat(elements.input2.value)
            ];
            const targets = [parseFloat(elements.expectedOutput.value)];
            network.train(inputs, targets);
        }

        updateDisplay();
        visualizer.update();
    }, 50);
}

/**
 * Stop auto training
 */
function stopAutoTrain() {
    autoTraining = false;
    elements.autoTrainBtn.textContent = 'Auto Train';
    elements.autoTrainBtn.classList.remove('btn-warning');
    elements.autoTrainBtn.classList.add('btn-info');

    if (autoTrainInterval) {
        clearInterval(autoTrainInterval);
        autoTrainInterval = null;
    }
}

/**
 * Load a preset dataset
 */
function loadDataset(name) {
    stopAutoTrain();
    network.reset();
    currentDataset = DATASETS[name];
    datasetIndex = 0;

    // Set initial values from first example
    const example = currentDataset[0];
    elements.input1.value = example.inputs[0];
    elements.input2.value = example.inputs[1];
    elements.expectedOutput.value = example.targets[0];
    elements.input1Value.textContent = example.inputs[0];
    elements.input2Value.textContent = example.inputs[1];
    elements.expectedOutputValue.textContent = example.targets[0];

    runForwardPass();

    // Show dataset info
    showDatasetInfo(name);
}

/**
 * Show information about the loaded dataset using safe DOM methods
 */
function showDatasetInfo(name) {
    const explanations = {
        XOR: {
            title: 'XOR Problem:',
            text: 'The network must learn that output is 1 only when inputs differ. This is the classic non-linearly separable problem that proved single-layer perceptrons have limitations.'
        },
        AND: {
            title: 'AND Gate:',
            text: 'Output is 1 only when both inputs are 1. This is linearly separable and easier to learn.'
        },
        OR: {
            title: 'OR Gate:',
            text: 'Output is 1 when either input is 1. Also linearly separable and quick to learn.'
        }
    };

    // Clear existing content
    while (elements.explanationText.firstChild) {
        elements.explanationText.removeChild(elements.explanationText.firstChild);
    }

    // Create explanation paragraph
    const p1 = document.createElement('p');
    const strong = document.createElement('strong');
    strong.textContent = explanations[name].title;
    p1.appendChild(strong);
    p1.appendChild(document.createTextNode(' ' + explanations[name].text));
    elements.explanationText.appendChild(p1);

    // Create instruction paragraph
    const p2 = document.createElement('p');
    p2.textContent = 'Click "Auto Train" to watch the network learn all 4 input combinations!';
    elements.explanationText.appendChild(p2);
}

/**
 * Update all display elements
 */
function updateDisplay() {
    // Update statistics
    elements.epochCount.textContent = network.epoch;
    elements.currentError.textContent = network.lastError.toFixed(6);

    const prediction = network.outputActivations[0];
    if (prediction !== undefined) {
        elements.prediction.textContent = prediction.toFixed(4);
        elements.prediction.style.color = prediction > 0.5 ? '#00ff88' : '#ff6b6b';
    }

    // Update weights display
    updateWeightsDisplay();
}

/**
 * Create a weight span element
 */
function createWeightSpan(label, value) {
    const span = document.createElement('span');
    span.className = 'weight ' + (value >= 0 ? 'positive' : 'negative');
    span.textContent = label + ': ' + value.toFixed(3);
    return span;
}

/**
 * Update the weights display panel using safe DOM methods
 */
function updateWeightsDisplay() {
    const weights = network.getWeights();

    // Clear existing content
    while (elements.weightsDisplay.firstChild) {
        elements.weightsDisplay.removeChild(elements.weightsDisplay.firstChild);
    }

    // Input to Hidden weights
    const group1 = document.createElement('div');
    group1.className = 'weight-group';
    const h4_1 = document.createElement('h4');
    h4_1.textContent = 'Input → Hidden';
    group1.appendChild(h4_1);

    for (let i = 0; i < network.inputSize; i++) {
        for (let j = 0; j < network.hiddenSize; j++) {
            group1.appendChild(createWeightSpan('W[' + i + '][' + j + ']', weights.inputHidden[i][j]));
        }
    }
    elements.weightsDisplay.appendChild(group1);

    // Hidden to Output weights
    const group2 = document.createElement('div');
    group2.className = 'weight-group';
    const h4_2 = document.createElement('h4');
    h4_2.textContent = 'Hidden → Output';
    group2.appendChild(h4_2);

    for (let j = 0; j < network.hiddenSize; j++) {
        for (let k = 0; k < network.outputSize; k++) {
            group2.appendChild(createWeightSpan('W[' + j + '][' + k + ']', weights.hiddenOutput[j][k]));
        }
    }
    elements.weightsDisplay.appendChild(group2);

    // Biases
    const group3 = document.createElement('div');
    group3.className = 'weight-group';
    const h4_3 = document.createElement('h4');
    h4_3.textContent = 'Biases';
    group3.appendChild(h4_3);

    weights.biasHidden.forEach((b, i) => {
        group3.appendChild(createWeightSpan('bH[' + i + ']', b));
    });
    weights.biasOutput.forEach((b, i) => {
        group3.appendChild(createWeightSpan('bO[' + i + ']', b));
    });
    elements.weightsDisplay.appendChild(group3);
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', init);
