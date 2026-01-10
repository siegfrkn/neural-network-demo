/**
 * Main Application
 * Connects the neural network with visualization and user controls
 */

// Initialize the neural network with 4 hidden nodes
const network = new NeuralNetwork(2, 4, 1);
let visualizer;
let currentDataset = null;
let currentProblemName = null;
let isTraining = false;
let trainInterval = null;

// DOM Elements
const elements = {
    truthTableSection: document.getElementById('truthTableSection'),
    truthTableBody: document.getElementById('truthTableBody'),
    problemName: document.getElementById('problemName'),
    weightsSection: document.getElementById('weightsSection'),
    weightsDisplay: document.getElementById('weightsDisplay'),
    explanationText: document.getElementById('explanationText'),
    epochCount: document.getElementById('epochCount'),
    accuracy: document.getElementById('accuracy'),
    avgError: document.getElementById('avgError'),
    learningRate: document.getElementById('learningRate'),
    lrValue: document.getElementById('lrValue'),
    trainUntilLearnedBtn: document.getElementById('trainUntilLearnedBtn'),
    trainStepBtn: document.getElementById('trainStepBtn'),
    train100Btn: document.getElementById('train100Btn'),
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

    setupEventListeners();
    visualizer.draw();
}

/**
 * Set up all event listeners
 */
function setupEventListeners() {
    // Problem selection buttons
    elements.xorBtn.addEventListener('click', () => selectProblem('XOR'));
    elements.andBtn.addEventListener('click', () => selectProblem('AND'));
    elements.orBtn.addEventListener('click', () => selectProblem('OR'));

    // Training controls
    elements.trainUntilLearnedBtn.addEventListener('click', trainUntilLearned);
    elements.trainStepBtn.addEventListener('click', trainOneEpoch);
    elements.train100Btn.addEventListener('click', () => trainEpochs(100));
    elements.resetBtn.addEventListener('click', resetNetwork);

    // Learning rate slider
    elements.learningRate.addEventListener('input', () => {
        const lr = parseFloat(elements.learningRate.value);
        elements.lrValue.textContent = lr.toFixed(1);
        network.setLearningRate(lr);
    });
}

/**
 * Select a problem to train on
 */
function selectProblem(name) {
    stopTraining();
    network.reset();
    currentDataset = DATASETS[name];
    currentProblemName = name;

    // Update UI
    elements.truthTableSection.style.display = 'block';
    elements.weightsSection.style.display = 'block';
    elements.problemName.textContent = name;

    // Highlight selected button
    document.querySelectorAll('.problem-btn').forEach(btn => btn.classList.remove('selected'));
    document.getElementById(name.toLowerCase() + 'Btn').classList.add('selected');

    // Update explanation
    updateExplanation(name);

    // Update displays
    updateTruthTable();
    updateStats();
    updateWeightsDisplay();
    visualizer.update();
}

/**
 * Update the explanation panel based on selected problem
 */
function updateExplanation(name) {
    const explanations = {
        XOR: {
            title: 'XOR (Exclusive OR)',
            points: [
                'Output is 1 only when inputs are different',
                'This is NOT linearly separable - requires hidden layer',
                'Classic problem that proved need for multi-layer networks'
            ]
        },
        AND: {
            title: 'AND Gate',
            points: [
                'Output is 1 only when BOTH inputs are 1',
                'Linearly separable - could be solved with single layer',
                'Learns quickly due to simple decision boundary'
            ]
        },
        OR: {
            title: 'OR Gate',
            points: [
                'Output is 1 when ANY input is 1',
                'Also linearly separable',
                'Very easy for the network to learn'
            ]
        }
    };

    const info = explanations[name];

    // Clear and rebuild
    while (elements.explanationText.firstChild) {
        elements.explanationText.removeChild(elements.explanationText.firstChild);
    }

    const title = document.createElement('p');
    const strong = document.createElement('strong');
    strong.textContent = info.title;
    title.appendChild(strong);
    elements.explanationText.appendChild(title);

    const ul = document.createElement('ul');
    info.points.forEach(point => {
        const li = document.createElement('li');
        li.textContent = point;
        ul.appendChild(li);
    });
    elements.explanationText.appendChild(ul);
}

/**
 * Train for one epoch (all 4 examples)
 */
function trainOneEpoch() {
    if (!currentDataset) return;

    for (const example of currentDataset) {
        network.train(example.inputs, example.targets);
    }

    updateTruthTable();
    updateStats();
    updateWeightsDisplay();
    visualizer.update();
}

/**
 * Train for multiple epochs
 */
function trainEpochs(count) {
    if (!currentDataset) return;

    for (let i = 0; i < count; i++) {
        for (const example of currentDataset) {
            network.train(example.inputs, example.targets);
        }
    }

    updateTruthTable();
    updateStats();
    updateWeightsDisplay();
    visualizer.update();
}

/**
 * Train until the network has learned (all predictions correct)
 */
function trainUntilLearned() {
    if (!currentDataset || isTraining) {
        stopTraining();
        return;
    }

    isTraining = true;
    elements.trainUntilLearnedBtn.textContent = 'Stop Training';
    elements.trainUntilLearnedBtn.classList.remove('btn-success');
    elements.trainUntilLearnedBtn.classList.add('btn-warning');

    trainInterval = setInterval(() => {
        // Train one full epoch
        for (const example of currentDataset) {
            network.train(example.inputs, example.targets);
        }

        updateTruthTable();
        updateStats();
        updateWeightsDisplay();
        visualizer.update();

        // Check if learned (all predictions within 0.1 of expected)
        const accuracy = calculateAccuracy();
        if (accuracy === 100 || network.epoch > 50000) {
            stopTraining();
        }
    }, 20);
}

/**
 * Stop auto training
 */
function stopTraining() {
    isTraining = false;
    if (trainInterval) {
        clearInterval(trainInterval);
        trainInterval = null;
    }
    elements.trainUntilLearnedBtn.textContent = 'Train Until Learned';
    elements.trainUntilLearnedBtn.classList.remove('btn-warning');
    elements.trainUntilLearnedBtn.classList.add('btn-success');
}

/**
 * Reset the network
 */
function resetNetwork() {
    stopTraining();
    network.reset();

    if (currentDataset) {
        updateTruthTable();
        updateStats();
        updateWeightsDisplay();
        visualizer.update();
    }
}

/**
 * Calculate accuracy (% of correct predictions)
 */
function calculateAccuracy() {
    if (!currentDataset) return 0;

    let correct = 0;
    for (const example of currentDataset) {
        const prediction = network.predict(example.inputs)[0];
        const expected = example.targets[0];
        // Consider correct if prediction rounds to expected value
        if (Math.round(prediction) === expected) {
            correct++;
        }
    }
    return Math.round((correct / currentDataset.length) * 100);
}

/**
 * Update the truth table display
 */
function updateTruthTable() {
    if (!currentDataset) return;

    // Clear existing rows
    while (elements.truthTableBody.firstChild) {
        elements.truthTableBody.removeChild(elements.truthTableBody.firstChild);
    }

    for (const example of currentDataset) {
        const prediction = network.predict(example.inputs)[0];
        const expected = example.targets[0];
        const isCorrect = Math.round(prediction) === expected;

        const row = document.createElement('tr');
        row.className = isCorrect ? 'correct' : 'incorrect';

        // Input 1
        const td1 = document.createElement('td');
        td1.textContent = example.inputs[0];
        row.appendChild(td1);

        // Input 2
        const td2 = document.createElement('td');
        td2.textContent = example.inputs[1];
        row.appendChild(td2);

        // Expected
        const td3 = document.createElement('td');
        td3.textContent = expected;
        row.appendChild(td3);

        // Prediction
        const td4 = document.createElement('td');
        td4.className = 'prediction-cell';
        td4.textContent = prediction.toFixed(3);
        row.appendChild(td4);

        // Status
        const td5 = document.createElement('td');
        td5.className = 'status-cell';
        td5.textContent = isCorrect ? 'Correct' : 'Learning...';
        row.appendChild(td5);

        elements.truthTableBody.appendChild(row);
    }
}

/**
 * Update statistics display
 */
function updateStats() {
    elements.epochCount.textContent = Math.floor(network.epoch / 4); // 4 examples per epoch
    elements.accuracy.textContent = calculateAccuracy() + '%';

    // Calculate average error across all examples
    if (currentDataset) {
        let totalError = 0;
        for (const example of currentDataset) {
            const prediction = network.predict(example.inputs)[0];
            totalError += Math.pow(example.targets[0] - prediction, 2);
        }
        elements.avgError.textContent = (totalError / currentDataset.length).toFixed(4);
    }
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
 * Update the weights display panel
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
    h4_1.textContent = 'Input to Hidden';
    group1.appendChild(h4_1);

    for (let i = 0; i < network.inputSize; i++) {
        for (let j = 0; j < network.hiddenSize; j++) {
            group1.appendChild(createWeightSpan('W' + i + '' + j, weights.inputHidden[i][j]));
        }
    }
    elements.weightsDisplay.appendChild(group1);

    // Hidden to Output weights
    const group2 = document.createElement('div');
    group2.className = 'weight-group';
    const h4_2 = document.createElement('h4');
    h4_2.textContent = 'Hidden to Output';
    group2.appendChild(h4_2);

    for (let j = 0; j < network.hiddenSize; j++) {
        group2.appendChild(createWeightSpan('W' + j, weights.hiddenOutput[j][0]));
    }
    elements.weightsDisplay.appendChild(group2);
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', init);
