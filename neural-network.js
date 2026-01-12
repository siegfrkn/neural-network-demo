/**
 * Deep Neural Network Implementation
 * Supports multiple hidden layers for complex pattern recognition
 */

class NeuralNetwork {
    constructor(layerSizes) {
        // layerSizes is an array like [25, 16, 8, 4] for input, hidden1, hidden2, output
        this.layerSizes = layerSizes;
        this.numLayers = layerSizes.length;
        this.learningRate = 0.5;

        // Initialize weights and biases for each layer transition
        this.weights = [];
        this.biases = [];

        for (let i = 0; i < this.numLayers - 1; i++) {
            this.weights.push(this.initializeWeights(layerSizes[i], layerSizes[i + 1]));
            this.biases.push(new Array(layerSizes[i + 1]).fill(0).map(() => Math.random() * 0.5 - 0.25));
        }

        // Store activations for each layer (for visualization)
        this.activations = layerSizes.map(size => new Array(size).fill(0));

        // Training statistics
        this.epoch = 0;
        this.lastError = 0;
    }

    /**
     * Initialize weights using Xavier initialization
     */
    initializeWeights(inputSize, outputSize) {
        const weights = [];
        const limit = Math.sqrt(6 / (inputSize + outputSize));

        for (let i = 0; i < inputSize; i++) {
            weights[i] = [];
            for (let j = 0; j < outputSize; j++) {
                weights[i][j] = (Math.random() * 2 - 1) * limit;
            }
        }
        return weights;
    }

    /**
     * Sigmoid activation function
     */
    sigmoid(x) {
        return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
    }

    /**
     * Derivative of sigmoid
     */
    sigmoidDerivative(x) {
        return x * (1 - x);
    }

    /**
     * Forward propagation through all layers
     */
    forward(inputs) {
        this.activations[0] = [...inputs];

        for (let layer = 0; layer < this.numLayers - 1; layer++) {
            const currentActivations = this.activations[layer];
            const nextSize = this.layerSizes[layer + 1];

            for (let j = 0; j < nextSize; j++) {
                let sum = this.biases[layer][j];
                for (let i = 0; i < currentActivations.length; i++) {
                    sum += currentActivations[i] * this.weights[layer][i][j];
                }
                this.activations[layer + 1][j] = this.sigmoid(sum);
            }
        }

        return this.activations[this.numLayers - 1];
    }

    /**
     * Backpropagation through all layers
     */
    backward(targets) {
        const gradients = [];

        // Calculate output layer gradients
        const outputLayer = this.numLayers - 1;
        gradients[outputLayer] = [];
        for (let k = 0; k < this.layerSizes[outputLayer]; k++) {
            const error = targets[k] - this.activations[outputLayer][k];
            gradients[outputLayer][k] = error * this.sigmoidDerivative(this.activations[outputLayer][k]);
        }

        // Calculate hidden layer gradients (backpropagate)
        for (let layer = outputLayer - 1; layer > 0; layer--) {
            gradients[layer] = [];
            for (let j = 0; j < this.layerSizes[layer]; j++) {
                let error = 0;
                for (let k = 0; k < this.layerSizes[layer + 1]; k++) {
                    error += gradients[layer + 1][k] * this.weights[layer][j][k];
                }
                gradients[layer][j] = error * this.sigmoidDerivative(this.activations[layer][j]);
            }
        }

        // Update weights and biases
        for (let layer = 0; layer < this.numLayers - 1; layer++) {
            for (let i = 0; i < this.layerSizes[layer]; i++) {
                for (let j = 0; j < this.layerSizes[layer + 1]; j++) {
                    this.weights[layer][i][j] += this.learningRate * gradients[layer + 1][j] * this.activations[layer][i];
                }
            }
            for (let j = 0; j < this.layerSizes[layer + 1]; j++) {
                this.biases[layer][j] += this.learningRate * gradients[layer + 1][j];
            }
        }
    }

    /**
     * Train on a single example
     */
    train(inputs, targets) {
        this.forward(inputs);
        this.backward(targets);

        // Calculate mean squared error
        let totalError = 0;
        const outputLayer = this.numLayers - 1;
        for (let k = 0; k < this.layerSizes[outputLayer]; k++) {
            totalError += Math.pow(targets[k] - this.activations[outputLayer][k], 2);
        }
        this.lastError = totalError / this.layerSizes[outputLayer];
        this.epoch++;

        return this.lastError;
    }

    /**
     * Get prediction
     */
    predict(inputs) {
        return this.forward(inputs);
    }

    /**
     * Get the index of the highest output (for classification)
     */
    classify(inputs) {
        const outputs = this.forward(inputs);
        let maxIndex = 0;
        let maxValue = outputs[0];
        for (let i = 1; i < outputs.length; i++) {
            if (outputs[i] > maxValue) {
                maxValue = outputs[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * Reset the network
     */
    reset() {
        for (let i = 0; i < this.numLayers - 1; i++) {
            this.weights[i] = this.initializeWeights(this.layerSizes[i], this.layerSizes[i + 1]);
            this.biases[i] = new Array(this.layerSizes[i + 1]).fill(0).map(() => Math.random() * 0.5 - 0.25);
        }
        this.activations = this.layerSizes.map(size => new Array(size).fill(0));
        this.epoch = 0;
        this.lastError = 0;
    }

    /**
     * Set learning rate
     */
    setLearningRate(rate) {
        this.learningRate = rate;
    }

    /**
     * Get all weights for visualization
     */
    getWeights() {
        return this.weights;
    }

    /**
     * Get all activations for visualization
     */
    getActivations() {
        return this.activations;
    }

    /**
     * Get layer sizes
     */
    getLayerSizes() {
        return this.layerSizes;
    }
}

// =============================================
// DATASETS
// =============================================

// Simple logic gate datasets (2 inputs -> 1 output)
const LOGIC_DATASETS = {
    XOR: [
        { inputs: [0, 0], targets: [0] },
        { inputs: [0, 1], targets: [1] },
        { inputs: [1, 0], targets: [1] },
        { inputs: [1, 1], targets: [0] }
    ],
    AND: [
        { inputs: [0, 0], targets: [0] },
        { inputs: [0, 1], targets: [0] },
        { inputs: [1, 0], targets: [0] },
        { inputs: [1, 1], targets: [1] }
    ],
    OR: [
        { inputs: [0, 0], targets: [0] },
        { inputs: [0, 1], targets: [1] },
        { inputs: [1, 0], targets: [1] },
        { inputs: [1, 1], targets: [1] }
    ]
};

// 5x5 pixel pattern datasets for image recognition
// Each pattern is a 25-element array (5x5 grid flattened)
// Outputs are one-hot encoded: [horizontal, vertical, diagonal, cross]

const PATTERN_NAMES = ['Horizontal', 'Vertical', 'Diagonal', 'Cross'];

// EXPANDED TRAINING DATASET - More variations = better generalization
const IMAGE_DATASET = [
    // ============ HORIZONTAL LINES ============
    // Full lines at every row
    { inputs: [1,1,1,1,1, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0], targets: [1,0,0,0], name: 'H row 0' },
    { inputs: [0,0,0,0,0, 1,1,1,1,1, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0], targets: [1,0,0,0], name: 'H row 1' },
    { inputs: [0,0,0,0,0, 0,0,0,0,0, 1,1,1,1,1, 0,0,0,0,0, 0,0,0,0,0], targets: [1,0,0,0], name: 'H row 2' },
    { inputs: [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 1,1,1,1,1, 0,0,0,0,0], targets: [1,0,0,0], name: 'H row 3' },
    { inputs: [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 1,1,1,1,1], targets: [1,0,0,0], name: 'H row 4' },
    // Partial lines (4 pixels)
    { inputs: [0,0,0,0,0, 0,0,0,0,0, 1,1,1,1,0, 0,0,0,0,0, 0,0,0,0,0], targets: [1,0,0,0], name: 'H partial L' },
    { inputs: [0,0,0,0,0, 0,0,0,0,0, 0,1,1,1,1, 0,0,0,0,0, 0,0,0,0,0], targets: [1,0,0,0], name: 'H partial R' },
    { inputs: [0,0,0,0,0, 1,1,1,1,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0], targets: [1,0,0,0], name: 'H partial top' },
    { inputs: [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,1,1,1,1, 0,0,0,0,0], targets: [1,0,0,0], name: 'H partial bot' },
    // Short lines (3 pixels)
    { inputs: [0,0,0,0,0, 0,0,0,0,0, 0,1,1,1,0, 0,0,0,0,0, 0,0,0,0,0], targets: [1,0,0,0], name: 'H short ctr' },
    { inputs: [0,0,0,0,0, 0,1,1,1,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0], targets: [1,0,0,0], name: 'H short top' },
    { inputs: [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,1,1,1,0, 0,0,0,0,0], targets: [1,0,0,0], name: 'H short bot' },
    { inputs: [0,0,0,0,0, 0,0,0,0,0, 1,1,1,0,0, 0,0,0,0,0, 0,0,0,0,0], targets: [1,0,0,0], name: 'H short L' },
    { inputs: [0,0,0,0,0, 0,0,0,0,0, 0,0,1,1,1, 0,0,0,0,0, 0,0,0,0,0], targets: [1,0,0,0], name: 'H short R' },

    // ============ VERTICAL LINES ============
    // Full lines at every column
    { inputs: [1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0], targets: [0,1,0,0], name: 'V col 0' },
    { inputs: [0,1,0,0,0, 0,1,0,0,0, 0,1,0,0,0, 0,1,0,0,0, 0,1,0,0,0], targets: [0,1,0,0], name: 'V col 1' },
    { inputs: [0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0], targets: [0,1,0,0], name: 'V col 2' },
    { inputs: [0,0,0,1,0, 0,0,0,1,0, 0,0,0,1,0, 0,0,0,1,0, 0,0,0,1,0], targets: [0,1,0,0], name: 'V col 3' },
    { inputs: [0,0,0,0,1, 0,0,0,0,1, 0,0,0,0,1, 0,0,0,0,1, 0,0,0,0,1], targets: [0,1,0,0], name: 'V col 4' },
    // Partial lines (4 pixels)
    { inputs: [0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,0,0,0], targets: [0,1,0,0], name: 'V partial top' },
    { inputs: [0,0,0,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0], targets: [0,1,0,0], name: 'V partial bot' },
    { inputs: [0,1,0,0,0, 0,1,0,0,0, 0,1,0,0,0, 0,1,0,0,0, 0,0,0,0,0], targets: [0,1,0,0], name: 'V partial L' },
    { inputs: [0,0,0,0,0, 0,0,0,1,0, 0,0,0,1,0, 0,0,0,1,0, 0,0,0,1,0], targets: [0,1,0,0], name: 'V partial R' },
    // Short lines (3 pixels)
    { inputs: [0,0,0,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,0,0,0], targets: [0,1,0,0], name: 'V short ctr' },
    { inputs: [0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,0,0,0, 0,0,0,0,0], targets: [0,1,0,0], name: 'V short top' },
    { inputs: [0,0,0,0,0, 0,0,0,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0], targets: [0,1,0,0], name: 'V short bot' },
    { inputs: [0,0,0,0,0, 0,1,0,0,0, 0,1,0,0,0, 0,1,0,0,0, 0,0,0,0,0], targets: [0,1,0,0], name: 'V short L' },
    { inputs: [0,0,0,0,0, 0,0,0,1,0, 0,0,0,1,0, 0,0,0,1,0, 0,0,0,0,0], targets: [0,1,0,0], name: 'V short R' },

    // ============ DIAGONAL LINES ============
    // Main diagonals
    { inputs: [1,0,0,0,0, 0,1,0,0,0, 0,0,1,0,0, 0,0,0,1,0, 0,0,0,0,1], targets: [0,0,1,0], name: 'D \\ main' },
    { inputs: [0,0,0,0,1, 0,0,0,1,0, 0,0,1,0,0, 0,1,0,0,0, 1,0,0,0,0], targets: [0,0,1,0], name: 'D / main' },
    // Thick diagonals
    { inputs: [1,1,0,0,0, 1,1,1,0,0, 0,1,1,1,0, 0,0,1,1,1, 0,0,0,1,1], targets: [0,0,1,0], name: 'D \\ thick' },
    { inputs: [0,0,0,1,1, 0,0,1,1,1, 0,1,1,1,0, 1,1,1,0,0, 1,1,0,0,0], targets: [0,0,1,0], name: 'D / thick' },
    // Shifted diagonals
    { inputs: [0,1,0,0,0, 0,0,1,0,0, 0,0,0,1,0, 0,0,0,0,1, 0,0,0,0,0], targets: [0,0,1,0], name: 'D \\ shift R' },
    { inputs: [0,0,0,0,0, 1,0,0,0,0, 0,1,0,0,0, 0,0,1,0,0, 0,0,0,1,0], targets: [0,0,1,0], name: 'D \\ shift D' },
    { inputs: [0,0,0,1,0, 0,0,1,0,0, 0,1,0,0,0, 1,0,0,0,0, 0,0,0,0,0], targets: [0,0,1,0], name: 'D / shift L' },
    { inputs: [0,0,0,0,0, 0,0,0,0,1, 0,0,0,1,0, 0,0,1,0,0, 0,1,0,0,0], targets: [0,0,1,0], name: 'D / shift D' },
    // Partial diagonals (4 pixels)
    { inputs: [1,0,0,0,0, 0,1,0,0,0, 0,0,1,0,0, 0,0,0,1,0, 0,0,0,0,0], targets: [0,0,1,0], name: 'D \\ partial' },
    { inputs: [0,0,0,0,0, 0,0,0,1,0, 0,0,1,0,0, 0,1,0,0,0, 1,0,0,0,0], targets: [0,0,1,0], name: 'D / partial' },
    // Short diagonals (3 pixels)
    { inputs: [0,0,0,0,0, 0,1,0,0,0, 0,0,1,0,0, 0,0,0,1,0, 0,0,0,0,0], targets: [0,0,1,0], name: 'D \\ short' },
    { inputs: [0,0,0,0,0, 0,0,0,1,0, 0,0,1,0,0, 0,1,0,0,0, 0,0,0,0,0], targets: [0,0,1,0], name: 'D / short' },
    { inputs: [1,0,0,0,0, 0,1,0,0,0, 0,0,1,0,0, 0,0,0,0,0, 0,0,0,0,0], targets: [0,0,1,0], name: 'D \\ corner' },
    { inputs: [0,0,0,0,1, 0,0,0,1,0, 0,0,1,0,0, 0,0,0,0,0, 0,0,0,0,0], targets: [0,0,1,0], name: 'D / corner' },

    // ============ CROSS PATTERNS ============
    // X patterns
    { inputs: [1,0,0,0,1, 0,1,0,1,0, 0,0,1,0,0, 0,1,0,1,0, 1,0,0,0,1], targets: [0,0,0,1], name: 'X full' },
    { inputs: [1,0,0,0,1, 0,1,0,1,0, 0,0,0,0,0, 0,1,0,1,0, 1,0,0,0,1], targets: [0,0,0,1], name: 'X hollow' },
    { inputs: [0,0,0,0,0, 0,1,0,1,0, 0,0,1,0,0, 0,1,0,1,0, 0,0,0,0,0], targets: [0,0,0,1], name: 'X small' },
    // + patterns
    { inputs: [0,0,1,0,0, 0,0,1,0,0, 1,1,1,1,1, 0,0,1,0,0, 0,0,1,0,0], targets: [0,0,0,1], name: '+ full' },
    { inputs: [0,0,1,0,0, 0,0,1,0,0, 1,1,0,1,1, 0,0,1,0,0, 0,0,1,0,0], targets: [0,0,0,1], name: '+ hollow' },
    { inputs: [0,0,0,0,0, 0,0,1,0,0, 0,1,1,1,0, 0,0,1,0,0, 0,0,0,0,0], targets: [0,0,0,1], name: '+ small' },
    // Thick crosses
    { inputs: [0,0,1,0,0, 0,1,1,1,0, 1,1,1,1,1, 0,1,1,1,0, 0,0,1,0,0], targets: [0,0,0,1], name: '+ thick' },
    { inputs: [1,0,0,0,1, 1,1,0,1,1, 0,0,1,0,0, 1,1,0,1,1, 1,0,0,0,1], targets: [0,0,0,1], name: 'X thick' },
    // Offset crosses
    { inputs: [0,1,0,0,0, 0,1,0,0,0, 1,1,1,1,0, 0,1,0,0,0, 0,1,0,0,0], targets: [0,0,0,1], name: '+ offset L' },
    { inputs: [0,0,0,1,0, 0,0,0,1,0, 0,1,1,1,1, 0,0,0,1,0, 0,0,0,1,0], targets: [0,0,0,1], name: '+ offset R' },
    { inputs: [0,0,1,0,0, 1,1,1,1,1, 0,0,1,0,0, 0,0,1,0,0, 0,0,0,0,0], targets: [0,0,0,1], name: '+ offset T' },
    { inputs: [0,0,0,0,0, 0,0,1,0,0, 0,0,1,0,0, 1,1,1,1,1, 0,0,1,0,0], targets: [0,0,0,1], name: '+ offset B' }
];

// TEST DATASET - Unseen variations to test generalization
const TEST_DATASET = [
    // Horizontal tests
    { inputs: [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 1,1,1,1,0, 0,0,0,0,0], targets: [1,0,0,0], name: 'H test 1' },
    { inputs: [1,1,1,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0], targets: [1,0,0,0], name: 'H test 2' },
    { inputs: [0,0,0,0,0, 0,0,1,1,1, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0], targets: [1,0,0,0], name: 'H test 3' },

    // Vertical tests
    { inputs: [0,0,0,0,0, 0,0,0,0,1, 0,0,0,0,1, 0,0,0,0,1, 0,0,0,0,0], targets: [0,1,0,0], name: 'V test 1' },
    { inputs: [1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0], targets: [0,1,0,0], name: 'V test 2' },
    { inputs: [0,0,0,0,0, 0,0,0,0,0, 0,1,0,0,0, 0,1,0,0,0, 0,1,0,0,0], targets: [0,1,0,0], name: 'V test 3' },

    // Diagonal tests
    { inputs: [0,0,0,0,0, 0,0,0,0,0, 1,0,0,0,0, 0,1,0,0,0, 0,0,1,0,0], targets: [0,0,1,0], name: 'D test 1' },
    { inputs: [0,0,1,0,0, 0,0,0,1,0, 0,0,0,0,1, 0,0,0,0,0, 0,0,0,0,0], targets: [0,0,1,0], name: 'D test 2' },
    { inputs: [0,0,0,0,0, 0,0,0,0,0, 0,0,1,0,0, 0,1,0,0,0, 1,0,0,0,0], targets: [0,0,1,0], name: 'D test 3' },

    // Cross tests
    { inputs: [0,0,0,0,0, 0,1,0,1,0, 0,0,1,0,0, 0,1,0,1,0, 0,0,0,0,0], targets: [0,0,0,1], name: 'X test 1' },
    { inputs: [0,0,1,0,0, 0,1,1,1,0, 0,0,1,0,0, 0,0,0,0,0, 0,0,0,0,0], targets: [0,0,0,1], name: '+ test 1' },
    { inputs: [0,0,0,0,0, 0,0,1,0,0, 1,1,1,0,0, 0,0,1,0,0, 0,0,1,0,0], targets: [0,0,0,1], name: '+ test 2' }
];
