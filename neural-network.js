/**
 * Neural Network Implementation
 * A simple feedforward neural network with one hidden layer
 * demonstrating forward propagation and backpropagation
 */

class NeuralNetwork {
    constructor(inputSize = 2, hiddenSize = 3, outputSize = 1) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.learningRate = 0.5;

        // Initialize weights with Xavier initialization
        this.weightsInputHidden = this.initializeWeights(inputSize, hiddenSize);
        this.weightsHiddenOutput = this.initializeWeights(hiddenSize, outputSize);

        // Initialize biases
        this.biasHidden = new Array(hiddenSize).fill(0).map(() => Math.random() * 0.5 - 0.25);
        this.biasOutput = new Array(outputSize).fill(0).map(() => Math.random() * 0.5 - 0.25);

        // Store activations for visualization
        this.inputActivations = [];
        this.hiddenActivations = [];
        this.outputActivations = [];

        // Store gradients for visualization
        this.outputGradients = [];
        this.hiddenGradients = [];

        // Training statistics
        this.epoch = 0;
        this.lastError = 0;
    }

    /**
     * Initialize weights using Xavier initialization
     * This helps with gradient flow during training
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
     * Maps any value to range (0, 1)
     */
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * Derivative of sigmoid function
     * Used in backpropagation to calculate gradients
     */
    sigmoidDerivative(x) {
        return x * (1 - x);
    }

    /**
     * Forward propagation
     * Pass input through the network to get output
     */
    forward(inputs) {
        this.inputActivations = [...inputs];

        // Calculate hidden layer activations
        this.hiddenActivations = [];
        for (let j = 0; j < this.hiddenSize; j++) {
            let sum = this.biasHidden[j];
            for (let i = 0; i < this.inputSize; i++) {
                sum += inputs[i] * this.weightsInputHidden[i][j];
            }
            this.hiddenActivations[j] = this.sigmoid(sum);
        }

        // Calculate output layer activations
        this.outputActivations = [];
        for (let k = 0; k < this.outputSize; k++) {
            let sum = this.biasOutput[k];
            for (let j = 0; j < this.hiddenSize; j++) {
                sum += this.hiddenActivations[j] * this.weightsHiddenOutput[j][k];
            }
            this.outputActivations[k] = this.sigmoid(sum);
        }

        return this.outputActivations;
    }

    /**
     * Backpropagation
     * Calculate gradients and update weights based on error
     */
    backward(targets) {
        // Calculate output layer error and gradients
        this.outputGradients = [];
        for (let k = 0; k < this.outputSize; k++) {
            const error = targets[k] - this.outputActivations[k];
            this.outputGradients[k] = error * this.sigmoidDerivative(this.outputActivations[k]);
        }

        // Calculate hidden layer gradients
        this.hiddenGradients = [];
        for (let j = 0; j < this.hiddenSize; j++) {
            let error = 0;
            for (let k = 0; k < this.outputSize; k++) {
                error += this.outputGradients[k] * this.weightsHiddenOutput[j][k];
            }
            this.hiddenGradients[j] = error * this.sigmoidDerivative(this.hiddenActivations[j]);
        }

        // Update weights: hidden -> output
        for (let j = 0; j < this.hiddenSize; j++) {
            for (let k = 0; k < this.outputSize; k++) {
                this.weightsHiddenOutput[j][k] += this.learningRate * this.outputGradients[k] * this.hiddenActivations[j];
            }
        }

        // Update biases: output
        for (let k = 0; k < this.outputSize; k++) {
            this.biasOutput[k] += this.learningRate * this.outputGradients[k];
        }

        // Update weights: input -> hidden
        for (let i = 0; i < this.inputSize; i++) {
            for (let j = 0; j < this.hiddenSize; j++) {
                this.weightsInputHidden[i][j] += this.learningRate * this.hiddenGradients[j] * this.inputActivations[i];
            }
        }

        // Update biases: hidden
        for (let j = 0; j < this.hiddenSize; j++) {
            this.biasHidden[j] += this.learningRate * this.hiddenGradients[j];
        }
    }

    /**
     * Train the network on a single example
     */
    train(inputs, targets) {
        this.forward(inputs);
        this.backward(targets);

        // Calculate mean squared error
        let totalError = 0;
        for (let k = 0; k < this.outputSize; k++) {
            totalError += Math.pow(targets[k] - this.outputActivations[k], 2);
        }
        this.lastError = totalError / this.outputSize;
        this.epoch++;

        return this.lastError;
    }

    /**
     * Train on a dataset for multiple epochs
     */
    trainDataset(dataset, epochs = 100) {
        let totalError = 0;

        for (let e = 0; e < epochs; e++) {
            totalError = 0;
            for (const example of dataset) {
                totalError += this.train(example.inputs, example.targets);
            }
            totalError /= dataset.length;
        }

        return totalError;
    }

    /**
     * Get prediction for given inputs
     */
    predict(inputs) {
        return this.forward(inputs);
    }

    /**
     * Reset the network with new random weights
     */
    reset() {
        this.weightsInputHidden = this.initializeWeights(this.inputSize, this.hiddenSize);
        this.weightsHiddenOutput = this.initializeWeights(this.hiddenSize, this.outputSize);
        this.biasHidden = new Array(this.hiddenSize).fill(0).map(() => Math.random() * 0.5 - 0.25);
        this.biasOutput = new Array(this.outputSize).fill(0).map(() => Math.random() * 0.5 - 0.25);
        this.epoch = 0;
        this.lastError = 0;
        this.inputActivations = [];
        this.hiddenActivations = [];
        this.outputActivations = [];
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
        return {
            inputHidden: this.weightsInputHidden,
            hiddenOutput: this.weightsHiddenOutput,
            biasHidden: this.biasHidden,
            biasOutput: this.biasOutput
        };
    }

    /**
     * Get all activations for visualization
     */
    getActivations() {
        return {
            input: this.inputActivations,
            hidden: this.hiddenActivations,
            output: this.outputActivations
        };
    }

    /**
     * Get gradients for visualization
     */
    getGradients() {
        return {
            hidden: this.hiddenGradients,
            output: this.outputGradients
        };
    }
}

// Preset datasets for demonstration
const DATASETS = {
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
