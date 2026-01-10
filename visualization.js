/**
 * Neural Network Visualization
 * Renders the network on a canvas with animations
 */

class NetworkVisualizer {
    constructor(canvasId, network) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.network = network;

        // Layout configuration
        this.padding = 60;
        this.neuronRadius = 25;

        // Colors
        this.colors = {
            background: '#1a1a2e',
            neuronFill: '#16213e',
            neuronStroke: '#0f3460',
            inputNeuron: '#00d9ff',
            hiddenNeuron: '#9d4edd',
            outputNeuron: '#00ff88',
            positiveWeight: '#00ff88',
            negativeWeight: '#ff6b6b',
            text: '#ffffff',
            activeGlow: '#00d9ff'
        };

        // Animation state
        this.animationPhase = 'idle'; // 'idle', 'forward', 'backward'
        this.animationProgress = 0;
        this.signalPositions = [];

        // Calculate neuron positions
        this.calculatePositions();
    }

    /**
     * Calculate positions for all neurons
     */
    calculatePositions() {
        const width = this.canvas.width;
        const height = this.canvas.height;

        // Layer x positions
        const layerX = [
            this.padding + 50,
            width / 2,
            width - this.padding - 50
        ];

        // Input layer (2 neurons)
        this.inputPositions = [];
        const inputSpacing = (height - 2 * this.padding) / 3;
        for (let i = 0; i < this.network.inputSize; i++) {
            this.inputPositions.push({
                x: layerX[0],
                y: this.padding + inputSpacing + i * inputSpacing
            });
        }

        // Hidden layer (3 neurons)
        this.hiddenPositions = [];
        const hiddenSpacing = (height - 2 * this.padding) / (this.network.hiddenSize + 1);
        for (let i = 0; i < this.network.hiddenSize; i++) {
            this.hiddenPositions.push({
                x: layerX[1],
                y: this.padding + hiddenSpacing + i * hiddenSpacing
            });
        }

        // Output layer (1 neuron)
        this.outputPositions = [];
        const outputSpacing = (height - 2 * this.padding) / 2;
        for (let i = 0; i < this.network.outputSize; i++) {
            this.outputPositions.push({
                x: layerX[2],
                y: height / 2
            });
        }
    }

    /**
     * Draw the entire network
     */
    draw() {
        // Clear canvas
        this.ctx.fillStyle = this.colors.background;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw connections first (behind neurons)
        this.drawConnections();

        // Draw neurons
        this.drawNeurons();

        // Draw signals if animating
        if (this.animationPhase !== 'idle') {
            this.drawSignals();
        }
    }

    /**
     * Draw all connections between neurons
     */
    drawConnections() {
        const weights = this.network.getWeights();

        // Input to Hidden connections
        for (let i = 0; i < this.network.inputSize; i++) {
            for (let j = 0; j < this.network.hiddenSize; j++) {
                const weight = weights.inputHidden[i][j];
                this.drawConnection(
                    this.inputPositions[i],
                    this.hiddenPositions[j],
                    weight,
                    this.animationPhase === 'forward' && this.animationProgress > i * 0.2
                );
            }
        }

        // Hidden to Output connections
        for (let j = 0; j < this.network.hiddenSize; j++) {
            for (let k = 0; k < this.network.outputSize; k++) {
                const weight = weights.hiddenOutput[j][k];
                this.drawConnection(
                    this.hiddenPositions[j],
                    this.outputPositions[k],
                    weight,
                    this.animationPhase === 'forward' && this.animationProgress > 0.5 + j * 0.1
                );
            }
        }
    }

    /**
     * Draw a single connection with weight visualization
     */
    drawConnection(from, to, weight, active = false) {
        const ctx = this.ctx;

        // Calculate line width based on weight magnitude
        const lineWidth = Math.min(Math.abs(weight) * 4 + 1, 6);

        // Color based on weight sign
        const color = weight >= 0 ? this.colors.positiveWeight : this.colors.negativeWeight;
        const alpha = Math.min(Math.abs(weight) * 0.5 + 0.2, 0.8);

        ctx.beginPath();
        ctx.moveTo(from.x + this.neuronRadius, from.y);
        ctx.lineTo(to.x - this.neuronRadius, to.y);

        ctx.strokeStyle = active
            ? color
            : this.hexToRgba(color, alpha);
        ctx.lineWidth = lineWidth;
        ctx.stroke();

        // Draw weight value at midpoint
        const midX = (from.x + to.x) / 2;
        const midY = (from.y + to.y) / 2;

        // Small background for text
        ctx.fillStyle = this.colors.background;
        ctx.fillRect(midX - 20, midY - 8, 40, 16);

        ctx.fillStyle = this.colors.text;
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(weight.toFixed(2), midX, midY);
    }

    /**
     * Draw all neurons
     */
    drawNeurons() {
        const activations = this.network.getActivations();

        // Draw input neurons
        this.inputPositions.forEach((pos, i) => {
            const activation = activations.input[i] || 0;
            this.drawNeuron(pos, activation, this.colors.inputNeuron, `I${i + 1}`);
        });

        // Draw hidden neurons
        this.hiddenPositions.forEach((pos, i) => {
            const activation = activations.hidden[i] || 0;
            this.drawNeuron(pos, activation, this.colors.hiddenNeuron, `H${i + 1}`);
        });

        // Draw output neurons
        this.outputPositions.forEach((pos, i) => {
            const activation = activations.output[i] || 0;
            this.drawNeuron(pos, activation, this.colors.outputNeuron, `O${i + 1}`);
        });
    }

    /**
     * Draw a single neuron
     */
    drawNeuron(pos, activation, color, label) {
        const ctx = this.ctx;

        // Glow effect based on activation
        if (activation > 0.1) {
            const gradient = ctx.createRadialGradient(
                pos.x, pos.y, this.neuronRadius,
                pos.x, pos.y, this.neuronRadius + 20
            );
            gradient.addColorStop(0, this.hexToRgba(color, activation * 0.5));
            gradient.addColorStop(1, 'transparent');

            ctx.beginPath();
            ctx.arc(pos.x, pos.y, this.neuronRadius + 20, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();
        }

        // Neuron body
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, this.neuronRadius, 0, Math.PI * 2);

        // Fill with gradient based on activation
        const fillGradient = ctx.createRadialGradient(
            pos.x - 10, pos.y - 10, 0,
            pos.x, pos.y, this.neuronRadius
        );
        fillGradient.addColorStop(0, this.hexToRgba(color, 0.3 + activation * 0.5));
        fillGradient.addColorStop(1, this.colors.neuronFill);

        ctx.fillStyle = fillGradient;
        ctx.fill();

        // Border
        ctx.strokeStyle = color;
        ctx.lineWidth = 2 + activation * 2;
        ctx.stroke();

        // Label
        ctx.fillStyle = this.colors.text;
        ctx.font = 'bold 12px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, pos.x, pos.y - 8);

        // Activation value
        ctx.font = '11px monospace';
        ctx.fillText(activation.toFixed(2), pos.x, pos.y + 8);
    }

    /**
     * Draw animated signals
     */
    drawSignals() {
        const ctx = this.ctx;

        this.signalPositions.forEach(signal => {
            ctx.beginPath();
            ctx.arc(signal.x, signal.y, 6, 0, Math.PI * 2);

            const gradient = ctx.createRadialGradient(
                signal.x, signal.y, 0,
                signal.x, signal.y, 8
            );
            gradient.addColorStop(0, '#ffffff');
            gradient.addColorStop(0.5, signal.color);
            gradient.addColorStop(1, 'transparent');

            ctx.fillStyle = gradient;
            ctx.fill();
        });
    }

    /**
     * Animate forward propagation
     */
    animateForward(callback) {
        this.animationPhase = 'forward';
        this.animationProgress = 0;
        this.signalPositions = [];

        const duration = 1500;
        const startTime = performance.now();

        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            this.animationProgress = Math.min(elapsed / duration, 1);

            // Update signal positions
            this.updateForwardSignals();

            this.draw();

            if (this.animationProgress < 1) {
                requestAnimationFrame(animate);
            } else {
                this.animationPhase = 'idle';
                this.signalPositions = [];
                this.draw();
                if (callback) callback();
            }
        };

        requestAnimationFrame(animate);
    }

    /**
     * Update signal positions during forward animation
     */
    updateForwardSignals() {
        this.signalPositions = [];

        // Input to hidden signals
        if (this.animationProgress < 0.5) {
            const progress = this.animationProgress * 2;
            for (let i = 0; i < this.network.inputSize; i++) {
                for (let j = 0; j < this.network.hiddenSize; j++) {
                    const from = this.inputPositions[i];
                    const to = this.hiddenPositions[j];
                    this.signalPositions.push({
                        x: from.x + (to.x - from.x) * progress,
                        y: from.y + (to.y - from.y) * progress,
                        color: this.colors.inputNeuron
                    });
                }
            }
        }
        // Hidden to output signals
        else {
            const progress = (this.animationProgress - 0.5) * 2;
            for (let j = 0; j < this.network.hiddenSize; j++) {
                for (let k = 0; k < this.network.outputSize; k++) {
                    const from = this.hiddenPositions[j];
                    const to = this.outputPositions[k];
                    this.signalPositions.push({
                        x: from.x + (to.x - from.x) * progress,
                        y: from.y + (to.y - from.y) * progress,
                        color: this.colors.hiddenNeuron
                    });
                }
            }
        }
    }

    /**
     * Animate backpropagation
     */
    animateBackward(callback) {
        this.animationPhase = 'backward';
        this.animationProgress = 0;
        this.signalPositions = [];

        const duration = 1500;
        const startTime = performance.now();

        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            this.animationProgress = Math.min(elapsed / duration, 1);

            // Update signal positions (reverse direction)
            this.updateBackwardSignals();

            this.draw();

            if (this.animationProgress < 1) {
                requestAnimationFrame(animate);
            } else {
                this.animationPhase = 'idle';
                this.signalPositions = [];
                this.draw();
                if (callback) callback();
            }
        };

        requestAnimationFrame(animate);
    }

    /**
     * Update signal positions during backward animation
     */
    updateBackwardSignals() {
        this.signalPositions = [];

        // Output to hidden signals
        if (this.animationProgress < 0.5) {
            const progress = this.animationProgress * 2;
            for (let j = 0; j < this.network.hiddenSize; j++) {
                for (let k = 0; k < this.network.outputSize; k++) {
                    const from = this.outputPositions[k];
                    const to = this.hiddenPositions[j];
                    this.signalPositions.push({
                        x: from.x + (to.x - from.x) * progress,
                        y: from.y + (to.y - from.y) * progress,
                        color: this.colors.negativeWeight
                    });
                }
            }
        }
        // Hidden to input signals
        else {
            const progress = (this.animationProgress - 0.5) * 2;
            for (let i = 0; i < this.network.inputSize; i++) {
                for (let j = 0; j < this.network.hiddenSize; j++) {
                    const from = this.hiddenPositions[j];
                    const to = this.inputPositions[i];
                    this.signalPositions.push({
                        x: from.x + (to.x - from.x) * progress,
                        y: from.y + (to.y - from.y) * progress,
                        color: this.colors.negativeWeight
                    });
                }
            }
        }
    }

    /**
     * Convert hex color to rgba
     */
    hexToRgba(hex, alpha) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    /**
     * Redraw immediately without animation
     */
    update() {
        this.draw();
    }
}
