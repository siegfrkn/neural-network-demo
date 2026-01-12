/**
 * Neural Network Visualization
 * Enhanced to show weight changes and pathway strengthening
 */

class NetworkVisualizer {
    constructor(canvasId, network) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.network = network;

        // Layout configuration
        this.padding = 40;
        this.maxNeuronRadius = 18;
        this.minNeuronRadius = 6;

        // Colors
        this.colors = {
            background: '#16213e',
            neuronFill: '#16213e',
            neuronStroke: '#0f3460',
            inputNeuron: '#00d9ff',
            hiddenNeuron: '#9d4edd',
            outputNeuron: '#00ff88',
            positiveWeight: '#00ff88',
            negativeWeight: '#ff6b6b',
            weightChange: '#ffd93d',
            text: '#ffffff'
        };

        // Track previous weights for change detection
        this.previousWeights = null;
        this.weightChanges = null;
        this.changeDecay = 0.85; // How quickly the change highlight fades

        // Calculate positions
        this.calculatePositions();
    }

    /**
     * Update the network reference and recalculate
     */
    setNetwork(network) {
        this.network = network;
        this.previousWeights = null;
        this.weightChanges = null;
        this.calculatePositions();
    }

    /**
     * Calculate positions for all neurons in all layers
     */
    calculatePositions() {
        const width = this.canvas.width;
        const height = this.canvas.height;
        const layerSizes = this.network.getLayerSizes();
        const numLayers = layerSizes.length;

        // Calculate neuron radius based on largest layer
        const maxNeurons = Math.max(...layerSizes);
        this.neuronRadius = Math.max(
            this.minNeuronRadius,
            Math.min(this.maxNeuronRadius, (height - 2 * this.padding) / (maxNeurons * 2.5))
        );

        // Calculate layer x positions
        const layerSpacing = (width - 2 * this.padding) / (numLayers - 1);

        this.layerPositions = [];

        for (let layer = 0; layer < numLayers; layer++) {
            const layerX = this.padding + layer * layerSpacing;
            const numNeurons = layerSizes[layer];
            const neuronSpacing = (height - 2 * this.padding) / (numNeurons + 1);

            const positions = [];
            for (let i = 0; i < numNeurons; i++) {
                positions.push({
                    x: layerX,
                    y: this.padding + neuronSpacing * (i + 1)
                });
            }
            this.layerPositions.push(positions);
        }
    }

    /**
     * Track weight changes for visualization
     */
    updateWeightTracking() {
        const currentWeights = this.network.getWeights();

        if (!this.previousWeights) {
            // First time - initialize tracking
            this.previousWeights = this.deepCopyWeights(currentWeights);
            this.weightChanges = this.createWeightChangeMatrix(currentWeights);
            return;
        }

        // Calculate changes and decay old changes
        for (let layer = 0; layer < currentWeights.length; layer++) {
            for (let i = 0; i < currentWeights[layer].length; i++) {
                for (let j = 0; j < currentWeights[layer][i].length; j++) {
                    const change = Math.abs(currentWeights[layer][i][j] - this.previousWeights[layer][i][j]);

                    // Accumulate change with decay
                    this.weightChanges[layer][i][j] =
                        this.weightChanges[layer][i][j] * this.changeDecay + change * 3;

                    // Cap the change intensity
                    this.weightChanges[layer][i][j] = Math.min(1, this.weightChanges[layer][i][j]);
                }
            }
        }

        this.previousWeights = this.deepCopyWeights(currentWeights);
    }

    /**
     * Deep copy weights array
     */
    deepCopyWeights(weights) {
        return weights.map(layer =>
            layer.map(row => [...row])
        );
    }

    /**
     * Create empty weight change matrix
     */
    createWeightChangeMatrix(weights) {
        return weights.map(layer =>
            layer.map(row => new Array(row.length).fill(0))
        );
    }

    /**
     * Draw the entire network
     */
    draw() {
        const ctx = this.ctx;

        // Update weight change tracking
        this.updateWeightTracking();

        // Clear canvas
        ctx.fillStyle = this.colors.background;
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw connections (back to front for proper layering)
        this.drawConnections();

        // Draw neurons
        this.drawNeurons();
    }

    /**
     * Draw all connections between layers with weight change visualization
     */
    drawConnections() {
        const weights = this.network.getWeights();
        const activations = this.network.getActivations();
        const numLayers = this.layerPositions.length;

        for (let layer = 0; layer < numLayers - 1; layer++) {
            const fromPositions = this.layerPositions[layer];
            const toPositions = this.layerPositions[layer + 1];
            const layerWeights = weights[layer];
            const fromActivations = activations[layer] || [];

            // For large networks, sample connections
            const skipFactor = (fromPositions.length > 12 || toPositions.length > 12) ? 2 : 1;

            for (let i = 0; i < fromPositions.length; i += skipFactor) {
                for (let j = 0; j < toPositions.length; j += skipFactor) {
                    const weight = layerWeights[i][j];
                    const activation = fromActivations[i] || 0;
                    const change = this.weightChanges ? this.weightChanges[layer][i][j] : 0;

                    this.drawConnection(
                        fromPositions[i],
                        toPositions[j],
                        weight,
                        activation,
                        change
                    );
                }
            }
        }
    }

    /**
     * Draw a single connection with weight strength and change visualization
     */
    drawConnection(from, to, weight, activation, change) {
        const ctx = this.ctx;
        const absWeight = Math.abs(weight);

        // Line thickness based on weight strength
        const baseWidth = Math.min(absWeight * 2 + 0.3, 4);
        const lineWidth = baseWidth + (change * 2); // Thicker when recently changed

        // Color based on weight sign
        const baseColor = weight >= 0 ? this.colors.positiveWeight : this.colors.negativeWeight;

        // Alpha based on weight strength and activation
        const strengthAlpha = Math.min(absWeight * 0.5 + 0.1, 0.8);
        const activationBoost = activation * 0.3;
        const alpha = Math.min(strengthAlpha + activationBoost, 0.9);

        // Draw the main connection
        ctx.beginPath();
        ctx.moveTo(from.x + this.neuronRadius, from.y);
        ctx.lineTo(to.x - this.neuronRadius, to.y);
        ctx.strokeStyle = this.hexToRgba(baseColor, alpha);
        ctx.lineWidth = lineWidth;
        ctx.stroke();

        // Draw change highlight (yellow glow for recently changed weights)
        if (change > 0.05) {
            ctx.beginPath();
            ctx.moveTo(from.x + this.neuronRadius, from.y);
            ctx.lineTo(to.x - this.neuronRadius, to.y);
            ctx.strokeStyle = this.hexToRgba(this.colors.weightChange, change * 0.6);
            ctx.lineWidth = lineWidth + 2;
            ctx.stroke();
        }

        // Draw activation flow (bright pulse along strong active connections)
        if (activation > 0.5 && absWeight > 0.3) {
            const gradient = ctx.createLinearGradient(from.x, from.y, to.x, to.y);
            gradient.addColorStop(0, this.hexToRgba(baseColor, activation * 0.8));
            gradient.addColorStop(0.5, this.hexToRgba('#ffffff', activation * 0.3));
            gradient.addColorStop(1, this.hexToRgba(baseColor, activation * 0.8));

            ctx.beginPath();
            ctx.moveTo(from.x + this.neuronRadius, from.y);
            ctx.lineTo(to.x - this.neuronRadius, to.y);
            ctx.strokeStyle = gradient;
            ctx.lineWidth = Math.max(1, lineWidth * 0.5);
            ctx.stroke();
        }
    }

    /**
     * Draw all neurons
     */
    drawNeurons() {
        const activations = this.network.getActivations();
        const layerSizes = this.network.getLayerSizes();
        const numLayers = layerSizes.length;

        for (let layer = 0; layer < numLayers; layer++) {
            const positions = this.layerPositions[layer];
            const layerActivations = activations[layer] || [];

            // Determine layer color
            let color;
            if (layer === 0) {
                color = this.colors.inputNeuron;
            } else if (layer === numLayers - 1) {
                color = this.colors.outputNeuron;
            } else {
                color = this.colors.hiddenNeuron;
            }

            // For large layers, show simplified view
            const simplified = positions.length > 10;

            for (let i = 0; i < positions.length; i++) {
                const activation = layerActivations[i] || 0;
                this.drawNeuron(positions[i], activation, color, simplified);
            }

            // Draw layer count if simplified
            if (simplified) {
                const ctx = this.ctx;
                ctx.fillStyle = color;
                ctx.font = 'bold 11px monospace';
                ctx.textAlign = 'center';
                ctx.fillText(
                    positions.length + ' neurons',
                    positions[0].x,
                    this.canvas.height - 15
                );
            }
        }
    }

    /**
     * Draw a single neuron with activation glow
     */
    drawNeuron(pos, activation, color, simplified = false) {
        const ctx = this.ctx;
        const radius = simplified ? this.neuronRadius * 0.6 : this.neuronRadius;

        // Outer glow for high activation
        if (activation > 0.3) {
            const glowRadius = radius + 8 + activation * 10;
            const gradient = ctx.createRadialGradient(
                pos.x, pos.y, radius,
                pos.x, pos.y, glowRadius
            );
            gradient.addColorStop(0, this.hexToRgba(color, activation * 0.5));
            gradient.addColorStop(1, 'transparent');

            ctx.beginPath();
            ctx.arc(pos.x, pos.y, glowRadius, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();
        }

        // Neuron body
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);

        // Fill based on activation
        const fillAlpha = 0.2 + activation * 0.7;
        ctx.fillStyle = this.hexToRgba(color, fillAlpha);
        ctx.fill();

        // Border
        ctx.strokeStyle = color;
        ctx.lineWidth = simplified ? 1 : 2;
        ctx.stroke();

        // Show activation value for non-simplified neurons
        if (!simplified && radius > 10) {
            ctx.fillStyle = this.colors.text;
            ctx.font = 'bold 9px monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(activation.toFixed(2), pos.x, pos.y);
        }
    }

    /**
     * Convert hex to rgba
     */
    hexToRgba(hex, alpha) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return 'rgba(' + r + ', ' + g + ', ' + b + ', ' + alpha + ')';
    }

    /**
     * Redraw
     */
    update() {
        this.draw();
    }
}
