/**
 * Neural Network Visualization
 * Renders networks with any number of layers on canvas
 */

class NetworkVisualizer {
    constructor(canvasId, network) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.network = network;

        // Layout configuration
        this.padding = 50;
        this.maxNeuronRadius = 20;
        this.minNeuronRadius = 8;

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
            text: '#ffffff'
        };

        // Calculate positions
        this.calculatePositions();
    }

    /**
     * Update the network reference and recalculate
     */
    setNetwork(network) {
        this.network = network;
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
            Math.min(this.maxNeuronRadius, (height - 2 * this.padding) / (maxNeurons * 3))
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
     * Draw the entire network
     */
    draw() {
        const ctx = this.ctx;

        // Clear canvas
        ctx.fillStyle = this.colors.background;
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw connections
        this.drawConnections();

        // Draw neurons
        this.drawNeurons();
    }

    /**
     * Draw all connections between layers
     */
    drawConnections() {
        const weights = this.network.getWeights();
        const numLayers = this.layerPositions.length;

        for (let layer = 0; layer < numLayers - 1; layer++) {
            const fromPositions = this.layerPositions[layer];
            const toPositions = this.layerPositions[layer + 1];
            const layerWeights = weights[layer];

            // For large networks, don't draw all connections (too cluttered)
            const skipConnections = fromPositions.length > 10 || toPositions.length > 10;

            for (let i = 0; i < fromPositions.length; i++) {
                for (let j = 0; j < toPositions.length; j++) {
                    // Skip some connections for clarity in large networks
                    if (skipConnections && (i + j) % 3 !== 0) continue;

                    const weight = layerWeights[i][j];
                    this.drawConnection(fromPositions[i], toPositions[j], weight, skipConnections);
                }
            }
        }
    }

    /**
     * Draw a single connection
     */
    drawConnection(from, to, weight, simplified = false) {
        const ctx = this.ctx;

        const lineWidth = simplified
            ? Math.min(Math.abs(weight) * 2 + 0.5, 3)
            : Math.min(Math.abs(weight) * 3 + 1, 5);

        const color = weight >= 0 ? this.colors.positiveWeight : this.colors.negativeWeight;
        const alpha = Math.min(Math.abs(weight) * 0.4 + 0.1, 0.6);

        ctx.beginPath();
        ctx.moveTo(from.x + this.neuronRadius, from.y);
        ctx.lineTo(to.x - this.neuronRadius, to.y);
        ctx.strokeStyle = this.hexToRgba(color, alpha);
        ctx.lineWidth = lineWidth;
        ctx.stroke();
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
                const label = simplified ? '' : this.getNeuronLabel(layer, i, numLayers);
                this.drawNeuron(positions[i], activation, color, label, simplified);
            }

            // Draw layer count if simplified
            if (simplified) {
                const centerY = this.canvas.height / 2;
                const ctx = this.ctx;
                ctx.fillStyle = color;
                ctx.font = 'bold 14px monospace';
                ctx.textAlign = 'center';
                ctx.fillText(positions.length + ' neurons', positions[0].x, centerY + this.canvas.height / 2 - 30);
            }
        }
    }

    /**
     * Get label for a neuron
     */
    getNeuronLabel(layer, index, numLayers) {
        if (layer === 0) return 'I' + (index + 1);
        if (layer === numLayers - 1) return 'O' + (index + 1);
        return 'H' + layer + '.' + (index + 1);
    }

    /**
     * Draw a single neuron
     */
    drawNeuron(pos, activation, color, label, simplified = false) {
        const ctx = this.ctx;
        const radius = simplified ? this.neuronRadius * 0.7 : this.neuronRadius;

        // Glow effect based on activation
        if (activation > 0.1 && !simplified) {
            const gradient = ctx.createRadialGradient(
                pos.x, pos.y, radius,
                pos.x, pos.y, radius + 15
            );
            gradient.addColorStop(0, this.hexToRgba(color, activation * 0.4));
            gradient.addColorStop(1, 'transparent');

            ctx.beginPath();
            ctx.arc(pos.x, pos.y, radius + 15, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();
        }

        // Neuron body
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);

        // Fill based on activation
        const fillAlpha = 0.3 + activation * 0.5;
        ctx.fillStyle = this.hexToRgba(color, fillAlpha);
        ctx.fill();

        // Border
        ctx.strokeStyle = color;
        ctx.lineWidth = simplified ? 1 : 2;
        ctx.stroke();

        // Label and value (only for non-simplified)
        if (!simplified && label) {
            ctx.fillStyle = this.colors.text;
            ctx.font = 'bold 10px monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            if (radius > 15) {
                ctx.fillText(label, pos.x, pos.y - 5);
                ctx.font = '9px monospace';
                ctx.fillText(activation.toFixed(2), pos.x, pos.y + 6);
            } else {
                ctx.font = '8px monospace';
                ctx.fillText(activation.toFixed(1), pos.x, pos.y);
            }
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
