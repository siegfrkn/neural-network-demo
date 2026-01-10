# Interactive Neural Network Visualization

An interactive web-based demonstration of how neural networks learn through forward propagation and backpropagation.

## Features

- **Visual Network Display**: See the neural network architecture with input, hidden, and output layers
- **Real-time Weight Visualization**: Watch weights update as the network learns (green = positive, red = negative)
- **Animated Training**: Step through training with animations showing signal flow
- **Interactive Controls**: Adjust inputs and learning rate in real-time
- **Preset Problems**: Try classic problems like XOR, AND, and OR gates

## How to Use

1. Open `index.html` in a web browser
2. Use the input sliders to set values for Input 1 and Input 2
3. Set the Expected Output to define what the network should learn
4. Click **Step Forward** to see animated forward and backward propagation
5. Click **Train (100 epochs)** for faster training without animation
6. Click **Auto Train** to watch continuous training
7. Try the preset problems (XOR, AND, OR) to see classic neural network challenges

## Understanding the Visualization

- **Neurons**: Circles show activation values (brighter = higher activation)
- **Connections**: Lines show weights (thicker = stronger, green = positive, red = negative)
- **Forward Propagation**: Data flows left to right through the network
- **Backpropagation**: Error signals flow right to left, adjusting weights

## Network Architecture

```
Input Layer (2 neurons) → Hidden Layer (3 neurons) → Output Layer (1 neuron)
```

The network uses:
- **Sigmoid activation function**: Maps values to (0, 1) range
- **Xavier initialization**: Better starting weights for training
- **Gradient descent**: Adjusts weights to minimize error

## Files

- `index.html` - Main HTML structure
- `styles.css` - Visual styling
- `neural-network.js` - Neural network implementation
- `visualization.js` - Canvas-based visualization
- `app.js` - Application logic and event handling

## Try These Experiments

1. **XOR Problem**: The classic non-linearly separable problem. Watch how the hidden layer learns to represent intermediate features.

2. **Learning Rate**: Try different learning rates to see how it affects training speed and stability.

3. **Weight Initialization**: Reset the network multiple times to see how random initialization affects training.

## License

MIT License - Feel free to use this for learning and teaching!
