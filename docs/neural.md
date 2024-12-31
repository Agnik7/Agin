# **Neural Network**

The `neural_network` module contains implementations of neural network and deep learning models. Currently, the package supports:
- **Neural Network**

## **NeuralNetwork**
The `NeuralNetwork` class implements a basic multi-layer perceptron (MLP) model for multi-class classification using gradient descent. It utilizes the tanh activation function in the hidden layers and the softmax function in the output layer. The model is trained using forward propagation, backward propagation (backpropagation), and gradient descent. It can be imported directly from `agin` or from `agin.neural_network`.

### **Usage**
The `NeuralNetwork` class can be imported directly from the `agin` package or from the `agin.neural_network` module:

```python
from agin import NeuralNetwork
# or
from agin.neural_network import NeuralNetwork
```

#### **Example**

```python
import numpy as np
from agin import NeuralNetwork

# Training data
x_train = np.random.randn(3, 1000)
y_train = np.eye(3)[np.random.choice(3, 1000)].T

x_test = np.random.randn(3, 200)
y_test = np.eye(3)[np.random.choice(3, 200)].T

# Neural network parameters
layer_sizes = [3, 5, 3]
learning_rate = 0.1
iterations = 2000

# Initialize the model
model = NeuralNetwork(layer_sizes, learning_rate=learning_rate, iterations=iterations)
model.fit(x_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(x_test)
accuracy, precision, recall, f1_score = model.metrics(y_pred, y_test)

print(f"\nAccuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1 Score: {f1_score:.2f}%")
```

### **Methods**
#### **`fit(x_train, y_train)`**
   - Trains the neural network by adjusting the weights using forward propagation, backward propagation, and gradient descent.
   - **Args**:
     - `x_train` (numpy.ndarray): Training feature data.
     - `y_train` (numpy.ndarray): One-hot encoded target labels.
   - **Returns**: None. The model's weights are updated during training.

#### **`predict(x)`**
   - Predicts class labels for the given feature data.
   - **Args**:
     - `x` (numpy.ndarray): Feature data.
   - **Returns**: numpy.ndarray of predicted class labels.

#### **`metrics(y_pred, y_test)`**
   - Computes evaluation metrics for classification.
   - **Args**:
     - `y_pred` (numpy.ndarray): Predicted labels.
     - `y_test` (numpy.ndarray): One-hot encoded true labels.
   - **Returns**: Tuple containing accuracy, precision, recall, and F1-score.

### **Parameters**
- `layer_sizes` (list of int): List specifying the number of neurons in each layer, including input and output layers. Example: `[3, 5, 3]` for a 3-input, 5-hidden, and 3-output layer network.
- `learning_rate` (float): The learning rate for gradient descent. Default is `0.01`.
- `iterations` (int): The number of training iterations. Default is `1000`.
- `init_method` (str): Initialization method for weights. Can be "he" or "xavier". Default is `"he"`.

### **Attributes**
- `layer_sizes` (list of int): The structure of the network, representing the number of neurons in each layer.
- `learning_rate` (float): The learning rate for gradient descent.
- `iterations` (int): The total number of training iterations.
- `layers` (list of `Neuron`): The layers of the neural network, each represented as a `Neuron` object.
- `costs` (list of float): The history of the cost function during training.
- `mean` (numpy.ndarray): The mean of the training data (used for normalization).
- `std` (numpy.ndarray): The standard deviation of the training data (used for normalization).

### **Notes**
The `NeuralNetwork` class supports two weight initialization methods (`he` and `xavier`) to ensure stable training for deeper networks. Hidden layers use the `tanh` activation function, while the output layer applies the `softmax` function for multi-class classification.

