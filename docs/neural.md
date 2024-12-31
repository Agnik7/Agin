# **Neural Network**

The `neural_network` module contains implementations of neural network and deep learning models. Currently, the package supports:
- **Neural Network**

## **Neural Network**
The `NeuralNetwork` class implements a basic multi-layer perceptron (MLP) model for multi-class classification using gradient descent. It utilizes the tanh activation function in the hidden layer and the softmax function in the output layer. The model is trained using forward propagation, backward propagation (backpropagation), and gradient descent. It can be imported directly from `agin` or from `agin.neural_network`.

### **Usage**
The `NeuralNetwork` class can be imported directly from the `agin` package or from the `agin.neural_network` module:

```python
from agin import NeuralNetwork
# or
from agin.neural_network import NeuralNetwork
```

#### **Example**

```python
# Option 1: Importing directly from agin
from agin import NeuralNetwork

# Option 2: Importing from agin.regression
from agin.neural_network import NeuralNetwork

# Training data
x_train = np.random.randn(3, 1000)
y_train = np.eye(3)[np.random.choice(3, 1000)].T

x_test = np.random.randn(3, 200)
y_test = np.eye(3)[np.random.choice(3, 200)].T

# Neural network parameters
n_x = x_train.shape[0]
n_h = 5
n_y = y_train.shape[0]
# Initialize the model
model = NeuralNetwork(n_x, n_h, n_y, learning_rate=0.1, iterations=2000)
model.fit(x_train, y_train)
# Evaluate the model
accuracy, precision, recall, f1_score = model.metrics(x_test, y_test)
print(f"\nAccuracy:  Custom: {accuracy:.2f}%")
print(f"Precision:  Custom: {precision:.2f}% ")
print(f"Recall:  Custom: {recall:.2f}%")
print(f"F1:  Custom: {f1_score:.2f}%")


```

### **Methods**
#### **`fit(x_train, y_train)`**
   - Trains the neural network by adjusting the weights using backpropagation.
   - **Args**:
     - `x_train` (numpy.ndarray or pandas.DataFrame): Training feature data.
     - `y_train` (numpy.ndarray or pandas.DataFrame): Target labels.
   - **Returns**: None. The model's weights are updated during training.

#### **`predict(x)`**
   - Predicts class labels for the given feature data.
   - **Args**:
     - `x` (numpy.ndarray or pandas.DataFrame): Feature data.
   - **Returns**: numpy.ndarray of predicted class labels.

#### **`metrics(y_pred, y_test)`**
   - Computes various evaluation metrics for classification.
   - **Args**:
     - `y_pred` (numpy.ndarray): Predicted labels.
     - `y_test` (numpy.ndarray): True labels.
   - **Returns**: Tuple containing accuracy, precision, recall, and F1-score.

### **Parameters**
- `layers` (list of int): List specifying the number of neurons in each layer, including input and output layers. Example: `[2, 4, 1]` for a 2-input, 4-hidden, and 1-output layer network.
- `activation` (str): The activation function to use for the hidden layers ('relu', 'sigmoid', 'tanh'). Default is 'relu'.
- `learning_rate` (float): The learning rate for gradient descent. Default is 0.01.
- `epochs` (int): The number of training iterations. Default is 1000.
- `batch_size` (int): The number of samples per gradient update. Default is 32.

### **Attributes**
- `weights` (list of numpy.ndarray): The weights for each layer of the network.
- `biases` (list of numpy.ndarray): The biases for each layer.
- `loss_history` (list): The history of the loss function during training.
- `accuracy_history` (list): The history of the accuracy during training.
