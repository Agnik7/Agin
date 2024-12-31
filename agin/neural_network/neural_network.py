import numpy as np
from ..utils import Neuron

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, iterations=1000, init_method="he"):
        """
        Initialize the Neural Network model with layers, weights, and hyperparameters.

        Args:
            layer_sizes (list): A list containing the number of neurons for each layer [n_x, n_h1, ..., n_hn, n_y].
            learning_rate (float, optional): The learning rate used for optimization (default is 0.01).
            iterations (int, optional): The number of iterations for training the model (default is 1000).
            init_method (str, optional): Initialization method for weights ("he" or "xavier"). Default is "he".
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.iterations = iterations

        self.layers = []
        for i in range(1, len(layer_sizes)):
            activation_function = np.tanh if i < len(layer_sizes) - 1 else self.softmax
            activation_derivative = lambda x: 1 - np.power(np.tanh(x), 2) if i < len(layer_sizes) - 1 else None
            self.layers.append(
                Neuron(layer_sizes[i - 1], layer_sizes[i], activation_function, activation_derivative, init_method)
            )

    @staticmethod
    def softmax(x):
        """
        Compute the softmax activation function.

        Args:
            x (np.ndarray): Input data array.

        Returns:
            np.ndarray: Softmax-transformed probabilities.
        """
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def normalize(self, x):
        """
        Normalize the input features for better optimization and accuracy.

        Args:
            x (np.ndarray): Input feature data for normalization.

        Returns:
            np.ndarray: Normalized input data.
        """
        self.mean = np.mean(x, axis=1, keepdims=True)
        self.std = np.std(x, axis=1, keepdims=True)
        return (x - self.mean) / (self.std + 1e-8)

    def forward_propagation(self, x):
        """
        Perform forward propagation through the neural network.

        Args:
            x (np.ndarray): Input feature data for the model.

        Returns:
            list: A list of activations from each layer.
        """
        activations = [x]
        for layer in self.layers:
            activations.append(layer.forward(activations[-1]))
        return activations

    def cost_function(self, a2, y):
        """
        Compute the cost function (cross-entropy loss) for the model.

        Args:
            a2 (np.ndarray): The predicted output probabilities from the model.
            y (np.ndarray): The true labels (one-hot encoded) for the data.

        Returns:
            float: The computed cost (cross-entropy loss).
        """
        m = y.shape[1]
        cost = -(1 / m) * np.sum(y * np.log(a2))
        return cost

    def backward_propagation(self, x, y, activations):
        """
        Perform backward propagation to compute gradients for weights and biases.

        Args:
            x (np.ndarray): The input feature data for the model.
            y (np.ndarray): The true labels (one-hot encoded) for the data.
            activations (list): A list of activations from each layer.

        Returns:
            None: Updates gradients in the model's layers.
        """
        dz = activations[-1] - y
        for i in reversed(range(len(self.layers))):
            prev_a = activations[i]
            dz = self.layers[i].backward(dz, prev_a)

    def update_parameters(self):
        """
        Update the parameters of all layers using the gradients.

        Returns:
            None: Updates the model's weights and biases.
        """
        for layer in self.layers:
            layer.update_parameters(self.learning_rate)

    def fit(self, x_train, y_train):
        """
        Train the neural network using gradient descent and forward/backward propagation.

        Args:
            x_train (np.ndarray): The input feature data for training.
            y_train (np.ndarray): The true labels (one-hot encoded) for the data.

        Returns:
            None: Updates the model's parameters during training.
        """
        x_train = self.normalize(x_train)
        self.costs = []

        for i in range(self.iterations):
            activations = self.forward_propagation(x_train)
            cost = self.cost_function(activations[-1], y_train)
            self.backward_propagation(x_train, y_train, activations)
            self.update_parameters()
            self.costs.append(cost)

            if i % (self.iterations // 10) == 0:
                print(f"Cost after iteration {i}: {cost:.4f}")

    def predict(self, x_test):
        """
        Predict the output classes for given input data.

        Args:
            x_test (np.ndarray): The input feature data for prediction.

        Returns:
            np.ndarray: The predicted class labels.
        """
        x_test = self.normalize(x_test)
        activations = self.forward_propagation(x_test)
        return np.argmax(activations[-1], axis=0)

    def metrics(self, y_pred, y_test):
        """
        Calculate performance metrics such as accuracy, precision, recall, and F1 score.

        Args:
            y_pred (np.ndarray): The predicted labels for the data.
            y_test (np.ndarray): The true labels (one-hot encoded) for the data.

        Returns:
            tuple: A tuple containing the following metrics:
                - accuracy (float): The accuracy of the model's predictions.
                - precision (float): The precision of the model.
                - recall (float): The recall of the model.
                - f1_score (float): The F1 score of the model.
        """
        #predictions = self.predict(x_test)
        labels = np.argmax(y_test, axis=0)

        accuracy = np.mean(y_pred == labels)

        precision = np.zeros(self.layer_sizes[-1])
        recall = np.zeros(self.layer_sizes[-1])
        f1_score = np.zeros(self.layer_sizes[-1])

        for i in range(self.layer_sizes[-1]):
            TP = np.sum((y_pred == i) & (labels == i))
            FP = np.sum((y_pred == i) & (labels != i))
            FN = np.sum((y_pred != i) & (labels == i))

            if TP + FP > 0:
                precision[i] = TP / (TP + FP)
            else:
                precision[i] = 0
            if TP + FN > 0:
                recall[i] = TP / (TP + FN)
            else:
                recall[i] = 0
            if precision[i] + recall[i] > 0:
                f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
            else:
                f1_score[i] = 0

        precision = np.mean(precision)
        recall = np.mean(recall)
        f1_score = np.mean(f1_score)

        return accuracy, precision, recall, f1_score
