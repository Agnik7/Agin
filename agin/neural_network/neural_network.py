import numpy as np

class NeuralNetwork:
    def __init__(self, n_x, n_h, n_y, learning_rate=0.01, iterations=1000):
        """
        Initialize the Neural Network model with weights, biases, and hyperparameters.

        Args:
            n_x (int): The number of input features for the model.
            n_h (int): The number of neurons in the hidden layer.
            n_y (int): The number of output neurons.
            learning_rate (float, optional): The learning rate used for optimization (default is 0.01).
            iterations (int, optional): The number of iterations for training the model (default is 1000).
        """
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        self.learning_rate = learning_rate
        self.iterations = iterations

        self.parameters = self.initialize_parameters()

    def initialize_parameters(self):
        """
        Initialize the weights and biases for the model.

        Returns:
            dict: A dictionary containing the initialized weights ('w1', 'w2') and biases ('b1', 'b2').
        """
        w1 = np.random.randn(self.n_h, self.n_x) * 0.01
        b1 = np.zeros((self.n_h, 1))
        w2 = np.random.randn(self.n_y, self.n_h) * 0.01
        b2 = np.zeros((self.n_y, 1))

        return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

    def tanh(self, x):
        """
        Compute the hyperbolic tangent (tanh) activation function.

        Args:
            x (np.ndarray): Input values to the activation function.

        Returns:
            np.ndarray: The tanh of each element in x.
        """
        return np.tanh(x)

    def softmax(self, x):
        """
        Compute the softmax activation function.

        Args:
            x (np.ndarray): Input values to the activation function.

        Returns:
            np.ndarray: The softmax probabilities for each element in x.
        """
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def derivative_tanh(self, x):
        """
        Compute the derivative of the tanh activation function.

        Args:
            x (np.ndarray): Input values to the derivative of the tanh function.

        Returns:
            np.ndarray: The derivative of tanh applied to each element in x.
        """
        return 1 - np.power(np.tanh(x), 2)

    def forward_propagation(self, x):
        """
        Perform forward propagation through the neural network.

        Args:
            x (np.ndarray): Input feature data for the model.

        Returns:
            dict: A dictionary containing the intermediate values during forward propagation ('z1', 'a1', 'z2', 'a2').
        """
        w1, b1, w2, b2 = self.parameters["w1"], self.parameters["b1"], self.parameters["w2"], self.parameters["b2"]

        z1 = np.dot(w1, x) + b1
        a1 = self.tanh(z1)

        z2 = np.dot(w2, a1) + b2
        a2 = self.softmax(z2)

        return {"z1": z1, "a1": a1, "z2": z2, "a2": a2}

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

    def backward_propagation(self, x, y, forward_cache):
        """
        Perform backward propagation to compute gradients for weights and biases.

        Args:
            x (np.ndarray): The input feature data for the model.
            y (np.ndarray): The true labels (one-hot encoded) for the data.
            forward_cache (dict): A dictionary containing the values from forward propagation.

        Returns:
            dict: A dictionary containing the gradients of weights and biases ('dw1', 'db1', 'dw2', 'db2').
        """
        w2 = self.parameters["w2"]

        a1, a2 = forward_cache["a1"], forward_cache["a2"]
        m = x.shape[1]

        dz2 = a2 - y
        dw2 = (1 / m) * np.dot(dz2, a1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

        dz1 = np.dot(w2.T, dz2) * self.derivative_tanh(a1)
        dw1 = (1 / m) * np.dot(dz1, x.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

        return {"dw1": dw1, "db1": db1, "dw2": dw2, "db2": db2}

    def update_parameters(self, gradients):
        """
        Update the model's weights and biases using the computed gradients and learning rate.

        Args:
            gradients (dict): A dictionary containing the gradients for the weights and biases.
        """
        for key in gradients:
            self.parameters[key.replace("d", "")] -= self.learning_rate * gradients[key]

    def fit(self, x, y):
        """
        Train the neural network using gradient descent and forward/backward propagation.

        Args:
            x (np.ndarray): The input feature data for training.
            y (np.ndarray): The true labels (one-hot encoded) for the data.
        """
        self.costs = []

        for i in range(self.iterations):
            forward_cache = self.forward_propagation(x)
            cost = self.cost_function(forward_cache["a2"], y)
            gradients = self.backward_propagation(x, y, forward_cache)
            self.update_parameters(gradients)
            self.costs.append(cost)

            if i % (self.iterations // 10) == 0:
                print(f"Cost after iteration {i}: {cost:.4f}")

    def predict(self, x):
        """
        Predict the output classes for given input data.

        Args:
            x (np.ndarray): The input feature data for prediction.

        Returns:
            np.ndarray: The predicted class labels.
        """
        forward_cache = self.forward_propagation(x)
        return np.argmax(forward_cache["a2"], axis=0)

    def metrics(self, x, y):
        """
        Calculate performance metrics such as accuracy, precision, recall, and F1 score.

        Args:
            x (np.ndarray): The input feature data for the model.
            y (np.ndarray): The true labels (one-hot encoded) for the data.

        Returns:
            tuple: A tuple containing the following metrics:
                - accuracy (float): The accuracy of the model's predictions.
                - precision (float): The precision of the model.
                - recall (float): The recall of the model.
                - f1_score (float): The F1 score of the model.
        """
        predictions = self.predict(x)
        labels = np.argmax(y, axis=0)

        # Accuracy
        accuracy = np.mean(predictions == labels)

        # Precision, Recall, F1 Score (for each class)
        precision = np.zeros(self.n_y)
        recall = np.zeros(self.n_y)
        f1_score = np.zeros(self.n_y)

        for i in range(self.n_y):
            TP = np.sum((predictions == i) & (labels == i))  # True Positives
            FP = np.sum((predictions == i) & (labels != i))  # False Positives
            FN = np.sum((predictions != i) & (labels == i))  # False Negatives
            TN = np.sum((predictions != i) & (labels != i))  # True Negatives

            # Precision, Recall, and F1 Score
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

        # Average Precision, Recall, and F1 Score
        precision = np.mean(precision)
        recall = np.mean(recall)
        f1_score = np.mean(f1_score)

        return accuracy, precision, recall, f1_score

