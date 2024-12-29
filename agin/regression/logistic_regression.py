import numpy as np
import pandas as pd
import copy
from ..preprocessing import MinMaxScaler
class LogisticRegression():
    def __init__(self):
        """
        Initializes the LogisticRegression model with empty lists to store 
        the loss values and training accuracies.

        Attributes:
            losses (list): List to store loss values during training.
            train_accuracies (list): List to store accuracy values during training.
            weights (numpy array): Array of model weights initialized to zeros.
            bias (float): Bias term initialized to zero.
        """
        self.loss = []
        self.train_acc = []
        self.minmax = MinMaxScaler()

    def fit(self, x, y, epochs=100):
        """ 
        Function to train the Logistic Regression model based on training data provided by the user.
        Calculates the model weights and bias using gradient descent optimization.
        
        Args: 
            x (numpy array or pandas DataFrame): Feature data for training.
            y (numpy array or pandas DataFrame): Target data for training.
            epochs (int, optional): The number of iterations for gradient descent (default is 100).

        Returns:
            None: The model parameters (weights and bias) are updated in place.
        """
        x,y = self.deepcopy(x,y)
        x = self.minmax.scale(x)
        self.weights = np.zeros(x.shape[1])
        self.bias = 0
        for i in range(epochs):
            x_dot_weights = np.matmul(self.weights, x.transpose()) + self.bias
            pred = self.sigmoid(x_dot_weights)
            loss = self.compute_loss(y, pred)
            error_w, error_b = self.compute_gradients(x, y, pred)
            self.update_params(error_w, error_b)

            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            self.train_acc.append(self.accuracy_score(y, pred_to_class))
            self.loss.append(loss)

    def compute_loss(self, y_true, y_pred):
        """ 
        Function to calculate the binary cross-entropy loss between true and predicted values.

        Args: 
            y_true (numpy array): Actual target values.
            y_pred (numpy array): Predicted probabilities for the target values.

        Returns:
            float: The calculated binary cross-entropy loss.
        """
        # binary cross entropy
        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (1-y_true) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)

    def compute_gradients(self, x, y_true, y_pred):
        """ 
        Function to compute the gradients of the loss with respect to the weights and bias.

        Args: 
            x (numpy array): Feature data.
            y_true (numpy array): Actual target values.
            y_pred (numpy array): Predicted values (probabilities) from the model.

        Returns:
            tuple: A tuple containing:
                - gradients_w (numpy array): The gradients with respect to weights.
                - gradient_b (float): The gradient with respect to the bias.
        """
        # derivative of binary cross entropy
        difference =  y_pred - y_true
        gradient_b = np.mean(difference)
        gradients_w = np.matmul(x.transpose(), difference)
        gradients_w = np.array([np.mean(grad) for grad in gradients_w])

        return gradients_w, gradient_b

    def update_params(self, error_w, error_b):
        """ 
        Function to update the model weights and bias using the gradients computed by gradient descent.

        Args:
            error_w (numpy array): Gradients of the weights.
            error_b (float): Gradient of the bias.

        Returns:
            None: The weights and bias are updated in place.
        """
        self.weights = self.weights - 0.1 * error_w
        self.bias = self.bias - 0.1 * error_b

    def predict(self, x):
        """ 
        Function to make predictions using the trained Logistic Regression model.

        Args: 
            x (numpy array or pandas DataFrame): Feature data to predict on.

        Returns:
            list: A list of predicted class labels (0 or 1).
        """
        x = self.minmax.scale(x.values)
        x_dot_weights = np.matmul(x, self.weights.transpose()) + self.bias
        probabilities = self.sigmoid(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in probabilities]

    def sigmoid(self, x):
        """ 
        Function to apply the sigmoid activation function to the input values.

        Args: 
            x (numpy array): Input values to be transformed by the sigmoid function.

        Returns:
            numpy array: The sigmoid-transformed values for each input.
        """
        return np.array([self.sigmoid_function(value) for value in x])

    def sigmoid_function(self, x):
        """ 
        Sigmoid function to map input values to probabilities in the range (0, 1).

        Args: 
            x (float): The input value to be transformed.

        Returns:
            float: The sigmoid-transformed value in the range (0, 1).
        """
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)
    def deepcopy(self, x, y):
        """ 
        Function to perform a deep copy of the input data (x and y) to avoid modifying the original data.

        Args: 
            x (numpy array or pandas DataFrame): Feature data.
            y (numpy array or pandas DataFrame): Target data.

        Returns:
            tuple: A tuple containing the deep-copied feature data (x) and target data (y).
        """
        if isinstance(x, pd.DataFrame):
            x = x.values  # Convert DataFrame to NumPy array
        if isinstance(y, pd.DataFrame):
            y = y.values  # Convert DataFrame to NumPy array
        x = copy.deepcopy(x)
        y = copy.deepcopy(y).reshape(y.shape[0], 1)
        return x, y
    
    def accuracy_score(self,y_true, y_pred):
        """ 
        Function to calculate the accuracy of the model's predictions by comparing 
        the predicted values to the true values.

        Args:
            y_true (numpy array or list): The actual target values.
            y_pred (numpy array or list): The predicted values by the model.

        Returns:
            float: The accuracy score as the ratio of correct predictions to total predictions.
                It ranges from 0 (no correct predictions) to 1 (all predictions correct).
        """
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred)        
        correct_predictions = np.sum(y_true == y_pred)
        total_predictions = len(y_true)
        return correct_predictions / total_predictions
    def metrics(self, y_pred, y_test):
        """ 
        Function to calculate and return accuracy, precision, recall, and F1 score

        Args: 
            y_pred (list or numpy array): Predicted labels from the model (0 or 1)
            y_test (list or numpy array): Actual labels (0 or 1)

        Returns:
            tuple: (accuracy, precision, recall, F1 score)
        """
        # Convert to numpy arrays for easier computation
        y_pred = np.array(y_pred)
        y_test = np.array(y_test)

        # True positives, false positives, true negatives, and false negatives
        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        tn = np.sum((y_pred == 0) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))

        # Accuracy
        accuracy = self.accuracy_score(y_test,y_pred)

        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return accuracy, precision, recall, f1_score

