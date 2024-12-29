import numpy as np
import pandas as pd
import copy
from ..preprocessing import MinMaxScaler

class LogisticRegression():
    def __init__(self, regularization='l2', C=1.0, max_iter=100, tol=1e-4, class_weight=None, random_state=None, l1_ratio=0.5):
        """
        Initializes the LogisticRegression model with hyperparameters for training and regularization.

        Args:
            regularization (str): Type of regularization ('l1', 'l2', 'elasticnet' or None). Default is 'l2'.
            C (float): Inverse of regularization strength. Smaller values indicate stronger regularization. Default is 1.0.
            max_iter (int): Maximum number of iterations for optimization. Default is 100.
            tol (float): Tolerance for stopping criteria. Default is 1e-4.
            class_weight (dict or 'balanced' or 'unbalanced'): Weights associated with classes. If 'balanced', class weights are
                                               computed inversely proportional to class frequencies. Default is None.
            random_state (int): Seed for random number generation to ensure reproducibility. Default is None.
            l1_ratio (float): The mixing parameter for elasticnet regularization. l1_ratio=1 corresponds to l1, 
                            while l1_ratio=0 corresponds to l2. Default is 0.5.
        """
        self.loss = []
        self.train_acc = []
        self.minmax = MinMaxScaler()
        self.regularization = regularization
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight
        self.random_state = random_state
        self.l1_ratio = l1_ratio
        
        if random_state is not None:
            np.random.seed(random_state)

    def compute_class_weights(self, y):
        """
        Computes class weights if the 'balanced' option is selected.

        Args:
            y (np.ndarray): Array of target labels.

        Returns:
            dict or None: A dictionary mapping class labels to their computed weights if 'balanced' is selected,
                          otherwise the original class weights.
        """
        if self.class_weight == 'balanced':
            classes = np.unique(y)
            class_weights = {}
            n_samples = len(y)
            for c in classes:
                class_weights[c] = n_samples / (len(classes) * np.sum(y == c))
            return class_weights
        elif self.class_weight == 'unbalanced':
            classes = np.unique(y)
            class_weights = {}
            for c in classes:
                class_weights[c] = np.sum(y == c) / len(y)
            return class_weights
        return self.class_weight

    def fit(self, x, y, epochs=None, learning_rate=0.1, batch_size=32):
        """ 
        Trains the Logistic Regression model using gradient descent with optional batch processing and regularization.

        Args:
            x (np.ndarray or pd.DataFrame): Training feature data.
            y (np.ndarray or pd.DataFrame): Target labels.
            epochs (int): Number of epochs for training. Default is the value of `max_iter`.
            learning_rate (float): Learning rate for gradient updates. Default is 0.1.
            batch_size (int): Size of batches for mini-batch gradient descent. Default is 32.

        Returns:
            self: The trained Logistic Regression model.
        """
        x, y = self.deepcopy(x, y)
        x = self.minmax.scale(x)
        
        if epochs is None:
            epochs = self.max_iter

        n_features = x.shape[1]
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
        
        class_weights = self.compute_class_weights(y)
        n_samples = x.shape[0]
        prev_loss = float('inf')
        
        for i in range(epochs):
            indices = np.random.permutation(n_samples)
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                x_batch = x[batch_indices]
                y_batch = y[batch_indices]
                
                # Ensure proper shapes
                x_dot_weights = np.dot(x_batch, self.weights.reshape(-1, 1)) + self.bias
                pred = self.sigmoid(x_dot_weights)
                
                sample_weights = np.ones(len(y_batch))
                if class_weights is not None:
                    for cls, weight in class_weights.items():
                        sample_weights[y_batch.ravel() == cls] = weight
                
                loss = self.compute_loss(y_batch, pred, sample_weights)
                error_w, error_b = self.compute_gradients(x_batch, y_batch, pred, sample_weights)
                
                # Add regularization term to gradients
                if self.regularization == 'l2':
                    error_w += (1 / (2 * self.C)) * self.weights
                elif self.regularization == 'l1':
                    error_w += (1 / (2 * self.C)) * np.sign(self.weights)
                elif self.regularization == 'elasticnet':
                    l1_term = self.l1_ratio * np.sign(self.weights)
                    l2_term = (1 - self.l1_ratio) * self.weights
                    error_w += (1 / (2 * self.C)) * (l1_term + l2_term)
                
                self.update_params(error_w, error_b, learning_rate)
            
            # Calculate full training metrics
            full_pred = self.sigmoid(np.dot(x, self.weights.reshape(-1, 1)) + self.bias)
            current_loss = self.compute_loss(y, full_pred)
            self.loss.append(current_loss)
            
            pred_to_class = [1 if p > 0.5 else 0 for p in full_pred]
            self.train_acc.append(self.accuracy_score(y, pred_to_class))
            
            if abs(prev_loss - current_loss) < self.tol:
                break
            prev_loss = current_loss
            
        return self

    def compute_loss(self, y_true, y_pred, sample_weights=None):
        """
        Computes the binary cross-entropy loss with optional regularization.

        Args:
            y_true (np.ndarray): Array of true labels.
            y_pred (np.ndarray): Array of predicted probabilities.
            sample_weights (np.ndarray, optional): Weights for individual samples. Default is None.

        Returns:
            float: The computed loss value.
        """
        if sample_weights is None:
            sample_weights = np.ones(len(y_true))
            
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        
        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (1 - y_true) * np.log(1 - y_pred + 1e-9)
        weighted_loss = -np.mean(sample_weights * (y_zero_loss + y_one_loss))
        
        if self.regularization == 'l2':
            reg_term = (1 / (2 * self.C)) * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            reg_term = (1 / (2 * self.C)) * np.sum(np.abs(self.weights))
        elif self.regularization == 'elasticnet':
            l1_term = self.l1_ratio * np.sum(np.abs(self.weights))
            l2_term = (1 - self.l1_ratio) * np.sum(self.weights ** 2)
            reg_term = (1 / (2 * self.C)) * (l1_term + l2_term)
        else:
            reg_term = 0
            
        return weighted_loss + reg_term

    def compute_gradients(self, x, y_true, y_pred, sample_weights=None):
        """
        Computes gradients of the loss function with respect to the model's parameters.

        Args:
            x (np.ndarray): Feature data.
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted probabilities.
            sample_weights (np.ndarray, optional): Weights for individual samples. Default is None.

        Returns:
            tuple: Gradients for weights and bias.
        """
        if sample_weights is None:
            sample_weights = np.ones(len(y_true))
        
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        sample_weights = sample_weights.ravel()
        
        difference = (y_pred - y_true) * sample_weights
        gradient_b = np.mean(difference)
        gradients_w = np.mean(x * difference.reshape(-1, 1), axis=0)
        
        return gradients_w, gradient_b

    def update_params(self, error_w, error_b, learning_rate):
        """
        Updates model parameters using computed gradients and learning rate.

        Args:
            error_w (np.ndarray): Gradients for weights.
            error_b (float): Gradient for bias.
            learning_rate (float): Learning rate for parameter updates.
        """
        self.weights = self.weights - learning_rate * error_w
        self.bias = self.bias - learning_rate * error_b

    def predict_probabilities(self, x):
        """
        Predicts probabilities for the given feature data.

        Args:
            x (np.ndarray or pd.DataFrame): Feature data.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        if isinstance(x, pd.DataFrame):
            x = x.values
        x = self.minmax.scale(x)
        return self.sigmoid(np.dot(x, self.weights.reshape(-1, 1)) + self.bias)

    def predict(self, x, threshold=0.5):
        """
        Predicts class labels for the given feature data.

        Args:
            x (np.ndarray or pd.DataFrame): Feature data.
            threshold (float): Threshold for converting probabilities to binary class labels. Default is 0.5.

        Returns:
            np.ndarray: Predicted class labels.
        """
        probas = self.predict_probabilities(x)
        return (probas >= threshold).astype(int)

    def sigmoid(self, x):
        """
        Applies the sigmoid function element-wise to the input.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Sigmoid-transformed values.
        """
        return np.array([self.sigmoid_function(value) for value in x])

    def sigmoid_function(self, x):
        """
        Computes the sigmoid function for a single input value.

        Args:
            x (float): Input value.

        Returns:
            float: Sigmoid-transformed value.

        """
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)

    def deepcopy(self, x, y):
        """
        Creates deep copies of input feature and target data.

        Args:
            x (np.ndarray or pd.DataFrame): Feature data.
            y (np.ndarray or pd.DataFrame): Target labels.

        Returns:
            tuple: Deep copies of feature data and target labels.
        """
        if isinstance(x, pd.DataFrame):
            x = x.values
        if isinstance(y, pd.DataFrame):
            y = y.values
        x = copy.deepcopy(x)
        y = copy.deepcopy(y).reshape(-1, 1)
        return x, y
    
    def accuracy_score(self, y_true, y_pred):
        """
        Computes the accuracy score for predictions.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Accuracy score.
        """
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred).ravel()
        return np.mean(y_true == y_pred)

    def metrics(self, y_pred, y_test):
        """
        Computes various evaluation metrics for classification.

        Args:
            y_pred (np.ndarray): Predicted labels.
            y_test (np.ndarray): True labels.

        Returns:
            tuple: Accuracy, precision, recall, and F1-score.
        """
        y_pred = np.array(y_pred).ravel()
        y_test = np.array(y_test).ravel()

        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        tn = np.sum((y_pred == 0) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))

        accuracy = self.accuracy_score(y_test, y_pred)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return accuracy, precision, recall, f1_score