import numpy as np

class NonLinearSVM:
    def __init__(self, learning_rate=0.01, regularization_strength=0.1, num_iterations=1000, gamma=1.0, C=1.0):
        """
        Initializes the Non-Linear SVM model with hyperparameters for training.

        Attributes:
            learning_rate (float): The step size for gradient descent during training.
            regularization_strength (float): The coefficient for the regularization term to control overfitting.
            num_iterations (int): The number of iterations for gradient descent optimization.
            gamma (float): The parameter for the RBF kernel that defines the influence of a single training sample.
            C (float): The regularization parameter that controls the trade-off between achieving a low error on the training data and minimizing model complexity.
            alpha (numpy.ndarray): The dual coefficients (Lagrange multipliers) of the model, initialized during training.
            b (float): The bias term of the decision function.
            x_train (numpy.ndarray): The training data, stored for kernel computations.
            y_train (numpy.ndarray): The labels of the training data, stored for kernel computations.
        """
        self.learning_rate = learning_rate
        self.regularization_strength = regularization_strength
        self.num_iterations = num_iterations
        self.gamma = gamma
        self.C = C
        self.alpha = None
        self.b = 0
        self.x_train = None
        self.y_train = None
    
    def rbf_kernel(self, X1, X2):
        """
        Computes the Radial Basis Function (RBF) kernel between two datasets.

        Args:
            X1 (numpy.ndarray): The first dataset of shape (n_samples_1, n_features).
            X2 (numpy.ndarray): The second dataset of shape (n_samples_2, n_features).

        Returns:
            numpy.ndarray: A kernel matrix of shape (n_samples_1, n_samples_2), where each element
            represents the similarity between a pair of samples from X1 and X2 based on the RBF kernel.
        """
        dist_sq = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * dist_sq)

    def fit(self, x_train, y_train):
        """
        Trains the Non-Linear SVM model by optimizing the dual coefficients and bias term.

        Args:
            x_train (numpy.ndarray): A 2D array containing the training data for independent variables.
            y_train (numpy.ndarray): A 1D array containing the class labels for the training data.

        Returns:
            None: The method updates the dual coefficients (alpha) and bias term (b).

        This method uses the RBF kernel to compute the decision boundary in a high-dimensional space and 
        applies gradient descent to maximize the margin while minimizing the hinge loss.
        """
        self.x_train = x_train
        self.y_train = y_train
        n_samples = x_train.shape[0]
        
        # Initialize alpha (dual coefficients)
        self.alpha = np.zeros(n_samples)
        
        # Compute the kernel matrix
        K = self.rbf_kernel(x_train, x_train)
        
        for _ in range(self.num_iterations):
            for i in range(n_samples):
                # Compute the margin
                decision_value = np.sum(self.alpha * y_train * K[:, i]) + self.b
                margin = y_train[i] * decision_value
                
                # Update rules for alpha and bias
                if margin < 1:
                    self.alpha[i] += self.learning_rate * (1 - margin - self.regularization_strength * self.alpha[i])
                    self.b += self.learning_rate * y_train[i]
                else:
                    self.alpha[i] -= self.learning_rate * self.regularization_strength * self.alpha[i]
            
            # Ensure alpha satisfies the constraints
            self.alpha = np.clip(self.alpha, 0, self.C)

    def predict(self, x_test):
        """
        Predicts the class labels for each sample in the test data using the trained Non-Linear SVM model.

        Args:
            x_test (numpy.ndarray): A 2D array containing the test data for independent variables.

        Returns:
            numpy.ndarray: A 1D array of predicted class labels for the test data.

        The prediction is based on the decision function:
            Predict class '1' if the decision value > 0, otherwise class '-1'.
        """
        K = self.rbf_kernel(self.x_train, x_test)
        decision_values = np.dot((self.alpha * self.y_train).T, K) + self.b
        return np.sign(decision_values)

    def metrics(self, y_pred, y_test):
        """
        Calculates the performance metrics for the Non-Linear SVM classifier.

        Args:
            y_test (numpy.ndarray): A 1D array containing the true class labels for the test data.
            y_pred (numpy.ndarray): A 1D array containing the predicted class labels from the model.

        Returns:
            tuple: A tuple containing the following metrics:
                - accuracy (float): The fraction of correct predictions.
                - precision (float): The ratio of true positives to the sum of true positives and false positives.
                - recall (float): The ratio of true positives to the sum of true positives and false negatives.
                - f1_score (float): The harmonic mean of precision and recall.

        Accuracy is computed as:
            accuracy = (number of correct predictions) / (total number of predictions)

        Precision is computed as:
            precision = true_positives / (true_positives + false_positives)

        Recall is computed as:
            recall = true_positives / (true_positives + false_negatives)

        F1 Score is computed as:
            f1_score = 2 * (precision * recall) / (precision + recall)
        """
        accuracy = np.mean(y_pred == y_test)
        true_positive = np.sum((y_test == 1) & (y_pred == 1))
        false_positive = np.sum((y_test == -1) & (y_pred == 1))
        false_negative = np.sum((y_test == 1) & (y_pred == -1))
        
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        return accuracy, precision, recall, f1_score
