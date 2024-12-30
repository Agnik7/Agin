import numpy as np

class ElasticNetRegression:
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        """
        Initializes the ElasticNetRegression model with intercept, coefficients, and regularization parameters.

        Attributes:
            slope (np.ndarray): The coefficients (slopes) of the model.
            intercept (float): The intercept (bias term) of the model.
            alpha (float): The overall regularization strength.
            l1_ratio (float): The ratio of L1 (Lasso) to L2 (Ridge) regularization. Must be between 0 and 1.
            max_iter (int): Maximum number of iterations for optimization.
            tol (float): Tolerance for optimization convergence.
        """
        self.slope = None
        self.intercept = None
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, x_train, y_train):
        """
        Function to train the Elastic Net Regression model using coordinate descent algorithm.

        Args:
            x_train (list or np.ndarray): List or array containing the training feature data (X values).
            y_train (list or np.ndarray): List or array containing the target data (Y values).

        Returns:
            None: This function updates the model's slope and intercept attributes.
        """
        # Convert to numpy arrays
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # Add bias term (column of ones) to X for intercept calculation
        X = np.column_stack((np.ones(x_train.shape[0]), x_train))
        n_samples, n_features = X.shape

        # Initialize coefficients
        coefficients = np.zeros(n_features)

        # L1 and L2 weights
        l1_weight = self.alpha * self.l1_ratio
        l2_weight = self.alpha * (1 - self.l1_ratio)

        # Coordinate Descent Algorithm
        for iteration in range(self.max_iter):
            coefficients_old = coefficients.copy()

            # Update each coefficient
            for j in range(n_features):
                residual = y_train - (X @ coefficients) + (X[:, j] * coefficients[j])
                rho_j = np.dot(X[:, j], residual)

                if j == 0:  # Intercept term
                    coefficients[j] = rho_j / n_samples
                else:
                    z_j = np.sum(X[:, j] ** 2) + l2_weight
                    if rho_j < -l1_weight:
                        coefficients[j] = (rho_j + l1_weight) / z_j
                    elif rho_j > l1_weight:
                        coefficients[j] = (rho_j - l1_weight) / z_j
                    else:
                        coefficients[j] = 0

            # Check for convergence
            if np.max(np.abs(coefficients - coefficients_old)) < self.tol:
                break

        # Extract intercept and slope
        self.intercept = coefficients[0]
        self.slope = coefficients[1:]

    def predict(self, x_test):
        """
        Function to find the value(s) predicted by the model based on the input feature data.

        Args:
            x_test (list or np.ndarray): List or array containing the test feature data (X values).

        Returns:
            np.ndarray: Returns predicted values based on the learned Elastic Net regression model.
        """
        x_test = np.array(x_test)

        # If a single feature, reshape to 2D for consistent matrix operations
        if x_test.ndim == 1:
            x_test = x_test.reshape(-1, 1)

        # Compute predictions
        return np.dot(x_test, self.slope) + self.intercept

    def metrics(self, y_pred, y_test):
        """
        Function to calculate the performance metrics of the model, including Mean Squared Error (MSE)
        and R-squared (R2) score.

        Args:
            y_pred (list or np.ndarray): List or array of predicted values.
            y_test (list or np.ndarray): List or array of actual values (ground truth).

        Returns:
            tuple: A tuple containing the following metrics:
                - MSE (float): The Mean Squared Error of the model.
                - R2_SCORE (float): The R-squared value of the model, indicating the proportion of variance explained.
        """
        # Calculate Mean Squared Error (MSE)
        squared_errors = (y_test - y_pred) ** 2
        mse = np.mean(squared_errors)

        # Calculate R2 Score (R2)
        total_variance = np.sum((y_test - np.mean(y_test)) ** 2)
        explained_variance = np.sum((y_pred - np.mean(y_test)) ** 2)
        r2_score = explained_variance / total_variance

        return mse, r2_score
