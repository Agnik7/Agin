import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        """
        Initializes the RidgeRegression model with intercept, coefficients, and regularization parameter.

        Attributes:
            slope (np.ndarray): The slope in the model.
            intercept (float): The intercept (bias term) of the model.
            alpha (float): The regularization parameter (penalty term).
        """
        self.slope = None
        self.intercept = None
        self.alpha = alpha

    def fit(self, x_train, y_train):
        """ 
        Trains the Ridge Regression model using the closed-form solution.

        Args: 
            x_train (list or np.ndarray): Training feature data.
            y_train (list or np.ndarray): Target data.
        """
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        if x_train.ndim == 1:
            x_train = x_train.reshape(-1, 1)

        # Add bias term
        X = np.column_stack((np.ones(x_train.shape[0]), x_train))

        # Closed-form solution for Ridge Regression
        identity = np.eye(X.shape[1])  # Identity matrix
        identity[0, 0] = 0  # No regularization for the intercept term

        ridge_matrix = np.linalg.inv(X.T @ X + self.alpha * identity)
        coefficients = ridge_matrix @ X.T @ y_train

        # Extract intercept and slope
        self.intercept = coefficients[0]
        self.slope = coefficients[1:]

    def predict(self, x_test):
        """ 
        Predicts target values for the test feature data.

        Args: 
            x_test (list or np.ndarray): Test feature data.
        """
        x_test = np.array(x_test)

        if x_test.ndim == 1:
            x_test = x_test.reshape(-1, 1)

        # Add bias term
        X = np.column_stack((np.ones(x_test.shape[0]), x_test))

        # Compute predictions
        return X @ np.concatenate(([self.intercept], self.slope))

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
        y_test = np.array(y_test)

        # Calculate MSE
        squared_errors = (y_test - y_pred) ** 2
        mse = np.mean(squared_errors)

        # Calculate R2 Score
        total_variance = np.sum((y_test - np.mean(y_test)) ** 2)
        explained_variance = np.sum((y_pred - np.mean(y_test)) ** 2)
        r2_score = explained_variance / total_variance

        return mse, r2_score

