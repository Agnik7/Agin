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
        Calculates performance metrics: Mean Squared Error (MSE) and R-squared (R2) score.

        Args: 
            y_pred (np.ndarray): Predicted values.
            y_test (np.ndarray): Actual values.
        """
        y_test = np.array(y_test)

        # Calculate MSE
        squared_errors = (y_test - y_pred) ** 2
        mse = np.mean(squared_errors)

        # Calculate R2 Score
        total_variance = np.sum((y_test - np.mean(y_test)) ** 2)
        explained_variance = np.sum((y_pred - np.mean(y_test)) ** 2)
        r2 = explained_variance / total_variance

        return mse, r2

if __name__ == "__main__":
    # Sample data
    x_train = [1, 2, 3, 4, 5]
    y_train = [2.2, 2.8, 3.6, 4.5, 5.1]

    x_test = [6, 7, 8]
    y_test = [5.9, 6.7, 7.6]

    # Regularization strength
    alpha = 0.1

    # Create and train the model
    model = RidgeRegression(alpha)
    model.fit(x_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(x_test)
    mse, r2 = model.metrics(y_pred, y_test)

    # Display results
    print("Trained Ridge Regression Model:")
    print(f"Slope: {model.slope}")
    print(f"Intercept: {model.intercept}\n")

    print("Test Results:")
    print(f"Predicted Values: {y_pred}")
    print(f"Actual Values: {y_test}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")
