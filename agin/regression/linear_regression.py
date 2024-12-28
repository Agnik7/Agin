import numpy as np

class LinearRegression:
    def __init__(self):
        """
        Initializes the LinearRegression model with intercept and coefficients set to None.
        
        Attributes:
            slope(float): The slope in the model.
            intercept (float): The intercept (bias term) of the model.
        """
        self.slope = None
        self.intercept = None

    def fit(self, x_train, y_train):
        """ 
        Function to train the Linear Regression model based on data given by the user. 
        Calculates and stores the slope and intercept of the linear equation.

        Args: 
            x_train (list or np.ndarray): List or array containing the training feature data (X values).
            y_train (list or np.ndarray): List or array containing the target data (Y values).
        
        Returns:
            None: This function updates the model's slope and intercept attributes.
        """
        # Convert to numpy arrays
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # Calculate the mean of x_train and y_train
        mean_value_x = np.mean(x_train)
        mean_value_y = np.mean(y_train)

        # Calculate deviations
        deviations_x = x_train - mean_value_x
        deviations_y = y_train - mean_value_y

        # Calculate the product of deviations and sum of squares
        product = np.sum(deviations_x * deviations_y)
        sum_of_squares_x = np.sum(deviations_x ** 2)

        # Calculate the slope (m) and intercept (b)
        self.slope = product / sum_of_squares_x
        self.intercept = mean_value_y - (self.slope * mean_value_x)

    def predict(self, x_test):
        """ 
        Function to find the value(s) predicted by the model based on the input feature data.

        Args: 
            x_test (list or np.ndarray): List or array containing the test feature data (X values).
        
        Returns:
            np.ndarray: Returns predicted values based on the learned linear regression model.
        """
        x_test = np.array(x_test)
        return (self.slope * x_test) + self.intercept

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
                - R2 (float): The R-squared value of the model, indicating the proportion of variance explained.
        """
        # Manually calculate Mean Squared Error (MSE)

        # Calculate Mean Squared Error (MSE)
        squared_errors = [(y_test - y_pred) ** 2 for y_true, y_pred in zip(y_test, y_pred)]
        mse = np.mean(squared_errors)

        # Calculate R2 Score (R2)
        total_variance = np.sum((y_test - np.mean(y_test)) ** 2)
        explained_variance = np.sum((y_pred - np.mean(y_test)) ** 2)
        r2 = explained_variance / total_variance
        return mse, r2