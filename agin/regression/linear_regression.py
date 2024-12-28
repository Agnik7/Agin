import numpy as np

class LinearRegression:
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, x, y):
        """ 
        Function to train the Linear Regression model based on data given by user. Calculates and stores the slope and intercept

        Args: 
            x,y: Lists containing the training X and Y datas
        
        Returns:
            None
        """
        # Convert to numpy arrays
        x_train = np.array(x)
        y_train = np.array(y)

        # Calculate the mean of x and y
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

    def predict(self, x):
        """ 
        Function to find the value(s) predicted by the model.

        Args: 
            x: List containing test_x data
        
        Returns:
            Float[]: Returns values predicted using Linear Regression.
        """
        x_test = np.array(x)
        return (self.slope * x_test) + self.intercept

    def metrics(self, y_pred, y_test):
        # Manually calculate Mean Squared Error (MSE)
        """ 
        Function to calculate the mean squared error

        Args: 
            y_pred,y_test: Lists containing predicted and expected output values respectively
        
        Returns:
            Float: Returns the mse of the model.
        """

        # Calculate Mean Squared Error (MSE)
        squared_errors = [(y_test - y_pred) ** 2 for y_true, y_pred in zip(y_test, y_pred)]
        mse = np.mean(squared_errors)

        # Calculate R2 Score (R2)
        total_variance = np.sum((y_test - np.mean(y_test)) ** 2)
        explained_variance = np.sum((y_pred - np.mean(y_test)) ** 2)
        r2 = explained_variance / total_variance
        return mse, r2