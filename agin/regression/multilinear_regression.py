import numpy as np

class MultilinearRegression:
    def __init__(self):
        """
        Initializes the MultilinearRegression model with intercept and coefficients set to None.
        
        Attributes:
            intercept (float): The intercept (bias term) of the model.
            coefficients (numpy.ndarray): The coefficients (slopes) for each feature in the model.
        """
        self.intercept=None
        self.coeffients=None
    def fit(self,x_train,y_train):
        """
        Trains the Multilinear Regression model using the Normal Equation to calculate coefficients.
        
        Args:
            x_train (list or numpy.ndarray): A 2D array containing the training data for independent variables.
            y_train (list or numpy.ndarray): A 1D array containing the true values for the dependent variable.
        
        Returns:
            None: This method updates the model's intercept and coefficients based on the input data.
        
        The Normal Equation is applied to solve for the coefficients of the linear regression model:
            coefficients = (X^T * X)^-1 * X^T * y
        """
        x_train=np.array(x_train)
        y_train=np.array(y_train)
        # Adding the column of bias with X train
        x_train_with_bias=np.c_[np.ones(x_train.shape[0]),x_train]
        
        # Apply the Normal Equation: coefficients = (X^T * X)^-1 * X^T * y
        
        x_transpose=x_train_with_bias.T
        x_transpose_x=np.dot(x_transpose,x_train_with_bias)
        x_inverse = np.linalg.pinv(x_transpose_x)
        x_transpose_y=np.dot(x_transpose,y_train)
        
        coefficients=np.dot(x_inverse,x_transpose_y)
        
        # Extract the intercept and the slopes
        self.intercept=coefficients[0]
        self.coeffients=coefficients[1:]
        
    def predict(self,x_test):
        """
        Predicts the dependent variable using the trained Multilinear Regression model.
        
        Args:
            x_test (list or numpy.ndarray): A 2D array containing test data for independent variables.
        
        Returns:
            numpy.ndarray: A 1D array of predicted values based on the model.
        
        This method uses the formula:
            prediction = X * coefficients + intercept
        where X is the input data and coefficients are the learned model parameters.
        """
        
        x_test=np.array(x_test)
        
        return np.dot(x_test,self.coeffients) + self.intercept
    
    def metrics(self,y_pred, y_test):
        """
        Calculates the Mean Squared Error (MSE) and R^2 score to evaluate the model's performance.
        
        Args:
            y_test (list or numpy.ndarray): A 1D array containing the true values for the dependent variable.
            y_pred (list or numpy.ndarray): A 1D array containing the predicted values from the model.
        
        Returns:
            tuple: A tuple containing two float values:
                - mse (float): The Mean Squared Error of the model.
                - r2_score (float): The R-squared score of the model.
        
        The Mean Squared Error (MSE) is computed as the average squared difference between the true and predicted values.
        The R-squared score is a measure of how well the model's predictions match the true values.
        """
        
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        
        # Mean Squared Error
        mse = np.mean((y_test - y_pred)** 2)
        
        # R^2 Score
        total_variance = np.sum((y_test - np.mean(y_test)) ** 2)
        explained_variance = np.sum((y_pred - np.mean(y_test)) ** 2)
        r2_score = explained_variance / total_variance

        return mse, r2_score