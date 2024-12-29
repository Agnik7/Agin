import numpy as np

class PolynomialRegression:
    def __init__(self,degree = 2):
        """
        Initializes the PolynomialRegression model with degree, intercept, and coefficients set to None.
        
        Args:
            degree (int): The degree of the polynomial to be fitted (default is 2).
        
        Attributes:
            degree (int): Degree of the polynomial model.
            intercept (float): The intercept (bias term) of the model.
            coefficients (numpy.ndarray): The coefficients for each polynomial term.
        """
        self.degree=degree
        self.coefficients=None
        self.intercept=None
        
    def fit(self,x_train,y_train):
        """
        Trains the Polynomial Regression model using the Normal Equation.
        
        Args:
            x_train (list or numpy.ndarray): A 1D array containing the independent variable.
            y_train (list or numpy.ndarray): A 1D array containing the dependent variable.
        
        Returns:
            None: This method updates the model's intercept and coefficients.
        """
        # Convert x_train and y_train to numpy arrays
        x_train=np.array(x_train)
        y_train=np.array(y_train)
        
        x_poly =np.column_stack([x_train**i for i in range(self.degree+1)])
        
        # Apply the normal equation for getting an linear algebra
        
        x_transpose = x_poly.T  
        x_transpose_x = np.dot(x_transpose,x_poly)
        x_inverse = np.linalg.pinv(x_transpose_x)
        x_transpose_y = np.dot(x_transpose,y_train)
        
        # Calculate coefficients
        coefficients = np.dot(x_inverse, x_transpose_y)
        
        self.intercept=coefficients[0]
        self.coefficients=coefficients[1:]
   
    def predict(self,x_test):
        
        """
        Predicts the dependent variable using the trained Polynomial Regression model.
        
        Args:
            x_test (list or numpy.ndarray): A 1D array of values for the independent variable.
        
        Returns:
            numpy.ndarray: A 1D array of predicted values.
        """
        
        x_test=np.array(x_test)
        
        x_poly=np.column_stack([x_test**i for i in range(1,self.degree+1)])
        
        return np.dot(x_poly,self.coefficients) + self.intercept
    
    def metrics(self,y_pred,y_test):
        """
        Calculates the Mean Squared Error (MSE) and R^2 score to evaluate the model's performance.
        
        Args:
            y_test (list or numpy.ndarray): True values of the dependent variable.
            y_pred (list or numpy.ndarray): Predicted values from the model.
        
        Returns:
            tuple: (mse, r2_score), where:
                - mse (float): Mean Squared Error
                - r2_score (float): R-squared score
        """
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)

        # Mean Squared Error
        mse = np.mean((y_test - y_pred)**2)

        # R^2 Score
        total_variance = np.sum((y_test - np.mean(y_test))**2)
        explained_variance = np.sum((y_pred - np.mean(y_test))**2)
        r2_score = explained_variance / total_variance

        return mse, r2_score
    
