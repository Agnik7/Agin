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
        
    def fit(self,x,y):
        """
        Trains the Polynomial Regression model using the Normal Equation.
        
        Args:
            x (list or numpy.ndarray): A 1D array containing the independent variable.
            y (list or numpy.ndarray): A 1D array containing the dependent variable.
        
        Returns:
            None: This method updates the model's intercept and coefficients.
        """
        # Convert x and y to numpy arrays
        x_train=np.array(x)
        y_train=np.array(y)
        
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
   
    def predict(self,x):
        
        """
        Predicts the dependent variable using the trained Polynomial Regression model.
        
        Args:
            x (list or numpy.ndarray): A 1D array of values for the independent variable.
        
        Returns:
            numpy.ndarray: A 1D array of predicted values.
        """
        
        x_test=np.array(x)
        
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
    
if __name__ == "__main__":
    # Sample Data: (e.g., for a quadratic relationship y = x^2 + 2x + 1)
    x_train = [1, 2, 3, 4, 5]
    y_train = [4, 11, 20, 31, 44]  # y = x^2 + 2x + 1

    # Create PolynomialRegression instance with degree 2 (quadratic)
    model = PolynomialRegression(degree=2)

    # Train the model
    model.fit(x_train, y_train)

    # Predict using the trained model
    x_test = [6, 7, 8]
    y_pred = model.predict(x_test)

    # Print predictions
    print("Predictions for x =", x_test, "are:", y_pred)

    # Evaluate the model
    mse, r2 = model.metrics(y_train, model.predict(x_train))
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")    
               