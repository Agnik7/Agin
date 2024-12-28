import numpy as np

class MultilinearRegression:
    def __init__(self):
        self.intercept=None
        self.coeffients=None
    def fit(self,x,y):
        x_train=np.array(x)
        y_train=np.array(y)
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
        
    def predict(self,x):
        """
        Predicts the dependent variable using the trained model.

        Args:
            X: 2D list or array containing test data for independent variables.

        Returns:
            List of predicted values.
        """
        
        x_test=np.array(x)
        
        return np.dot(x_test,self.coeffients) + self.intercept
    
    def metrics(self,y_test,y_pred):
        """
        Calculates the Mean Squared Error (MSE) and R^2 score.

        Args:
            y_pred: List of predicted values.
            y_test: List of true values.

        Returns:
            Tuple containing (mse, r2_score).
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
    



        
    
    
                 
        
     