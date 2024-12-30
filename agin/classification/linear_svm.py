import numpy as np

class LinearSVMClassifier:
    def __init__(self,learning_rate=0.01,regularization_strength=0.1,num_iterations=1000):
        """
        Initializes the Linear SVM model with parameters for training.

        Attributes:
            learning_rate (float): The learning rate for gradient descent.
            regularization_strength (float): The regularization strength to avoid overfitting.
            num_iterations (int): The number of iterations for training.
            w (numpy.ndarray): The weight vector for the model.
            b (float): The bias term for the model.
        """
        self.learning_rate=learning_rate
        self.regularization_strength=regularization_strength
        self.num_iterations=num_iterations
        self.w=None
        self.b=None
        
    def fit(self,x_train,y_train):
        """
        Trains the Linear SVM model by optimizing the weight vector and bias term using gradient descent.

        Args:
            x_train (list or numpy.ndarray): A 2D array containing the training data for independent variables.
            y_train (list or numpy.ndarray): A 1D array containing the true class labels for the dependent variable.

        Returns:
            None: This method updates the model's weight vector and bias term.

        This method optimizes the following objective:
            Minimize: 1/2 * ||w||^2 + C * Σ(max(0, 1 - y_i * (w^T * x_i + b)))
        """
        x_train=np.array(x_train)
        y_train=np.array(y_train)
        
        # Initialize weight vector and bias term
        self.w=np.zeros(x_train.shape[1])
        self.b=0
        
        for _ in range(self.num_iterations):
            for i in range(x_train.shape[0]):
                # Compute the margin (w * x_i + b)
                margin=y_train[i] * (np.dot(x_train[i],self.w) + self.b)
                
                if margin >=1 :
                    # Correct classification, just regularization
                    dw = self.regularization_strength * self.w 
                    db = 0
                    
                else :
                    # Incorrect classification, hinge loss  
                    dw = (self.regularization_strength * self.w) - np.dot(y_train[i],x_train[i])
                    db = - y_train[i]
                
                # Update the parameters using gradient descent    
                self.w-=self.learning_rate*dw
                self.b-=self.learning_rate*db
                
    def predict(self,x_test):
        
        """
        Predicts the class label for each sample in the test data using the trained Linear SVM model.

        Args:
            x_test (list or numpy.ndarray): A 2D array containing test data for independent variables.

        Returns:
            numpy.ndarray: A 1D array of predicted class labels for each sample.

        The prediction is based on the decision function:
            Predict class '1' if w^T * x + b > 0, otherwise class '-1'.
        """
        
        x_test=np.array(x_test)
        
        predictions=np.dot(x_test,self.w) + self.b
        return np.where(predictions > 0,1,-1)
    
    def metrics(self, y_pred, y_test):
        """
        Calculates the accuracy, precision, recall, and F1 score of the Linear SVM classifier.

        Args:
            y_test (list or numpy.ndarray): A 1D array containing the true class labels for the dependent variable.
            y_pred (list or numpy.ndarray): A 1D array containing the predicted class labels from the model.

        Returns:
            tuple: A tuple containing the following metrics:
                - accuracy (float): The fraction of correct predictions.
                - precision (float): The ratio of true positives to the sum of true positives and false positives.
                - recall (float): The ratio of true positives to the sum of true positives and false negatives.
                - f1_score (float): The harmonic mean of precision and recall, giving a balanced score.

        Accuracy is computed as:
            accuracy = (number of correct predictions) / (total number of predictions)

        Precision is computed as:
            precision = true_positives / (true_positives + false_positives)

        Recall is computed as:
            recall = true_positives / (true_positives + false_negatives)

        F1 Score is computed as:
            f1_score = 2 * (precision * recall) / (precision + recall)
        """
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)

        accuracy = np.sum(y_test == y_pred) / len(y_test)
        
        true_positives = np.sum((y_test == 1) & (y_pred == 1))
        false_positives = np.sum((y_test == -1) & (y_pred == 1))
        false_negatives = np.sum((y_test == 1) & (y_pred == -1))
        true_negatives = np.sum((y_test == -1) & (y_pred == -1))
        
        # Precision
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
        
        # Recall
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
        
        # F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        
        return accuracy, precision, recall, f1_score
