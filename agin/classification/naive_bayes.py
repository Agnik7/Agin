import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        """
        Initializes the Naive Bayes model with class probabilities and feature likelihoods.
        
        Attributes:
            class_probs (dict): The probabilities of each class in the dataset.
            feature_probs (dict): The likelihood of each feature given the class.
        """
        self.class_probs = None
        self.feature_probs = None
    
    def fit(self, X, y):
        """
        Trains the Naive Bayes model by calculating class probabilities and feature likelihoods.
        
        Args:
            X (list or numpy.ndarray): A 2D array containing the training data for independent variables.
            y (list or numpy.ndarray): A 1D array containing the true class labels for the dependent variable.
        
        Returns:
            None: This method updates the model's class probabilities and feature likelihoods.
        
        This method calculates:
            - class probabilities: P(class)
            - feature likelihoods: P(feature | class)
        """
        X_train = np.array(X)
        y_train = np.array(y)
        
        # Calculate class probabilities: P(class)
        class_labels = np.unique(y_train)
        class_probs = {}
        for label in class_labels:
            class_probs[label] = np.sum(y_train == label) / len(y_train)
        
        # Calculate feature probabilities: P(feature | class)
        feature_probs = {}
        for label in class_labels:
            X_class = X_train[y_train == label]
            feature_probs[label] = {}
            for feature_idx in range(X_train.shape[1]):
                feature_values = X_class[:, feature_idx]
                unique_vals, counts = np.unique(feature_values, return_counts=True)
                feature_probs[label][feature_idx] = dict(zip(unique_vals, counts / len(feature_values)))
        
        self.class_probs = class_probs
        self.feature_probs = feature_probs
    
    def predict(self, X):
        """
        Predicts the class label for each sample in the test data using the trained Naive Bayes model.
        
        Args:
            X (list or numpy.ndarray): A 2D array containing test data for independent variables.
        
        Returns:
            numpy.ndarray: A 1D array of predicted class labels for each sample.
        
        The prediction is based on the formula:
            P(class | X) ∝ P(class) * P(X | class)
        where X is the input data and P(X | class) is the product of individual feature likelihoods.
        """
        X_test = np.array(X)
        predictions = []
        
        for sample in X_test:
            class_scores = {}
            for label, class_prob in self.class_probs.items():
                score = np.log(class_prob)  # Using log to avoid underflow
                
                for feature_idx, feature_value in enumerate(sample):
                    if feature_value in self.feature_probs[label][feature_idx]:
                        score += np.log(self.feature_probs[label][feature_idx].get(feature_value, 1e-5))  # Adding smoothing
                    else:
                        score += np.log(1e-5)  # Adding smoothing for unseen features
                
                class_scores[label] = score
            
            # Select the class with the highest score
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
    def metrics(self, y_test, y_pred):
        """
        Calculates the accuracy of the Naive Bayes classifier.
        
        Args:
            y_test (list or numpy.ndarray): A 1D array containing the true class labels for the dependent variable.
            y_pred (list or numpy.ndarray): A 1D array containing the predicted class labels from the model.
        
        Returns:
            float: The accuracy of the model, which is the fraction of correct predictions.
        
        Accuracy is computed as:
            accuracy = (number of correct predictions) / (total number of predictions)
        """
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        
        accuracy = np.sum(y_test == y_pred) / len(y_test)
        return accuracy


    
if __name__ == "__main__":
    # Sample dataset
    X_train = [
        ["Sunny", "Hot", "High", "Weak"],
        ["Sunny", "Hot", "High", "Strong"],
        ["Overcast", "Hot", "High", "Weak"],
        ["Rainy", "Mild", "High", "Weak"],
        ["Rainy", "Cool", "Normal", "Weak"],
        ["Rainy", "Cool", "Normal", "Strong"],
        ["Overcast", "Cool", "Normal", "Strong"],
        ["Sunny", "Mild", "High", "Weak"],
        ["Sunny", "Cool", "Normal", "Weak"],
        ["Rainy", "Mild", "Normal", "Weak"],
        ["Sunny", "Mild", "Normal", "Strong"],
        ["Overcast", "Mild", "High", "Strong"],
        ["Overcast", "Hot", "Normal", "Weak"],
        ["Rainy", "Mild", "High", "Strong"]
    ]
    y_train = ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]

    X_test = [["Sunny", "Cool", "High", "Strong"]]
    
    # Train the Naive Bayes Classifier
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.fit(X_train, y_train)
    
    # Make predictions
    predictions = nb_classifier.predict(X_test)
    
    # Print the results
    print("Predicted class for the test samples:", predictions)

        
        