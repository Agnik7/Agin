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
    
    def fit(self, x_train, y_train):
        """
        Trains the Naive Bayes model by calculating class probabilities and feature likelihoods.
        
        Args:
            x_train (list or numpy.ndarray): A 2D array containing the training data for independent variables.
            y_train (list or numpy.ndarray): A 1D array containing the true class labels for the dependent variable.
        
        Returns:
            None: This method updates the model's class probabilities and feature likelihoods.
        
        This method calculates:
            - class probabilities: P(class)
            - feature likelihoods: P(feature | class)
        """
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        
        # Calculate class probabilities: P(class)
        class_labels = np.unique(y_train)
        class_probs = {}
        for label in class_labels:
            class_probs[label] = np.sum(y_train == label) / len(y_train)
        
        # Calculate feature probabilities: P(feature | class)
        feature_probs = {}
        for label in class_labels:
            X_class = x_train[y_train == label]
            feature_probs[label] = {}
            for feature_idx in range(x_train.shape[1]):
                feature_values = X_class[:, feature_idx]
                unique_vals, counts = np.unique(feature_values, return_counts=True)
                feature_probs[label][feature_idx] = dict(zip(unique_vals, counts / len(feature_values)))
        
        self.class_probs = class_probs
        self.feature_probs = feature_probs
    
    def predict(self, x_train):
        """
        Predicts the class label for each sample in the test data using the trained Naive Bayes model.
        
        Args:
            x_train (list or numpy.ndarray): A 2D array containing test data for independent variables.
        
        Returns:
            numpy.ndarray: A 1D array of predicted class labels for each sample.
        
        The prediction is based on the formula:
            P(class | X) ∝ P(class) * P(X | class)
        where X is the input data and P(X | class) is the product of individual feature likelihoods.
        """
        x_test = np.array(x_train)
        predictions = []
        
        for sample in x_test:
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
    
    def metrics(self, y_pred, y_test):
        """
        Calculates the accuracy, precision, recall, and F1 score of the Naive Bayes classifier.
        
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
        # Precision, Recall, F1 Score (assuming positive class is 'Yes' and negative class is 'No')
        true_positives = np.sum((y_test == 'Yes') & (y_pred == 'Yes'))
        false_positives = np.sum((y_test == 'No') & (y_pred == 'Yes'))
        false_negatives = np.sum((y_test == 'Yes') & (y_pred == 'No'))
        true_negatives = np.sum((y_test == 'No') & (y_pred == 'No'))
        
        # Precision
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
        
        # Recall
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
        
        # F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        
        return accuracy, precision, recall, f1_score