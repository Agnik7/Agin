import numpy as np
import pandas as pd
from collections import Counter

class KNNClassifier:
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean', random_state=None):
        """
        Initialize the K-Nearest Neighbors classifier.

        Args:
            n_neighbors (int): Number of neighbors to consider for classification. 
                This determines how many closest data points influence the prediction. Default is 5.
            weights (str): Weight function used in prediction. Possible values:
                - 'uniform': All neighbors contribute equally to the prediction.
                - 'distance': Closer neighbors contribute more to the prediction, inversely proportional to their distance.
                Default is 'uniform'.
            metric (str): Distance metric to use for calculating the similarity between points. Possible values:
                - 'euclidean': Uses Euclidean distance (straight-line distance in space).
                - 'manhattan': Uses Manhattan distance (sum of absolute differences across dimensions).
                Default is 'euclidean'.
            random_state (int): Seed for random number generation to ensure reproducibility. Default is None.
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        
        # Set the random seed if provided
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, x_train, y_train):
        """
        Fit the K-Nearest Neighbors classifier using the training dataset.

        This method stores the training data, which will later be used to calculate distances and make predictions.
        
        Args:
            x_train (np.ndarray or pd.DataFrame): Training data features.
            y_train (np.ndarray or pd.DataFrame): Corresponding labels for the training data.
        
        Returns:
            self: The fitted KNN classifier object.
        """
        # Validate and standardize input data
        x_train, y_train = self.validate_data(x_train, y_train)

        # Store training data as class attributes
        self.x_train = x_train
        self.y_train = y_train
        
        return self

    def predict(self, x_test):
        """
        Predict the class labels for the provided test data.

        For each test sample, the method calculates distances to all training samples,
        identifies the k nearest neighbors, and determines the majority or weighted vote for classification.
        
        Args:
            x_test (np.ndarray or pd.DataFrame): Test data to classify.
        
        Returns:
            np.ndarray: Predicted class labels for each test sample.
        """
        # Validate and standardize input data
        x_test = self.validate_data(x_test)[0]

        predictions = []  # List to store predictions for each test sample
        
        # Iterate over each test sample
        for x in x_test:
            # Compute distances from the test sample to all training samples
            distances = self.calculate_distances(x)
            
            # Identify the indices of the k nearest neighbors
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_neighbors = self.y_train[nearest_indices]

            # Determine prediction based on weights
            if self.weights == 'distance':
                # Compute weights as the inverse of distances, with a small constant to avoid division by zero
                weights = 1 / (distances[nearest_indices] + 1e-8)
                
                # Weighted voting: Aggregate the weighted votes of the neighbors
                votes = {}
                for neighbor, weight in zip(nearest_neighbors, weights):
                    votes[neighbor] = votes.get(neighbor, 0) + weight
                prediction = max(votes.items(), key=lambda x: x[1])[0]
            else:
                # Uniform voting: Determine the majority class among the neighbors
                prediction = Counter(nearest_neighbors).most_common(1)[0][0]

            predictions.append(prediction)  # Append the prediction to the results
        
        return np.array(predictions)

    def calculate_distances(self, x):
        """
        Calculate distances between a single test sample and all training samples.

        Depending on the specified distance metric (Euclidean or Manhattan), this method
        computes the similarity or difference between the test sample and all training samples.
        
        Args:
            x (np.ndarray): A single test sample.
        
        Returns:
            np.ndarray: An array of distances from the test sample to each training sample.
        """
        if self.metric == 'euclidean':
            # Compute Euclidean distance
            return np.sqrt(np.sum((self.x_train - x) ** 2, axis=1))
        elif self.metric == 'manhattan':
            # Compute Manhattan distance
            return np.sum(np.abs(self.x_train - x), axis=1)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def validate_data(self, X, y=None):
        """
        Validate and convert input data into a standardized format (numpy arrays).

        Ensures that both features (X) and labels (y) are numpy arrays, which simplifies computations.
        If y is provided, it also ensures it has the correct shape.
        
        Args:
            X (np.ndarray or pd.DataFrame): Input features.
            y (np.ndarray or pd.DataFrame, optional): Corresponding target labels.
        
        Returns:
            tuple or np.ndarray: Validated and standardized input features (and labels, if provided).
        """
        # Convert features to numpy array if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Convert labels to numpy array and flatten if provided
        if y is not None:
            if isinstance(y, pd.DataFrame):
                y = y.values
            y = y.ravel()
            return X, y
        return X,

    def metrics(self, y_pred, y_test):
        """
        Calculate performance metrics to evaluate the classifier.

        This method computes metrics such as accuracy, precision, recall, and F1-score
        to assess how well the classifier performs on a given test set.
        
        Args:
            y_pred (np.ndarray): Predicted labels from the classifier.
            y_test (np.ndarray): True labels for the test set.
        
        Returns:
            tuple: A tuple containing the following metrics:
                - accuracy (float): Fraction of correctly predicted samples.
                - precision (float): Proportion of true positives among predicted positives.
                - recall (float): Proportion of true positives among actual positives.
                - f1_score (float): Harmonic mean of precision and recall.
        """
        y_pred = np.array(y_pred).ravel()
        y_test = np.array(y_test).ravel()
        
        # Calculate true positives, false positives, true negatives, false negatives
        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        tn = np.sum((y_pred == 0) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        
        # Compute performance metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return accuracy, precision, recall, f1_score

    def get_neighbors(self, x):
        """
        Identify the k-nearest neighbors for a given input sample.

        This method computes the distances between the input sample and all training samples,
        then identifies the indices and distances of the k nearest neighbors.
        
        Args:
            x (np.ndarray): Input sample to find neighbors for.
        
        Returns:
            tuple: A tuple containing:
                - Indices of the k-nearest neighbors.
                - Distances to these neighbors.
        """
        x = self.validate_data(x)[0]

        # Ensure input sample has the correct shape
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        # Calculate distances to all training samples
        distances = self.calculate_distances(x[0])

        # Identify the indices of the k nearest neighbors
        nearest_indices = np.argsort(distances)[:self.n_neighbors]
        
        return nearest_indices, distances[nearest_indices]
