import numpy as np
import pandas as pd
import copy

class KNNRegressor:
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean', random_state=None):
        """
        Initialize the K-Nearest Neighbors regressor.

        Args:
            n_neighbors (int): Number of neighbors to use for prediction. Default is 5.
                - A higher value results in smoother predictions but may lose local accuracy.
            weights (str): Weight function used in prediction. Possible values:
                - 'uniform': All neighbors contribute equally.
                - 'distance': Closer neighbors contribute more than distant ones.
                Default is 'uniform'.
            metric (str): Distance metric to use. Possible values:
                - 'euclidean': Straight-line distance between points.
                - 'manhattan': Sum of absolute differences along each dimension.
                Default is 'euclidean'.
            random_state (int): Seed for random number generation. Default is None.
                - Useful for reproducibility when using random elements.
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric

        # Set the random seed if specified
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, x_train, y_train):
        """
        Fit the k-nearest neighbors regressor from the training dataset.

        Args:
            x_train (np.ndarray or pd.DataFrame): Training data features.
                - Should be of shape (n_samples, n_features).
            y_train (np.ndarray or pd.Series): Target values.
                - Should be of shape (n_samples,).

        Returns:
            self: The fitted regressor instance.
        """
        # Convert input data to numpy arrays if needed
        x_train, y_train = self.validate_data(x_train, y_train)

        # Store the training data for later use during prediction
        self.x_train = x_train
        self.y_train = y_train

        return self

    def predict(self, x_test):
        """
        Predict target values for the provided data.

        Args:
            x_test (np.ndarray or pd.DataFrame): Test samples.
                - Should be of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted target values for each input sample.
        """
        # Validate and process the input data
        x_test = self.validate_data(x_test)[0]
        predictions = []

        # Loop over each test sample
        for x in x_test:
            # Calculate distances to all training samples
            distances = self.calculate_distances(x)

            # Identify indices of the k nearest neighbors
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_neighbors = self.y_train[nearest_indices]

            if self.weights == 'distance':
                # Calculate weights inversely proportional to distances
                weights = 1 / (distances[nearest_indices] + 1e-8)  # Add small constant to avoid division by zero
                weights = weights / np.sum(weights)  # Normalize weights to sum to 1
                prediction = np.sum(nearest_neighbors * weights)
            else:
                # Compute simple average for uniform weights
                prediction = np.mean(nearest_neighbors)

            # Append the prediction to the list
            predictions.append(prediction)

        return np.array(predictions)

    def calculate_distances(self, x):
        """
        Calculate distances between a test sample and all training samples.

        Args:
            x (np.ndarray): Single test sample.
                - Should be of shape (n_features,).

        Returns:
            np.ndarray: Array of distances to all training samples.
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
        Convert input data to numpy arrays and validate shapes.

        Args:
            X (np.ndarray or pd.DataFrame): Input features.
                - Can be a DataFrame or ndarray of shape (n_samples, n_features).
            y (np.ndarray or pd.Series, optional): Target values.
                - Can be a Series or ndarray of shape (n_samples,).

        Returns:
            tuple or np.ndarray: Processed input data.
        """
        # Convert DataFrame to numpy array if necessary
        if isinstance(X, pd.DataFrame):
            X = X.values
        if y is not None:
            # Convert Series to numpy array if necessary
            if isinstance(y, (pd.DataFrame, pd.Series)):
                y = y.values
            y = y.ravel()
            return X, y
        return X,

    def get_neighbors(self, x):
        """
        Get the k-nearest neighbors for a single sample.

        Args:
            x (np.ndarray): Input sample.
                - Should be of shape (n_features,).

        Returns:
            tuple: (indices of nearest neighbors, distances to nearest neighbors).
        """
        # Validate the input sample
        x = self.validate_data(x)[0]

        # Reshape if the input is a 1D array
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        # Calculate distances to all training samples
        distances = self.calculate_distances(x[0])

        # Get indices of the k nearest neighbors
        nearest_indices = np.argsort(distances)[:self.n_neighbors]

        return nearest_indices, distances[nearest_indices]

    def score(self, X, y):
        """
        Calculate the coefficient of determination (R^2) for the prediction.

        Args:
            X (np.ndarray or pd.DataFrame): Test samples.
                - Should be of shape (n_samples, n_features).
            y (np.ndarray or pd.Series): True target values.
                - Should be of shape (n_samples,).

        Returns:
            float: R^2 score indicating model performance.
                - Higher values indicate better performance, with 1.0 being perfect.
        """
        # Validate and process test data and true labels
        X, y = self.validate_data(X, y)
        y_pred = self.predict(X)

        # Calculate residual sum of squares and total sum of squares
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        # Compute R^2 score
        r2 = 1 - (ss_res / (ss_tot + 1e-8))  # Add small constant to avoid division by zero

        return r2

    def metrics(self, y_pred, y_test):
        """
        Calculate various regression metrics including MSE, RMSE, MAE, and R^2 score.

        Args:
            y_pred (np.ndarray): Predicted target values.
                - Should be of shape (n_samples,).
            y_test (np.ndarray): True target values.
                - Should be of shape (n_samples,).

        Returns:
            tuple: (mse r2)
                - mse: Mean Squared Error.
                - r2_score: The R-squared value of the model, indicating the proportion of variance explained.
        """
        # Ensure inputs are numpy arrays
        y_test = np.array(y_test).ravel()
        y_pred = np.array(y_pred).ravel()

        # Calculate mean squared error
        mse = np.mean((y_test - y_pred) ** 2)

        # Calculate R^2 score
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2_score = 1 - (ss_res / (ss_tot + 1e-8))

        return mse, r2_score
