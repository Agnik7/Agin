import numpy as np
import pandas as pd
from collections import Counter
from ..utils.node import Node

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=None):
        """
        Initialize the Decision Tree regressor.

        Args:
            max_depth (int): Maximum depth of the tree. If None, nodes are expanded until 
                all leaves contain less than min_samples_split samples. Default is None.
            min_samples_split (int): Minimum number of samples required to split an internal node.
                Default is 2.
            min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
                Default is 1.
            random_state (int): Seed for random number generation to ensure reproducibility.
                Default is None.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, x_train, y_train):
        """
        Build the decision tree using the training data.

        Args:
            x_train (np.ndarray or pd.DataFrame): Training data features.
            y_train (np.ndarray or pd.Series): Training data target values.

        Returns:
            self: The fitted Decision Tree regressor object.
        """
        # Validate and standardize input data
        self.x_train, self.y_train = self.validate_data(x_train, y_train)
        
        # Build the tree recursively
        self.root = self.grow_tree(self.x_train, self.y_train, depth=0)
        
        return self

    def predict(self, x_test):
        """
        Predict target values for samples in x_test.

        Args:
            x_test (np.ndarray or pd.DataFrame): Samples to make predictions for.

        Returns:
            np.ndarray: Predicted target value for each sample in x_test.
        """
        # Validate and standardize input data
        x_test = self.validate_data(x_test)[0]
        
        # Make predictions for each sample
        return np.array([self.traverse_tree(x, self.root) for x in x_test])

    def grow_tree(self, X, y, depth):
        """
        Recursively grow the decision tree by selecting the best splits.

        Args:
            X (np.ndarray): Feature matrix of the current node.
            y (np.ndarray): Target values of the current node.
            depth (int): Current depth in the tree.

        Returns:
            Node: A decision tree node (either a leaf or an internal node with splits).
        """
        n_samples, n_features = X.shape

        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split:
            leaf_value = self.mean_value(y)
            return Node(value=leaf_value)

        # Find the best split
        feature_idxs = np.arange(n_features)
        best_feature, best_threshold = self.best_split(X, y, feature_idxs)
        
        if best_feature is None:  # No valid split found
            leaf_value = self.mean_value(y)
            return Node(value=leaf_value)

        # Create child splits
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = ~left_idxs
        
        # Check min_samples_leaf criterion
        if np.sum(left_idxs) < self.min_samples_leaf or np.sum(right_idxs) < self.min_samples_leaf:
            leaf_value = self.mean_value(y)
            return Node(value=leaf_value)

        # Recursively grow the left and right subtrees
        left = self.grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self.grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def best_split(self, X, y, feature_idxs):
        """
        Find the best split for a node by minimizing MSE.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target values.
            feature_idxs (np.ndarray): Features to consider for splitting.

        Returns:
            tuple: Best feature index and threshold value for splitting.
        """
        best_mse = float('inf')
        best_feature = None
        best_threshold = None

        # Calculate parent MSE
        parent_mse = self.mse(y)

        # Try each feature
        for feature in feature_idxs:
            thresholds = np.unique(X[:, feature])
            
            # Try each threshold
            for threshold in thresholds:
                # Create children
                left_mask = X[:, feature] < threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                # Calculate MSE for children
                left_mse = self.mse(y[left_mask])
                right_mse = self.mse(y[right_mask])
                
                # Weighted average of children MSE
                n = len(y)
                n_l, n_r = np.sum(left_mask), np.sum(right_mask)
                weighted_mse = (n_l * left_mse + n_r * right_mse) / n
                
                # Update best split if this split is better
                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def mse(self, y):
        """
        Calculate the Mean Squared Error (MSE) of a node.

        Args:
            y (np.ndarray): Target values in the node.

        Returns:
            float: MSE value.
        """
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def mean_value(self, y):
        """
        Calculate the mean value for a leaf node.

        Args:
            y (np.ndarray): Target values in the node.

        Returns:
            float: Mean value.
        """
        return np.mean(y)

    def traverse_tree(self, x, node):
        """
        Traverse the decision tree to make a prediction for a single sample.

        Args:
            x (np.ndarray): Single sample to predict.
            node (Node): Current node in the tree.

        Returns:
            float: Predicted target value.
        """
        if node.value is not None:
            return node.value

        if x[node.feature] < node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)

    def validate_data(self, X, y=None):
        """
        Validate and convert input data into a standardized format (numpy arrays).

        Args:
            X (np.ndarray or pd.DataFrame): Input features.
            y (np.ndarray or pd.DataFrame, optional): Input target values.

        Returns:
            tuple or np.ndarray: Validated and standardized input data.
        """
        # Convert features to numpy array if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Convert target values to numpy array if provided
        if y is not None:
            if isinstance(y, (pd.DataFrame, pd.Series)):
                y = y.values
            y = y.ravel()
            return X, y
        return X,

    def metrics(self, y_pred, y_test):
        """
        Calculate performance metrics for the regressor.

        Args:
            y_pred (np.ndarray): Predicted target values.
            y_test (np.ndarray): True target values.

        Returns:
            tuple: Contains the following metrics:
                - mse (float): Mean Squared Error.
                - r2_score (float): R-squared score (coefficient of determination).
        """
        y_pred = np.array(y_pred).ravel()
        y_test = np.array(y_test).ravel()
        
        # Calculate MSE
        mse = np.mean((y_test - y_pred) ** 2)
        
        # Calculate R2 score
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        ss_res = np.sum((y_test - y_pred) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return mse, r2_score