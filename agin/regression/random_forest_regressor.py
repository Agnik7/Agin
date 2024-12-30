import numpy as np
import pandas as pd
from collections import Counter
from .decision_tree_regressor import DecisionTreeRegressor
class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True, random_state=None):
        """
        Initialize the Random Forest regressor.

        Args:
            n_estimators (int): Number of trees in the forest. Default is 100.
            max_depth (int): Maximum depth of each tree. If None, nodes are expanded until 
                all leaves contain less than min_samples_split samples. Default is None.
            min_samples_split (int): Minimum number of samples required to split an internal node.
                Default is 2.
            min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
                Default is 1.
            max_features (str or int): Number of features to consider when looking for the best split.
                If 'sqrt', then max_features=sqrt(n_features).
                If 'log2', then max_features=log2(n_features).
                If int, then consider max_features at each split. Default is 'sqrt'.
            bootstrap (bool): Whether bootstrap samples are used when building trees.
                Default is True.
            random_state (int): Seed for random number generation. Default is None.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y):
        """
        Build a forest of trees from the training set (X, y).

        Args:
            X (np.ndarray or pd.DataFrame): Training data features.
            y (np.ndarray or pd.Series): Training data target values.

        Returns:
            self: The fitted Random Forest regressor.
        """
        # Validate and standardize input data
        X, y = self.validate_data(X, y)
        n_samples, n_features = X.shape
        
        # Calculate max_features if string
        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                self.max_features_value = int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                self.max_features_value = int(np.log2(n_features))
        else:
            self.max_features_value = min(self.max_features, n_features)
        
        # Build each tree in the forest
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            
            # Create bootstrap sample if bootstrap=True
            if self.bootstrap:
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                sample_X = X[indices]
                sample_y = y[indices]
            else:
                sample_X = X
                sample_y = y
            
            # Train the tree on the bootstrap sample
            tree.fit(sample_X, sample_y)
            self.trees.append(tree)
        
        return self

    def predict(self, X):
        """
        Predict regression target for X.

        Args:
            X (np.ndarray or pd.DataFrame): Samples to make predictions for.

        Returns:
            np.ndarray: Predicted target values for each sample in X.
        """
        # Validate and standardize input data
        X = self.validate_data(X)[0]
        
        # Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Take mean of predictions
        return np.mean(predictions, axis=0)

    def validate_data(self, X, y=None):
        """
        Validate and convert input data into a standardized format (numpy arrays).

        Args:
            X (np.ndarray or pd.DataFrame): Input features.
            y (np.ndarray or pd.DataFrame, optional): Input target values.

        Returns:
            tuple or np.ndarray: Validated and standardized input data.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
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

    def feature_importance(self, X, y):
        """
        Calculate feature importance using mean decrease in variance.

        Args:
            X (np.ndarray or pd.DataFrame): Training data features.
            y (np.ndarray or pd.Series): Training data target values.

        Returns:
            np.ndarray: Importance of each feature.
        """
        X, y = self.validate_data(X, y)
        n_features = X.shape[1]
        
        # Initialize feature importance array
        importance = np.zeros(n_features)
        
        # Sum importance from each tree
        for tree in self.trees:
            tree_importance = self.calculate_tree_feature_importance(tree.root, n_features)
            importance += tree_importance
        
        # Normalize importance
        importance = importance / self.n_estimators
        importance = importance / np.sum(importance)
        
        return importance

    def calculate_tree_feature_importance(self, node, n_features, weighted=True):
        """
        Recursively calculate feature importance for a single tree.

        Args:
            node: Current node in the tree.
            n_features (int): Total number of features.
            weighted (bool): Whether to weight importance by reduction in variance.

        Returns:
            np.ndarray: Feature importance for this tree.
        """
        importance = np.zeros(n_features)
        
        def recurse(node, importance):
            if node.value is not None:  # Leaf node
                return
            
            # Add importance of this split
            importance[node.feature] += 1
            
            # Recurse on children
            recurse(node.left, importance)
            recurse(node.right, importance)
        
        recurse(node, importance)
        return importance

    def get_bootstrap_indices(self, X):
        """
        Get the indices of samples that were used to train each tree.

        Args:
            X (np.ndarray or pd.DataFrame): Training data features.

        Returns:
            list: List of arrays containing indices used for each tree.
        """
        n_samples = len(X)
        bootstrap_indices = []
        
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_indices.append(indices)
        
        return bootstrap_indices

    def out_of_bag_score(self, X, y):
        """
        Calculate the out-of-bag (OOB) score for the training dataset.

        Args:
            X (np.ndarray or pd.DataFrame): Training data features.
            y (np.ndarray or pd.Series): Training data target values.

        Returns:
            float: Out-of-bag R-squared score.
        """
        X, y = self.validate_data(X, y)
        n_samples = X.shape[0]
        
        # Initialize predictions and counts
        predictions_sum = np.zeros(n_samples)
        n_predictions = np.zeros(n_samples)
        
        # Get predictions from each tree
        for i, tree in enumerate(self.trees):
            oob_mask = ~np.isin(np.arange(n_samples), self.get_bootstrap_indices(X)[i])
            if np.any(oob_mask):
                predictions = tree.predict(X[oob_mask])
                predictions_sum[oob_mask] += predictions
                n_predictions[oob_mask] += 1
        
        # Calculate final predictions
        valid_mask = n_predictions > 0
        if np.any(valid_mask):
            oob_predictions = predictions_sum[valid_mask] / n_predictions[valid_mask]
            oob_score = 1 - (np.sum((y[valid_mask] - oob_predictions) ** 2) / 
                           np.sum((y[valid_mask] - np.mean(y[valid_mask])) ** 2))
            return oob_score
        return 0.0