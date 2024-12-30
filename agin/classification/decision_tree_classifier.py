import numpy as np
import pandas as pd
from collections import Counter
from ..utils.node import Node
class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=None):
        """
        Initialize the Decision Tree classifier.

        Args:
            max_depth (int): Maximum depth of the tree. If None, nodes are expanded until 
                all leaves are pure or contain less than min_samples_split samples. Default is None.
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
            y_train (np.ndarray or pd.Series): Training data labels.

        Returns:
            self: The fitted Decision Tree classifier object.
        """
        # Validate and standardize input data
        self.x_train, self.y_train = self.validate_data(x_train, y_train)
        self.n_classes = len(np.unique(y_train))
        
        # Build the tree recursively
        self.root = self.grow_tree(self.x_train, self.y_train, depth=0)
        
        return self

    def predict(self, x_test):
        """
        Predict class labels for samples in x_test.

        Args:
            x_test (np.ndarray or pd.DataFrame): Samples to make predictions for.

        Returns:
            np.ndarray: Predicted class label for each sample in x_test.
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
            y (np.ndarray): Labels of the current node.
            depth (int): Current depth in the tree.

        Returns:
            Node: A decision tree node (either a leaf or an internal node with splits).
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_classes == 1:
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)

        # Find the best split
        feature_idxs = np.arange(n_features)
        best_feature, best_threshold = self.best_split(X, y, feature_idxs)
        
        if best_feature is None:  # No valid split found
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)

        # Create child splits
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = ~left_idxs
        
        # Check min_samples_leaf criterion
        if np.sum(left_idxs) < self.min_samples_leaf or np.sum(right_idxs) < self.min_samples_leaf:
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)

        # Recursively grow the left and right subtrees
        left = self.grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self.grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def best_split(self, X, y, feature_idxs):
        """
        Find the best split for a node by maximizing information gain.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Labels.
            feature_idxs (np.ndarray): Features to consider for splitting.

        Returns:
            tuple: Best feature index and threshold value for splitting.
        """
        best_gain = -1
        best_feature = None
        best_threshold = None

        # Calculate parent entropy
        parent_entropy = self.entropy(y)

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

                # Calculate information gain
                left_entropy = self.entropy(y[left_mask])
                right_entropy = self.entropy(y[right_mask])
                
                # Weighted average of children entropy
                n = len(y)
                n_l, n_r = np.sum(left_mask), np.sum(right_mask)
                child_entropy = (n_l * left_entropy + n_r * right_entropy) / n
                
                # Calculate information gain
                info_gain = parent_entropy - child_entropy
                
                # Update best split if this split is better
                if info_gain > best_gain:
                    best_gain = info_gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def entropy(self, y):
        """
        Calculate the entropy of a node.

        Args:
            y (np.ndarray): Labels in the node.

        Returns:
            float: Entropy value.
        """
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def most_common_label(self, y):
        """
        Find the most common label in a node.

        Args:
            y (np.ndarray): Labels in the node.

        Returns:
            int: Most common label.
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def traverse_tree(self, x, node):
        """
        Traverse the decision tree to make a prediction for a single sample.

        Args:
            x (np.ndarray): Single sample to predict.
            node (Node): Current node in the tree.

        Returns:
            int: Predicted class label.
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
            y (np.ndarray or pd.DataFrame, optional): Input labels.

        Returns:
            tuple or np.ndarray: Validated and standardized input data.
        """
        # Convert features to numpy array if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Convert labels to numpy array if provided
        if y is not None:
            if isinstance(y, (pd.DataFrame, pd.Series)):
                y = y.values
            y = y.ravel()
            return X, y
        return X,

    def metrics(self, y_pred, y_test):
        """
        Calculate performance metrics for the classifier.

        Args:
            y_pred (np.ndarray): Predicted labels.
            y_test (np.ndarray): True labels.

        Returns:
            tuple: Contains the following metrics:
                - accuracy (float): Overall prediction accuracy.
                - precision (float): Precision score.
                - recall (float): Recall score.
                - f1_score (float): F1 score.
        """
        y_pred = np.array(y_pred).ravel()
        y_test = np.array(y_test).ravel()
        
        # Calculate confusion matrix elements
        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        tn = np.sum((y_pred == 0) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return accuracy, precision, recall, f1_score