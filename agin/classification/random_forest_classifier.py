import numpy as np
import pandas as pd
from collections import Counter
from .decision_tree_classifier import DecisionTreeClassifier
class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True, random_state=None):
        """
        Initialize the Random Forest classifier.

        Args:
            n_estimators (int): Number of trees in the forest. Default is 100.
            max_depth (int): Maximum depth of each tree. If None, nodes are expanded until 
                all leaves are pure or contain less than min_samples_split samples. Default is None.
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

    def fit(self, x_train, y_train):
        """
        Build a forest of trees from the training set (x_train, y).

        Args:
            x_train (np.ndarray or pd.DataFrame): Training data features.
            y_train (np.ndarray or pd.Series): Training data labels.

        Returns:
            self: The fitted Random Forest classifier.
        """
        # Validate and standardize input data
        x_train, y_train = self.validate_data(x_train, y_train)
        self.n_classes = len(np.unique(y_train))
        n_samples, n_features = x_train.shape
        
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
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            
            # Create bootstrap sample if bootstrap=True
            if self.bootstrap:
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                sample_X = x_train[indices]
                sample_y = y_train[indices]
            else:
                sample_X = x_train
                sample_y = y_train
            
            # Train the tree on the bootstrap sample
            tree.fit(sample_X, sample_y)
            self.trees.append(tree)
        
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
        
        # Get predictions from all trees
        predictions = np.array([tree.predict(x_test) for tree in self.trees])
        
        # Take majority vote for final prediction
        final_predictions = []
        for sample_pred in predictions.T:
            counter = Counter(sample_pred)
            final_predictions.append(counter.most_common(1)[0][0])
            
        return np.array(final_predictions)

    def predict_probabilities(self, X):
        """
        Predict class probabilities for samples in X.

        Args:
            X (np.ndarray or pd.DataFrame): Samples to make predictions for.

        Returns:
            np.ndarray: Probability of each class for each sample in X.
        """
        # Validate and standardize input data
        X = self.validate_data(X)[0]
        
        # Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Calculate probabilities
        probabilities = []
        for sample_pred in predictions.T:
            counter = Counter(sample_pred)
            proba = [counter[i] / len(self.trees) for i in range(self.n_classes)]
            probabilities.append(proba)
            
        return np.array(probabilities)

    def validate_data(self, X, y=None):
        """
        Validate and convert input data into a standardized format (numpy arrays).

        Args:
            X (np.ndarray or pd.DataFrame): Input features.
            y (np.ndarray or pd.DataFrame, optional): Input labels.

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

    def feature_importance(self, X, y):
        """
        Calculate feature importance using mean decrease in impurity.

        Args:
            X (np.ndarray or pd.DataFrame): Training data features.
            y (np.ndarray or pd.Series): Training data labels.

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
            weighted (bool): Whether to weight importance by node samples.

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
    