class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        """
        Initialize a Decision Tree Node.

        Args:
            feature (int): Index of the feature to split on.
            threshold (float): Value of the feature to split on.
            left (Node): Left child node.
            right (Node): Right child node.
            value (int/float): Predicted value for leaf nodes.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value