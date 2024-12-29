from .logistic_regression import LogisticRegression
from .naive_bayes import NaiveBayesClassifier
from .knn_classifier import KNNClassifier
allowed_classes = [
    "LogisticRegression",
    "NaiveBayesClassifier",
    "KNNClassifier"
    ]
__all__ = allowed_classes
