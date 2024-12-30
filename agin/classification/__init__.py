from .logistic_regression import LogisticRegression
from .naive_bayes import NaiveBayesClassifier
from .knn_classifier import KNNClassifier
from .linear_svm import LinearSVMClassifier
allowed_classes = [
    "LogisticRegression",
    "NaiveBayesClassifier",
    "KNNClassifier",
    "LinearSVMClassifier"
    ]
__all__ = allowed_classes
