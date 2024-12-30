from .logistic_regression import LogisticRegression
from .naive_bayes import NaiveBayesClassifier
from .knn_classifier import KNNClassifier
from .linear_svm import LinearSVMClassifier
from .nonlinear_svm import NonLinearSVM
allowed_classes = [
    "LogisticRegression",
    "NaiveBayesClassifier",
    "KNNClassifier",
    "LinearSVMClassifier",
    "NonLinearSVM"
    ]
__all__ = allowed_classes
