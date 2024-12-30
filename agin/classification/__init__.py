from .logistic_regression import LogisticRegression
from .naive_bayes import NaiveBayesClassifier
from .knn_classifier import KNNClassifier
from .linear_svm import LinearSVMClassifier
from .nonlinear_svm import NonLinearSVM
from .decision_tree_classifier import DecisionTreeClassifier
from .linear_svm import LinearSVMClassifier
from .nonlinear_svm import NonLinearSVM
from .decision_tree_classifier import DecisionTreeClassifier
from .random_forest_classifier import RandomForestClassifier
allowed_classes = [
    "LogisticRegression",
    "NaiveBayesClassifier",
    "KNNClassifier",
    "LinearSVMClassifier",
    "NonLinearSVM",
    "DecisionTreeClassifier",
    "RandomForestClassifier"
    ]
__all__ = allowed_classes
