from .logistic_regression import LogisticRegression
from .naive_bayes import NaiveBayesClassifier
from .knn_classifier import KNNClassifier
<<<<<<< HEAD
from .linear_svm import LinearSVMClassifier
from .nonlinear_svm import NonLinearSVM
=======
from .decision_tree_classifier import DecisionTreeClassifier
>>>>>>> f17b7b2 (feat: implement Decision Tree Classifier)
allowed_classes = [
    "LogisticRegression",
    "NaiveBayesClassifier",
    "KNNClassifier",
<<<<<<< HEAD
    "LinearSVMClassifier",
    "NonLinearSVM"
=======
    "DecisionTreeClassifier"
>>>>>>> f17b7b2 (feat: implement Decision Tree Classifier)
    ]
__all__ = allowed_classes
