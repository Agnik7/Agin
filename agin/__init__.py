# This is the import line
from .utils.health import Health
from .regression import (
    LinearRegression,
    MultilinearRegression,
    PolynomialRegression,
    KNNRegressor
)
from .preprocessing import(
    MinMaxScaler
)
from .classification import (
    LogisticRegression,
    NaiveBayesClassifier,
    KNNClassifier,
    LinearSVMClassifier
    
)

# End of import line
allowed_classes = [
    "Health", 
    "LinearRegression", 
    "MultilinearRegression",
    "PolynomialRegression",
    "MinMaxScaler",
    "LogisticRegression",
    "NaiveBayesClassifier",
    "KNNClassifier",
    "KNNRegressor",
    "LinearSVMClassifier"
    
    ] # List of all public facing classes
__all__ = allowed_classes