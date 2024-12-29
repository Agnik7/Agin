from .linear_regression import LinearRegression
from .multilinear_regression import MultilinearRegression
from .logistic_regression import LogisticRegression
from .polynomial_regression import PolynomialRegression
allowed_classes = [
    "LinearRegression", 
    "MultilinearRegression",
    "LogisticRegression",
    "PolynomialRegression"
    ]
__all__ = allowed_classes
