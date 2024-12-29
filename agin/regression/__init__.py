from .linear_regression import LinearRegression
from .multilinear_regression import MultilinearRegression
from .polynomial_regression import PolynomialRegression
allowed_classes = [
    "LinearRegression", 
    "MultilinearRegression",
    "PolynomialRegression"
    ]
__all__ = allowed_classes
