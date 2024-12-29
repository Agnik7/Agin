from .linear_regression import LinearRegression
from .multilinear_regression import MultilinearRegression
from .polynomial_regression import PolynomialRegression
from .knn_regressor import KNNRegressor
allowed_classes = [
    "LinearRegression", 
    "MultilinearRegression",
    "PolynomialRegression",
    "KNNRegressor"
    ]
__all__ = allowed_classes
