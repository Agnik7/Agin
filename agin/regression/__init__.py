from .linear_regression import LinearRegression
from .multilinear_regression import MultilinearRegression
from .polynomial_regression import PolynomialRegression
from .knn_regressor import KNNRegressor
from .decision_tree_regressor import DecisionTreeRegressor
allowed_classes = [
    "LinearRegression", 
    "MultilinearRegression",
    "PolynomialRegression",
    "KNNRegressor",
    "DecisionTreeRegressor"
    ]
__all__ = allowed_classes
