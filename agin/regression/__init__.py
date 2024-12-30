from .linear_regression import LinearRegression
from .multilinear_regression import MultilinearRegression
from .polynomial_regression import PolynomialRegression
from .knn_regressor import KNNRegressor
from .decision_tree_regressor import DecisionTreeRegressor
from .random_forest_regressor import RandomForestRegressor
allowed_classes = [
    "LinearRegression", 
    "MultilinearRegression",
    "PolynomialRegression",
    "KNNRegressor",
    "DecisionTreeRegressor",
    "RandomForestRegressor"
    ]
__all__ = allowed_classes
