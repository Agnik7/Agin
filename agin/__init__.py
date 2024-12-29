# This is the import line
from .utils.health import Health
from .regression.linear_regression import LinearRegression
from .regression.multilinear_regression import MultilinearRegression
from .regression.polynomial_regression import PolynomialRegression
from .regression import (
    LinearRegression,
    LogisticRegression,
    MultilinearRegression,
    PolynomialRegression
)
from .preprocessing import(
    MinMaxScaler
)
# End of import line
allowed_classes = [
    "Health", 
    "LinearRegression", 
    "MultilinearRegression",
    "PolynomialRegression",
    "LogisticRegression",
    "MinMaxScaler"
    ] # List of all public facing classes
__all__ = allowed_classes