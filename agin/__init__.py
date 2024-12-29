# This is the import line
from .utils.health import Health
from .regression.linear_regression import LinearRegression
from .regression.multilinear_regression import MultilinearRegression
from .regression.polynomial_regression import PolynomialRegression
# End of import line
allowed_classes = ["Health", "LinearRegression", "MultilinearRegression","PolynomialRegression"] # List of all public facing classes
__all__ = allowed_classes