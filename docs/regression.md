# **Regression**

The `regression` module contains implementations of regression models. Currently, the package supports:

- **Linear Regression**
- **Multilinear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **ElasticNet Regression**
- **Polynomial Regression**
- **K-Nearest Neighbors (KNN) Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**

## **Linear Regression**
The `LinearRegression` class provides methods to train a linear regression model, make predictions, and evaluate the model's performance. It can be imported directly from `agin` or from `agin.regression`.

### **Usage**
The `LinearRegression` class can be imported directly from the `agin` package or from the `agin.regression` module:

```python
from agin import LinearRegression
# or
from agin.regression import LinearRegression
```

#### **Example**

```python
# Option 1: Importing directly from agin
from agin import LinearRegression

# Option 2: Importing from agin.regression
from agin.regression import LinearRegression

# Training data
x_train = [1, 2, 3, 4, 5]
y_train = [2, 4, 6, 8, 10]

# Initialize the model
model = LinearRegression()

# Fit the model
model.fit(x_train, y_train)

# Predict using the model
x_test = [0, 1, 2, 3, 4, 5]
y_test = [1, 3, 5, 7, 9, 11]

y_pred = model.predict(x_test)

print("Predictions:", y_pred)

# Evaluate the model metrics
mse, r2 = model.metrics(y_pred, y_test)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
```

### **Methods**
#### **`fit(x_train, y_train)`**
   - Trains the model based on the input data.
   - **Args**: `x_train` (list or numpy.ndarray), `y_train` (list or numpy.ndarray)
   - **Returns**: None

#### **`predict(x_test)`**
   - Predicts outputs for given input data.
   - **Args**: `x_test` (list or numpy.ndarray)
   - **Returns**: List of predicted values

#### **`metrics(y_pred, y_test)`**
   - Calculates evaluation metrics like Mean Squared Error (MSE) and R² Score.
   - **Args**: `y_pred` (list or numpy.ndarray), `y_test` (list or numpy.ndarray)
   - **Returns**: Tuple containing MSE and R² Score.

## **Multilinear Regression**
The `MultilinearRegression` class provides methods to train a multilinear regression model, make predictions, and evaluate the model's performance. It can be imported directly from `agin` or from `agin.regression`.

### **Usage**
The `MultilinearRegression` class can be imported directly from the `agin` package or from the `agin.regression` module:

```python
from agin import MultilinearRegression
# or
from agin.regression import MultilinearRegression
```

#### **Example**

```python
# Option 1: Importing directly from agin
from agin import MultilinearRegression

# Option 2: Importing from agin.regression
from agin.regression import MultilinearRegression

# Training data
x_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
y_train = [3, 5, 7, 9]

# Initialize the model
model = MultilinearRegression()

# Fit the model
model.fit(x_train, y_train)

# Predict using the model
x_test = [[5, 6], [6, 7]]

y_pred = model.predict(x_test)

print("Predictions:", y_pred)

# Evaluate the model metrics
y_test = [11, 13]
mse, r2 = model.metrics(y_pred, y_test)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
```

### **Methods**
#### **`fit(x_train, y_train)`**
   - Trains the model using the Normal Equation.
   - **Args**: `x_train` (list or numpy.ndarray), `y_train` (list or numpy.ndarray)
   - **Returns**: None

#### **`predict(x_test)`**
   - Predicts outputs for given input data.
   - **Args**: `x_test` (list or numpy.ndarray)
   - **Returns**: List of predicted values

#### **`metrics(y_pred, y_test)`**
   - Calculates evaluation metrics like Mean Squared Error (MSE) and R² Score.
   - **Args**: `y_pred` (list or numpy.ndarray), `y_test` (list or numpy.ndarray)
   - **Returns**: Tuple containing MSE and R² Score.

## **Ridge Regression**
The `RidgeRegression` class provides methods to train a ridge regression model, make predictions, and evaluate the model's performance using L2 regularization. Unlike linear regression, which minimizes only the sum of squared residuals, ridge regression adds a penalty for large coefficients, helping to mitigate overfitting and improve predictions when multicollinearity or small sample sizes are present. It can be imported directly from `agin` or from `agin.regression`.

### **Usage**
The `RidgeRegression` class can be imported directly from the `agin` package or from the `agin.regression` module:

```python
from agin import RidgeRegression
# or
from agin.regression import RidgeRegression
```

#### **Example**

```python
# Option 1: Importing directly from agin
from agin import RidgeRegression

# Option 2: Importing from agin.regression
from agin.regression import RidgeRegression

# Training data
x_train = [1, 2, 3, 4, 5]
y_train = [2, 4, 6, 8, 10]

# Initialize the model with regularization parameter alpha
model = RidgeRegression(alpha=1.0)

# Fit the model
model.fit(x_train, y_train)

# Predict using the model
x_test = [0, 1, 2, 3, 4, 5]
y_test = [1, 3, 5, 7, 9, 11]

y_pred = model.predict(x_test)

print("Predictions:", y_pred)

# Evaluate the model metrics
mse, r2_score = model.metrics(y_pred, y_test)
print("Mean Squared Error:", mse)
print("R2 Score:", r2_score)
```

### **Methods**
#### **`fit(x_train, y_train)`**
   - Trains the model using the closed-form solution for ridge regression.
   - **Args**: `x_train` (list or numpy.ndarray), `y_train` (list or numpy.ndarray)
   - **Returns**: None

#### **`predict(x_test)`**
   - Predicts outputs for given input data.
   - **Args**: `x_test` (list or numpy.ndarray)
   - **Returns**: List of predicted values

#### **`metrics(y_pred, y_test)`**
   - Calculates evaluation metrics like Mean Squared Error (MSE) and R² Score.
   - **Args**: `y_pred` (list or numpy.ndarray), `y_test` (list or numpy.ndarray)
   - **Returns**: Tuple containing MSE and R² Score.

#### **Attributes**
- **`alpha`**: Regularization parameter controlling the strength of the L2 penalty.
- **`slope`**: Coefficients of the model.
- **`intercept`**: Intercept (bias term) of the model.

## **Lasso Regression**

The `LassoRegression` class implements the Lasso regression model using the coordinate descent algorithm. It supports training the model, making predictions, and evaluating its performance. The model introduces L1 regularization, which encourages sparsity in the coefficients by penalizing the absolute values of the coefficients. This makes it particularly useful for feature selection.

### **Usage**
The `LassoRegression` class can be imported directly from the `agin` package or from the `agin.regression` module:

```python
from agin import LassoRegression
# or
from agin.regression import LassoRegression
```

#### **Example**

```python
# Option 1: Importing directly from agin
from agin import LassoRegression

# Option 2: Importing from agin.regression
from agin.regression import LassoRegression

# Training data
x_train = [1, 2, 3, 4, 5]
y_train = [2.1, 4.2, 6.1, 8.2, 10.1]

# Initialize the model with regularization parameter alpha
model = LassoRegression(alpha=0.5, max_iter=1000, tol=1e-4)

# Fit the model
model.fit(x_train, y_train)

# Predict using the model
x_test = [0, 1, 2, 3, 4, 5]
y_test = [1.1, 3.2, 5.1, 7.2, 9.1, 11.2]

y_pred = model.predict(x_test)

print("Predictions:", y_pred)

# Evaluate the model metrics
mse, r2_score = model.metrics(y_pred, y_test)
print("Mean Squared Error:", mse)
print("R2 Score:", r2_score)
```

### **Methods**
#### **`fit(x_train, y_train)`**
   - Trains the model using the coordinate descent algorithm for Lasso regression.
   - **Args**:
     - `x_train` (list or numpy.ndarray): Training feature data (X values).
     - `y_train` (list or numpy.ndarray): Target data (Y values).
   - **Returns**: None

#### **`predict(x_test)`**
   - Predicts outputs for given input feature data.
   - **Args**:
     - `x_test` (list or numpy.ndarray): Test feature data (X values).
   - **Returns**: numpy.ndarray of predicted values.

#### **`metrics(y_pred, y_test)`**
   - Calculates evaluation metrics like Mean Squared Error (MSE) and R² Score.
   - **Args**:
     - `y_pred` (list or numpy.ndarray): Predicted values.
     - `y_test` (list or numpy.ndarray): Actual values (ground truth).
   - **Returns**: Tuple containing:
     - `MSE` (float): Mean Squared Error of the model.
     - `R²` (float): R-squared value, indicating the proportion of variance explained by the model.

#### **Attributes**
- **`alpha`**: Regularization parameter that controls the strength of the L1 penalty.
- **`max_iter`**: Maximum number of iterations for the optimization algorithm.
- **`tol`**: Tolerance for convergence of the optimization.
- **`slope`**: Coefficients of the model (excluding the intercept).
- **`intercept`**: Intercept (bias term) of the model.

## **ElasticNet Regression**
The `ElasticNetRegression` class implements the Elastic Net model, which combines the strengths of both Ridge (L2 regularization) and Lasso (L1 regularization) regression. It is particularly useful when dealing with datasets that exhibit multicollinearity and when feature selection is desired.

### **Usage**
The `ElasticNetRegression` class can be imported directly from the `agin` package or from the `agin.regression` module:

```python
from agin import ElasticNetRegression
# or
from agin.regression import ElasticNetRegression
```

#### **Example**

```python
# Option 1: Importing directly from agin
from agin import ElasticNetRegression

# Option 2: Importing from agin.regression
from agin.regression import ElasticNetRegression

# Training data
x_train = [1, 2, 3, 4, 5]
y_train = [2.5, 4.0, 6.5, 8.0, 10.5]

# Initialize the model with alpha and l1_ratio
model = ElasticNetRegression(alpha=1.0, l1_ratio=0.5)

# Fit the model
model.fit(x_train, y_train)

# Predict using the model
x_test = [0, 1, 2, 3, 4, 5]
y_test = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]

y_pred = model.predict(x_test)

print("Predictions:", y_pred)

# Evaluate the model metrics
mse, r2_score = model.metrics(y_pred, y_test)
print("Mean Squared Error:", mse)
print("R2 Score:", r2_score)
```

### **Methods**
#### **`fit(x_train, y_train)`**
   - Trains the model using a combination of L1 and L2 penalties.
   - **Args**:
     - `x_train` (list or numpy.ndarray): Training feature data (X values).
     - `y_train` (list or numpy.ndarray): Target data (Y values).
   - **Returns**: None

#### **`predict(x_test)`**
   - Predicts outputs for given input data.
   - **Args**: `x_test` (list or numpy.ndarray)
   - **Returns**: List of predicted values

#### **`metrics(y_pred, y_test)`**
   - Calculates evaluation metrics like Mean Squared Error (MSE) and R² Score.
   - **Args**:
     - `y_pred` (list or numpy.ndarray): Predicted values.
     - `y_test` (list or numpy.ndarray): Actual values (ground truth).
   - **Returns**: Tuple containing MSE and R² Score.

#### **Attributes**
- **`alpha`**: Regularization strength parameter. Higher values imply stronger regularization.
- **`l1_ratio`**: The mixing ratio between L1 and L2 regularization. Ranges from 0 (Ridge regression) to 1 (Lasso regression).
- **`slope`**: Coefficients of the model (excluding the intercept).
- **`intercept`**: Intercept (bias term) of the model.

## **Polynomial Regression**
The `PolynomialRegression` class provides methods to train a polynomial regression model of any degree, make predictions, and evaluate the model's performance. It can be imported directly from `agin` or from the `agin.regression`.

### **Usage**
The `PolynomialRegression` class can be imported directly from the `agin` package or from the `agin.regression` module:

```python
from agin import PolynomialRegression
# or
from agin.regression import PolynomialRegression
```

#### **Example**

```python
# Option 1: Importing directly from agin
from agin import PolynomialRegression

# Option 2: Importing from agin.regression
from agin.regression import PolynomialRegression

# Training data
x_train = [1, 2, 3, 4, 5]
y_train = [2, 5, 10, 17, 26]

# Initialize the model
model = PolynomialRegression(degree=2)

# Fit the model
model.fit(x_train, y_train)

# Predict using the model
x_test = [6, 7, 8]

y_pred = model.predict(x_test)

print("Predictions:", y_pred)

# Evaluate the model metrics
y_test = [37, 50, 65]
mse, r2 = model.metrics(y_pred, y_test)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
```

### **Methods**
#### **`fit(x_train, y_train)`**
   - Trains the model using the Normal Equation for polynomial features.
   - **Args**: `x_train` (list or numpy.ndarray), `y_train` (list or numpy.ndarray)
   - **Returns**: None

#### **`predict(x_test)`**
   - Predicts outputs for given input data.
   - **Args**: `x_test` (list or numpy.ndarray)
   - **Returns**: List of predicted values

#### **`metrics(y_pred, y_test)`**
   - Calculates evaluation metrics like Mean Squared Error (MSE) and R² Score.
   - **Args**: `y_pred` (list or numpy.ndarray), `y_test` (list or numpy.ndarray)
   - **Returns**: Tuple containing MSE and R² Score.

## **K-Nearest Neighbors (KNN) Regression**
The `KNNRegressor` class provides a flexible implementation of the k-nearest neighbors algorithm for regression tasks. It supports both uniform and distance-based weighting schemes.

### **Usage**
The `KNNRegressor` class can be imported directly from the `agin` package or from the `agin.regression` module:

```python
from agin import KNNRegressor
# or
from agin.regression import KNNRegressor
```

#### **Example**

```python
# Option 1: Importing directly from agin
from agin import KNNRegressor

# Option 2: Importing from agin.regression
from agin.regression import KNNRegressor

# Training data
x_train = [[1.0], [2.0], [3.0], [4.0], [5.0]]
y_train = [1.5, 2.5, 3.5, 4.5, 5.5]

# Initialize the model
model = KNNRegressor(n_neighbors=3, weights='distance', metric='euclidean')

# Fit the model
model.fit(x_train, y_train)

# Predict using the model
x_test = [[1.5], [2.5], [3.5]]
y_pred = model.predict(x_test)

print("Predictions:", y_pred)

# Evaluate the model metrics
y_test = [1.8, 2.8, 3.8]
mse, r2 = model.metrics(y_pred, y_test)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
```

### **Methods**
#### **`fit(x_train, y_train)`**
   - Stores the training data for use during prediction.
   - **Args**: `x_train` (numpy.ndarray or pandas.DataFrame), `y_train` (numpy.ndarray or pandas.Series)
   - **Returns**: The fitted regressor instance.

#### **`predict(x_test)`**
   - Predicts target values for given test samples.
   - **Args**: `x_test` (numpy.ndarray or pandas.DataFrame)
   - **Returns**: numpy.ndarray of predicted target values.

#### **`metrics(y_pred, y_test)`**
   - Calculates evaluation metrics like Mean Squared Error (MSE) and R² Score.
   - **Args**: `y_pred` (numpy.ndarray), `y_test` (numpy.ndarray)
   - **Returns**: Tuple containing MSE and R² Score.

#### **`score(x_test, y_test)`**
   - Computes the coefficient of determination (R²) score.
   - **Args**: `x_test` (numpy.ndarray or pandas.DataFrame), `y_test` (numpy.ndarray or pandas.Series)
   - **Returns**: R² score as a float.

## **Decision Tree Regressor**

The `DecisionTreeRegressor` class provides a regression model using a decision tree algorithm. It supports custom configurations like maximum depth, minimum samples per split, and minimum samples per leaf for controlling tree growth and preventing overfitting.

### **Usage**
The `DecisionTreeRegressor` class can be imported directly from the `agin` package or from the `agin.regression` module:

```python
from agin import DecisionTreeRegressor
# or
from agin.regression import DecisionTreeRegressor
```

#### **Example**

```python
# Option 1: Importing directly from agin
from agin import DecisionTreeRegressor

# Option 2: Importing from agin.regression
from agin.regression import DecisionTreeRegressor

# Training data
x_train = [[1], [2], [3], [4], [5]]
y_train = [2.5, 3.5, 7.5, 9.0, 12.0]

# Initialize the model
model = DecisionTreeRegressor(max_depth=3, min_samples_split=2, random_state=42)

# Fit the model
model.fit(x_train, y_train)

# Predict using the model
x_test = [[1.5], [2.5], [3.5]]
y_pred = model.predict(x_test)

print("Predictions:", y_pred)

# Evaluate the model metrics
y_test = [3.0, 6.0, 8.5]
mse, r2 = model.metrics(y_pred, y_test)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
```

### **Methods**

#### **`fit(x_train, y_train)`**
   - Builds the decision tree from training data.
   - **Args**: 
     - `x_train` (numpy.ndarray or pandas.DataFrame): Training features.
     - `y_train` (numpy.ndarray or pandas.Series): Training target values.
   - **Returns**: The fitted `DecisionTreeRegressor` instance.

#### **`predict(x_test)`**
   - Predicts target values for the given test data using the decision tree.
   - **Args**: 
     - `x_test` (numpy.ndarray or pandas.DataFrame): Test data features.
   - **Returns**: numpy.ndarray of predicted target values.

#### **`metrics(y_pred, y_test)`**
   - Calculates evaluation metrics like Mean Squared Error (MSE) and R² Score.
   - **Args**: 
     - `y_pred` (numpy.ndarray): Predicted target values.
     - `y_test` (numpy.ndarray): True target values.
   - **Returns**: Tuple containing MSE and R² Score.

#### **Attributes**
- **`max_depth`**: The maximum depth of the tree.
- **`min_samples_split`**: Minimum number of samples required to split a node.
- **`min_samples_leaf`**: Minimum number of samples required to form a leaf node.
- **`random_state`**: Random seed for reproducibility.
- **`root`**: The root node of the decision tree.

## **Random Forest Regressor**

The `RandomForestRegressor` class implements a regression model using an ensemble of decision trees. It provides methods to train the model, make predictions, and evaluate its performance. This regressor is robust to overfitting and can capture non-linear relationships effectively.

### **Usage**

The `RandomForestRegressor` class can be imported directly from the `agin` package or from the `agin.regression` module:

```python
from agin import RandomForestRegressor
# or
from agin.regression import RandomForestRegressor
```

#### **Example**

```python
# Option 1: Importing directly from agin
from agin import RandomForestRegressor

# Option 2: Importing from agin.regression
from agin.regression import RandomForestRegressor

# Training data
x_train = [[1], [2], [3], [4], [5]]
y_train = [2.5, 3.5, 7.5, 9.0, 12.0]

# Initialize the model
model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

# Fit the model
model.fit(x_train, y_train)

# Predict using the model
x_test = [[1.5], [2.5], [3.5]]
y_pred = model.predict(x_test)

print("Predictions:", y_pred)

# Evaluate the model metrics
y_test = [3.0, 6.0, 8.5]
mse, r2 = model.metrics(y_pred, y_test)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
```

### **Methods**

#### **`fit(x_train, y_train)`**
   - Trains the random forest model using the training data.
   - **Args**: 
     - `x_train` (numpy.ndarray or pandas.DataFrame): Training features.
     - `y_train` (numpy.ndarray or pandas.Series): Training target values.
   - **Returns**: The fitted `RandomForestRegressor` instance.

#### **`predict(x_test)`**
   - Predicts target values for the given test data using the trained ensemble of trees.
   - **Args**: 
     - `x_test` (numpy.ndarray or pandas.DataFrame): Test data features.
   - **Returns**: numpy.ndarray of predicted target values.

#### **`metrics(y_pred, y_test)`**
   - Calculates evaluation metrics like Mean Squared Error (MSE) and R² Score.
   - **Args**: 
     - `y_pred` (numpy.ndarray): Predicted target values.
     - `y_test` (numpy.ndarray): True target values.
   - **Returns**: Tuple containing MSE and R² Score.

#### **Attributes**
- **`n_estimators`**: The number of trees in the forest (default: 100).
- **`max_depth`**: The maximum depth of each tree (default: None).
- **`min_samples_split`**: Minimum number of samples required to split a node (default: 2).
- **`min_samples_leaf`**: Minimum number of samples required to form a leaf node (default: 1).
- **`random_state`**: Random seed for reproducibility.