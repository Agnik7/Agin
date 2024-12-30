# **Regression**

The `regression` module contains implementations of regression models. Currently, the package supports:

- **Linear Regression**
- **Multilinear Regression**
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