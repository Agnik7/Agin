# **Regression**

The `regression` module contains implementations of regression models. Currently, the package supports:

- **Linear Regression**

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
#### **`fit(x, y)`**
   - Trains the model based on the input data.
   - **Args**: `x` (list), `y` (list)
   - **Returns**: None

#### **`predict(x)`**
   - Predicts outputs for given input data.
   - **Args**: `x` (list)
   - **Returns**: List of predicted values

#### **`metrics(y_pred, y_test)`**
   - Calculates evaluation metrics like Mean Squared Error (MSE) and R² Score.
   - **Args**: `y_pred` (list), `y_test` (list)
   - **Returns**: Tuple containing MSE and R² Score.
