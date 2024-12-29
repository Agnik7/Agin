# **Classification**

The `classification` module contains implementations of classification models. Currently, the package supports:

- **Logistic Regression**

## **Logistic Regression**
The `LogisticRegression` class provides methods to train a logistic regression model using gradient descent with optional regularization, make predictions, and evaluate the model's performance. It can be imported directly from `agin` or from `agin.regression`.

### **Usage**
The `LogisticRegression` class can be imported directly from the `agin` package or from the `agin.regression` module:

```python
from agin import LogisticRegression
# or
from agin.regression import LogisticRegression
```

#### **Example**

```python
# Option 1: Importing directly from agin
from agin import LogisticRegression

# Option 2: Importing from agin.regression
from agin.regression import LogisticRegression

# Training data
x_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
y_train = [0, 1, 0, 1]

# Initialize the model
model = LogisticRegression(regularization='l2', C=1.0, max_iter=100)

# Fit the model
model.fit(x_train, y_train)

# Predict using the model
x_test = [[5, 6], [6, 7]]
y_pred = model.predict(x_test)

print("Predictions:", y_pred)

# Evaluate the model metrics
y_test = [1, 0]
accuracy, precision, recall, f1_score = model.metrics(y_pred, y_test)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
```

### **Methods**
#### **`fit(x_train, y_train, epochs=None, learning_rate=0.1, batch_size=32)`**
   - Trains the logistic regression model using gradient descent.
   - **Args**: 
     - `x_train` (numpy.ndarray or pandas.DataFrame): Training feature data.
     - `y_train` (numpy.ndarray or pandas.DataFrame): Target labels.
     - `epochs` (int): Number of epochs for training. Default is the value of `max_iter`.
     - `learning_rate` (float): Learning rate for gradient updates. Default is 0.1.
     - `batch_size` (int): Size of batches for mini-batch gradient descent. Default is 32.
   - **Returns**: The trained LogisticRegression model.

#### **`predict_probabilities(x)`**
   - Predicts probabilities for the given feature data.
   - **Args**: `x` (numpy.ndarray or pandas.DataFrame): Feature data.
   - **Returns**: numpy.ndarray of predicted probabilities.

#### **`predict(x, threshold=0.5)`**
   - Predicts class labels for the given feature data.
   - **Args**: 
     - `x` (numpy.ndarray or pandas.DataFrame): Feature data.
     - `threshold` (float): Threshold for converting probabilities to binary class labels. Default is 0.5.
   - **Returns**: numpy.ndarray of predicted class labels.

#### **`metrics(y_pred, y_test)`**
   - Computes various evaluation metrics for classification.
   - **Args**: 
     - `y_pred` (numpy.ndarray): Predicted labels.
     - `y_test` (numpy.ndarray): True labels.
   - **Returns**: Tuple containing accuracy, precision, recall, and F1-score.

### **Parameters**
- `regularization` (str): Type of regularization ('l1', 'l2', 'elasticnet' or None). Default is 'l2'.
- `C` (float): Inverse of regularization strength. Smaller values indicate stronger regularization. Default is 1.0.
- `max_iter` (int): Maximum number of iterations for optimization. Default is 100.
- `tol` (float): Tolerance for stopping criteria. Default is 1e-4.
- `class_weight` (dict or 'balanced' or 'unbalanced'): Weights associated with classes. If 'balanced', class weights are computed inversely proportional to class frequencies. Default is None.
- `random_state` (int): Seed for random number generation to ensure reproducibility. Default is None.
- `l1_ratio` (float): The mixing parameter for elasticnet regularization. l1_ratio=1 corresponds to l1, while l1_ratio=0 corresponds to l2. Default is 0.5.

### **Attributes**
- `weights` (numpy.ndarray): Model coefficients for features.
- `bias` (float): Model intercept term.
- `loss` (list): Training loss history.
- `train_acc` (list): Training accuracy history.

### **Notes**
- The model supports three types of regularization: L1 (Lasso), L2 (Ridge), and Elastic Net.
- Class weights can be automatically computed using the 'balanced' option for imbalanced datasets.
- The model uses mini-batch gradient descent for optimization, with customizable batch sizes.
- Early stopping is implemented based on the tolerance parameter.
- Features are automatically scaled using MinMaxScaler during training and prediction.