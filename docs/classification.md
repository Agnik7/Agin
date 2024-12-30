
# **Classification**

The `classification` module contains implementations of classification models. Currently, the package supports:

- **Logistic Regression**
- **Naive Bayes Classifier**
- **K-Nearest Neighbors (KNN) Classifier**
- **Linear SVM Classifier**
- **Non-Linear SVM Classifier**

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

## **Naive Bayes Classifier**
The `NaiveBayesClassifier` class provides methods to train a Naive Bayes model, make predictions, and evaluate its performance. It can be imported directly from `agin` or from `agin.classification`.

### **Usage**
The `NaiveBayesClassifier` class can be imported directly from the `agin` package or from the `agin.classification` module:

```python
from agin import NaiveBayesClassifier
# or
from agin.classification import NaiveBayesClassifier
```

#### **Example**

```python
# Option 1: Importing directly from agin
from agin import NaiveBayesClassifier

# Option 2: Importing from agin.classification
from agin.classification import NaiveBayesClassifier

# Training data
x_train = [[1, 2], [2, 2], [3, 1], [4, 1]]
y_train = ['Yes', 'No', 'Yes', 'No']

# Initialize the model
model = NaiveBayesClassifier()

# Fit the model
model.fit(x_train, y_train)

# Predict using the model
x_test = [[2, 2], [3, 1]]
y_pred = model.predict(x_test)

print("Predictions:", y_pred)

# Evaluate the model metrics
y_test = ['No', 'Yes']
accuracy, precision, recall, f1_score = model.metrics(y_pred, y_test)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
```

### **Methods**
#### **`fit(x_train, y_train)`**
   - Trains the Naive Bayes model by calculating class probabilities and feature likelihoods.
   - **Args**:
     - `x_train` (list or numpy.ndarray): A 2D array containing the training data for independent variables.
     - `y_train` (list or numpy.ndarray): A 1D array containing the true class labels for the dependent variable.
   - **Returns**: None. Updates the model's class probabilities and feature likelihoods.

#### **`predict(x_test)`**
   - Predicts the class label for each sample in the test data using the trained Naive Bayes model.
   - **Args**:
     - `x_test` (list or numpy.ndarray): A 2D array containing test data for independent variables.
   - **Returns**: numpy.ndarray of predicted class labels.

#### **`metrics(y_pred, y_test)`**
   - Calculates the accuracy, precision, recall, and F1 score of the Naive Bayes classifier.
   - **Args**:
     - `y_pred` (list or numpy.ndarray): A 1D array containing the predicted class labels from the model.
     - `y_test` (list or numpy.ndarray): A 1D array containing the true class labels for the dependent variable.
   - **Returns**: Tuple containing accuracy, precision, recall, and F1-score.

## **K-Nearest Neighbors (KNN) Classifier**
The `KNNClassifier` class implements the K-Nearest Neighbors algorithm for classification. It calculates the distances between test samples and training samples to identify the `k` nearest neighbors and predict labels using majority or weighted voting.

### **Usage**
The `KNNClassifier` class can be imported directly from the `agin` package or from the `agin.classification` module:

```python
from agin import KNNClassifier
# or
from agin.classification import KNNClassifier
```

#### **Example**

```python
# Option 1: Importing directly from agin
from agin import KNNClassifier

# Option 2: Importing from agin.classification
from agin.classification import KNNClassifier

# Training data
x_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
y_train = [0, 1, 0, 1]

# Initialize the model
model = KNNClassifier(n_neighbors=3, weights='distance', metric='euclidean')

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
#### **`fit(x_train, y_train)`**
   - Stores the training data for use in distance calculations and predictions.
   - **Args**:
     - `x_train` (list or numpy.ndarray): A 2D array containing the training data for independent variables.
     - `y_train` (list or numpy.ndarray): A 1D array containing the true class labels for the dependent variable.
   - **Returns**: None. Updates the model with training data.

#### **`predict(x_test)`**
   - Predicts the class label for each sample in the test data.
   - **Args**:
     - `x_test` (list or numpy.ndarray): A 2D array containing the test data for independent variables.
   - **Returns**: numpy.ndarray of predicted class labels.

#### **`metrics(y_pred, y_test)`**
   - Calculates the accuracy, precision, recall, and F1 score of the KNN classifier.
   - **Args**:
     - `y_pred` (list or numpy.ndarray): A 1D array containing the predicted class labels from the model.
     - `y_test` (list or numpy.ndarray): A 1D array containing the true class labels for the dependent variable.
   - **Returns**: Tuple containing accuracy, precision, recall, and F1-score.

## **Linear SVM Classifier**
The `LinearSVMClassifier` class implements the Linear Support Vector Machine (SVM) algorithm for classification tasks. It separates classes by finding the hyperplane that maximizes the margin between them.

### **Usage**
The `LinearSVMClassifier` class can be imported directly from `agin` or from `agin.classification`.

```python
from agin import LinearSVMClassifier
# or
from agin.classification import LinearSVMClassifier
```

#### **Example**

```python
# Option 1: Importing directly from agin
from agin import LinearSVMClassifier

# Option 2: Importing from agin.classification
from agin.classification import LinearSVMClassifier

# Training data
x_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
y_train = [0, 1, 0, 1]

# Initialize the model
model = LinearSVMClassifier(C=1.0, max_iter=100)

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
#### **`fit(x_train, y_train)`**
   - Trains the SVM model by finding the optimal hyperplane.
   - **Args**:
     - `x_train` (numpy.ndarray or pandas.DataFrame): Training feature data.
     - `y_train` (numpy.ndarray or pandas.DataFrame): Target labels.
   - **Returns**: The trained LinearSVMClassifier model.

#### **`predict(x)`**
   - Predicts class labels for the given

 feature data.
   - **Args**:
     - `x` (numpy.ndarray or pandas.DataFrame): Feature data.
   - **Returns**: numpy.ndarray of predicted class labels.

#### **`metrics(y_pred, y_test)`**
   - Calculates the accuracy, precision, recall, and F1 score of the KNN classifier.
   - **Args**:
     - `y_pred` (list or numpy.ndarray): A 1D array containing the predicted class labels from the model.
     - `y_test` (list or numpy.ndarray): A 1D array containing the true class labels for the dependent variable.
   - **Returns**: Tuple containing accuracy, precision, recall, and F1-score.

## **Linear SVM Classifier**
The `LinearSVMClassifier` class implements the Linear Support Vector Machine (SVM) algorithm for classification tasks. It separates classes by finding the hyperplane that maximizes the margin between them.

### **Usage**
The `LinearSVMClassifier` class can be imported directly from `agin` or from `agin.classification`.

```python
from agin import LinearSVMClassifier
# or
from agin.classification import LinearSVMClassifier
```

#### **Example**

```python
# Option 1: Importing directly from agin
from agin import LinearSVMClassifier

# Option 2: Importing from agin.classification
from agin.classification import LinearSVMClassifier

# Training data
x_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
y_train = [0, 1, 0, 1]

# Initialize the model
model = LinearSVMClassifier(C=1.0, max_iter=100)

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
#### **`fit(x_train, y_train)`**
   - Trains the SVM model by finding the optimal hyperplane.
   - **Args**:
     - `x_train` (numpy.ndarray or pandas.DataFrame): Training feature data.
     - `y_train` (numpy.ndarray or pandas.DataFrame): Target labels.
   - **Returns**: The trained LinearSVMClassifier model.

#### **`predict(x)`**
   - Predicts class labels for the given feature data.
   - **Args**: `x` (numpy.ndarray or pandas.DataFrame): Feature data.
   - **Returns**: numpy.ndarray of predicted class labels.

#### **`metrics(y_pred, y_test)`**
   - Computes accuracy, precision, recall, and F1-score for classification.
   - **Args**:
     - `y_pred` (numpy.ndarray): Predicted labels.
     - `y_test` (numpy.ndarray): True labels.
   - **Returns**: Tuple containing accuracy, precision, recall, and F1-score.

## **Non-Linear SVM Classifier**
The `NonLinearSVMClassifier` class implements the Support Vector Machine (SVM) algorithm using a non-linear kernel (e.g., RBF, polynomial) to classify data that cannot be separated by a straight line. The class supports multiple kernel types and hyperparameter tuning.

### **Usage**
The `NonLinearSVMClassifier` class can be imported directly from `agin` or from `agin.classification`.

```python
from agin import NonLinearSVMClassifier
# or
from agin.classification import NonLinearSVMClassifier
```

#### **Example**

```python
# Option 1: Importing directly from agin
from agin import NonLinearSVMClassifier

# Option 2: Importing from agin.classification
from agin.classification import NonLinearSVMClassifier

# Training data
x_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
y_train = [0, 1, 0, 1]

# Initialize the model
model = NonLinearSVMClassifier(kernel='rbf', C=1.0, gamma='scale', max_iter=100)

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
#### **`fit(x_train, y_train)`**
   - Trains the non-linear SVM model using the specified kernel.
   - **Args**:
     - `x_train` (numpy.ndarray or pandas.DataFrame): Training feature data.
     - `y_train` (numpy.ndarray or pandas.DataFrame): Target labels.
   - **Returns**: The trained NonLinearSVMClassifier model.

#### **`predict(x)`**
   - Predicts class labels for the given feature data.
   - **Args**:
     - `x` (numpy.ndarray or pandas.DataFrame): Feature data.
   - **Returns**: numpy.ndarray of predicted class labels.

#### **`metrics(y_pred, y_test)`**
   - Computes various evaluation metrics for classification.
   - **Args**:
     - `y_pred` (numpy.ndarray): Predicted labels.
     - `y_test` (numpy.ndarray): True labels.
   - **Returns**: Tuple containing accuracy, precision, recall, and F1-score.

### **Parameters**
- `kernel` (str): The kernel function to use ('linear', 'poly', 'rbf', 'sigmoid'). Default is 'rbf'.
- `C` (float): Regularization parameter. Default is 1.0.
- `gamma` (str or float): Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. Default is 'scale'.
- `degree` (int): Degree of the polynomial kernel function ('poly'). Default is 3.
- `max_iter` (int): Maximum number of iterations for optimization. Default is 1000.
