import numpy as np

class LinearRegressionModel:
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, x, y):
        # Convert to numpy arrays
        x_train = np.array(x)
        y_train = np.array(y)

        # Calculate the mean of x and y
        mean_value_x = np.mean(x_train)
        mean_value_y = np.mean(y_train)

        # Calculate deviations
        deviations_x = x_train - mean_value_x
        deviations_y = y_train - mean_value_y

        # Calculate the product of deviations and sum of squares
        product = np.sum(deviations_x * deviations_y)
        sum_of_squares_x = np.sum(deviations_x ** 2)

        # Calculate the slope (m) and intercept (b)
        self.slope = product / sum_of_squares_x
        self.intercept = mean_value_y - (self.slope * mean_value_x)

    def predict(self, x):
        x_test = np.array(x)
        return (self.slope * x_test) + self.intercept

    def metrics(self, y_pred, y_test):
        # Manually calculate Mean Squared Error (MSE)
        squared_errors = [(y_true - y_pred) ** 2 for y_true, y_pred in zip(y_test, y_pred)]
        mse = np.mean(squared_errors)
        print(f"Mean Squared Error: {mse:.4f}")

if __name__ == "__main__":
    # Training data
    x_train = [1, 2, 3, 4, 5]
    y_train = [2, 4, 6, 8, 10]

    # Testing data
    x_test = [0, 1, 2, 3, 4, 5]
    y_test = [1, 3, 5, 7, 9, 11]

    # Initialize the model
    model = LinearRegressionModel()

    # Fit the model
    model.fit(x_train, y_train)

    # Predict using the model
    y_pred = model.predict(x_test)
    print("Predicted values:", y_pred)

    # Calculate and print metrics (MSE)
    model.metrics(y_pred, y_test)
