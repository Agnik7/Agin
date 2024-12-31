import numpy as np

class NeuralNetwork:
    def __init__(self, n_x, n_h, n_y, learning_rate=0.01, iterations=1000):
        """
        Initialize the Neural Network model with weights, biases, and hyperparameters.

        Args:
            n_x (int): Number of input features.
            n_h (int): Number of neurons in the hidden layer.
            n_y (int): Number of output neurons.
            learning_rate (float): Learning rate for optimization.
            iterations (int): Number of training iterations.
        """
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        self.learning_rate = learning_rate
        self.iterations = iterations

        self.parameters = self.initialize_parameters()

    def initialize_parameters(self):
        """Initialize weights and biases."""
        w1 = np.random.randn(self.n_h, self.n_x) * 0.01
        b1 = np.zeros((self.n_h, 1))
        w2 = np.random.randn(self.n_y, self.n_h) * 0.01
        b2 = np.zeros((self.n_y, 1))

        return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

    def tanh(self, x):
        return np.tanh(x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def derivative_tanh(self, x):
        return 1 - np.power(np.tanh(x), 2)

    def forward_propagation(self, x):
        """Perform forward propagation."""
        w1, b1, w2, b2 = self.parameters["w1"], self.parameters["b1"], self.parameters["w2"], self.parameters["b2"]

        z1 = np.dot(w1, x) + b1
        a1 = self.tanh(z1)

        z2 = np.dot(w2, a1) + b2
        a2 = self.softmax(z2)

        return {"z1": z1, "a1": a1, "z2": z2, "a2": a2}

    def cost_function(self, a2, y):
        """Compute the cost function."""
        m = y.shape[1]
        cost = -(1 / m) * np.sum(y * np.log(a2))
        return cost

    def backward_propagation(self, x, y, forward_cache):
        """Perform backward propagation."""
        w2 = self.parameters["w2"]

        a1, a2 = forward_cache["a1"], forward_cache["a2"]
        m = x.shape[1]

        dz2 = a2 - y
        dw2 = (1 / m) * np.dot(dz2, a1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

        dz1 = np.dot(w2.T, dz2) * self.derivative_tanh(a1)
        dw1 = (1 / m) * np.dot(dz1, x.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

        return {"dw1": dw1, "db1": db1, "dw2": dw2, "db2": db2}

    def update_parameters(self, gradients):
        """Update weights and biases using gradients."""
        for key in gradients:
            self.parameters[key.replace("d", "")] -= self.learning_rate * gradients[key]

    def fit(self, x, y):
        """Train the model using gradient descent."""
        self.costs = []

        for i in range(self.iterations):
            forward_cache = self.forward_propagation(x)
            cost = self.cost_function(forward_cache["a2"], y)
            gradients = self.backward_propagation(x, y, forward_cache)
            self.update_parameters(gradients)
            self.costs.append(cost)

            if i % (self.iterations // 10) == 0:
                print(f"Cost after iteration {i}: {cost:.4f}")

    def predict(self, x):
        """Predict outputs for given inputs."""
        forward_cache = self.forward_propagation(x)
        return np.argmax(forward_cache["a2"], axis=0)

    def metrics(self, x, y):
        """Calculate model accuracy."""
        predictions = self.predict(x)
        labels = np.argmax(y, axis=0)
        accuracy = np.mean(predictions == labels) * 100
        return accuracy

if __name__ == "__main__":
    # Sample dataset
    x_train = np.random.randn(3, 1000)
    y_train = np.eye(3)[np.random.choice(3, 1000)].T

    x_test = np.random.randn(3, 200)
    y_test = np.eye(3)[np.random.choice(3, 200)].T

    # Neural network parameters
    n_x = x_train.shape[0]
    n_h = 5
    n_y = y_train.shape[0]

    # Initialize and train the model
    model = NeuralNetwork(n_x, n_h, n_y, learning_rate=0.1, iterations=2000)
    model.fit(x_train, y_train)

    # Evaluate the model
    acc = model.metrics(x_test, y_test)
    print(f"Model Accuracy: {acc:.2f}%")
