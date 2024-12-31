import numpy as np

class Neuron:
    def __init__(self, input_size, output_size, activation_function, activation_derivative, init_method="he"):
        """
        Initializes a Neuron with specified weight initialization and activation function.

        Attributes:
            weights (np.ndarray): Weight matrix of shape (output_size, input_size) initialized based on the specified method.
            biases (np.ndarray): Bias vector of shape (output_size, 1) initialized to zeros.
            activation_function (callable): Activation function for forward propagation.
            activation_derivative (callable): Derivative of the activation function for backpropagation.

        Args:
            input_size (int): Number of inputs to the neuron.
            output_size (int): Number of outputs from the neuron.
            activation_function (callable): Activation function for forward propagation.
            activation_derivative (callable): Derivative of the activation function for backpropagation.
            init_method (str): Initialization method for weights ("he" or "xavier"). Default is "he".

        Raises:
            ValueError: If the provided `init_method` is neither "he" nor "xavier".
        """
        if init_method == "he":
            self.weights = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
        elif init_method == "xavier":
            self.weights = np.random.randn(output_size, input_size) * np.sqrt(1 / input_size)
        else:
            raise ValueError("Invalid initialization method. Choose 'he' or 'xavier'.")

        self.biases = np.zeros((output_size, 1))
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative

    def forward(self, x):
        """
        Performs the forward propagation step for the neuron.

        Args:
            x (np.ndarray): Input data of shape (input_size, number_of_samples).

        Returns:
            np.ndarray: Output of the neuron after applying the activation function, 
            of shape (output_size, number_of_samples).

        Side Effects:
            Stores the linear combination (z) and activated output (a) for use in backpropagation.
        """
        self.z = np.dot(self.weights, x) + self.biases
        self.a = self.activation_function(self.z)
        return self.a

    def backward(self, dz, prev_a):
        """
        Performs the backward propagation step for the neuron to calculate gradients.

        Args:
            dz (np.ndarray): Gradient of the loss with respect to the neuron's output, 
            of shape (output_size, number_of_samples).
            prev_a (np.ndarray): Activated output from the previous layer (input to this neuron), 
            of shape (input_size, number_of_samples).

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of the neuron (dz_prev), 
            of shape (input_size, number_of_samples).

        Side Effects:
            Updates the attributes `dw` (weight gradients) and `db` (bias gradients).
        """
        m = prev_a.shape[1]
        self.dw = (1 / m) * np.dot(dz, prev_a.T)
        self.db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        dz_prev = np.dot(self.weights.T, dz)
        return dz_prev

    def update_parameters(self, learning_rate):
        """
        Updates the weights and biases of the neuron using the calculated gradients.

        Args:
            learning_rate (float): Learning rate for gradient descent.

        Side Effects:
            Updates the attributes `weights` and `biases` by subtracting the scaled gradients.
        """
        self.weights -= learning_rate * self.dw
        self.biases -= learning_rate * self.db
