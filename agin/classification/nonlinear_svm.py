import numpy as np

class NonLinearSVM:
    def __init__(self, learning_rate=0.01, regularization_strength=0.1, num_iterations=1000, gamma=1.0, C=1.0):
        self.learning_rate = learning_rate
        self.regularization_strength = regularization_strength
        self.num_iterations = num_iterations
        self.gamma = gamma
        self.C = C
        self.alpha = None  # Lagrange multipliers
        self.b = 0  # Bias term
        self.X_train = None  # Training data
        self.y_train = None  # Training labels
    
    def rbf_kernel(self, X1, X2):
        """
        Computes the Radial Basis Function (RBF) kernel.
        """
        dist_sq = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * dist_sq)

    def fit(self, X_train, y_train):
        """
        Trains the Non-Linear SVM model.
        """
        self.X_train = X_train
        self.y_train = y_train
        n_samples = X_train.shape[0]
        
        # Initialize alpha (dual coefficients)
        self.alpha = np.zeros(n_samples)
        
        # Compute the kernel matrix
        K = self.rbf_kernel(X_train, X_train)
        
        for _ in range(self.num_iterations):
            for i in range(n_samples):
                # Compute the margin
                decision_value = np.sum(self.alpha * y_train * K[:, i]) + self.b
                margin = y_train[i] * decision_value
                
                # Update rules for alpha and bias
                if margin < 1:
                    self.alpha[i] += self.learning_rate * (1 - margin - self.regularization_strength * self.alpha[i])
                    self.b += self.learning_rate * y_train[i]
                else:
                    self.alpha[i] -= self.learning_rate * self.regularization_strength * self.alpha[i]
            
            # Ensure alpha satisfies the constraints
            self.alpha = np.clip(self.alpha, 0, self.C)

    def predict(self, X_test):
        """
        Predicts the class labels for the test data.
        """
        K = self.rbf_kernel(self.X_train, X_test)
        decision_values = np.dot((self.alpha * self.y_train).T, K) + self.b
        return np.sign(decision_values)

    def metrics(self, y_pred, y_test):
        """
        Calculates accuracy, precision, recall, and F1 score.
        """
        accuracy = np.mean(y_pred == y_test)
        true_positive = np.sum((y_test == 1) & (y_pred == 1))
        false_positive = np.sum((y_test == -1) & (y_pred == 1))
        false_negative = np.sum((y_test == 1) & (y_pred == -1))
        
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        return accuracy, precision, recall, f1_score


# Example usage
if __name__ == "__main__":
    # Generate synthetic dataset
    np.random.seed(42)
    
    # Create two classes: Class +1 and Class -1
    class1 = np.random.randn(10, 2) + [2, 2]  # Center around (2,2)
    class2 = np.random.randn(10, 2) + [-2, -2]  # Center around (-2,-2)

    X_train = np.vstack((class1, class2))
    y_train = np.hstack((np.ones(class1.shape[0]), -np.ones(class2.shape[0])))

    # Test set: Additional points for evaluation
    X_test = np.array([
        [3, 3],  # Close to Class +1
        [-3, -3],  # Close to Class -1
        [0, 0],  # Neutral, may vary based on decision boundary
        [2, -2],  # Borderline case
    ])
    y_test = np.array([1, -1, -1, -1])  # True labels based on visual observation

    # Create and train the SVM model
    svm = NonLinearSVM(learning_rate=0.01, regularization_strength=0.1, num_iterations=1000, gamma=0.5, C=1.0)
    svm.fit(X_train, y_train)

    # Make predictions
    y_pred = svm.predict(X_test)

    # Evaluate performance
    accuracy, precision, recall, f1_score = svm.metrics(y_pred, y_test)

    # Output results
    print("Training Data:\n", X_train)
    print("Training Labels:\n", y_train)
    print("Test Data:\n", X_test)
    print("Predictions:", y_pred)
    print("True Labels:", y_test)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")


