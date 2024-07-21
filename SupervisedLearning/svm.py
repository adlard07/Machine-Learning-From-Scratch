import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Ensure labels are -1 or 1
        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]
                    
        logging.info(f"Model parameters: weights = {self.w}, bias = {self.b}")

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


if __name__ == '__main__':
    # Generate some linearly separable data
    X = np.random.randint(low=0, high=100, size=(5, 10))
    y = np.array([1, -1, -1, -1, 1, 1, 1, -1, 1, -1])

    svm = SVM()
    svm.fit(X, y)

    # Predict using the SVM model
    predictions = svm.predict(X)
    logging.info(f"Actuals: {y}")
    logging.info(f"Predictions: {predictions}")
