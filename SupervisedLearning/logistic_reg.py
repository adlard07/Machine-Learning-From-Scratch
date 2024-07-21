import pandas as pd
import numpy as np
import logging
import math

from linear_reg import LinearRegression

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%d/%m/%y %I:%M:%S %p')

class LogisticRegression:
    def __init__(self):
        self.B = None

    def fit(self, X_train, Y_train):
        # Linear regression to get the coefficients
        if X_train.shape[1] > 1:
            self.B, _ = LinearRegression().linear_regression(X_train, Y_train)
        else:
            self.B, _, _ = LinearRegression().linear_regression(X_train, Y_train)

        logging.info(f"B(slope) -> {self.B}")

    def predict_probabilities(self, X):
        if self.B is None:
            raise Exception("Model not trained. Call 'fit' with appropriate data.")

        z = np.dot(X, self.B)
        logging.info(f"z (linear combination) -> {z}")

        # Sigmoid function
        probabilities = 1 / (1 + np.exp(-z))
        logging.info(f"Output probabilities -> {probabilities}")

        return probabilities

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_probabilities(X)
        predictions = (probabilities >= threshold).astype(int)
        logging.info(f"Predictions -> {predictions}")

        return predictions


if __name__ == "__main__":
    # Train data
    X_train = pd.DataFrame(np.random.randint(low=0, high=100, size=(10, 5)))
    Y_train = np.random.randint(low=0, high=2, size=10)

    # Test data
    X_test = pd.DataFrame(np.random.randint(low=0, high=100, size=(5, 5)))
    Y_test = np.random.randint(low=0, high=2, size=5)

    # Create and train the logistic regression model
    logistic_reg = LogisticRegression()
    logistic_reg.fit(X_train, Y_train)

    # Predict probabilities on the test set
    probabilities = logistic_reg.predict_probabilities(X_test)
    logging.info(f"Test Probabilities -> {probabilities}")

    # Predict class labels on the test set
    predictions = logistic_reg.predict(X_test)
    logging.info(f"Test Predictions -> {predictions}")

    # Optionally, compare predictions with the actual test labels
    accuracy = np.mean(predictions == Y_test)
    logging.info(f"Accuracy -> {accuracy}")
