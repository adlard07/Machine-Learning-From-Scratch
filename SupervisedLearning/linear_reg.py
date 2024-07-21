import pandas as pd
import numpy as np
import logging

from utils import save_model, load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')

class LinearRegression:
    def fit(self, X, Y):
        try:
            if isinstance(Y, pd.DataFrame):
                Y = Y.to_numpy().ravel()
            else:
                Y = np.array(Y).ravel()
                
            X = X.to_numpy()
            n, m = X.shape

            if m > 1:
                # -------------multivariate regression-------------
                try:
                    X_transpose = X.T
                    X_trans_X = X_transpose @ X
                    X_trans_X_inv = np.linalg.pinv(X_trans_X)
                    X_trans_Y = X_transpose @ Y
                    B = X_trans_X_inv @ X_trans_Y
                    logging.info(f"The coefficients (Rate of change of all coefficients) -> {B}")

                    # Prediction
                    y_pred = X @ B
                    logging.info(f"Predictions -> {y_pred}")

                    # Difference between actual and predicted
                    residuals = Y - y_pred
                    MSE = np.mean(residuals**2)
                    logging.info(f"Mean Squared Error -> {MSE}")

                    return B, MSE
                except Exception as e:
                    logging.error(f'Exception occurred during multivariate regression: {e}')
                    return None, None

            else:
                # -------------simple linear regression-------------
                X = X.ravel()
                sum_X = np.sum(X)
                sum_Y = np.sum(Y)
                sum_XY = np.sum(X * Y)
                sum_X_sqr = np.sum(X * X)

                B = ((n * sum_XY) - (sum_X * sum_Y)) / ((n * sum_X_sqr) - (sum_X ** 2))
                logging.info(f"Slope (Rate of change) -> {B}")

                intercept = (sum_Y - (B * sum_X)) / n
                logging.info(f"Intercept (coefficient) -> {intercept}")

                y_pred = intercept + B * X
                logging.info(f"Predictions -> {y_pred}")

                residuals = y_pred - Y
                MSE = np.mean(residuals**2)
                logging.info(f"Mean Squared Error -> {MSE}")

                return B, intercept, MSE

        except Exception as e:
            logging.error(f'An exception occurred: {e}')
            return None, None, None

if __name__ == '__main__':
    # train
    X = pd.DataFrame(np.random.randint(low=0, high=100, size=(10, 5)))
    Y = np.random.randint(low=0, high=10, size=(10, 1))

    # test 
    x_test = pd.DataFrame(np.random.randint(low=0, high=100, size=(10, 5)))
    y_test = np.random.randint(low=0, high=2, size=(10, 1))

    linear_reg = LinearRegression()
    result = linear_reg.fit(X, Y)

    if len(result) == 3:
        B, intercept, MSE = result
        logging.info(f"Model parameters: B = {B}, intercept = {intercept}, MSE = {MSE}")
    else:
        B, MSE = result
        logging.info(f"Model parameters: B = {B}, MSE = {MSE}")

    save_model(B, 'Supervised Learning/model')
    model = load_model('Supervised Learning/model.json')
