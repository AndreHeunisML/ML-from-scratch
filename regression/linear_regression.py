
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from pylab import *


def la_fit(diabetes_X_train, diabetes_X_test, diabetes_y_train, lambda_l2):

    theta = np.dot(
        np.dot(
            np.linalg.inv(np.dot(
                x_train_standardised.T,
                x_train_standardised)),
            x_train_standardised.T),
        diabetes_y_train)

    theta_l2 = np.dot(
        np.dot(
            np.linalg.inv(np.dot(
                x_train_standardised.T,
                x_train_standardised) + lambda_l2 * np.eye(diabetes_X_train.shape[1])),
            x_train_standardised.T),
        diabetes_y_train)

    y_test_pred_without_bias = theta * x_test_standardised
    y_test_pred_without_bias_l2 = theta_l2 * x_test_standardised

    bias = np.mean(diabetes_y_train - theta * x_train_standardised)
    bias2 = np.mean(diabetes_y_train - theta_l2 * x_train_standardised)

    return y_test_pred_without_bias + bias, y_test_pred_without_bias_l2 + bias2


if __name__ == "__main__":
    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()

    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr_l2 = linear_model.Ridge()

    # Standardise training data for sklearn fitting
    x_train_mean = np.mean(diabetes_X_train, axis=0)
    x_train_std = np.std(diabetes_X_train, axis=0)
    x_train_standardised = (diabetes_X_train - x_train_mean) / x_train_std
    x_test_standardised = (diabetes_X_test - x_train_mean) / x_train_std

    # Train the model using the training sets
    regr.fit(x_train_standardised, diabetes_y_train)
    diabetes_y_pred = regr.predict(x_test_standardised)

    regr_l2.fit(x_train_standardised, diabetes_y_train)
    diabetes_y_pred_l2 = regr_l2.predict(x_test_standardised)

    # Fit using linear algebra
    la_pred, la_pred_l2 = la_fit(diabetes_X_train, diabetes_X_test, diabetes_y_train, 1.0)

    print('Coefficients: \n', regr.coef_)
    print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

    print(diabetes_y_pred)

    # Plot outputs
    figure()
    subplot(211)
    scatter(diabetes_X_test, diabetes_y_test,  color='black')
    plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3, alpha=0.5, label='sklearn')
    plot(diabetes_X_test, la_pred, color='red', linewidth=3, alpha=0.5, label='linear')
    legend()
    subplot(212)
    scatter(diabetes_X_test, diabetes_y_test, color='black')
    plot(diabetes_X_test, diabetes_y_pred_l2, color='blue', linewidth=3, alpha=0.5, label='sklearn')
    plot(diabetes_X_test, la_pred_l2, color='red', linewidth=3, alpha=0.5, label='ridge')
    plt.show()
