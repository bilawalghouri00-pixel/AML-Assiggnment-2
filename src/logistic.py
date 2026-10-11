"""Task 3 - logistic regression written by hand with NumPy."""
import numpy as np


def add_bias(X):
    """Put x0 = 1 in front of every row."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def sigmoid(z):
    z = np.clip(z, -500, 500)            # stops overflow warnings
    return 1.0 / (1.0 + np.exp(-z))


def hypothesis(X, theta):
    return sigmoid(X @ theta)


def cost(X, y, theta):
    """J(theta) = -(1/n) * sum[ y log h + (1-y) log(1-h) ]"""
    h = np.clip(hypothesis(X, theta), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(h) + (1 - y) * np.log(1 - h)))


def fit_logistic(X, y, alpha, iters):
    """Batch gradient descent. theta_j := theta_j + alpha/n * sum (y - h) x_j"""
    n = X.shape[0]
    theta = np.zeros(X.shape[1])
    history = []
    for _ in range(iters):
        h = hypothesis(X, theta)
        theta = theta + (alpha / n) * (X.T @ (y - h))
        history.append(cost(X, y, theta))      # J after every iteration
    return theta, np.array(history)


def predict(X, theta):
    return (hypothesis(X, theta) >= 0.5).astype(int)
