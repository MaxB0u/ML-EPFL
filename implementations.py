from src.helpers import *


def ridge_regression(y, tx, lambda_):
    """Ridge regression learning using normal equations"""

    lambda_prime = lambda_ * 2 * len(y)
    a = tx.T @ tx + lambda_prime * np.eye(len(tx.T))
    b = tx.T @ y

    w_opt = np.linalg.solve(a, b)
    loss_opt = compute_mse_loss(y, tx, w_opt)

    return w_opt, loss_opt


def logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_=0, epsilon=0.01):
    """The Gradient Descent (GD) algorithm for logistic regression."""

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = [compute_loss_logistic(y, tx, initial_w, lambda_)]
    w = initial_w

    sgd = False
    for n_iter in range(max_iters):
        # Compute gradient and loss
        if sgd:
            # If no batch size is passed it defaults to 1 which is the case for SGD
            for batch_y, batch_tx in batch_iter(y, tx):
                grad = compute_gradient_logistic(batch_y, batch_tx, w)
                # In SGD loss is computed only on sampled data
        else:
            grad = compute_gradient_logistic(y, tx, w)

        # Update gradient
        w -= gamma * (grad + 2 * lambda_ * w)
        loss = compute_loss_logistic(y, tx, w, lambda_)

        # store w and loss
        ws.append(w)
        losses.append(loss)
        # print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

        # Exit if no significant change in model weights

    return ws[-1], losses[-1]


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, epsilon=0.01):
    """The Gradient Descent (GD) algorithm for logistic regression."""
    return logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_, epsilon)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """ "The Gradient Descent (GD) algorithm for linear regression which uses the least squares loss function."""

    ws = [initial_w]
    losses = [compute_mse_loss(y, tx, initial_w)]
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient_mse_loss_function(y, tx, w)

        w = w - gamma * gradient
        loss = compute_mse_loss(y, tx, w)

        ws.append(w)
        losses.append(loss)

    return ws[-1], losses[-1]


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic gradient descent (SGD) algorithm for linear regression which uses the least squares loss function."""

    batch_size = 1  # SGD uses batch_size of 1
    ws = [initial_w]
    losses = [compute_mse_loss(y, tx, initial_w)]
    w = initial_w

    for n_iter in range(max_iters):
        minibatch_y, minibatch_tx = next(batch_iter(y, tx, batch_size))
        minibatch_gradient = compute_gradient_mse_loss_function(
            minibatch_y, minibatch_tx, w
        )

        # Update w from the computed gradients
        w = w - gamma * minibatch_gradient
        loss = compute_mse_loss(minibatch_y, minibatch_tx, w)

        losses.append(loss)
        ws.append(w)
    return ws[-1], losses[-1]


def least_squares(y, tx):
    """The Least Squares regression learning using normal equations"""

    w_optimal = np.linalg.inv(tx.transpose() @ tx) @ tx.transpose() @ y
    num_samples = y.shape[0]
    mse = compute_mse_loss(y, tx, w_optimal)
    return w_optimal, mse
