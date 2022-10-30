from src.utils import *


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm for logistic regression.

    Args:
        y:  shape=(N, 1) -> y-data
        tx: shape=(N, D) -> x-data
        initial_w:  shape=(D, 1) -> Initial model weights
        max_iters: int -> Maximum number of training iterations
        gamma: float -> Step-size

    Returns:
        loss: scalar number -> The final training loss
        w: shape=(D, 1) -> The trained weights
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    loss = compute_mse_loss(y, tx, initial_w)
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w - gamma * grad

        loss = compute_mse_loss(y, tx, w)
        # store w and loss
        ws.append(w)
        losses.append(loss)

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """The Stochastic Gradient Descent (SGD) algorithm for logistic regression.

    Args:
        y:  shape=(N, 1) -> y-data
        tx: shape=(N, D) -> x-data
        initial_w:  shape=(D, 1) -> Initial model weights
        max_iters: int -> Maximum number of training iterations
        gamma: float -> Step-size
        batch_size: int -> Number of samples on which to calculate the gradient

    Returns:
        loss: scalar number -> The final training loss
        w: shape=(D, 1) -> The trained weights
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    loss = compute_mse_loss(y, tx, w)

    for n_iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, batch_size):
            grad = compute_gradient(batch_y, batch_tx, w)
            # In SGD loss is computed only on sampled data

            w = w - gamma * grad

            loss = compute_mse_loss(batch_y, batch_tx, w)

        # store w and loss
        ws.append(w)
        losses.append(loss)

    return w, loss


def least_squares(y, tx):
    """Least squares using normal equations
    Args:
        y:  shape=(N, 1) -> y-data
        tx: shape=(N, D) -> x-data

    Returns:
        loss: scalar number -> The final training loss
        w: shape=(D, 1) -> The trained weights
    """
    a = tx.T @ tx
    b = tx.T @ y
    w_opt = np.linalg.solve(a, b)
    loss_opt = compute_mse_loss(y, tx, w_opt)
    return w_opt, loss_opt


def ridge_regression(y, tx, lambda_):
    """Ridge regression learning using normal equations
    Args:
        y:  shape=(N, 1) -> y-data
        tx: shape=(N, D) -> x-data
        lambda_: float -> Regularization parameter

    Returns:
        loss: scalar number -> The final training loss
        w: shape=(D, 1) -> The trained weights
    """

    lambda_prime = lambda_ * 2 * len(y)
    a = tx.T @ tx + lambda_prime * np.eye(len(tx.T))
    b = tx.T @ y

    w_opt = np.linalg.solve(a, b)
    loss_opt = compute_mse_loss(y, tx, w_opt)

    return w_opt, loss_opt


def logistic_regression(
    y, tx, initial_w, max_iters, gamma, lambda_=0.0, newton_method=False
):
    """The Gradient Descent (GD) algorithm for logistic regression.

    Args:
        y:  shape=(N, 1) -> y-data
        tx: shape=(N, D) -> x-data
        initial_w:  shape=(D, 1) -> Initial model weights
        max_iters: int -> Maximum number of training iterations
        gamma: float -> Step-size
        lambda_: float -> Regularization parameter
        newton_method: bool -> Whether to use Newton's method for gradient descent or not

    Returns:
        loss: scalar number -> The final training loss
        w: shape=(D, 1) -> The trained weights
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = [compute_loss_logistic(y, tx, initial_w)]
    w = initial_w
    grad = 0

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
        if newton_method:
            # No regularization for Newton's method
            h = compute_hessian_logistic(tx, w)
            diff = np.linalg.solve(h, gamma * grad)
            w -= diff
        else:
            w -= gamma * (grad + 2 * lambda_ * w)

        # Compute loss
        loss = compute_loss_logistic(y, tx, w)

        # store w and loss
        ws.append(w)
        losses.append(loss)

    return ws[-1], losses[-1]


def reg_logistic_regression(
    y, tx, lambda_, initial_w, max_iters, gamma, newton_method=False
):
    """The Gradient Descent (GD) algorithm for logistic regression.
    Args:
        y:  shape=(N, 1) -> y-data
        tx: shape=(N, D) -> x-data
        initial_w:  shape=(D, 1) -> Initial model weights
        max_iters: int -> Maximum number of training iterations
        gamma: float -> Step-size
        lambda_: float -> Regularization parameter
        newton_method: bool -> Whether to use Newton's method for gradient descent or not

    Returns:
        loss: scalar number -> The final training loss
        w: shape=(D, 1) -> The trained weights
    """
    return logistic_regression(
        y, tx, initial_w, max_iters, gamma, lambda_, newton_method
    )
