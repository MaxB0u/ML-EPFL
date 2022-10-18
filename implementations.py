from src.helpers import *


def ridge_regression(y, tx, lambda_):
    lambda_prime = lambda_ * 2 * len(y)
    a = tx.T @ tx + lambda_prime * np.eye(len(tx.T))
    b = tx.T @ y

    w_opt = np.linalg.solve(a, b)
    loss_opt = compute_loss(y, tx, w_opt)

    return w_opt, loss_opt


def logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_=0):
    """The Gradient Descent (GD) algorithm for logistic regression."""

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    if max_iters == 0:
        loss = compute_loss_logistic(y, tx, w, lambda_)
        losses.append(loss)
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
        w -= gamma * (grad + 2*lambda_*w)
        loss = compute_loss_logistic(y, tx, w, lambda_)

        # store w and loss
        ws.append(w)
        losses.append(loss)
        # print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

        # Exit if no significant change in model weights
        #if n_iter > 0 and np.sum(np.square(w - ws[n_iter-1])) < epsilon:
        #    print(n_iter)
        #    break

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, epsilon=0.01):
    """The Gradient Descent (GD) algorithm for logistic regression."""
    return logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_, epsilon)
