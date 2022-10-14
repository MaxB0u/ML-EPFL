from src.helpers import *


def logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_=0):
    """The Gradient Descent (GD) algorithm for logistic regression."""

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    if max_iters == 0:
        loss = compute_loss_logistic(y, tx, w)
        losses.append(loss)
    sgd = False
    for n_iter in range(max_iters):
        # Compute gradient and loss
        if sgd:
            # If no batch size is passed it defaults to 1 which is the case for SGD
            for batch_y, batch_tx in batch_iter(y, tx):
                grad = compute_gradient_logistic(batch_y, batch_tx, w)
                # In SGD loss is computed only on sampled data
                loss = compute_loss(batch_y, batch_tx, w)
        else:
            grad = compute_gradient_logistic(y, tx, w)
            loss = compute_loss_logistic(y, tx, w)

        # Update gradient
        w -= gamma * grad + 2*lambda_*w

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm for logistic regression."""
    return logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_)


#y = np.array([[0.], [1.], [1.]])
#tx = np.array([[2.3, 3.2], [1., 0.1], [1.4, 2.3]])
#initial_w = np.array([[0.463156], [0.939874]])
#MAX_ITERS = 0
#GAMMA = 0.1

#print(logistic_regression(y, tx, initial_w, MAX_ITERS, GAMMA))