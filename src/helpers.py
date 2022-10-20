import numpy as np
import math


def compute_loss(y, tx, w):
    # Compute MSE loss

    e = y - tx @ w
    loss = 1 / (2 * len(y)) * (e.T @ e)

    return loss[0][0]


def compute_loss_mae(y, tx, w):
    # Compute MAE loss
    e = y - tx @ w
    loss = np.mean(np.abs(e))
    return loss


def compute_loss_logistic(y, tx, w, lambda_=0):
    # Compute loss for logistic regression
    # Assumes y is either 0 or 1
    # loss = - y.T @ np.log(sigma(tx, w)) - (1 - y.T) @ np.log(1 - sigma(tx, w))
    Z = tx @ w
    # loss = 0
    # for z in Z:
    #    if z[0] < 0:
    #        loss = math.log(1 + math.exp(z[0]))
    #    else:
    #        loss = math.log(1 + math.exp(-z[0])) - math.log(math.exp(-z[0]))
    # loss -= y.T @ z
    loss = np.sum(np.log(1 + np.exp(Z))) - y.T @ Z
    return loss[0][0] / len(y) + lambda_ * np.sum(np.square(w))


def load_data():
    """load data."""
    data = np.loadtxt("dataEx3.csv", delimiter=",", skiprows=1, unpack=True)
    x = data[0]
    y = data[1]
    return x, y


def load_data(path_dataset, x_cols, y_col, id_col):
    """Load data and convert it to the metric system."""

    x = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=x_cols)
    y = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=y_col, dtype=str
    )
    id = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=id_col)

    return x, y, id


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def robust_standardize(x):
    median = np.median(x, axis=0)
    p25 = np.percentile(x, 25, axis=0)
    p75 = np.percentile(x, 75, axis=0)

    x = (x - median) / (p75 - p25)

    return x


def build_model_data(x, y):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return tx


def compute_gradient_logistic(y, tx, w):
    """Computes the gradient at w."""
    # w is 1xd, tx is nxd -> sigma is 1xn
    sig = sigmoid(tx, w)
    # sigma is nx1, y is nx1, tx is nxd -> grad is 1xd
    grad = tx.T @ (sig - y)

    return grad / len(y)


def sigmoid(tx, w):
    # To prevent overflow
    Z = tx @ w
    # for z in Z:
    #    if z[0] < 0:
    #        sig = math.exp(z[0]) / (1 + math.exp(z[0]))
    #    else:
    #        sig = 1 / (1 + math.exp(-z[0]))
    # return sig
    return 1 / (1 + np.exp(-Z))


def batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True):
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def build_k_indices(y, k_fold, seed=-1):
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    # Used seed for reproducibility if needed
    if seed != -1:
        np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def scale(x):
    x = x / np.amax(np.abs(x), axis=0)
    return x


def remove_outliers(x, y, num_std_dev):
    std_x = np.std(x, axis=0)
    x_no_outliers = []
    y_no_outliers = []
    for i in range(len(y)):
        add = True
        for j in range(len(x[0])):
            if x[i][j] < -num_std_dev * std_x[j] or x[i][j] > num_std_dev * std_x[j]:
                add = False
                break

        if add:
            x_no_outliers.append(x[i])
            y_no_outliers.append(y[i])

    x_no_outliers = np.array(x_no_outliers)
    y_no_outliers = np.array(y_no_outliers)
    return x_no_outliers, y_no_outliers


def clip_outliers(x, y, num_std_dev):
    std_x = np.std(x, axis=0)
    for i in range(len(y)):
        add = True
        for j in range(len(x[0])):
            if x[i][j] < -num_std_dev * std_x[j]:
                x[i][j] = -num_std_dev * std_x[j]
            elif x[i][j] > num_std_dev * std_x[j]:
                x[i][j] = num_std_dev * std_x[j]

    return x, y


def impute_outliers(x, y, num_std_dev):
    std_x = np.std(x, axis=0)
    median_x = np.median(x, axis=0)
    for i in range(len(y)):
        for j in range(len(x[0])):
            if x[i][j] < -num_std_dev * std_x[j] or x[i][j] > num_std_dev * std_x[j]:
                x[i][j] = median_x[j]

    return x, y


def replace_invalid_values(x):
    mean_x = np.mean(x, axis=0)
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i][j] == -999.0:
                x[i][j] = mean_x[j]

    return x




def compute_mse_loss(y, tx, w):
    
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    error = y - tx@ w
    mse = np.sum(np.square(error))
    num_samples = y.shape[0]
    return mse/(2*num_samples)

def compute_gradient_linear_regression(y, tx, w):
    """Computes the gradient at w.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        
    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """

    num_data_points = y.shape[0]
    error = y - tx@ w
    gradient = -1*((np.transpose(tx))@ error)/(num_data_points)
    return gradient
    

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD 
    """

    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient_linear_regression(y, tx, w)
        loss = compute_mse_loss(y, tx, w)
        w = w - gamma*gradient
        ws.append(w)
        losses.append(loss)
    return losses, ws

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]



def stochastic_gradient_descent_linear_regression(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).
            
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD 
    """

    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        iterator = batch_iter(y, tx, batch_size)
        loss = compute_mse_loss(y, tx, w)
        y_batch, x_batch = next(iterator)
        stoch_gradient = compute_gradient_linear_regression(y_batch, x_batch, w)
        w = w - gamma*stoch_gradient
        losses.append(loss)
        ws.append(w)
    return losses, ws