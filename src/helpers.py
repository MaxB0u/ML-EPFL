import numpy as np


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


def compute_loss_logistic(y, tx, w, lambda_ = 0):
    # Compute loss for logistic regression
    # Assumes y is either 0 or 1
    loss = - y.T @ np.log(sigma(tx, w)) - (1 - y.T) @ np.log(1 - sigma(tx, w))
    return loss[0][0] / len(y) + lambda_ * np.sum(np.square(w))


def load_data():
    """load data."""
    data = np.loadtxt("dataEx3.csv", delimiter=",", skiprows=1, unpack=True)
    x = data[0]
    y = data[1]
    return x, y


def load_data(path_dataset, x_cols_name, y_col_name):
    """Load data and convert it to the metric system."""

    x = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=x_cols_name)
    y = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=y_col_name,
        converters={0: lambda x: 0 if b"Male" in x else 1})

    return x, y


def standardize(x_raw):
    """Standardize the original data set."""
    mean_x = np.mean(x_raw)
    x = x_raw - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x


def build_model_data(x, y):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


def compute_gradient_logistic(y, tx, w):
    """Computes the gradient at w."""
    # w is 1xd, tx is nxd -> sigma is 1xn
    sig = sigma(tx, w)
    # sigma is nx1, y is nx1, tx is nxd -> grad is 1xd
    grad = tx.T @ (sig - y)

    return grad / len(y)


def sigma(tx, w):
    return 1 / (1 + np.exp(-(tx @ w)))


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



def split_data(dataframe, k):
    '''
        Splitting dataset into training and validation set
    '''
    df = dataframe.copy()

    # y data is assumed to be in the last column of the dataframe
    y_val = df[k].loc['Prediction']
    x_val = df[k].drop(['Id', 'Prediction'], axis=1)

    # Remove the val data, the rest is used for training
    df.pop(k)
    df_tr = pd.concat(df)
    y_tr = df_tr.loc['Prediction']
    x_tr = df_tr.drop(['Id', 'Prediction'], axis=1)

    return x_tr, y_tr, x_val, y_val


