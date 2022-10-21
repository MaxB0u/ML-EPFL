import numpy as np


def compute_mse_loss(y, tx, w):
    """Compute MSE loss.

    Args:
        y: shape(N,1) -> y-data
        tx: shape(N,D) -> x-data
        w: shape(D,1) -> model weights

    Returns:
        loss: float -> MSE loss
    """
    # Compute MSE loss
    e = y - (tx @ w)
    loss = 1 / (2 * len(y)) * (e.T @ e)

    return loss[0][0]


def compute_loss_mae(y, tx, w):
    """Compute MAE loss

        Args:
        y: shape(N,1) -> y-data
        tx: shape(N,D) -> x-data
        w: shape(D,1) -> model weights

    Returns:
        loss: float -> MAE loss
    """
    e = y - (tx @ w)
    loss = np.mean(np.abs(e))
    return loss


def compute_loss_logistic(y, tx, w):
    """Compute the cross-entropy loss for logistic regression

        Args:
        y: shape(N,1) -> y-data
        tx: shape(N,D) -> x-data
        w: shape(D,1) -> model weights

    Returns:
        loss: float -> Logistic regression loss
    """
    # Compute loss for logistic regression
    # Assumes y is either 0 or 1
    Z = tx @ w
    loss = np.sum(np.log(1 + np.exp(Z))) - y.T @ Z
    return loss[0][0] / len(y)


def load_data(path_dataset, x_cols, y_col, id_col):
    """Load data and convert it to the metric system.

    Args:
        path_dataset:  str -> Path to load the dataset from
        x_cols: tuple -> Tuple with the number of all the columns to use for the x-data in the dataset
        y_col: tuple -> Tuple with the number of the column to use in the dataset as the y-data
        id_col: tuple -> Tuple with the number of the column to use in the dataset as the IDs

    Returns:
        x: shape(N,D) -> Preprocessed x-data
        y: shape(N,1) -> Preprocessed y-data
        data_id: shape(N,1) -> IDs of the x and y data
    """

    x = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=x_cols)
    y = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=y_col, dtype=str
    )
    data_id = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=id_col)

    return x, y, data_id


def standardize(x):
    """Standardize the original data set.

    Args:
        x: shape(N,D) -> x-data

    Returns:
        x: shape(N,D) -> x-data with mean 0 and variance 1 for each feature mean_x: shape(D,1) -> Mean of each
    of the features before standardization
        std_x: shape(D,1) -> Standard deviation of each of the features before
    standardization assuming Gaussian distribution
    """
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def robust_standardize(x):
    """Standardize the original data set with robust statistics to outliers.

    Args:
        x: shape(N,D) -> x-data

    Returns:
        x: shape(N,D) -> x-data standardized with robust statistics to outliers
    """
    median = np.median(x, axis=0)
    p25 = np.percentile(x, 25, axis=0)
    p75 = np.percentile(x, 75, axis=0)

    x = (x - median) / (p75 - p25)

    return x


def build_model_data(x, y):
    """Form (y,tX) to get regression data in matrix form.

    Args:
        x: shape(N,D) -> x-data
        y: shape(N,1) -> y-data

    Returns:
        tx: shape(N,D+1) -> x-data with a prepended column of 1's (w_0)
    """
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return tx


def compute_gradient_logistic(y, tx, w):
    """Computes the gradient at w.

        Args:
        y: shape(N,1) -> y-data
        tx: shape(N,D) -> x-data
        w: shape(D,1) -> model weights

    Returns:
        grad: shape(D,1) -> Gradient for logistic regression
    """
    # w is 1xd, tx is nxd -> sigma is 1xn
    sig = sigmoid(tx @ w)
    # sigma is nx1, y is nx1, tx is nxd -> grad is 1xd
    grad = tx.T @ (sig - y)

    return grad / len(y)


def compute_hessian_logistic(tx, w):
    """Computes the gradient at w.

    Args:
        tx: shape=(N, D) -> x-data
        w:  shape=(D, 1) -> model weights

    Returns:
        h: shape(D, D) -> The Hessian of X
    """
    z = sigmoid(tx @ w)
    s = np.diag((z * (1 - z)).T[0])
    h = tx.T @ s @ tx

    return 0.5 * h


def sigmoid(t):
    """Sigmod of a scalar, or possibly sigmoid applied element-wise to a vector

    Args:
        t: shape(N,1) -> x @ w in logistic regression

    Returns:
        sigmoid(t): shape(N,1) -> Element-wise sigmoid of the array t

    """
    return np.exp(t) / (1 + np.exp(t))


def batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True):
    """Get an iterator that returns batches of the data for SGD or mini-batch gd.

    Args:
            y:  shape=(N, 1) -> y-data
            tx: shape=(N, D) -> x-data
            num_batches: int -> NUmber of batches to return in the iterator
            batch_size: int -> Number of samples on which to calculate the gradient
            shuffle: bool -> Whether to shuffle the data before sampling or not

    Returns:
        Iterator of the x and y data batches
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


def build_k_indices(y, k_fold, seed=-1):
    """Get k indices used to split data for cross validation.


    Args:
            y:  shape=(N, 1) -> y-data
            k_fold: int -> Number of folds that will be used in the cross validation
            seed: int -> Random seed that can be set for reproducibility. Set by default to -1 (not used)

    Returns:
        k_indices: shape(k_fold, N/k_fold) -> Data indices to use for each fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    # Used seed for reproducibility if needed
    if seed != -1:
        np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def scale(x):
    """Scale each feature so its values lie in [0,1].

    Args:
        x: shape(N,D) -> x-data

    Returns:
        x: shape(N,D) -> x-data that has been scaled to lie in [0,1] for each feature
    """
    x = x / np.amax(np.abs(x), axis=0)
    return x


def remove_outliers(x, y, num_std_dev):
    """Remove samples with outliers.

    Args:
        x: shape(N,D) -> x-data
        y: shape(N,1) -> y-data
        num_std_dev: float -> The number of standard deviations from which data is considered to be an outlier
        (assuming Gaussian distribution)

    Returns:
        x_no_outliers: shape(N',D) -> x-data without outliers
        y_no_outliers: shape(N',1) -> Corresponding y-data
    """
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
    """Clip the outliers to a specified number of standard deviations.

    Args:
        x: shape(N,D) -> x-data
        y: shape(N,1) -> y-data
        num_std_dev: float -> The number of standard deviations from which data is considered to be an outlier
        (assuming Gaussian distribution)

    Returns:
        x: shape(N,D) -> x-data with clipped outliers
        y: shape(N,1) -> Corresponding y-data
    """
    std_x = np.std(x, axis=0)
    for i in range(len(y)):
        for j in range(len(x[0])):
            if x[i][j] < -num_std_dev * std_x[j]:
                x[i][j] = -num_std_dev * std_x[j]
            elif x[i][j] > num_std_dev * std_x[j]:
                x[i][j] = num_std_dev * std_x[j]

    return x, y


def impute_outliers(x, y, num_std_dev):
    """Replace outliers y the median of their feature.

    Args:
        x: shape(N,D) -> x-data
        y: shape(N,1) -> y-data
        num_std_dev: float -> The number of standard deviations from which data is considered to be an outlier
        (assuming Gaussian distribution)

    Returns:
        x: shape(N,D) -> x-data with outliers replaced by the median of the features
        y: shape(N,1) -> Corresponding y-data
    """
    std_x = np.std(x, axis=0)
    median_x = np.median(x, axis=0)
    for i in range(len(y)):
        for j in range(len(x[0])):
            if x[i][j] < -num_std_dev * std_x[j] or x[i][j] > num_std_dev * std_x[j]:
                x[i][j] = median_x[j]

    return x, y


def replace_invalid_values(x):
    """Replace invalid values (-999.0) by the median of the feature.

    Args:
        x: shape(N,D) -> x-data

    Returns:
        x: shape(N,D) -> x-data with invalid values replaced by the median of the features
    """
    median_x = np.median(x, axis=0)
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i][j] == -999.0:
                x[i][j] = median_x[j]

    return x


def compute_gradient(y, tx, w):
    """Computes the gradient in linear regression.

        Args:
        y: shape(N,1) -> y-data
        tx: shape(N,D) -> x-data
        w: shape(D,1) -> model weights

    Returns:
        grad: shape(D,1) -> Gradient for linear regression
    """
    e = y - tx @ w
    return -1 / len(y) * (tx.T @ e)


def build_poly(x, degree):
    """Polynomial feature expansion of x.

        Args:
        x: shape(N,D) -> x-data
        degree: int -> degree of the resulting polynomial

    Returns:
        poly: shape(N,D*d) -> Polynomial expansion of x
    """
    poly = np.hstack([np.vstack(x**d) for d in range(1, degree + 1)])
    return poly


def log_features(x):
    """Feature expansion with the log of the input.

        Args:
        x: shape(N,D) -> x-data

    Returns:
        x: shape(N,D) -> Element-wise log of the original x-data
    """
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i][j] > 0:
                x[i][j] = np.log(x[i][j])
            elif x[i][j] < 0:
                x[i][j] = -np.log(-x[i][j])
    return x


def get_pca_transformation(x, var_needed):
    """Get the transformation matrix of the input to be used for PCA.

        Args:
        x: shape(N,D) -> x-data
        var_needed: float -> Explained variance of the original space in the transformed space

    Returns:
        W: shape(D,D') -> Transformation matrix for X
    """
    cov = np.cov(x.T)
    eig_val, eig_vect = np.linalg.eig(cov)

    var_explained = []
    sum_eig = sum(eig_val)
    for eig in eig_val:
        var_explained.append(eig / sum_eig * 100)

    # Get the projection matrix
    num_comp = 0
    tot_var = 0
    for v in var_explained:
        num_comp += 1
        tot_var += v
        if tot_var > var_needed:
            break

    W = (eig_vect[:num_comp][:]).T
    # print(var_explained)

    return W
