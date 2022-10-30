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


def load_headers(path_dataset, column_indxs):
    """Loads the headers in the dataset file

    Args:
        path_dataset:   str -> Path to the file containing the dataset
        column_indxs:   Tuple -> Tuple with the indexes of columns that contains feature information

    Returns:
        headers:    List -> Returns the list of header names for the dataset
    """
    with open(path_dataset, "r") as datareader:
        header_line = datareader.readline().strip()
        headers = header_line.split(",")
        headers = [headers[c_idx] for c_idx in column_indxs]

        return headers


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
    s = (z * (1 - z)).T[0]
    h = (tx.T * s) @ tx

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

    # Sort eigenvectors based on the descencing order of their corresponding eigenvalues
    # That way we could capture the required cumulative variance in the least number of components
    idx = eig_val.argsort()[::-1]
    eig_val = eig_val[idx]
    eig_vect = eig_vect[:, idx]

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


def get_pca_transformation_with_dim(x, expected_dim):
    """Get the transformation matrix of the input to be used for PCA.

        Args:
        x: shape(N,D) -> x-data
        expected_dim: Int -> The final dimension of features D'

    Returns:
        W: shape(D,D') -> Transformation matrix for X
    """
    cov = np.cov(x.T)
    eig_val, eig_vect = np.linalg.eig(cov)

    # Sort eigenvectors based on the descencing order of their corresponding eigenvalues
    # That way we could capture the required cumulative variance in the least number of components
    idx = eig_val.argsort()[::-1]
    eig_val = eig_val[idx]
    eig_vect = eig_vect[:, idx]

    if expected_dim > len(eig_val):
        print(
            "Error: Expected dimenstions {}, Number of eigenvalues in the input {}"
            .format(expected_dim, len(eig_val))
        )
        return None

    W = (eig_vect[:expected_dim][:]).T
    # print(var_explained)

    return W


def compare_train_validation_err_logistic(x, y, params):
    """
    Splits data into training and validation for logistic regression to compare the loss values
    over different training iterations

    Args:
        y: shape(N,1) -> y-data
        x: shape(N,D) -> x-data
        params: dict -> training / validation parameters

    Returns:
        losses_tr, losses_val:  Tuple[np.array, np.array]: Tuple of training and validation losses
        over multiple training iterations
    """

    losses_tr = []
    losses_val = []
    max_iters = int(params["max_iters"])
    gamma = float(params["gamma"])
    lambda_ = float(params["lambda_"])
    split_ratio = float(params["split_ratio"])
    split_idx = int(y.shape[0] * split_ratio)
    w = np.array([np.array([[0.0] for _ in range(len(x[0]))])])
    grad = 0
    newton_method = False

    for n_iter in range(max_iters):
        indices = np.random.permutation(np.arange(y.shape[0]))
        y_train = y[indices[:split_idx]]
        x_train = x[indices[:split_idx]]
        y_val = y[indices[split_idx + 1 :]]
        x_val = x[indices[split_idx + 1 :]]
        # Compute gradient and loss
        grad = compute_gradient_logistic(y_train, x_train, w)

        # Update gradient
        if newton_method:
            # No regularization for Newton's method
            h = compute_hessian_logistic(x_train, w)
            diff = np.linalg.solve(h, gamma * grad)
            w -= diff
        else:
            w -= gamma * (grad + 2 * lambda_ * w)

        # Compute loss on training and validation sets
        loss_tr = compute_loss_logistic(y_train, x_train, w)
        loss_val = compute_loss_logistic(y_val, x_val, w)
        losses_tr.append(loss_tr)
        losses_val.append(loss_val)

    return losses_tr, losses_val


def get_euclidean_distance(x1, x2):
    """Get the squared euclidean distance for KNN

        Args:
        x1: shape(1,D) -> first data point
        x2: shape(1,D) -> second data point

    Returns:
        dist: float -> Euclidean distance between x1 and x2
    """
    dist = np.sum(np.square(x1 - x2), axis=1)
    return dist


def predict_knn(y, x_tr, x_te, k):
    """Get the k nearest neighbors of x_te

        Args:
        y : shape(N,1) -> y-data of training data
        x_tr: shape(N,D) -> x-data used for training
        x_te: shape(N',D) -> x-data used for validation or testing
        k: int -> Number of nearest neighbors to keep, should be odd

    Returns:
        neighbors: shape(N',k) -> class of the k nearest neighbors for each of the testing data points
    """
    predictions = np.zeros(len(x_te))
    for i in range(len(x_te)):
        distances = get_euclidean_distance(x_tr, x_te[i])
        idx = np.argpartition(distances, k)
        pred = np.sign((np.mean(y[idx[:k]]) - 0.5))
        predictions[i] = pred

    return predictions


def get_raw_data(path_dataset):
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

    x = np.genfromtxt(path_dataset, delimiter=",", usecols=tuple(range(2, 32)))

    return x


def get_f1_score(y, y_hat):
    TP, FP, FN = 0, 0, 0
    for i in range(len(y)):
        TP += (y[i] == y_hat[i] and y[i] == 1) * 1.0
        FP += (y[i] != y_hat[i] and y_hat[i] == 1) * 1.0
        FN += (y[i] != y_hat[i] and y_hat[i] == -1) * 1.0

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
