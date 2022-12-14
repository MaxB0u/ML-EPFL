from src.utils import *


def preprocess(path_dataset):
    """Loads and preprocesses the data so it can then be used for training or testing.

    Args:
        path_dataset:  str -> Path to load the dataset from

    Returns:
        x: shape(N,D) -> Preprocessed x-data
        y: shape(N,1) -> Preprocessed y-data
        data_id: shape(N,1) -> IDs of the x and y data
    """
    # For now use all columns, later do something better
    print("Loading data from " + path_dataset)
    # x_col = tuple(range(2, 32))
    x_col = tuple(range(2, 16))
    y_col = 1
    id_col = 0

    try:
        x, y, data_id = load_data(path_dataset, x_col, y_col, id_col)
    except OSError:
        print(
            "Dataset not found. Please make sure train.csv and test.csv are located in"
            " the /dataset folder"
        )
        quit()

    print("Finished loading data")

    print("Preprocessing...")
    y = np.array([[1 if x == "s" else -1] for x in y])

    x = replace_invalid_values(x)
    # x = robust_standardize(x)

    num_std_dev = 3
    # x, y = remove_outliers(x, y, num_std_dev)
    # x, y = impute_outliers(x, y, num_std_dev)
    x, y = clip_outliers(x, y, num_std_dev)
    x, _, _ = standardize(x)
    # Take only the main PCA
    # x = x @ get_pca_transformation(x, 95)

    # x = scale(x)
    x = build_poly(x, 4)
    # x, _, _ = standardize(x)
    x = build_model_data(x, y)
    print("Finished preprocessing")

    return x, y, data_id


def preprocess_with_pca(path_dataset, num_samples=-1):
    """Loads and preprocesses the data so it can then be used for training or testing.

    Args:
        path_dataset:  str -> Path to load the dataset from
        num_samples:   int -> Number of data samples to load if left -1, it loads all available samples

    Returns:
        x: shape(N,D) -> Preprocessed x-data
        y: shape(N,1) -> Preprocessed y-data
        data_id: shape(N,1) -> IDs of the x and y data
    """
    # For now use all columns, later do something better
    print("Loading data from " + path_dataset)
    # x_col = tuple(range(2, 32))
    x_col = tuple(range(2, 16))
    y_col = 1
    id_col = 0

    x, y, data_id = load_data(path_dataset, x_col, y_col, id_col)
    if num_samples != -1:
        indices = np.random.permutation(np.arange(num_samples))
        x = x[indices[:num_samples]]
        y = y[indices[:num_samples]]
        data_id = data_id[indices[:num_samples]]
    print("Finished loading data")

    print("Preprocessing...")
    y = np.array([[1 if x == "s" else -1] for x in y])

    x = replace_invalid_values(x)
    # x = robust_standardize(x)

    num_std_dev = 3
    x, y = clip_outliers(x, y, num_std_dev)
    x, _, _ = standardize(x)
    # PCA for dimensionality reduction
    x = x @ get_pca_transformation_with_dim(x, 2)

    # x = scale(x)
    x = build_poly(x, 4)
    # x, _, _ = standardize(x)
    x = build_model_data(x, y)
    print("Finished preprocessing")

    return x, y, data_id


def preprocess_knn(path_dataset):
    """Loads and preprocesses the data so it can then be used for training or testing.
    For the special case of knn.

    Args:
        path_dataset:  str -> Path to load the dataset from

    Returns:
        x: shape(N,D) -> Preprocessed x-data
        y: shape(N,1) -> Preprocessed y-data
        data_id: shape(N,1) -> IDs of the x and y data
    """
    # For now use all columns, later do something better
    print("Loading data from " + path_dataset)
    # x_col = tuple(range(2, 32))
    x_col = tuple(range(2, 16))
    y_col = 1
    id_col = 0

    x, y, data_id = load_data(path_dataset, x_col, y_col, id_col)
    print("Finished loading data")

    print("Preprocessing...")
    y = np.array([[1 if x == "s" else -1] for x in y])

    x = replace_invalid_values(x)
    x, _, _ = standardize(x)

    # Take only the main PCA
    x = x @ get_pca_transformation(x, 99)

    # x = scale(x)
    # x = build_poly(x, 4)
    print("Finished preprocessing")

    return x, y, data_id
