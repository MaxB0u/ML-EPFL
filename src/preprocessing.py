from src.helpers import *


def preprocess(path_dataset):
    # For now use all columns, later do something better
    print("Loading data from " + path_dataset)
    # x_col = tuple(range(2, 32))
    x_col = tuple(range(2, 16))
    y_col = 1
    id_col = 0

    x, y, id = load_data(path_dataset, x_col, y_col, id_col)
    print("Finished loading data")

    print("Preprocessing...")
    y = np.array([[1 if x == "s" else -1] for x in y])
    # if path_dataset == './dataset/train.csv':
    #    num_std_dev = 3
    #    #x, y = remove_outliers(x, y, num_std_dev)
    #    x, y = clip_outliers(x, y, num_std_dev)
    #    x, _, _ = standardize(x)
    # else:
    #    num_std_dev = 2
    #    x, y = clip_outliers(x, y, num_std_dev)
    #    x, _, _ = standardize(x)
    x = replace_invalid_values(x)

    # x = robust_standardize(x)

    num_std_dev = 3
    # x, y = remove_outliers(x, y, num_std_dev)
    # x, y = impute_outliers(x, y, num_std_dev)
    x, y = clip_outliers(x, y, num_std_dev)
    x, _, _ = standardize(x)
    # Take only the main PCA
    # x = x @ get_explained_variance(x, 95)

    # x = scale(x)

    x = build_poly(x, 4)
    # x, _, _ = standardize(x)
    x = build_model_data(x, y)
    print("Finished preprocessing")

    # test_decisiontree(tx, y.T[0])
    print(x.shape)
    print(y.shape)

    return x, y, id


def get_explained_variance(x, var_needed):
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
