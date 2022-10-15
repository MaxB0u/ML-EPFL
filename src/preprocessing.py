from src.helpers import *


def preprocess(path_dataset):
    # For now use all columns, later do something better
    print("Loading data from " + path_dataset)
    x_col = tuple(range(2, 32))
    y_col = 1
    id_col = 0

    x_raw, y_raw, id = load_data(path_dataset, x_col, y_col, id_col)
    y = np.array([[1 if x == 's' else -1] for x in y_raw])
    x = standardize(x_raw)
    tx = build_model_data(x, y)

    print("Finished loading data")

    return tx, y, id