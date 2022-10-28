from src.train import *
from src.test import *
from src.preprocessing import *
from src.plots import *
from src.helpers import *
import sys


def run():
    """Runs the training/testing pipeline

    Returns:
         predictions: shape=(N, 2) -> Data id and corresponding predictions
    """
    # Paths of the datasets to use for training and testing
    path_dataset_tr = "./dataset/train.csv"
    path_dataset_te = "./dataset/test.csv"

    # Cha=eck if a model was specified on the command line or not
    if len(sys.argv) == 2:
        model_name = sys.argv[1]
    else:
        model_name = "logistic_regression"

    experimenting = True

    print("Training model...")
    trained_weights = train_model(path_dataset_tr, model_name, experimenting)

    print("Testing model...")
    predictions = test_model(path_dataset_te, model_name, trained_weights)

    return predictions


def train_model(path_dataset_tr, model_name, experimenting=False):
    """Trains a given model and return the trained weights.
    Uses grid search to select the best hyperparameters when experimenting.
    Args:
        path_dataset_tr:  str -> Path to load the training dataset from
        model_name: str -> Name of the model to use
        experimenting: bool -> Whether to search for the best hyperparameters or not

    Returns:
        w: shape=(D, 1) -> The trained weights of the model
    """

    if model_name == "KNN" or model_name == "knn":
        x, y, _ = preprocess_knn(path_dataset_tr)
        params = dict()
        num_neighbors = range(1, 7, 2)
        best_k = 1
        best_acc = 0
        for k in num_neighbors:
            params["k"] = k
            kfold = KFoldCrossValidation(x, y, model_name, params)
            _, acc_val_avg = kfold.knn()
            if acc_val_avg > best_acc:
                best_k = k
        return best_k

    x, y, _ = preprocess(path_dataset_tr)
    # y should be either 0 or 1 when training
    y = (y > 0) * 1.0

    if experimenting:

        # Get the best hyperparameters
        # Gamma (and lambda if needed)
        gammas = [1.0]
        lambdas = [1.0]
        params = dict()

        # Specify model parameters
        if (
            model_name == "logistic_regression"
            or model_name == "reg_logistic_regression"
            or model_name == "mean_squared_error_gd"
            or model_name == "mean_squared_error_sgd"
        ):
            gammas = np.logspace(-2, -1, 5)
            params["initial_w"] = np.array([[0.0] for _ in range(len(x[0]))])
            params["max_iters"] = 20

        if model_name == "ridge_regression" or model_name == "reg_logistic_regression":
            lambdas = np.logspace(-4, -1, 3)

        # Grid search for hyperparameters
        rmse_tr = []
        rmse_val = []
        for lambda_ in lambdas:
            rmse_tr_g = []
            rmse_val_g = []
            params["lambda_"] = lambda_
            for gamma in gammas:
                params["gamma"] = gamma
                kfold = KFoldCrossValidation(x, y, model_name, params)
                weights, _, _, loss_tr_avg, loss_val_avg = kfold.run()
                rmse_tr_g.append(loss_tr_avg)
                rmse_val_g.append(loss_val_avg)
            rmse_tr.append(rmse_tr_g)
            rmse_val.append(rmse_val_g)

        min_idx = np.argmin(rmse_val)
        best_lambda = lambdas[min_idx // len(gammas)]
        best_gamma = gammas[min_idx % len(gammas)]
        best_rmse = min(np.array(rmse_val).flatten())

        params["lambda_"] = best_lambda
        params["gamma"] = best_gamma
        #  Calculate accuracy on the trained model
        print(
            "The best parameters are: Lambda = "
            + str(best_lambda)
            + ", Gamma = "
            + str(best_gamma)
            + " yielding an loss of "
            + str(best_rmse)
        )

    else:
        params = dict()
        params["initial_w"] = np.array([[0.0] for _ in range(len(x[0]))])
        params["max_iters"] = 100
        params["lambda_"] = 0.1
        params["gamma"] = 0.01

    w, loss = get_model_weights(x, y, params, model_name)
    acc = np.sum((y == get_predictions(x, w, model_name)) * 1.0) / len(y)

    print("Accuracy over the training set:" + str(acc))

    return w


def test_model(path_dataset_te, model_name, trained_weights):
    """Generates predictions on the test set.
    Args:
        path_dataset_te:  str -> Path to load the test dataset from
        model_name: str -> Name of the model to use
        trained_weights:  shape=(D, 1) -> Trained weights of the model

    Returns:
        predictions: shape=(N, 2) -> Data id and corresponding predictions
    """
    # Get and preprocess the test data
    x, y, data_id = preprocess(path_dataset_te)
    # Put the ids in a 2d array
    data_id = np.reshape(data_id, (len(data_id), 1))

    # Re-train model with all parameters
    predictions = test(x, trained_weights, model_name, data_id)

    return predictions


def visualize(model_name="knn"):
    if model_name == "knn" or model_name == "KNN":
        x_tr, y_tr, _ = preprocess_knn("./dataset/train.csv")
        k_indices = build_k_indices(y_tr, 1)
        x_tr = scale(x_tr)

        # Shuffle data before sampling
        x_tr = x_tr[k_indices[0], :2]
        y_tr = y_tr[k_indices[0], :2]
        # Sample
        sample = 2500

        print(x_tr.shape)
        print(x_tr[:, 0])
        print(x_tr[:, 1])

        k = 3
        visualize_knn(y_tr, x_tr, 3)

    elif model_name == "logistic_regression":
        x, y, _ = preprocess("./dataset/train.csv")
        # y should be either 0 or 1 when training
        y = (y > 0) * 1.0

        params = {"max_iters": 250, "gamma": 0.036, "lambda_": 0.00, "split_ratio": 0.9}
        losses_tr, losses_val = compare_train_validation_err_logistic(x, y, params)

        # Visualise the losses collected over the training iterations
        plot_train_validation_losses(np.array(losses_tr), np.array(losses_val))


def visualize_feature_covariance(path_dataset_tr):
    """
    Generates a visualisation plot for the correlation heatmap of the input features

    Args:
        path_dataset_tr:    str -> Path to the train dataset

    Returns:
        Shows a heatmap that represents the correlation coefficients amongst input features
    """
    columns_with_input_features = tuple(range(2, 32))
    y_column = 1
    id_column = 0

    x_tr, y_tr, _ = load_data(
        path_dataset_tr, columns_with_input_features, y_column, id_column
    )
    feature_headers = load_headers(path_dataset_tr, columns_with_input_features)

    plot_features_covariance(x_tr, feature_headers)


# Run the script
run()
