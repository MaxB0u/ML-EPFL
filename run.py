from src.train import *
from src.test import *
from src.preprocessing import *


def run():
    path_dataset_tr = "./dataset/train.csv"
    path_dataset_te = "./dataset/test.csv"

    model_name = "logistic_regression"

    print("Training model...")
    trained_weights = train_model(path_dataset_tr, model_name)

    print("Testing model...")
    predictions = test_model(path_dataset_te, model_name, trained_weights)

    return predictions


def train_model(path_dataset_tr, model_name):

    x, y, _ = preprocess(path_dataset_tr)

    # Get the best hyperparameters
    # Gamma (and lambda if needed
    gammas = [1.0]
    lambdas = [1.0]
    params = dict()

    # Search for gamma if needed

    if model_name == 'logistic_regression' or model_name == 'reg_logistic_regression':
        gammas = np.logspace(-2, -1, 7)
        params['initial_w'] = np.array([[0.] for _ in range(len(x[0]))])
        params['max_iters'] = 100
        y = (y > 0) * 1.0

    if model_name == 'ridge_regression' or model_name == 'reg_logistic_regression':
        lambdas = np.logspace(-4, -1, 3)

    rmse_tr = []
    rmse_val = []
    for lambda_ in lambdas:
        rmse_tr_g = []
        rmse_val_g = []
        params["lambda_"] = lambda_
        for gamma in gammas:
            params["gamma"] = gamma
            kfold = KFoldCrossValidation(x, y, model_name, params)
            w, _, _, loss_tr_avg, loss_val_avg = kfold.run()
            rmse_tr_g.append((loss_tr_avg))
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
    w, loss = get_model_weights(x, y, params, model_name)
    acc = np.sum((y == get_predictions(x, w, model_name)) * 1.0) / len(y)
    print(
        "The best parameters are: Lambda = "
        + str(best_lambda)
        + ", Gamma = "
        + str(best_gamma)
        + " yielding an loss of "
        + str(best_rmse)
    )
    print("Accuracy over the training set:" + str(acc))

    return w


def test_model(path_dataset_te, model_name, trained_weights):
    x, y, id = preprocess(path_dataset_te)

    id = np.reshape(id, (len(id), 1))

    # Re-train model with all parameters
    predictions = test(x, trained_weights, model_name, id)

    return predictions


# Run the script
run()
