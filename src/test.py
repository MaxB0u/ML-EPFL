from src.helpers import *


def test(x, w, model_name, data_id):
    """Given a pandas dataframe and a model
    Predicts output on a test set and formats the output for submission

    Args:
        x: shape=(N, D) -> x-data
        y:  shape=(N, 1) -> y-data
        model_name: str -> Name of the model to use
        data_id: shape(N,1) -> IDs of the data to predict on

    Returns:
        res: shape(N,2) -> Data id and corresponding predictions
    """
    pred = get_predictions(x, w, model_name)

    # Predictions are in the {-1,1} set
    pred = np.sign(pred - 0.5)

    # id and pred are 1xn arrays
    res = np.stack((data_id.flatten(), pred.flatten()), axis=1)

    # Write the result to a csv file

    np.savetxt(
        "./dataset/submission.csv",
        res,
        delimiter=",",
        fmt="%i",
        header="Id,Prediction",
        comments="",
    )

    return res


def get_predictions(x, w, model_name):
    """Gets predictions using either linear or logistic regression.
    Args:
        x: shape=(N, D) -> x-data
        y:  shape=(N, 1) -> y-data
        model_name: str -> Name of the model to use

    Returns:
        pred: shape(N,1) -> Label predictions
    """
    # Linear regression has a decision threshold of 0 (symmetric around 0) since y-values are unbounded
    # Logistic regression has a decision threshold of 0.5 (logistic function)
    # Could be adapted if class imbalance in the data

    if model_name == "logistic_regression" or model_name == "reg_logistic_regression":
        y_hat = sigmoid(x @ w)
        pred = (y_hat > 0.5) * 1.0
    else:
        y_hat = x @ w
        pred = (y_hat > 0.5) * 1.0

    return pred
