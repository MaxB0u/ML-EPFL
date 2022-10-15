import numpy as np
from src.helpers import sigma


def test(x, w, model_name, id):
    """ Given a pandas dataframe and a model
        Predicts output on a test set and formats the output for submission
        """
    pred = get_predictions(x, w, model_name)

    print(pred.shape)
    print(id.shape)

    # id and pred are 1xn arrays
    res = np.stack((id, pred), axis=1)

    return res


def get_predictions(x, w, model_name):
    # Linear regression has a decision threshold of 0 (symmetric around 0) since y-values are unbounded
    # Logistic regression has a decision threshold of 0.5 (logistic function)
    # Could be adapted if class imbalance in the data
    pred = np.zeros(len(x))
    if model_name == 'ridge_regression':
        y_hat = x @ w
        pred = [1 if x > 0 else -1 for x in y_hat]
    elif model_name == 'logistic_regression' or model_name == 'reg_logistic_regression':
        y_hat = sigma(x, w)
        pred = (y_hat > 0.5) * 1.0

    return pred