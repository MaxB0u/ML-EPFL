from src.helpers import *


def test(x, w, model_name, id):
    """ Given a pandas dataframe and a model
        Predicts output on a test set and formats the output for submission
        """
    pred = get_predictions(x, w, model_name)

    if model_name == 'logistic_regression' or model_name == "reg_logistic_regression":
        pred = np.sign(pred - 0.5)

    # id and pred are 1xn arrays
    res = np.stack((id.flatten(), pred.flatten()), axis=1)

    # Write the result to a csv file
    np.savetxt('./dataset/submission.csv', res, delimiter=',', fmt='%i', header='Id,Prediction', comments='')

    return res


def get_predictions(x, w, model_name):
    # Linear regression has a decision threshold of 0 (symmetric around 0) since y-values are unbounded
    # Logistic regression has a decision threshold of 0.5 (logistic function)
    # Could be adapted if class imbalance in the data
    pred = np.zeros(len(x))
    if model_name == 'ridge_regression':
        y_hat = x @ w
        pred = np.sign(y_hat)
    elif model_name == 'logistic_regression' or model_name == 'reg_logistic_regression':
        y_hat = sigmoid(x, w)
        pred = (y_hat > 0.5) * 1.0
        #pred = np.zeros((len(y_hat), 1))

    return pred