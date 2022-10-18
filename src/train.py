from implementations import *
from src.test import get_predictions


class KFoldCrossValidation:
    '''
        Class for K-Fold Cross Validation
        x : x-data
        y: y-data
        model_name: string with the name of the model to use for training
        params: dict with the params needed in the model (lambda, gamma, initial_w, etc)
    '''

    def __init__(self, x, y, model_name, params):
        self.x = x
        self.y = y
        self.model_name = model_name
        self.params = params

    def run(self, k_folds=5, verbose=True):
        # Shuffle data
        k_indices = build_k_indices(self.y, k_folds)

        # train and validation accuracy for each fold
        loss_val = np.zeros(k_folds)
        loss_tr = np.zeros(k_folds)
        acc_val = np.zeros(k_folds)
        acc_tr = np.zeros(k_folds)

        # Save trained models for each fold
        weights = []

        for k in range(k_folds):

            # Get data for the fold
            x_val = self.x[k_indices[k]]
            y_val = self.y[k_indices[k]]
            tr_indices = np.delete(k_indices, k, axis=0).flatten()
            x_tr = self.x[tr_indices]
            y_tr = self.y[tr_indices]

            # Would be nice if the model was a separate class
            # See the models.py file to see the interface for it
            w, _ = get_model_weights(x_tr, y_tr, self.params, self.model_name)
            weights.append(w)

            if self.model_name == 'logistic_regression':
                loss_tr[k] = compute_loss_logistic(y_tr, x_tr, w)
                loss_val[k] = compute_loss_logistic(y_val, x_val, w)
            elif self.model_name == 'reg_logistic_regression':
                loss_tr[k] = compute_loss_logistic(y_tr, x_tr, w, self.params['lambda_'])
                loss_val[k] = compute_loss_logistic(y_val, x_val, w, self.params['lambda_'])
            else:
                loss_tr[k] = np.sqrt(2 * compute_loss(y_tr, x_tr, w))
                loss_val[k] = np.sqrt(2 * compute_loss(y_val, x_val, w))

            acc_tr[k] = np.sum((y_tr == get_predictions(x_tr, w, self.model_name)) * 1.0) / len(y_tr)
            acc_val[k] = np.sum((y_val == get_predictions(x_val, w, self.model_name)) * 1.0) / len(y_val)

            #print(y_tr)
            #print(get_predictions(x_val, w, self.model_name))
            #print(np.sum((y_tr == get_predictions(x_tr, w, self.model_name)) * 1.0))
            #print(len(y_tr))

            if verbose:
                print("Fold " + str(k)+ ", training loss = " + str(loss_tr[k]) + ", training acc = " + str(acc_tr[k]) +
                      ", val loss = " + str(loss_val[k]) + ", val acc = " + str(acc_val[k]))

        # average accuracy
        loss_tr_avg = np.average(loss_tr)
        loss_val_avg = np.average(loss_val)

        # average performance over all folds
        if verbose:
            print("\nAverage training loss:")
            print(str(loss_tr_avg))

            print("\nAverage validation loss:")
            print(str(loss_val_avg))

        return weights, loss_tr, loss_val, loss_tr_avg, loss_val_avg


def get_model_weights(x, y, params, model_name):
    w = np.zeros(len(x[0]))
    loss = 0
    if model_name == 'ridge_regression':
        w, loss = ridge_regression(y, x, params['lambda_'])
    elif model_name == 'logistic_regression':
        w, loss = logistic_regression(y, x, params['initial_w'], params['max_iters'], params['gamma'])
    elif model_name == 'reg_logistic_regression':
        w, loss = reg_logistic_regression(y, x, params['lambda_'], params['initial_w'], params['max_iters'], params['gamma'])

    return w, loss

