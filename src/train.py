from src.helpers import *
import numpy as np


class KFoldCrossValidation:
    '''
        Class for K-Fold Cross Validation
    '''

    def __init__(self, df_data):
        self.df_data = df_data

    def run(self, k_folds=10, verbose=True, lambda_=0, reg_type=''):
        # shuffle data (samples all data in random order)
        df_shuffled = self.df_data.sample(frac=1)

        # split fractions for 10 (default) fold cross validations
        split_set = np.array_split(df_shuffled, k_folds)

        # train and validation accuracy for each fold
        loss_val = np.zeros(k_folds)
        loss_tr = np.zeros(k_folds)

        # Save trained models for each fold
        weights = []

        for k in range(k_folds):

            # Get data for the fold
            x_tr, y_tr, x_val, y_val = split_data(split_set, k)

            # Would be nice if the model was a separate class
            # See the models.py file to see the interface for it
            w = ridge_regression(y_tr, x_tr, lambda_)
            weights.append(w)

            loss_tr[i] = np.sqrt(2 * compute_loss(y_tr, x_tr, w))
            loss_val[i] = np.sqrt(2 * compute_loss(y_val, x_val, w))

            if verbose: print("Fold " + str(i)+ ", training_rmse = " + str(
                loss_tr[i]) + ", val. loss = " + str(loss_val[i]))

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


def hyperparam_search(params, df_data, k_folds, param_name=''):

    loss_tr_p = []
    loss_val_p = []

    for p in params:
        kfold = KFoldCrossValidation(df_data.copy())

        if param_name == 'lambda':
            _, _, _, loss_tr, loss_val = kfold.run(k_folds=k_folds, lambda_=p, verbose=False)

        loss_tr_p.append(loss_tr)
        loss_val_p.append(loss_val)

    best_loss = min(loss_val_p)
    best_p = params[np.argmin(loss_val_p)]

    return best_loss, best_p, loss_tr_p, loss_val_p
