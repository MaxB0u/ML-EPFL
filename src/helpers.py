import numpy as np
import pandas as pd


def compute_loss(y, tx, w):
    # Compute MSE loss

    e = y - tx @ w
    loss = 1 / (2 * len(y)) * (e.T @ e)

    return loss


def compute_loss_mae(y, tx, w):
    # Compute MAE loss
    e = y - tx @ w
    loss = np.mean(np.abs(e))
    return loss


def split_data(dataframe, k):
    '''
        Splitting dataset into training and validation set
    '''
    df = dataframe.copy()

    # y data is assumed to be in the last column of the dataframe
    y_val = df[k].loc['Prediction']
    x_val = df[k].drop(['Id', 'Prediction'], axis=1)

    # Remove the val data, the rest is used for training
    df.pop(k)
    df_tr = pd.concat(df)
    y_tr = df_tr.loc['Prediction']
    x_tr = df_tr.drop(['Id', 'Prediction'], axis=1)

    return x_tr, y_tr, x_val, y_val


