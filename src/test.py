def test(dataframe, model):
    """ Given a pandas dataframe and a model
        Predicts output on a test set and formats the output for submission
        """
    df = dataframe.copy()

    x_te = df.drop(['Id', 'Prediction'], axis=1)
    y_te = model.predict(x_te)

    df_pred = df.loc['Id'].copy()
    df_pred['Prediction'] = y_te

    return df_pred
