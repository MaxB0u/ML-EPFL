from implementations import *
from src.test import get_predictions


class KFoldCrossValidation:
    """
    Class for K-Fold Cross Validation
    """

    def __init__(self, x, y, model_name, params):
        """Constructor.
        Args:
            x: shape(N,D) -> x-data
            y: shape(N,1) -> y-data
            model_name: str -> The name of the model to use for training
            params: dict -> The params needed in the model (lambda, gamma, initial_w, etc)
        """
        self.x = x
        self.y = y
        self.model_name = model_name
        self.params = params

    def run(self, k_folds=5, verbose=True):
        """Runs k-fold cross validation on the data.
        Args:
            k_folds: int -> Indicates the number of folds
            verbose: bool -> Indicates whether to print information about the training or not

        Returns:
            weights: shape(k,D,1) -> The trained weights for each fold
            loss_tr: shape(k,1) -> The training loss for each fold
            loss_val: shape(k,1) -> The average loss for each fold
            loss_tr_avg: float -> The average training loss over al folds
            loss_val_avg: float -> The average validayion loss over al folds
        """
        # Shuffle data
        k_indices = build_k_indices(self.y, k_folds)

        # train and validation accuracies for each fold
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

            # Compute the loss differnetly depending on the model
            if (
                self.model_name == "logistic_regression"
                or self.model_name == "reg_logistic_regression"
            ):
                loss_tr[k] = compute_loss_logistic(y_tr, x_tr, w)
                loss_val[k] = compute_loss_logistic(y_val, x_val, w)
            else:
                loss_tr[k] = np.sqrt(2 * compute_mse_loss(y_tr, x_tr, w))
                loss_val[k] = np.sqrt(2 * compute_mse_loss(y_val, x_val, w))

            # Compute the training and validation accuracies
            acc_tr[k] = np.sum(
                (y_tr == get_predictions(x_tr, w, self.model_name)) * 1.0
            ) / len(y_tr)
            acc_val[k] = np.sum(
                (y_val == get_predictions(x_val, w, self.model_name)) * 1.0
            ) / len(y_val)

            if verbose:
                print(
                    "Fold "
                    + str(k)
                    + ", training loss = "
                    + str(loss_tr[k])
                    + ", training acc = "
                    + str(acc_tr[k])
                    + ", val loss = "
                    + str(loss_val[k])
                    + ", val acc = "
                    + str(acc_val[k])
                )

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

    def knn(self, k_folds=5, verbose=True):
        """Runs k-fold cross validation on the data for KNN.
           Handled separately since there are no model weights or loss
                Args:
                    k_folds: int -> Indicates the number of folds
                    verbose: bool -> Indicates whether to print information about the training or not

                Returns:
                    weights: shape(k,D,1) -> The trained weights for each fold
                    loss_tr: shape(k,1) -> The training loss for each fold
                    loss_val: shape(k,1) -> The average loss for each fold
                    loss_tr_avg: float -> The average training loss over al folds
                    loss_val_avg: float -> The average validayion loss over al folds
                """
        # Shuffle data
        k_indices = build_k_indices(self.y, k_folds)

        # train and validation accuracy for each fold
        acc = np.zeros(k_folds)

        for k in range(k_folds):

            # Get data for the fold
            x_val = self.x[k_indices[k]]
            y_val = self.y[k_indices[k]]
            tr_indices = np.delete(k_indices, k, axis=0).flatten()
            x_tr = self.x[tr_indices]
            y_tr = self.y[tr_indices]

            predictions = predict_knn(y_tr, x_tr, x_val, self.params["k"])

            # Compute the validation accuracy
            acc[k] = np.sum(
                (y_val.T[0] == predictions) * 1.0
            ) / len(y_val)

            if verbose:
                print(
                    "Fold "
                    + str(k)
                    + ", val acc = "
                    + str(acc[k])
                )

        # average accuracy
        acc_val_avg = np.average(acc)

        # average performance over all folds
        if verbose:
            print("\nAverage validation accuracy:")
            print(str(acc_val_avg))

        return acc, acc_val_avg


def get_model_weights(x, y, params, model_name):
    """Train the model for 1 fold of cross validation.
    Args:
            x: shape(N,D) -> x-data
            y: shape(N,1) -> y-data
            params: dict -> The params needed in the model (lambda, gamma, initial_w, etc)
            model_name: str -> The name of the model to use for training

    Returns:
        loss: scalar number -> The final training loss
        w: shape=(D, 1) -> The trained weights
    """
    # Initialize weight vector with placeholder values
    w = np.zeros(len(x[0]))
    loss = 0
    # Train the weights depending on the model chosen
    if model_name == "mean_squared_error_gd":
        w, loss = mean_squared_error_gd(
            y, x, params["initial_w"], params["max_iters"], params["gamma"]
        )
    elif model_name == "mean_squared_error_sgd":
        w, loss = mean_squared_error_sgd(
            y, x, params["initial_w"], params["max_iters"], params["gamma"]
        )
    elif model_name == "least_squares":
        w, loss = least_squares(y, x)
    elif model_name == "ridge_regression":
        w, loss = ridge_regression(y, x, params["lambda_"])
    elif model_name == "logistic_regression":
        w, loss = logistic_regression(
            y, x, params["initial_w"], params["max_iters"], params["gamma"]
        )
    elif model_name == "reg_logistic_regression":
        w, loss = reg_logistic_regression(
            y,
            x,
            params["lambda_"],
            params["initial_w"],
            params["max_iters"],
            params["gamma"],
        )

    return w, loss
