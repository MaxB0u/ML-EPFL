from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def fit(self, y_tr, x_tr, lambda_, reg_type=""):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, x_new):
        """Predict data from the trained model.
        x_new is a vector of new data.
        Return a vector of the predictions
        """
        pass
