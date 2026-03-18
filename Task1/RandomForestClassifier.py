import numpy as np
from sklearn.ensemble import RandomForestClassifier
from MnistClassifierInterface import MnistClassifierInterface

class RandomForestModel(MnistClassifierInterface):
    def __init__(self, n_estimators=100):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state=1
        )
        self.model_type = "RandomForest"

    @staticmethod
    def reshape_data(x):
        x_array = np.array(x)

        # (N, 784) -> just normalize
        if len(x_array.shape) == 2 and x_array.shape[1] == 784:
            return x_array / 255.0

        # (N, 28, 28) -> reshape to (N, 784) and normalize
        if len(x_array.shape) == 3:
            return x_array.reshape(x_array.shape[0], -1) / 255.0

        # If one image (28, 28) -> reshape to (784) and normalize
        if len(x_array.shape) == 2 and x_array.shape[0] == 28:
            return x_array.reshape(1, -1) / 255.0

        return x_array / 255.0

    def train(self, x_train, y_train):
        x_reshaped = RandomForestModel.reshape_data(x_train)
        self.model.fit(x_reshaped, y_train)

    def predict(self, x_test):
        x_reshaped = RandomForestModel.reshape_data(x_test)
        return self.model.predict(x_reshaped)