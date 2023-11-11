import numpy as np
from sklearn import ensemble
from InterfaceMLModel import IMLModel
from CustomException import CustomException

MSG_PATH = "MSG_CONSTANT.json"


class RandomForest(IMLModel):
    _model = None

    def __init__(self, **kwargs) -> None:
        self._model = ensemble.RandomForestClassifier(**kwargs)
        return

    def set_model(self, model: ensemble.RandomForestClassifier, **kwargs) -> None:
        self._model = model

    def fit(self, x: np.array, y: np.array, **kwargs) -> None:
        try:
            self._model.fit(x, y, **kwargs)
        except Exception as e:
            raise CustomException(MSG_PATH).update_exception(e, "FIT_MODEL_ERROR")

    def predict(self, x: np.array, **kwargs) -> np.array:
        try:
            return self._model.predict(x, **kwargs)
        except Exception as e:
            raise CustomException(MSG_PATH).update_exception(e, "PREDICT_MODEL_ERROR")


if __name__ == "__main__":
    x = np.array([[1], [2], [3], [4]])
    new_x = np.array([[0], [1], [2], [3], [4], [5]])
    y = np.array([0, 1, 1, 1])

    model = RandomForest(random_state=0)
    # model.load_model("ada.pickle")
    # model.save_model("a.pickle")
    model.fit(x, y)
    new_y = model.predict(new_x)
    print(new_y)