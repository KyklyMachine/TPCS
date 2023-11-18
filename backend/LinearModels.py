import numpy as np
from sklearn import linear_model
from InterfaceMLModel import IMLModel
from CustomException import CustomException

MSG_PATH = "MSG_CONSTANT.json"


class LogisticRegression(IMLModel):
    _model = None

    def __init__(self, **kwargs) -> None:
        self._model = linear_model.LogisticRegression(**kwargs)
        return

    def set_model(self, model: linear_model.LogisticRegression, **kwargs) -> None:
        self._model = model

    def fit(self, x: np.array, y: np.array, **kwargs) -> None:
        if len(set(y.tolist())) <= 1:
            raise CustomException(MSG_PATH).update_exception(Exception(), "Y_HAS_1_OR_0_CLASSES")
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
    # для обучения модели (TODO: попробуйте разные данные)
    x_train = np.array([[1], [2], [3], [4]])
    y_train = np.array([0, 1, 1, 1])

    # для предсказания меток модели
    x_test = np.array([[-1], [0], [1], [2], [3], [4], [5], [6]])

    # инициализация модели
    model = LogisticRegression(random_state=0)

    # загрузка модели (TODO: попробуйте разные имена и директории)
    # model.load_model("ada.pickle")

    # сохранение модели (TODO: попробуйте разные имена и директории)
    # model.save_model("a.pickle")

    # обучение модели
    model.fit(x_train, y_train)
    new_y = model.predict(x_test)
    print(new_y)

