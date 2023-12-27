import numpy as np
from abc import abstractmethod, ABC
import pickle
from backend.CustomException import CustomException

MSG_PATH = "MSG_CONSTANT.json"


class IMLModel:
    _model = None

    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError

    def load_model(self, path: str) -> None:
        try:
            self._model = pickle.load(open(path, 'rb'))
        except FileNotFoundError as e:
            raise CustomException(MSG_PATH).update_exception(e, "LOAD_MODEL_ERROR")

    def save_model(self, path: str) -> None:
        try:
            pickle.dump(self._model, open(path, 'wb'))
        except FileNotFoundError as e:
            raise CustomException(MSG_PATH).update_exception(e, "SAVE_MODEL_ERROR")

    @abstractmethod
    def set_model(self, model) -> None:
        raise NotImplementedError

    @abstractmethod
    def fit(self, x, y) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.array) -> np.array:
        raise NotImplementedError
