import numpy as np
from sklearn import model_selection
from sklearn import preprocessing
from backend.CustomException import CustomException


MSG_PATH = "MSG_CONSTANT.json"


class CVRatio:
    test: float
    val: float
    train: float
    help: float

    def __init__(self, test, val, train, help):
        self.set_ratio(test, val, train, help)

    def set_ratio(self, test, val, train, help):
        if test + val + train + help != 1:
            raise CustomException(MSG_PATH).update_exception(ValueError(), "TEST_VAL_TRAIN_HELP_RATIO_ERROR")
        self.test = test
        self.val = val
        self.train = train
        self.help = help


class DataPreprocessor:
    _cv_method = None
    _preprocessing_method = None
    _cv_ratio: CVRatio

    def __init__(self, preprocessing_method, cv_ratio: list):
        if len(cv_ratio) != 4:
            raise CustomException(MSG_PATH).update_exception(ValueError(), "CV_RATIO_LIST_LEN_ERROR")
        self.set_preprocessing_method(preprocessing_method)
        self.set_cv_ratio(cv_ratio[0], cv_ratio[1], cv_ratio[2], cv_ratio[3])
        pass

    def set_cv_ratio(self, test, val, train, help):
        self._cv_ratio = CVRatio(test, val, train, help)

    def get_cv_ratio(self):
        return self._cv_ratio

    def set_preprocessing_method(self, method):
        self._preprocessing_method = method

    def get_preprocessing_method(self, method):
        return self._preprocessing_method

    def set_cv_method(self, method):
        self._cv_method = method

    def get_cv_method(self, method):
        return self._cv_method

    def get_need_marked_exemplars(self):
        test = int(1 / self._cv_ratio.test + 1)
        val = int(1 / self._cv_ratio.val + 1)
        train = int(1 / self._cv_ratio.train + 1)
        help = int(1 / self._cv_ratio.help + 1)
        return test + val + train + help

    def cv_data(self, x: np.array, y: np.array):
        # сначала выделим данные под help
        if self._cv_ratio.help * x.shape[0] < 0:
            # FIXME: что-то придумать с малым кол-вом изображений
            pass
        else:
            x_help, x_tvt, y_help, y_tvt = model_selection.train_test_split(x, y, train_size=self._cv_ratio.help)

        # далее под train
        new_train = self._cv_ratio.train / (self._cv_ratio.train + self._cv_ratio.val + self._cv_ratio.test)
        x_train, x_ct, y_train, y_ct = model_selection.train_test_split(x_tvt, y_tvt, train_size=new_train)

        # далее test и val
        new_test = self._cv_ratio.test / (self._cv_ratio.test + self._cv_ratio.val)
        x_test, x_val, y_test, y_val = model_selection.train_test_split(x_ct, y_ct, train_size=new_test)

        return [[x_train, y_train], [x_val, y_val], [x_test, y_test], [x_help, y_help]]

    def union_rgb(self, r, g, b, n_samples):
        """res_i = []
        for i in range(n_samples):
            r_i = r[i]
            g_i = g[i]
            b_i = b[i]
            res_j = []
            for j in range(len(r_i)):
                res_j.append([r_i[j], g_i[j], b_i[j]])
            res_i.append(res_j)
        res = np.array(res_i)"""
        data_r = r.reshape(-1)
        data_g = g.reshape(-1)
        data_b = b.reshape(-1)
        res = np.concatenate([data_r, data_g, data_b]).reshape(3, -1).transpose().reshape(n_samples, -1, 3)
        return res

    def preprocess_data(self, data, fit_preprocessor=True):
        if fit_preprocessor:
            self._preprocessing_method.fit(data)
        return self._preprocessing_method.transform(data)

    def reshape_data(self, data: np.array):
        return data.reshape(data.shape[0], -1)


if __name__ == "__main__":
    dp = DataPreprocessor(preprocessing_method=preprocessing.StandardScaler(), cv_ratio=[0.25, 0.25, 0.25, 0.25])

    start_data = np.array(
        [[['r11' 'g11' 'b11'],
          ['r12' 'g12' 'b12'],
          ['r13' 'g13' 'b13']],

         [['r21' 'g21' 'b21'],
            ['r22' 'g22' 'b22'],
            ['r23' 'g23' 'b23']],

         [['r31' 'g31' 'b31'],
            ['r32' 'g32' 'b32'],
            ['r33' 'g33' 'b33']],

         [['r41' 'g41' 'b41'],
            ['r42' 'g42' 'b42'],
            ['r43' 'g43' 'b43']]],
    )

    data_r = np.array(
        [
            ["r11", "r12", "r13"],
            ["r21", "r22", "r23"],
            ["r31", "r32", "r33"],
            ["r41", "r42", "r43"]
        ]
    )

    data_g = np.array(
        [
            ["g11", "g12", "g13"],
            ["g21", "g22", "g23"],
            ["g31", "g32", "g33"],
            ["g41", "g42", "g43"]
        ]
    )

    data_b = np.array(
        [
            ["b11", "b12", "b13"],
            ["b21", "b22", "b23"],
            ["b31", "b32", "b33"],
            ["b41", "b42", "b43"]
        ]
    )
    """data_r = data_r.reshape(-1)
    data_g = data_g.reshape(-1)
    data_b = data_b.reshape(-1)
    res = np.concatenate([data_r, data_g, data_b]).reshape(3, -1).transpose().reshape(4, 3, -1)
    print(res)"""
    start_data = dp.union_rgb(data_r, data_g, data_b, 4)
    print(start_data)
