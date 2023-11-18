import numpy as np

from Preprocessing import DataPreprocessor
from Validation import ModelValidator
from DataManipulation import DataManipulator
from LinearModels import LogisticRegression
from sklearn import preprocessing


class PostLabeler:
    pass


if __name__ == "__main__":
    # get data and transform to np.array (DataManipulator)
    dm = DataManipulator("imgs")
    dm.transform_file_to_df(compress_images=True)
    _x = dm.get_df()["data"].to_numpy()
    x = np.array([img for img in _x])
    y = dm.get_df()["class"].to_numpy()

    # preprocess data (DataPreprocessor)
    pr = DataPreprocessor(preprocessing_method=preprocessing.StandardScaler(), cv_ratio=[0.25, 0.25, 0.25, 0.25])
    [[x_train, y_train], [x_val, y_val], [x_test, y_test], [x_help, y_help]] = pr.cv_data(x, y)

    x_train_preprocessed = pr.preprocess_data(x_train)
    x_val_preprocessed = pr.preprocess_data(x_val)
    x_test_preprocessed = pr.preprocess_data(x_test)
    x_help_preprocessed = pr.preprocess_data(x_help)

    # fit + predict (LinearModels)
    # FIXME: сделать обработку если только 1 класс в данных
    print("Start learning")
    log_reg = LogisticRegression()
    log_reg.fit(pr.preprocess_data(x), y)







