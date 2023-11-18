import pandas as pd
import os
from CustomException import CustomException
from PIL import Image
import numpy as np
from CustomException import CustomException


MSG_PATH = "MSG_CONSTANT.json"


class DataManipulator:
    _df_info: pd.DataFrame
    _filepath: str

    def __init__(self, filepath=""):
        if filepath:
            self.set_filepath(filepath)

    def set_filepath(self, filepath):
        if "unlabeled" not in os.listdir(filepath):
            raise CustomException(MSG_PATH).update_exception(FileNotFoundError("file" + filepath + " not found"),
                                                             "UNLABELED_NOT_FOUND")
        if len(os.listdir(filepath)) < 2:
            raise CustomException(MSG_PATH).update_exception(FileNotFoundError(),
                                                             "CLASS_FOLDERS_NOT_FOUND")
        self._filepath = filepath

    def get_file(self):
        return self._filepath

    def get_avg_image_sizes(self):
        if self._filepath:

            w = []
            h = []
            folders = os.listdir(self._filepath)
            folders = [fold for fold in folders if fold[0] != "."]

            for fold in folders:
                fold_path = self._filepath + "/" + fold
                images = os.listdir(fold_path)
                images = [image for image in images if image[0] != "."]

                for img in images:
                    img_path = fold_path + "/" + img
                    size_i = Image.open(img_path, "r").size
                    w.append(size_i[0])
                    h.append(size_i[1])
            return int(np.mean(w)), int(np.mean(h))
        return 0, 0

    def transform_file_to_df(self, compress_images=False):
        folders = os.listdir(self._filepath)
        folders = [fold for fold in folders if fold[0] != "."]

        df_info = pd.DataFrame(columns=["img_mode", "class", "name", "data"])

        pixels_data_list = []

        if compress_images:
            avg_sizes = self.get_avg_image_sizes()

        for fold in folders:
            fold_path = self._filepath + "/" + fold
            images = os.listdir(fold_path)
            images = [image for image in images if image[0] != "."]

            for img in images:
                img_path = fold_path + "/" + img
                im = Image.open(img_path, "r").copy()

                if compress_images:
                    im = im.resize(avg_sizes)

                np_data = np.asarray(im).copy()
                if len(np_data.shape) == 2:
                    np_data = np.reshape(np_data, (np_data.shape[0], np_data.shape[1], -1))

                np_data = np.reshape(np_data, (-1, np_data.shape[2]))

                if np_data.shape[1] == 1:
                    np_data = np.array([np_data.tolist(),] * 3).transpose().reshape(-1, 3)
                if np_data.shape[1] > 3:
                    np_data = np_data[:, 0:3]

                pixels_data_list.append(np_data)
                if len(pixels_data_list) >= 2:
                    if pixels_data_list[-1].shape != pixels_data_list[-2].shape:
                        raise CustomException(MSG_PATH).update_exception(Exception(), "IMAGE_SHAPE_NOT_CORRECT")

                df_add = pd.DataFrame(columns=["img_mode", "class", "name", "data"],
                                      data=[[im.mode, fold, img, np_data]])
                df_info = pd.concat([df_info, df_add])

        self._df_info = df_info

    def get_df(self):
        if not self._df_info.empty:
            return self._df_info
        else:
            raise CustomException(MSG_PATH).update_exception(Exception(), "PIXELS_ARRAY_NOT_INIT")

    def save_df(self, df_path: str):
        self._df_info.to_csv(df_path)


if __name__ == "__main__":
    # инициализация класса и выбор директории (TODO: попробуйте разные директории и их наполнение)
    dm = DataManipulator("imgs")

    # трансформация изображений в пиксели
    dm.transform_file_to_df(compress_images=True)
    dm.get_df()
