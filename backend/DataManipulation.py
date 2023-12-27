import copy
import pandas as pd
import os
from PIL import Image
import numpy as np
import zipfile
import random as rnd

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from backend.CustomException import CustomException
import backend.InterfaceMLModel


MSG_PATH = "MSG_CONSTANT.json"


class DataManipulator:
    _df_info: pd.DataFrame
    _filepath: str
    _zip: zipfile.ZipFile

    def set_filepath(self, filepath):
        if "unlabeled" not in os.listdir(filepath):
            raise CustomException(MSG_PATH).update_exception(FileNotFoundError("file" + filepath + " not found"),
                                                             "UNLABELED_NOT_FOUND")
        if len(os.listdir(filepath)) < 2:
            raise CustomException(MSG_PATH).update_exception(FileNotFoundError(),
                                                             "CLASS_FOLDERS_NOT_FOUND")
        self._filepath = filepath

    def set_zip(self, zip):
        self._zip = zip

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

    def get_avg_image_sizes_zip(self):
        if self._zip:
            w = []
            h = []
            with zipfile.ZipFile(self._zip) as dataset_zip:
                image_paths = [name for name in dataset_zip.namelist() if name.endswith('.jpg') and "MACOS" not in name]

                for img_path in image_paths:
                    file = dataset_zip.open(img_path)
                    size_i = Image.open(file, "r").size
                    w.append(size_i[0])
                    h.append(size_i[1])
                dataset_zip.read(image_paths[0])
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

    def get_classes(self):
        if self._zip:
            with zipfile.ZipFile(self._zip) as dataset_zip:
                root_paths = set([s.split("/")[0] for s in dataset_zip.namelist()])
                image_classes = set(name.split("/")[1] for name in dataset_zip.namelist()
                                if name.split("/")[1] != ""
                                and not name.split("/")[1].startswith(".")
                                and name.split("/")[1] not in root_paths)
                return image_classes
        else:
            raise CustomException(MSG_PATH).update_exception(Exception(), "ZIP_FILE_NOT_EXIST")

    def transform_file_to_df_zip(self, compress_images=False):
        if self._zip:
            folders = self.get_classes()
            df_info = pd.DataFrame(columns=["img_mode", "class", "name", "data"])
            pixels_data_list = []
            if compress_images:
                avg_sizes = self.get_avg_image_sizes_zip()

            with zipfile.ZipFile(self._zip) as dataset_zip:
                image_paths = [name for name in dataset_zip.namelist() if name.endswith('.jpg') and "MACOS" not in name]

                for img_path in image_paths:
                    file = dataset_zip.open(img_path)
                    im = Image.open(file, "r").copy()
                    if compress_images:
                        im = im.resize(avg_sizes)
                    np_data = np.asarray(im).copy()

                    if len(np_data.shape) == 2:
                        np_data = np.reshape(np_data, (np_data.shape[0], np_data.shape[1], -1))

                    np_data = np.reshape(np_data, (-1, np_data.shape[2]))

                    if np_data.shape[1] == 1:
                        np_data = np.reshape(np_data, (-1))
                        #np_data = np.array([np_data.tolist(), ] * 3).transpose().reshape(-1, 3)
                        pass
                    elif np_data.shape[1] == 3:
                        np_data = np.array([0.2126*R + 0.7152*G + 0.0722*B for R, G, B in zip(np_data[:,0], np_data[:,1], np_data[:,2])])
                        #np_data = np.array([np_data.tolist(), ] * 3).transpose().reshape(-1, 3)
                        pass
                    elif np_data.shape[1] == 4:
                        np_data = np.array([0.2126*R + 0.7152*G + 0.0722*B for R, G, B in zip(np_data[:,0], np_data[:,1], np_data[:,2])])
                        #np_data = np_data[:, 0:3]
                    pixels_data_list.append(np_data)
                    if len(pixels_data_list) >= 2:
                        if pixels_data_list[-1].shape != pixels_data_list[-2].shape:
                            raise CustomException(MSG_PATH).update_exception(Exception(), "IMAGE_SHAPE_NOT_CORRECT")

                    df_add = pd.DataFrame(columns=["img_mode", "class", "name", "data"],
                                          data=[[im.mode, img_path.split("/")[-2], img_path, np_data]])
                    df_info = pd.concat([df_info, df_add])

                self._df_info = df_info

    def get_df(self):
        if not self._df_info.empty:
            return self._df_info
        else:
            raise CustomException(MSG_PATH).update_exception(Exception(), "PIXELS_ARRAY_NOT_INIT")

    def save_df(self, df_path: str):
        self._df_info.to_csv(df_path)

    def get_images_to_show(self, random=True):
        if self._zip:
            with zipfile.ZipFile(self._zip) as dataset_zip:
                image_paths = [name for name in dataset_zip.namelist() if name.endswith('.jpg') and "MACOS" not in name]
                if random:
                    return image_paths[0:100]
    
    def get_images_to_label(self):
        if not self._df_info.empty:
            return self._df_info[self._df_info["class"] == "unlabeled"]["name"]
            """if self._zip:
            with zipfile.ZipFile(self._zip) as dataset_zip:
                image_paths = [name for name in dataset_zip.namelist() if name.endswith('.jpg') and "MACOS" not in name]
                image_paths_unlabaled = [image for image in image_paths if "unlabeled" in image]
                return image_paths_unlabaled"""
        else:
            raise CustomException(MSG_PATH).update_exception(Exception(), "DF_NOT_EXIST")

    def get_marked_img_count(self):
        if not self._df_info.empty:
            return self._df_info[self._df_info["class"] != "unlabeled"].shape[0]
        else:
            raise CustomException(MSG_PATH).update_exception(Exception(), "DATAFRAME_ERROR?")

    def set_label(self, label, name):
        if not self._df_info.empty:
            self._df_info.loc[self._df_info["name"] == name, "class"] = label
        else:
            raise CustomException(MSG_PATH).update_exception(Exception(), "DATAFRAME_ERROR?")



    


    


    


    def get_images_to_postlabel(self, model_start, x_labeled, y_labeled, x_test, y_test, x_helper, y_helper, alpha, number_markedup_ex):
        
        def Gini(folder, class_, cm, classes):
            cm_folder = np.array([])
            for i in range(len(classes)):
                cm_folder = np.append(cm_folder, cm[i][folder])
            cm_folder_norm = cm_folder/sum(cm_folder)
            cm_folder_norm_del = np.delete(cm_folder_norm, class_)
            return 1-sum(cm_folder_norm_del**2)

        def choose_data_for_markup(x_labeled, y_labeled, classes, number_markedup_ex):
            costs, score = [], []
            acc0 = accuracy_score(y_test, model.predict(x_test))
            N_ = len(y_helper)

            for j in range(K):
                for i in range(K):
                    #массивы (размерностью от 0 до number_markedup_ex) примеров из j папки i класса, которые будут добавлены на данном шаге.
                    x_ji, y_ji = get_example_from_folder_and_class(x_helper, y_helper, j, i, model, number_markedup_ex, classes)

                    N_ji = len(y_ji)
                    c_ji = cost(j, i, classes, model, x_test, y_test) #стоимость доразметки одного примера
                    costs.append(c_ji)

                    if N_ji > 0: #иначе в обучающий набор не добавится ничего нового
                        #обучаем модель на объединенных данных
                        model_U_ji = model_start()
                        model_U_ji.fit(np.concatenate((x_labeled, x_ji)), np.concatenate((y_labeled, y_ji)))
                        score.append(get_score(accuracy_score(y_test, model_U_ji.predict(x_test)), c_ji * N_ji / N_, alpha))
                        #print('x_ji', x_ji, 'y_ji', y_ji)
                    else:
                        score.append(get_score(acc0, c_ji / N_, alpha)) #не домножаем на N_ji т.к. оно равно нулю, но стоимость не нулевая
            #Находим оптимальные примеры для доразметки
            index_score_min = None
            try:
                index_score_min = np.reshape(np.where(score == np.min(score))[0], (-1))[0]
            except:
                index_score_min = 0
            print(index_score_min)
            cost0 = costs[index_score_min]
            i_class = index_score_min % K
            j_folder = index_score_min // K
            return j_folder, i_class, cost0
        
        def get_score(acc, cost, alpha):
            return alpha*(1-acc) + (1 - alpha)*cost

        def cost(j, i, classes, model_learned, x_test, y_test):
            K = len(classes)
            T = (K-2)/(K-1)
            p_cm = confusion_matrix(y_test, model_learned.predict(x_test))
            G = Gini(j,i,p_cm, classes)
            p = confusion_matrix(y_test, model_learned.predict(x_test), normalize='pred')[i][j]
            return (G**2 - p**2 - 2*p - p*(G**2) + 3)/((T**2)+3)

        def get_example_from_folder_and_class(x_from, y_from, folder, class_, model, amount, classes):
            y_pred = model.predict(x_from)
            cm = confusion_matrix(y_from, y_pred)
            x_out, y_out = [], []
            #создать массив индексов всех подходящих под условие примеров, а затем выбрать из них рандомные в нужном количестве
            i_arr = []
            for i in range(len(y_from)):
                if y_from[i] == classes[class_] and y_pred[i] == classes[folder]:
                    i_arr.append(i)
            #print(i_arr)
            if i_arr:
                i_arr = rnd.sample(i_arr, min(cm[class_][folder], amount, len(i_arr))) # type: ignore
            #print(i_arr)
            for k in i_arr:
                x_out.append(x_from[k])
                y_out.append(y_from[k])
            return x_out, y_out

        if not self._df_info.empty:
            model = model_start()
            model.fit(x_labeled, y_labeled)
            classes = list(self.get_classes())
            classes.remove("unlabeled")
            K = len(classes)
            T = (K-2)/(K-1)
            return choose_data_for_markup(x_labeled, y_labeled, classes, number_markedup_ex)
        else:
            raise CustomException(MSG_PATH).update_exception(Exception(), "DATAFRAME_ERROR?")


    def get_random_class(self):
        return rnd.choice(list(self.get_classes()))


    def get_i_folders(self, learned_model, x_unlabeled, names_unlabeled, folder_class):
        y_unlabeled = learned_model.predict(x_unlabeled)
        res_names = []
        for i in range(len(names_unlabeled)):
            name_i = names_unlabeled[i]
            class_i = y_unlabeled[i]
            if class_i == folder_class:
                res_names.append(name_i)
        return res_names



        
