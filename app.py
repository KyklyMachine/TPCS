import numpy as np
import streamlit as st
import pandas as pd
from backend.DataManipulation import DataManipulator
from backend.Preprocessing import DataPreprocessor
from backend.Validation import ModelValidator
from backend.LinearModels import LogisticRegression
from backend.EnsambleModels import RandomForest
import sklearn as skl
import extra_streamlit_components as stx
import zipfile
from streamlit_modal import Modal
import plotly.express as px
from streamlit_extras.stylable_container import stylable_container


CV_RATIO=[0.25, 0.25, 0.25, 0.25]
MIN_IMG_PER_CLASS = 20
TODO = 1


st.set_page_config(layout="wide")
with open("styles.css") as f:
        st.write(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# STATE MANIPULATION
        

def add_acc(acc: float):
    if "acc" not in st.session_state:
        st.session_state["acc"] = []
    st.session_state["acc"].append(acc)


def add_n_postlabeled(number: int):
    if "n_postlabeled" not in st.session_state:
        st.session_state["n_postlabeled"] = []
    st.session_state["n_postlabeled"].append(number)


def plot_acc_count(acc: list, counts: list):
    data = {'точность': acc, 'количество размеченных данных': counts}

    df = pd.DataFrame.from_dict(data)
    fig = px.line(df, x='количество размеченных данных', y='точность', markers=True)
    fig.update_layout(
        margin=dict(l=5, r=5, t=0, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# PICTURE DROWING

def picture_element(path: str, src):
    with st.container(border=True):

        st.image(src.read(path), use_column_width=True)

        def change_vis(name):
            st.session_state[name] = not st.session_state[name]

        if path not in st.session_state:
            st.session_state[path] = False
        if not st.session_state[path]:
            st.button("Выбрать", key="button-" + path, use_container_width=True, on_click=change_vis, kwargs={"name": path})
        if st.session_state[path]:
            st.button("Убрать", key="button-" + path, use_container_width=True, type="primary", on_click=change_vis,
                               kwargs={"name": path})


def calculate_rows_count(n_pics, n_cols):
    res = n_pics // n_cols
    res += 1 if n_pics % n_cols != 0 else 0
    return res


def pictures_container(pictures, src, n_cols=3):        
        with zipfile.ZipFile(src) as dataset_zip:
            with stylable_container("images-container", 
                                    """
                                    {
                                        max-height: 500px;
                                        border: 1px solid rgba(49, 51, 63, 0.2);
                                        overflow-y: scroll;
                                        overflow-x: hidden;
                                        padding: 10px;
                                        scrollbar-color: blue;
                                    }
                                    """):
                columns = st.columns(n_cols)
                column_num = 0
                for picture in pictures:
                    with columns[column_num % len(columns)]:
                        picture_element(picture, dataset_zip)
                        column_num += 1


def show_images_to_labeling(files, dm, pr, img_per_row):
    if files:
        # TODO: 0:100??? нужно добавить перелистывание страниц
        im_to_show = dm.get_images_to_label()[0:100]
        pictures_container(im_to_show, files, img_per_row)
    

def show_images_to_postlabeling(files, dm, pr, img_per_row, image_to_show):
    if files:
        # TODO: 0:100??? нужно добавить перелистывание страниц  
        image_to_show = image_to_show[:100]
        pictures_container(image_to_show, files, img_per_row)


# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# HELPERS

def get_nlabeled_per_class():
    classes = list(dm.get_classes())
    classes = sorted(classes) # type: ignore
    c = {}
    df = dm.get_df()
    for cls in classes:
        count_cls = int(df[df["class"] == cls].shape[0])
        c[cls] = count_cls
    return c


def labeling_left_panel(dm: DataManipulator):
    
    classes = list(dm.get_classes())
    classes.remove("unlabeled")
    classes = sorted(classes) # type: ignore
    c = []
    df = dm.get_df()
    for cls in classes:
        count_cls = int(df[df["class"] == cls].shape[0])
        if count_cls < MIN_IMG_PER_CLASS:
            c.append(fr"{cls} :red[{count_cls}/{MIN_IMG_PER_CLASS}]")
        else:
            c.append(fr"{cls} :green[{count_cls}/{MIN_IMG_PER_CLASS}]")
    
    class_marked = st.radio("Выберите класс", classes)
    
    st.markdown("  \n".join(c))
    
    return class_marked


def postlabeling_left_panel(dm: DataManipulator, class_to_postlabeling):
    st.radio("Класс для доразметки", [class_to_postlabeling])
    return class_to_postlabeling


def markup_images(img_class, dm: DataManipulator):
    # get all buttons (and img) that has True in session_state
    names = [btn_name for btn_name in st.session_state.keys() if st.session_state[btn_name] == True
                    and type(st.session_state[btn_name]) == bool]
    
    df = dm.get_df()
    for name in names:
        df.loc[df["name"] == name, "class"] = img_class
        st.session_state.pop(name)
    

# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# PAGES 

def dataset_marking_page(files, dm, pr: DataPreprocessor, preprocessing, ml_model, cv, priority, img_per_row):
    modal = Modal(
        "Информация о разметке", 
        key="demo-modal",
        padding=20,
        max_width=744
        )
    
    col1, col2 = st.columns([8, 1])
    with col1:
        st.write(f"Разметьте по 20 изображений в каждом классе.")
    with col2:
        open_modal = st.button("Помощь")

    if open_modal:
        modal.open()
    if modal.is_open():
        with modal.container():
            st.write("На данном экране Вам предлагается провести самостоятельную разметку датасета. \n" +
                        "1. Вам нужно будет указать класс, для которого проводится разметка. \n" +
                        "2. Выбрать изображения, которые на ваш взгляд относятся к выбранному классу. \n" + 
                        "3. Нажать на кнопку Разметить. \n" +
                        "4. Продолжать разметку до тех пор, пока не будет размечено требуемое количество изображений.")

    df = dm.get_df()
    col1, col2 = st.columns([1, 5])
    with col1:
        class_marked = labeling_left_panel(dm)
    with col2:
        show_images_to_labeling(files, dm, pr, img_per_row)
    button_cols = st.columns(3)
    with button_cols[1]:
        markup_button = st.button("Разметить", use_container_width=True)
        if markup_button:
            markup_images(class_marked, dm)
            st.rerun()


def dataset_addmarking_page(files, dm, pr, preprocessing, ml_model, cv, priority, img_per_row):

    if "preprocessed" not in st.session_state:
        st.session_state["preprocessed"] = False
        df = dm.get_df()
        df = df[df["class"] != "unlabeled"]
        df = df.to_numpy()
        x = df[:,3]
        x = np.array([np.array(xi) for xi in x])
        y = np.array(df[:,1])
        [[x_train, y_train], [x_val, y_val], [x_test, y_test], [x_help, y_help]] = pr.cv_data(x, y)

        x_train_pr = pr.preprocess_data(x_train, fit_preprocessor=True)
        x_val_pr = pr.preprocess_data(x_train, fit_preprocessor=False)
        x_test_pr = pr.preprocess_data(x_train, fit_preprocessor=False)
        x_help_pr = pr.preprocess_data(x_train, fit_preprocessor=False)

        model = ml_model()
        model.fit(x_train, y_train) 
        y_pred = model.predict(x_test)
        mv = ModelValidator(set([]), y_test, y_pred)
        
        j_folder, i_class, cost0 = dm.get_images_to_postlabel(ml_model, x_train, y_train, x_test, y_test, x_help, y_help, priority, TODO)
        
        # get images
        df = dm.get_df()
        df = df[df["class"] == "unlabeled"]
        df = df.to_numpy()
        x = df[:,3]
        x = np.array([np.array(xi) for xi in x])
        names = df[:,2].tolist()
        
        classes = list(dm.get_classes())
        classes.remove("unlabeled")

        img_names_to_show = dm.get_i_folders(model, x, names, classes[j_folder])

        st.session_state["preprocessed"] = True
        st.session_state["img_names_to_show"] = img_names_to_show
        st.session_state["class_to_postlabeling"] = classes[i_class]



    modal = Modal(
        "Информация о доразметке", 
        key="demo-modal",
        padding=20,
        max_width=744
        )
    
    col1, col2 = st.columns([8, 1])
    with col1:
        st.write(f"Доразметьте изображения предоставленного класса.")
    with col2:
        open_modal = st.button("Помощь")

    if open_modal:
        modal.open()
    if modal.is_open():
        with modal.container():
            st.write("На данном экране Вам предлагается провести самостоятельную доразметку датасета. \n" +
                        "1. Выберите изображения, которые на ваш взгляд относятся к предоставленному классу " + 
                        "(таких изображений может и не юыть). \n" + 
                        "2. Нажмите на кнопку Доразметить. \n" +
                        "Для просмотра результатов итерации перейдите на страницу 'Результаты'")

    
    col1, col2 = st.columns([1, 5])
    with col1:
        class_marked = postlabeling_left_panel(dm, st.session_state["class_to_postlabeling"])
    with col2:
        show_images_to_postlabeling(files, dm, pr, img_per_row, st.session_state["img_names_to_show"])
    button_cols = st.columns(3)

    with button_cols[1]:
        markup_button = st.button("Доразметить", use_container_width=True)
        if markup_button:
            markup_images(class_marked, dm)
            df = dm.get_df()
            df = df[df["class"] != "unlabeled"]
            df = df.to_numpy()
            x = df[:,3]
            x = np.array([np.array(xi) for xi in x])
            y = np.array(df[:,1])
            [[x_train, y_train], [x_val, y_val], [x_test, y_test], [x_help, y_help]] = pr.cv_data(x, y)
            
            x_train_pr = pr.preprocess_data(x_train, fit_preprocessor=True)
            x_val_pr = pr.preprocess_data(x_train, fit_preprocessor=False)
            x_test_pr = pr.preprocess_data(x_train, fit_preprocessor=False)
            x_help_pr = pr.preprocess_data(x_train, fit_preprocessor=False)

            model = ml_model()
            model.fit(np.concatenate([x_train, x_val, x_help]), np.concatenate([y_train, y_val, y_help])) 
            y_pred = model.predict(x_test)
            mv = ModelValidator(set([]), y_test, y_pred)
            add_acc(mv.get_acc())
            add_n_postlabeled(x.shape[0])

            if "preprocessed" in st.session_state:
                st.session_state.pop("preprocessed")
            if "img_names_to_show" in st.session_state:
                st.session_state.pop("img_names_to_show")
            if "class_to_postlabeling" in st.session_state:
                st.session_state.pop("class_to_postlabeling")

            st.rerun()


def results_page(file, dm, pr, preprocessing, ml_model, cv, priority, img_per_row):

    st.header("Результаты доразметки")

    cols = st.columns([1, 3])
    with cols[0]:
        with st.container(border=True):
            if "acc" in st.session_state:
                now = st.session_state["acc"][-1]
                if len(st.session_state["acc"]) > 1:
                    delta = now - st.session_state["acc"][-2]
                    st.metric("Текущая точность", now, delta)
                else:
                    st.metric("Текущая точность", now, "–", "off")
            else:
                st.metric("Текущая точность", "–", "–", "off")
        with st.container(border=True):
            if "n_postlabeled" in st.session_state:
                now = st.session_state["n_postlabeled"][-1]
                if len(st.session_state["n_postlabeled"]) > 1:
                    delta = now - st.session_state["n_postlabeled"][-2]
                    st.metric("Количество размеченных данных", now, delta)
                else:
                    st.metric("Количество размеченных данных", now, "–", "off")
            else:
                st.metric("Количество размеченных данных", "–", "–", "off")
    with cols[1]:
        if ("acc" in st.session_state) and ("n_postlabeled" in st.session_state):
            plot_acc_count(st.session_state["acc"], st.session_state["n_postlabeled"])
        else:
            plot_acc_count([], [])



    button_cols = st.columns(5)
    with button_cols[0]:
        bt_prev = st.button("Назад", use_container_width=True)
        if bt_prev:
            if st.session_state["stage"] != 1:
                st.session_state["stage"] -= 1
                st.rerun()

    with button_cols[4]:
        bt_next = st.button("Готово", use_container_width=True)
        if bt_next:
            if st.session_state["stage"] != 5:
                st.session_state["stage"] += 1
                st.rerun()
            else:
                st.session_state["stage"] = 1
                st.rerun()


# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# DATA LOADING

@st.cache_resource
def load_file(files):
    dm = DataManipulator()
    dm.set_zip(files)
    dm.transform_file_to_df_zip(compress_images=True)
    return dm


def sidebar():
    with st.sidebar:
        # TODO: add a menu
        #selected = option_menu("Меню", ["Настройки алгоритма", 'Настройки графики'], 
        #icons=['wrench', 'gear'], menu_icon="cast", default_index=1)

        # file loading
        file = st.file_uploader("Загрузите .zip архив")
        data_manipulator = load_file(file)

        # count images per row
        img_per_row = st.slider("Количество изображений в строке", 2, 10, 4, 1)

        # preprocessing
        # TODO: Add VGG16
        preprocessing_dict = {
            "Масштабирование": skl.preprocessing.MinMaxScaler, 
        }
        preprocessing = st.selectbox("Выберите алгоритм предобработки", 
                                     preprocessing_dict.keys(),
                                     index=0,
                                     placeholder="Алгоритм предобработки...")
        
        # ml algorythm
        # TODO: Add Naive Bayes, SVM
        ml_model_dict = {
            "Логистическая регрессия": LogisticRegression, 
            "Случайный лес": RandomForest
        }
        ml_model = st.selectbox("Выберите ML-алгоритм", 
                                ml_model_dict.keys(),
                                index=0,
                                placeholder="ML-алгоритм...")
        # cross-validation
        # TODO: Add K-fold, One-leave-out
        cv_dict = {
            "Монте-Карло": "Monte-Carlo"
        }
        cv = st.selectbox("Выберите алгоритм кросс-валидации", 
                          cv_dict.keys(),
                          index=0,
                          placeholder="Алгоритм кросс-валидации...")

        # priority
        priority = st.slider('Приоритет точности над стоимостью', 0., 1., 0.5, 0.1)

        preprocessing_res = preprocessing_dict[preprocessing] # type: ignore
        ml_model_res = ml_model_dict[ml_model] # type: ignore
        cv_res = cv_dict[cv] # type: ignore

        if file:
            st.download_button("Скачать доразмеченный датасет", file, disabled=True)

        data_preprocessor = DataPreprocessor(preprocessing_method=preprocessing_res(), cv_ratio=CV_RATIO)

    return file, data_manipulator, data_preprocessor, ml_model_res, cv_res, priority, img_per_row
    
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# COMPONENTS

def tab_bar_loader(pages_dict: dict, default_stage: int):
    # stage initialization
    if "stage" not in st.session_state:
        st.session_state["stage"] = default_stage

    # tab bar initialization
    tabs = [stx.TabBarItemData(id=key, title=pages_dict[key], description="") for key in pages_dict.keys()]
    chosen_id = stx.tab_bar(data=tabs, default=st.session_state["stage"], return_type=int) # type: ignore

    if int(chosen_id) != st.session_state["stage"]: # type: ignore
        st.session_state["stage"] = int(chosen_id) # type: ignore
        st.rerun()


def page_loader(file, dm, pr, preprocessing, ml_model, cv, priority, pages_dict: dict, img_per_row):
    for page_key in pages_dict.keys():
        if st.session_state["stage"] == page_key:
            pages_dict[page_key](file, dm, pr, preprocessing, ml_model, cv, priority, img_per_row)


if __name__ == "__main__":
    
    # sidebar with params
    with st.spinner('Загрузка страницы...'):
        file, dm, preprocessing, ml_model, cv, priority, img_per_row = sidebar()

    # check is dataset loaded
    if file:
        if dm._df_info is not None:
            _x = dm.get_df()["data"].to_numpy()
            x = np.array([img for img in _x])
            y = dm.get_df()["class"].to_numpy()
            pr = DataPreprocessor(preprocessing_method=skl.preprocessing.StandardScaler(), cv_ratio=[0.25, 0.25, 0.25, 0.25])
            min_marked_exemplars = pr.get_need_marked_exemplars()
            count_marked_exemplars = dm.get_marked_img_count()
            nlabeled_per_class = get_nlabeled_per_class()
            nlabeled_per_class.pop("unlabeled")
            enought_to_postlabeling = all([cls >= MIN_IMG_PER_CLASS for cls in nlabeled_per_class.values()])
            if not enought_to_postlabeling:
                pages = {
                    1: "Разметка датасета",
                }

                func_pages = {
                    1: dataset_marking_page,
                }
            else:
                pages = {
                    1: "Доразметка датасета",
                    2: "Результаты"
                }

                func_pages = {
                    1: dataset_addmarking_page,
                    2: results_page
                }

            

            tab_bar_loader(pages, 1)
            page_loader(file, dm, pr, preprocessing, ml_model, cv, priority, func_pages, img_per_row)
        else:
            st.info("Ожидание обработки файла")
    else:
        st.info("""Загрузите архив.
                Поддерживаются архивы формата .zip, внутри должны находиться изображения в папках,
                которые названы как классы изображений. Неразмеченные изображения должны находиться
                в папке unlabeled.""")
        

