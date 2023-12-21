import sys
import streamlit as st
import pandas as pd
from backend.DataManipulation import DataManipulator
import hydralit_components as hc
import extra_streamlit_components as stx
import zipfile


st.set_page_config(layout="wide")


def picture_element(path: str, src):
    with st.container(border=True):

        st.image(src.read(path), use_column_width=True)

        def change_vis(name):
            st.session_state[name] = not st.session_state[name]

        if path not in st.session_state:
            st.session_state[path] = False
        if not st.session_state[path]:
            button = st.button("Выбрать", key="button-" + path, use_container_width=True, on_click=change_vis, kwargs={"name": path})
        if st.session_state[path]:
            button = st.button("Убрать", key="button-" + path, use_container_width=True, type="primary", on_click=change_vis,
                               kwargs={"name": path})


def calculate_rows_count(n_pics, n_cols):
    res = n_pics // n_cols
    res += 1 if n_pics % n_cols != 0 else 0
    return res


def pictures_container(pictures, src, n_cols=3):
        picture_container = st.container(border=True)
        st.container()
        
        with zipfile.ZipFile(src) as dataset_zip:
            with picture_container:
                st.write("<div class='all-images-container'>", unsafe_allow_html=True)
                columns = st.columns(n_cols)
                column_num = 0
                for picture in pictures:
                    with columns[column_num % len(columns)]:
                        picture_element(picture, dataset_zip)
                        column_num += 1
                st.write("</div>", unsafe_allow_html=True)


def load_file(files):
    dm = DataManipulator()
    dm.set_zip(files)
    return dm


def show_random_images(files):
    if files:
        dm = load_file(files)
        """st.write(dm.transform_file_to_df_zip())
        dm.save_df("res_test.csv")"""
        im_to_show = dm.get_images_to_show()
        pictures_container(im_to_show, files, 4)


def load_file_test():
    files = st.file_uploader("Загрузите .zip файл")
    return files


def load_dataset_page():
    st.title("Загрузка датасета")
    with open("styles.css") as f:
        st.write(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    new_files = st.file_uploader("Загрузите .zip файл")
    if new_files:
        if new_files != st.session_state["file"]:
            st.session_state["file"] = new_files

def initial_setup_page():
    st.title("Начальная настройка алгоритма")
    
def dataset_marking_page(files):
    st.title("Разметка датасета")
    #show_random_images(files)

def dataset_addmarking_page(files):
    st.title("Доразметка датасета")
    show_random_images(files)

def results_page():
    st.title("Результаты итерации")


if __name__ == "__main__":
    if "stage" not in st.session_state:
        st.session_state["stage"] = 1

    chosen_id = stx.tab_bar(data=[
    stx.TabBarItemData(id=1, title="1", description="Загрузка датасета"),
    stx.TabBarItemData(id=2, title="2", description="Начальная настройка алгоритма"),
    stx.TabBarItemData(id=3, title="3", description="Разметка датасета"),
    stx.TabBarItemData(id=4, title="4", description="Доразметка датасета"),
    stx.TabBarItemData(id=5, title="5", description="Результаты"),
    ], default=st.session_state["stage"])

    if int(chosen_id) != st.session_state["stage"]:
        st.session_state["stage"] = int(chosen_id)
        st.rerun()
        
    if "file" not in st.session_state: 
        st.session_state["file"] = None

    if st.session_state["stage"] == 1:
        load_dataset_page()
    if st.session_state["stage"] == 2:
        initial_setup_page()
    if st.session_state["stage"] == 3:
        dataset_marking_page(st.session_state["file"])
    if st.session_state["stage"] == 4:
        dataset_addmarking_page(st.session_state["file"])
    if st.session_state["stage"] == 5:
        results_page()
    

    button_cols = st.columns(5)
    with button_cols[0]:
        bt_prev = st.button("Назад", use_container_width=True)
        if bt_prev:
            if st.session_state["stage"] != 1:
                st.session_state["stage"] -= 1
                st.rerun()

    with button_cols[4]:
        bt_next = st.button("Далее", use_container_width=True)
        if bt_next:
            if st.session_state["stage"] != 5:
                st.session_state["stage"] += 1
                st.rerun()
            else:
                st.session_state["stage"] = 1
                st.rerun()
    














