import inspect
import textwrap

import streamlit as st

import pandas as pd

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def show_code(demo):
    """Showing the code of the demo."""
    show_code = st.sidebar.checkbox("Show code", True)
    if show_code:
        # Showing the code of the demo.
        st.markdown("## Code")
        sourcelines, _ = inspect.getsourcelines(demo)
        st.code(textwrap.dedent("".join(sourcelines[1:])))

def get_dataframe():
    # чтение датасета
    dataframe = pd.read_csv("data/diabetes.csv")
    # приведем названия колонок к нижнему регистру и удалим знаки препинания
    dataframe.columns = dataframe.columns.str.lower().str.replace('[^\w\s]', '', regex=True)

    return dataframe

def bagging_model_load():
    loaded = joblib.load('bagging_model.pkl') #классическое  машинное
    #h5, keras - для нейронок
    return loaded


def knn_model_load():
    #загрузка файла модели knn
    knn_loaded = joblib.load('knn_best_model.pkl') #классическое  машинное
    #h5, keras - для нейронок
    return knn_loaded

def evaluate(model, test_data, y_data):

    y_test_pred = model.predict(test_data)

    # st.write("Предсказание модели на текущих данных" + y_test_pred)

    st.write("РЕЗУЛЬТАТЫ НА ВЫБОРКЕ: \n===============================")
    clf_report = pd.DataFrame(classification_report(y_data, y_test_pred, output_dict=True))
    st.table(clf_report)
    st.write(f"МАТРИЦА ОШИБОК (CONFUSION MATRIX):\n{confusion_matrix(y_data, y_test_pred)}")
    st.write(f"ACCURACY ПАРАМЕТР:\n{accuracy_score(y_data, y_test_pred):.4f}")
    st.write(f"PRECISION ПАРАМЕТР:\n{precision_score(y_data, y_test_pred):.4f}")
    st.write(f"RECALL ПАРАМЕТР:\n{recall_score(y_data, y_test_pred):.4f}")
    st.write(f"F1 МЕРА:\n{f1_score(y_data, y_test_pred):.4f}")
    st.write(f"ОТЧЕТ О КЛАССИФИКАЦИИ:\n{clf_report}")