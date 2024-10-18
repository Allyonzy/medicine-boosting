import streamlit as st
from utils import get_dataframe

import seaborn as sns
from matplotlib import pyplot as plt
# importing libraries
import altair as alt

from urllib.error import URLError

df = get_dataframe()
st.title('Предварительная обработка данных (Препроцессинг)')

st.table(df.head(10))

st.write('Сколько пропущенных значений отсутствует в каждом элементе')

feature_columns = df.columns


for column in feature_columns:
    st.write(f"{column} ==> Пропущенные значения вариант1 : {df.isna().mean()}")
    st.write("============================================")
    st.write(f"{column} ==> Значения  вариант2 равные 0: {len(df.loc[df[column] == 0])}")

st.write('Посмотрим корреляцию по значениям в исходном датасете')
st.write('Таблица корреляции')
st.table(df.corr())

st.write('График корреляции')
sns.heatmap(df.corr(), cmap="viridis", annot=True, linewidth=0.5)

st.pyplot(plt)