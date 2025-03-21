import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Загрузка датасета Titanic
@st.cache_data  # Кэширование данных для ускорения работы
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

df = load_data()

st.title("Дашбоард для анализа данных пассажиров Титаника")

# Описательная статистика
st.header("Описательная статистика")
st.write("Форма таблицы:", df.shape)
st.write("Колонки и типы данных:")
st.write(df.dtypes)

st.header("Гистограмма возраста пассажиров")
fig1, ax1 = plt.subplots()
sns.histplot(df['Age'].dropna(), kde=True, ax=ax1)
st.pyplot(fig1)

st.header("Круговая диаграмма выживших")
survived_counts = df['Survived'].value_counts()
fig2, ax2 = plt.subplots()
ax2.pie(survived_counts, labels=['Не выжили', 'Выжили'], autopct='%1.1f%%')
st.pyplot(fig2)

st.header("Точечный график: Возраст vs Стоимость билета")
fig3 = px.scatter(df, x='Age', y='Fare', color='Survived', title="Возраст vs Стоимость билета")
st.plotly_chart(fig3)

st.header("Интерактивный график: Распределение по классу билета")
selected_class = st.selectbox("Выберите класс билета", df['Pclass'].unique())
filtered_df = df[df['Pclass'] == selected_class]
fig4 = px.histogram(filtered_df, x='Age', nbins=20, title=f"Распределение возраста для класса {selected_class}")
st.plotly_chart(fig4)

st.header("Boxplot: Возраст по классам билета")
fig5, ax5 = plt.subplots()
sns.boxplot(x='Pclass', y='Age', data=df, ax=ax5)
st.pyplot(fig5)

st.header("Вывод строк таблицы")
n_rows = st.number_input("Введите количество строк для отображения", min_value=1, max_value=len(df), value=5)
st.write(df.head(n_rows))
