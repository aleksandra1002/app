import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Загрузка датасета Titanic
@st.cache_data  # Кэширование данных для ускорения работы
def load_data():
    # качаю табличку
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    
    # возвращаю датасет
    return pd.read_csv(url)

df = load_data()


# чтобы переходить на разные странички
page = st.sidebar.selectbox("Выбрать страницу", 
                            ["Вступление",
                             "Описательная статистика", 
                             "Гистограмма возраста пассажиров",
                             "Круговая диаграмма выживших",
                             "Возраст vs Стоимость билета",
                             "Распределение по классу билета",
                             "Возраст по классам билета",
                             "Вывод строк таблицы",
                             "Поблагодарить"])


if page == "Вступление":
    st.title("Дашбоард для анализа данных пассажиров Титаника 🚢")

    st.image(
        'https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/b5229bb5-6ca7-41f8-8e49-59bd2a0f4737/dh8iilh-c9cf1533-4888-4c0f-b6c6-f608bc26cda8.jpg/v1/fill/w_1213,h_659,q_70,strp/on_this_night__112_years_ago__the_titanic_sank__by_robloxfan333172_dh8iilh-pre.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9MTA4NiIsInBhdGgiOiJcL2ZcL2I1MjI5YmI1LTZjYTctNDFmOC04ZTQ5LTU5YmQyYTBmNDczN1wvZGg4aWlsaC1jOWNmMTUzMy00ODg4LTRjMGYtYjZjNi1mNjA4YmMyNmNkYTguanBnIiwid2lkdGgiOiI8PTIwMDAifV1dLCJhdWQiOlsidXJuOnNlcnZpY2U6aW1hZ2Uub3BlcmF0aW9ucyJdfQ.4C9QHAOryXN_Jv43Q6G-6y7hOZgPP7XU8SSs-d_zpRg', 
         caption = '',
         use_container_width=True
    )


# Описательная статистика
elif page == "Описательная статистика":
    st.title("Описательная статистика 📈")
    st.write("Форма таблицы:", df.shape)
    st.write("Колонки и типы данных:")
    # метод Pandas, который возвращает типы данных каждого столбца:
    st.write(df.dtypes)
    st.subheader('👆 DataFrame показывает, в какой колонке какой тип данных')

elif page == "Гистограмма возраста пассажиров":
    st.title("Гистограмма возраста пассажиров 📊")
    fig1, ax1 = plt.subplots()

    # из библиотеки seaborn для построения гистограммы:
    # удаляю пропуски из исходного датафрейма, kde=True - гладкая прямая сверху гистограммы
    sns.histplot(df['Age'].dropna(), kde=True, ax=ax1)
    # функция для отображения matplotlib-графиков
    st.pyplot(fig1)

    st.subheader('Гистограмма отражает количество пассажиров определённого возраста 👨🏻‍🦳')

elif page == "Круговая диаграмма выживших":

    st.title("Круговая диаграмма отношения выживших")

    # подсчёт уникальных значений из столбца с выжившими
    survived_counts = df['Survived'].value_counts()

    # выбор цвета для диаграммы через интерфейс 

    # надпись
    st.header("Настройкa цветов")
    st.subheader('Выберите цвет диаграммы 👇')
    # выбор для не выживших с помощью color_picker()
    color_not_survived = st.color_picker(
        "Цвет для 'Не выжили'", 
        "#ff9999"  # Значение по умолчанию (светло-красный)
    )

    # выбор для выживших с помощью color_picker()
    color_survived = st.color_picker(
        "Цвет для 'Выжили'", 
        "#66b3ff"  # Значение по умолчанию (светло-голубой)
    )

    # Создание диаграммы с выбранными цветами
    fig2, ax2 = plt.subplots()
    ax2.pie(
        survived_counts,
        labels=['Не выжили', 'Выжили'],
        autopct='%1.1f%%',
        colors=[color_not_survived, color_survived],
    )
    # вывожу
    st.pyplot(fig2)
    st.subheader('Диаграмма отражает процентное соотношение выживших и не выживших 🚢')

elif page == "Возраст vs Стоимость билета":

    st.title("Возраст👩🏻‍🦳 vs Стоимость билета💰")

    # создание точечного графика px.scatter
    # color='Survived' - раскраска по выживанию
    fig3 = px.scatter(df, x='Age', y='Fare', color='Survived', title="Возраст vs Стоимость билета")
    st.plotly_chart(fig3)
    st.subheader('Точечный график показывает распределение по возрасту и стоимости билета 👱🏻‍♂️')
    st.subheader('Не выжившие - тёмно-синие точки, выжившие - белые')

elif page == "Распределение по классу билета":

    st.title("Распределение по классу билета 📉")
    selected_class = st.selectbox("Выберите класс билета", df['Pclass'].unique())

    # Выбор цвета гистограммы
    hist_color = st.color_picker(
        "Выберите цвет гистограммы",
        "#1f77b4"  # Стандартный синий Plotly
    )
    filtered_df = df[df['Pclass'] == selected_class]

    # построение гистограммы по данным filtered_df с 
    fig4 = px.histogram(
        filtered_df, 
        x='Age', 
        title=f"Распределение возраста для класса {selected_class}", 
        color_discrete_sequence=[hist_color]
    )
    # настройка отступов между столбцами (правильный способ)
    fig4.update_layout(
        bargap=0.1,  # Расстояние между столбцами (действительно существует в Plotly!)
        bargroupgap=0.1,  # Расстояние между группами столбцов
        xaxis_title="Возраст",
        yaxis_title="Количество пассажиров"
    )
    # вывод
    st.plotly_chart(fig4)

    # прикольная штучка с ползунком
    optionals = st.expander("Optional Configurations", True)

    fare_min = optionals.slider(
    "Минимальная цена",
    min_value=float(df['Fare'].min()),
    max_value=float(df['Fare'].max())
    )
    fare_max = optionals.slider(
    "Максимальная цена",
    min_value=float(df['Fare'].min()),
    max_value=float(df['Fare'].max())
    )
    subset_fare = df[(df['Fare'] <= fare_max) & (fare_min <= df['Fare'])]

    st.subheader(f"Число записей по цене билета между {fare_min} и {fare_max}: {subset_fare.shape[0]}")


elif page == "Возраст по классам билета":

    st.title("Возраст по классам билета 🛳️")
    fig5, ax5 = plt.subplots()
    sns.boxplot(x='Pclass', y='Age', data=df, ax=ax5)
    st.pyplot(fig5)
    st.subheader('⬆️ График показывет распределение по классу билета и возрасту')

elif page == "Вывод строк таблицы":

    st.title("Вывод строк таблицы")
    # интерактивный ввод числа строк
    st.subheader('А вот и сама таблица')
    st.subheader("Выберите количество строк для отображения ⬇️")
    n_rows = st.number_input('', min_value=1, max_value=len(df), value=5)
    # печатат первые n_rows из таблицы
    st.write(df.head(n_rows))

elif page == "Поблагодарить":

    st.title('Спасибо за внимание!!!')
    st.image(
        'https://steamuserimages-a.akamaihd.net/ugc/2056493493099136845/4F5CA933CF2D2C5039DAEE1B15E48F9BE9491888/?imw=512&amp;&amp;ima=fit&amp;impolicy=Letterbox&amp;imcolor=%23000000&amp;letterbox=false', 
         caption = '',
         use_container_width=True
    )