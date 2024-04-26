import pandas as pd
import numpy as np
import spacy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from text_classifier import load_model_label, create_model
from news_scrapper import parse_news, load_data_json

# розрахунок по категоріям для однієї доби окремого сайту
def calc_predictions_for_date(ser, model, lables):
    count = {}
    prediction = list(model.predict(ser))
    for i in lables.index:
        count[i] = prediction.count(i)
    return pd.Series(count)

# розрахунок кількості новин по категоріям
def calc_predictions(df, model, labels):
    pred_df = pd.DataFrame(0, index=df.index, columns = labels.index)
    for row in df.index:
        for colum in df.columns:
            pred_df.loc[row]+=calc_predictions_for_date(df.loc[row, colum],
                                                        model, labels)
    return pred_df

def plot3D(df, labels):
    list_tmp = []
    for i in range(len(df.index)):
        list_tmp.append(df.index[i].date())
    # товщина стовпців
    dx, dy = .4, .4

    # підготовка вісей
    fig = plt.figure(figsize=(10,6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    # розмітка вісей
    xpos=np.arange(df.shape[0])
    ypos=np.arange(df.shape[1])

    ax.set_xticks(xpos + dx/2)
    ax.set_yticks(ypos + dy/2)

    # створення meshgrid
    xpos, ypos = np.meshgrid(xpos, ypos)
    xpos = xpos.flatten()
    ypos = ypos.flatten()

    # початок стовпців
    zpos=np.zeros(df.shape).flatten()

    # висота стовпців
    dz = df.values.ravel()
    # plot
    ax.bar3d(xpos,ypos,zpos,dx,dy,dz)

    # спрявжні лейбли вісей
    ax.w_yaxis.set_ticklabels(labels.to_numpy(),  rotation=5)
    ax.w_xaxis.set_ticklabels(list_tmp)

    ax.set_zlabel('Кількість новин')

    plt.show()

if __name__ == '__main__':
    print('Оберіть функціонал:')
    print('1 - Створення та навчання моделі класифікації заголовків')
    print('2 - Парсинг даних з 2 новиних сатів')
    print('3 - Аналіз отриманих даних з використанням OLAP')
    mode = int(input('Оберіть режим роботи:'))
    if(mode == 1):
        create_model()
    if(mode == 2):
        parse_news()
    if(mode == 3):
        df = load_data_json('news_data.json')
        model, labels = load_model_label()
        data = calc_predictions(df, model, labels)
        print(data)
        plot3D(data, labels)
