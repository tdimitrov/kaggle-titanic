import titanic.plots as tpl
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/input/train.csv')

tpl.binned_column(data, 'Age', 10)
tpl.binned_column(data, 'Fare', 25)
tpl.column(data, 'Sex')
tpl.column(data, 'Pclass')
tpl.column(data, 'SibSp')

plt.show()
