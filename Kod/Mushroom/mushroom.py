import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
plt.style.use('ggplot') # Have a nicer style of the plots

from sklearn import tree

from pandas import DataFrame 

filename_mushrooms = './agaricus-lepiota.csv'


df_mushrooms = pd.read_csv(filename_mushrooms)
df_mushrooms.head()
display(df_mushrooms.head())