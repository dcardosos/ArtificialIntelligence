import pandas as pd
import numpy as np
from scipy.sparse import data
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE

# constante de endereço
SRC_DIR = os.path.join( os.path.abspath( '.'), 'src')
SRC_DIR = os.path.dirname( os.path.abspath(__file__)) 
BASE_DIR = os.path.dirname( SRC_DIR )
DATA_DIR = os.path.join( BASE_DIR, 'data')


# read csv
df = pd.read_csv(os.path.join( DATA_DIR, 'df.csv'))

# separando as features
df.select_dtypes('object')

cat_features = ['Branch', 'Component', 'Gender', 'BMI_class', 'Height_class']
num_features = set(df.columns) - set(cat_features)

# get df numeric
df_numeric = df[num_features]


# change comma to dot in columns = [ BMI, weight_kg, sature_m ]
cols_to_change = [ 'BMI', 'weight_kg', 'stature_m' ]
df_numeric[cols_to_change] = df_numeric[cols_to_change].apply(lambda x: x.str.replace(',', '.'))

### t-SNE------
m = TSNE(learning_rate = 50)
tsne_features = m.fit_transform(df_numeric)

df['x'] = tsne_features[:,0]
df['y'] = tsne_features[:,1]

# plot with BMI_class tonalization 
sns.scatterplot(x = 'x', y = 'y', hue = 'BMI_class', data = df)
plt.show()

sns.scatterplot(x = 'x', y = 'y', hue = 'Height_class', data = df)
plt.show()