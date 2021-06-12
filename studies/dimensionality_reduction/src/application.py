import pandas as pd
import numpy as np
from scipy.sparse import data
import seaborn as sns
import matplotlib.pyplot as plt
import os
from seaborn.matrix import heatmap
from sklearn.manifold import TSNE
from sklearn.feature_selection import VarianceThreshold

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
df_numeric.loc[:, cols_to_change] = df_numeric.loc[:, cols_to_change].apply(lambda x: x.str.replace(',', '.')).astype(float)

## changes in original df
df.loc[:, cols_to_change] = df.loc[:, cols_to_change].apply(lambda x: x.str.replace(',', '.')).astype(float)

### t-SNE------
m = TSNE(learning_rate = 50)
tsne_features = m.fit_transform(df_numeric)

df['x'] = tsne_features[:,0]
df['y'] = tsne_features[:,1]

# plot with BMI_class tonalization 
sns.scatterplot(x = 'x', y = 'y', hue = 'BMI_class', data = df)
plt.show()

# plot with Height_class tonalization 
sns.scatterplot(x = 'x', y = 'y', hue = 'Height_class', data = df)
plt.show()

###  VarianceThreshold ------
sel = VarianceThreshold(threshold=1)
sel.fit(df_numeric)

mask = sel.get_support()

reduce_df = df_numeric.loc[:, mask]

# sel normalization
normalized_df_numeric = df_numeric / df_numeric.mean()
normalized_df_numeric.var().sort_values()

sel = VarianceThreshold(threshold=.002)
sel.fit(normalized_df_numeric)

mask_normalized = sel.get_support()
normalized_reduce_df = df_numeric.loc[:, mask_normalized]


### --- NaN findings
df.isna().sum().sum() # nenhum nan


## correlation plot ------------
df_corr = df.corr()
mask = np.triu(np.ones_like(df_corr, dtype=bool))

sns.heatmap(df_corr, mask = mask)
plt.show()

# matrix correlation with filter - para features com alta correlação entre si
corr_matrix = tri_df = df.corr().abs()
mask_m = np.triu(np.ones_like(corr_matrix, dtype=bool))
tri_df = corr_matrix.mask(mask_m)

to_drop = [c for c in df_numeric.columns if any(tri_df[c] > .95)] # mais ou menos arbitrario

# cara, para ver quantas relações entre as colunas são maior que corr .95 por exemplo
# (trim_df > .95).sum().sum()
# se tu botar apenas um .sum() ele vai dar por coluna, aqui tem mtas, é difícil de analisar, mas é assim que faz

reduce_tri_df = df.drop(to_drop, axis = 1)

# diferenças
print(f'Before {df.shape[1]} -> After {reduce_tri_df.shape[1]}')