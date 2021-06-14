# librarys
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import psutil
import re
plt.style.use('ggplot')


SRC_DIR = os.path.join( os.path.abspath( '.'), 'src')
SRC_DIR = os.path.dirname( os.path.abspath( __file__ )) 
BASE_DIR = os.path.dirname( SRC_DIR )
DATA_DIR = os.path.join( BASE_DIR, 'data')
IMG_DIR = os.path.join( BASE_DIR, 'img')


df = pd.read_csv(os.path.join( DATA_DIR, 'pesquisa_valores.csv' ), skiprows=1, index_col=0)
print(f'rows: {df.shape[0]}\ncolumns: {df.shape[1]}')

#isso aqui embaixo é uma abstração, se pá vou usar em algum momento

# with open(os.path.join( DATA_DIR, 'pesquisa_valores.csv' ), newline='') as f:
#     reader = csv.reader(f)
#     header_legend = next(reader)


# Colocando a coluna de mediana no dataframe estatístico
median_row = df.median().values.tolist() # mediana de cada feature
median_row.insert(0, 'median')

b_stats = df.describe()[1:3].reset_index()

b_stats.loc[-1] = median_row # new line
b_stats.index += 1 # arrumando o index

b_stats = b_stats.set_index('index').transpose()
b_stats.head()

# Quais as maiores média e mediana? ------------------------------

def sort_values_stats(n, medida = 'mean', best = False):
    
    if best:
         
         return(b_stats[medida].sort_values(ascending = False)[1:n])
    
    else:

        return(b_stats[medida].sort_values(ascending = False)[-n:])
    

## 10 melhores
### media
sort_values_stats(11, best = True)

### mediana
sort_values_stats(11, 'median', best = True)

## 10 piores
### media
sort_values_stats(10)

### mediana
sort_values_stats(10, 'median')

# Dummies ------------------------------
'''
Setores:
 - Privado: 0
 - Publico: 1
 - Ambos: 2
 
Trabalho:
 - Não estou trabalhando: 0
 - Privada: 1
 - Pública: 2

Disponibilizar dados:
 - Não: 0
 - Sim: 1
'''

## definindo os maps
map_setor = {'Ambos': 2, 'Setor privado': 0, 'Setor público':1}
map_trabalho = {'Não estou trabalhando': 0, 'Privada': 1, 'Pública':2}
map_dados = {'Não': 0, 'Sim': 1}

## colunas grande demais para serem repetidas mais que uma vez, vou definir, obg
col_setor = 'Atualmente, o setor que prefiro me inserir para trabalhar é: '
col_org = 'Minha atual organização é:'
col_dados = 'Disponibilizar dados'

## aplicando o map!
df[col_setor] = df[col_setor].map(map_setor)
df[col_org] = df[col_org].map(map_trabalho)
df[col_dados] = df[col_dados].map(map_dados)

df.to_csv(os.path.join( DATA_DIR, 'cleaned_data.csv'), index= False)

# Correlation ------------------------------
corr_matrix = df.corr().abs()
upper = corr_matrix.where(
    np.triu(
        np.triu(
            np.ones(corr_matrix.shape), k=1).astype(np.bool_)))

most_correlated_series = upper.unstack().sort_values(ascending=False)
df_most_correlated = pd.DataFrame(most_correlated_series).dropna()
df_most_correlated.columns = ['corr']

boolean_query = df_most_correlated > 0.6
df_maior_60_corr = df_most_correlated[boolean_query['corr']]
print(df_maior_60_corr)

# df_maior_60_corr.to_csv(os.path.join(DATA_DIR, 'df_maior_60_corr.csv'))
