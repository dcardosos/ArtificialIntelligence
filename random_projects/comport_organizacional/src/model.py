import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn.relational import scatterplot
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

SRC_DIR = os.path.join( os.path.abspath( '.'), 'src')
SRC_DIR = os.path.dirname( os.path.abspath( __file__ )) 
BASE_DIR = os.path.dirname( SRC_DIR )
DATA_DIR = os.path.join( BASE_DIR, 'data')
IMG_DIR = os.path.join( BASE_DIR, 'img')

df = pd.read_csv(os.path.join( DATA_DIR, 'cleaned_data.csv'))

df.select_dtypes('object') # nenhuma coluna como object, tudo otimo

# bem, sendo assim, vamos tacar do TSNE neah

m  = TSNE(learning_rate= 50)  # 50 pq tem tão pouquinho dado neah
tsne_features = m.fit_transform(df) # criou um array com duas colunas só em

# vamos add duas colunas 'x' e 'y' no nosso df pra fazer uns plot bele
# resolviq eu é melhor fazer uma copia tals

df_tsne = df.copy()
df_tsne['x'] = tsne_features[:, 0]
df_tsne['y'] = tsne_features[:, 1]

# bora de plots
# vou faze uma função pq n quero ficar igual um bobo
def scatterplot_vscode(hue_):
    '''
    A ideia é o seguinte: nesse vscode eu preciso ficar tascando o `plt.show()`
    tooooda hora, não curto e vou precisar fazer bastante gráfico, "ah mas pq n usa notebook'
    porque não, espero ter sido claro, obrigado

    O único argumento aqui é hue pq é o que vai diferenciar
    '''
    sns.scatterplot(x = 'x', y = 'y', hue = hue_, data = df_tsne)

    plt.show()

# ok, agora o hue_ precisa ser colunas que são tipo, classes manja, são categóricas
# são aquelas que eu saquei dos map's no arquivo analysis.py, por isso vou só copiar
# e colar as variaveis da coluna la

## colunas
col_setor = ['Atualmente, o setor que prefiro me inserir para trabalhar é: ']
col_org = ['Minha atual organização é:']
col_dados = ['Disponibilizar dados']   # agora que percebi que isso é *muito* inutil

cols_to_hue = col_setor + col_dados + col_org

# kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk
# NADA NADA NADA NADA NADA NADA QUE ÓDIO MEU DEUS
list(map(scatterplot_vscode, cols_to_hue))


# ta bom, vamos ver VarianceThreshold agora
sel = VarianceThreshold(threshold = 1) 
sel.fit(df)

mask = sel.get_support()

### sem normalizar so joga 5 colunas fora neah
print(f'Before {df.shape[1]} -> After {df.loc[:, mask].shape[1]}')

# agora normalizando os dados
# eu vou usar z_score se pa? NÃO DOUGLAS, PQ AI A VAR DE TODOS VAI SER 1, VC N SABE O QUE 
# É Z_SCORE?????? TA CHEGANDO 00:00 VAMO AGILIZAR
normalized_df = ((df  - df.min() ) / ( df.max() - df.min() ))
normalized_df.var().sort_values() # o que to fazendo aqui é vendo qual threshold eu coloco


### ta vou botar 0.05
sel = VarianceThreshold(threshold = .05) 
sel.fit(normalized_df)

mask = sel.get_support()

### é, tacou fora 18 colunas ne?
print(f'Before {normalized_df.shape[1]} -> After {normalized_df.loc[:, mask].shape[1]}')

## vamo tentar heatmap ai
reduced_df = normalized_df.loc[:, mask]
mask_m = np.triu(np.ones_like(reduced_df.corr(), dtype = bool))

plt.rcParams.update({'font.size': 7})
fig, ax = plt.subplots(figsize=(12,7))   
sns.heatmap(reduced_df.corr(), mask = mask_m, ax = ax) 
plt.tight_layout() 
plt.show()  


# vamos ter que ver PCA pra aplicar em, seria bom!, vou aplicar um basicão
pca = PCA(2)
pca.fit(df)

pca_data = pd.DataFrame(pca.transform(df))

# vou tacar da minha classe n sou bobo
# usei no terminal blz
#wget 'https://github.com/douglacardoso/ArtificialIntelligence/blob/master/studies/cluster/cluster_two.py'

# from cluster_two import ClusterTwo
