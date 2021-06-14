import pandas as pd
import numpy as np
import csv
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('random_projects/comport_organizacional/pesquisa_valores.csv', skiprows=1, index_col=0)
df.head()

print(f'rows: {df.shape[0]}\ncolumns: {df.shape[1]}')

#### feature engineering
map_setor = {'Ambos': 2, 'Setor privado': 0, 'Setor público':1}
map_trabalho = {'Não estou trabalhando': 0, 'Privada': 1, 'Pública':2}
map_dados = {'Não': 0, 'Sim': 1}

df['Atualmente, o setor que prefiro me inserir para trabalhar é: '] = df['Atualmente, o setor que prefiro me inserir para trabalhar é: '].map(map_setor) 
df['Minha atual organização é:'] = df['Minha atual organização é:'].map(map_trabalho)
df['Disponibilizar dados'] = df['Disponibilizar dados'].map(map_dados)

# definindo X e y
y = df['Atualmente, o setor que prefiro me inserir para trabalhar é: '].values
X = df.drop('Atualmente, o setor que prefiro me inserir para trabalhar é: ', axis = 1).values

# Recursive Feature Elimination
rfe = RFE(estimator= LogisticRegression(), n_features_to_select= 10, verbose= 1)
rfe.fit(X, y)

## ranking
print(dict(zip(df.columns, rfe.ranking_)))

# mask
lg_mask = rfe.support_

# dataframe reduzido
reduced_df = df.drop('Atualmente, o setor que prefiro me inserir para trabalhar é: ', axis = 1).loc[:, rfe.support_]

# ----------------------------------------------
### usando random forest classifier
rfe_rf = RFE(estimator = RandomForestClassifier(),n_features_to_select = 10, step = 10, verbose = 1)

rfe_rf.fit(X, y)

print(rfe_rf.ranking_)

rf_mask = rfe_rf.support_

# ----------------------------------------------
### usando gradient boosting
rfe_gb = RFE(estimator = GradientBoostingClassifier(), n_features_to_select = 10, step = 10, verbose = 1)

rfe_gb.fit(X, y)

gb_mask = rfe_gb.support_


### --------------------------
# votes
votes = np.sum([lg_mask, rf_mask, gb_mask], axis = 0)

# aqui, podemos selecionar nosso critério de quais features pegar,
# as que tiveram pelo menos 2 votos, por exemplo

mask = votes >= 2

df_droped = df.drop('Atualmente, o setor que prefiro me inserir para trabalhar é: ', axis = 1) 
reduced_X = df_droped.loc[:, mask]

# Plug the reduced dataset into a linear regression pipeline
X_train, X_test, y_train, y_test = train_test_split(reduced_X, y, test_size=0.3, random_state=0)

scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)


gb = GradientBoostingClassifier()
gb.fit(X_train_std, y_train)

r_squared = gb.score(X_test_std, y_test)

print('The model can explain {0:.1%} of the variance in the test set using {1:} features.'.format(r_squared, len(gb.feature_importances_)))


### ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

gb = GradientBoostingClassifier()
gb.fit(X_train_std, y_train)

r_squared = gb.score(X_test_std, y_test)

print('The model can explain {0:.1%} of the variance in the test set using {1:} features.'.format(r_squared, len(gb.feature_importances_)))
