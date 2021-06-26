from numpy.core.numeric import cross
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from scipy.sparse.construct import random
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

plt.style.use('ggplot')

class TimeSeriesAnalysis:

    def __init__(self, cod1, cod2):

        self.df1 = self.get_bc_data(cod1)
        self.df2 = self.get_bc_data(cod2)
    
    def get_bc_data(self, cod, type = 'linear'):
        link = f'http://api.bcb.gov.br/dados/serie/bcdata.sgs.{cod}/dados?formato=json'
        
        df = pd.read_json(link)
        df['data'] = pd.to_datetime(df.data, format= ('%d/%m/%Y'))
        df = df.set_index('data')['1995':].reset_index()
        
        if any(df.isna().sum() < 0) != False:
            df = self.interpolate_and_plot(df, type=type) 
        return df

    def join_inner_dfs(self, colname1, colname2, type = 'linear'):
        self.df = pd.concat(
            [self.df1.set_index('data'), self.df2.set_index('data')], 
            axis = 1, join= 'inner').reset_index()

        self.df.columns = ['data', colname1, colname2]
        
        self.X = self.df[colname1].values.reshape(-1, 1)
        self.y = self.df[colname2].values.reshape(-1, 1)

        if any(self.df.isna().sum() < 0) != False:
            self.interpolate_and_plot(self.df, type = type)

        return self.df    

    def interpolate_and_plot(self, data, type = 'linear'):
        missing_values = data.isna()
        self.values_interp = data.interpolate(type)
        

        ## plot com go.Figure()
        fig, ax = plt.subplots(figsize=(10, 5))
        self.values_interp.plot(color='k', alpha=.6, ax=ax, legend=False)

        self.values_interp[missing_values].plot(color='r', lw = 3, ax=ax, legend=False)

        plt.show()
        return self.values_interp

    def euclidean_distance(self):
        return np.sqrt(np.sum((self.df1['valor'] - self.df2['valor'])**2))


    def ridge_model(self, alpha = .1):
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = .25, shuffle = False, random_state = 146)

        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        y_hat = ridge.predict(X_test)
        self.plot_predict(y_test, y_hat)
        return r2_score(y_test, y_hat)


    def plotly_two(self):
        fig = go.Figure()
    
        fig.add_trace(
        go.Scatter(x = self.df1['data'], y = self.df1['valor']))
    
        fig.add_trace(
        go.Scatter(x = self.df2['data'], y = self.df2['valor']))
    
        fig.update_xaxes(rangeslider_visible=True)   
        fig.show()
    

    def scatterplot(self):
        df_temp = self.df[:]
        x = df_temp.columns.tolist()[1]
        y = df_temp.columns.tolist()[2]
    
        fig = px.scatter(df_temp, x = x, y = y, color = df_temp.index)
        fig.show()
        
    def plot_predict(self, y_test, y_hat):
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(y_test, color = 'k', lw = 3, label = 'y')
        ax.plot(y_hat, color = 'r', lw = 2, label = 'y_hat')
        ax.legend()
        plt.show()


## energia e metal
energia_metal = TimeSeriesAnalysis(27577, 27576)
energia_metal.plotly()
energia_metal.euclidean_distance()

## CDI e SELIC: anualizada base 252, %a.a.
cdi_selic = TimeSeriesAnalysis(4389, 1178)

### plot
cdi_selic.plotly_two()

### distance
cdi_selic.euclidean_distance()

### self.df, x, y
cdi_selic.join_inner_dfs('cdi', 'selic')

### plot
cdi_selic.scatterplot()

### RIDGE
cdi_selic.ridge_model()

### Messy data exemplo
dados_teste = cdi_selic.df.drop(labels = range(500, 600), axis = 0).set_index('data')
cdi_selic.interpolate_and_plot(dados_teste)

## Dynamic Time Warping
## Dynamic time warping finds the optimal non-linear alignment between two time series.
