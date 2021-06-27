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

class TimeSeriesAnalysis:

    def __init__(self, cod1, name_cod1, cod2, name_cod2):

        self.name_cod1 = name_cod1
        self.name_cod2 = name_cod2
        self.df1 = self.get_bc_data(cod1)
        self.df2 = self.get_bc_data(cod2)
        self.df = self.join_inner_dfs(name_cod1, name_cod2) 

    def get_bc_data(self, cod, type = 'linear'):
        link = f'http://api.bcb.gov.br/dados/serie/bcdata.sgs.{cod}/dados?formato=json'
        
        df = pd.read_json(link)
        df['data'] = pd.to_datetime(df.data, format= ('%d/%m/%Y'))
        df = df.set_index('data')['1995':]
        
        if any(df.isna().sum() < 0) != False:
            df = self.interpolate_and_plot(df, type=type) 
        return df


    def join_inner_dfs(self, colname1, colname2, type = 'linear'):
        self.df = pd.concat([self.df1, self.df2], axis = 1, join= 'inner')
        self.df.columns = [colname1, colname2]
        
        self.X = self.df[colname1].values.reshape(-1, 1)
        self.y = self.df[colname2].values.reshape(-1, 1)

        if any(self.df.isna().sum() < 0) != False:
            self.interpolate_and_plot(self.df, type = type)

        return self.df


    def values_pct(self, window = 20):

        values = self.df.rolling(window= window).agg(self.percent_change)
        
        self.values_pct = values.agg(self.replace_outlier)

        self.values_pct.plot()
        plt.show()

        return self.values_pct    


    def percent_change(self, series):

        previous_values = series[:-1]
        last_value = series[-1]

        percent_change = (last_value - np.mean(previous_values)) / np.mean(previous_values)

        return percent_change
        
        
    def replace_outlier(self, series):

        abs_diff_from_mean = np.abs(series - np.mean(series))
        this_mask = abs_diff_from_mean > (np.std(series) * 3)
        
        series[this_mask] = np.nanmedian(series)

        return series
        

    def interpolate_and_plot(self, data, type = 'linear'):
        missing_values = data.isna()
        self.values_interp = data.interpolate(type)
        
        ## plot com go.Figure()
        fig, ax = plt.subplots(figsize=(10, 5))
        
        self.values_interp.plot(color='k', alpha=.6, ax=ax, legend=False)

        self.values_interp[missing_values].plot(color='r', lw = 3, ax=ax, legend=False)

        plt.show()
        return self.values_interp


    def create_features(self, window:int = 21, features:list = [np.mean]):
        data_rolling = self.df.rolling(window)
        self.data_features = data_rolling.agg(features)
        self.data_features.columns = ['_'.join(col) for col in self.data_features]
        return self.data_features


    def euclidean_distance(self):
        return np.sqrt(np.sum((self.df1['valor'] - self.df2['valor'])**2))


    def ridge_model(self, alpha = .1):
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = .25, shuffle = False, random_state = 146)

        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        y_hat = ridge.predict(X_test)
        self.plot_predict(y_test, y_hat)
        return r2_score(y_test, y_hat)


    def plot_features(self):
        ax = self.data_features.plot()
        self.df.plot(ax = ax, color = 'k', alpha= .2, lw = 3)
        plt.show()
        

    def plotly_two(self):
        fig = go.Figure()
    
        fig.add_trace(
        go.Scatter(x = self.df1.index, y = self.df1['valor'], name = self.name_cod1))
    
        fig.add_trace(
        go.Scatter(x = self.df2.index, y = self.df2['valor'], name = self.name_cod2))
    
        fig.update_xaxes(rangeslider_visible=True)   
        fig.show()
    

    def scatterplot(self):
        x = self.df.columns.tolist()[0]
        y = self.df.columns.tolist()[1]
    
        fig = px.scatter(self.df, x = x, y = y, color = self.df.reset_index().index)
        fig.show()
        
    def plot_predict(self, y_test, y_hat):
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(y_test, color = 'k', lw = 3, label = 'y')
        ax.plot(y_hat, color = 'r', lw = 2, label = 'y_hat')
        ax.legend()
        plt.show()