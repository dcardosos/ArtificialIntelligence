reload from importlib import reload

from src.time_serie_analysis import TimeSeriesAnalysis
reload(src.time_serie_analysis)

## energia e metal
energia_metal = TimeSeriesAnalysis(27577, 27576)
energia_metal.plotly()
energia_metal.euclidean_distance()

## CDI e SELIC: anualizada base 252, %a.a.
cdi_selic = TimeSeriesAnalysis(4389, 'cdi', 1178, 'selic')

### plot
cdi_selic.plotly_two()

### distance
cdi_selic.euclidean_distance()

### plot
cdi_selic.scatterplot()

### RIDGE
cdi_selic.ridge_model()

### Messy data exemplo
import numpy as np
dados_teste = cdi_selic.df[:]
dados_teste.iloc[range(3500, 3600), :] = np.nan

cdi_selic.interpolate_and_plot(dados_teste, 'quadratic')

### percent change in data raw
cdi_selic.values_pct()

### moving average
cdi_selic.create_features(window = 200)

cdi_selic.plot_features()

## Dynamic Time Warping
## Dynamic time warping finds the optimal non-linear alignment between two time series.
