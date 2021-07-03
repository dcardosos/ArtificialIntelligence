from src.time_serie_analysis import TimeSeriesAnalysis

## energia e metal
energia_metal = TimeSeriesAnalysis(27577, 'energia', 27576, 'metal')
energia_metal.plotly_two()
energia_metal.euclidean_distance()

## CDI e SELIC: anualizada base 252, %a.a.
cdi_selic = TimeSeriesAnalysis(4389, 'cdi', 1178, 'selic')

cdi_selic.df
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
import numpy as np
cdi_selic.create_features(window = 365, features=[np.mean, np.max, np.min])
cdi_selic.data_features

cdi_selic.plot_features()

### euro libra
euro_libra = TimeSeriesAnalysis(21619, 'euro', 21623, 'libra')

euro_libra.euclidean_distance()
euro_libra.plotly_two()
euro_libra.scatterplot()
euro_libra.values_pct(200)
euro_libra.ridge_model()
euro_libra.create_features(window= 200)
euro_libra.create_percentiles()
euro_libra.plot_features()


## Dynamic Time Warping
## Dynamic time warping finds the optimal non-linear alignment between two time series.




