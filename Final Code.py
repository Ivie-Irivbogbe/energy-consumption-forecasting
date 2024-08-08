#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:33:22 2024

@author: ivieirivbogbe
"""
############################################################
#   Working Directory Setup 
############################################################
# Getting the current working directory   
import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split

# Statistical modeling
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import Holt, ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Forecasting
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.base import ForecastingHorizon 
from sktime.utils.plotting import plot_series, plot_correlations

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

os.chdir('/Users/ivieirivbogbe/Desktop/SPRING 2024/Forecasting Methods - 202410 BANA-7350-1QA/Project')  
os.getcwd()

# Setting up charting formats:
# Optional plot parameters
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['figure.figsize'] = [16.0, 8.0]
plt.rcParams['font.size']= 20
plt.style.available     # Check the styles available for Chart formats
plt.style.use('fivethirtyeight')       # Assign FiveThirtyEight 

#Load the CSV file in Python as a DataFrame
df = pd.read_csv('NATURALGASD11.csv', index_col=0, parse_dates=True)
df = df.rename(columns={'NATURALGASD11': 'NaturalGasConsumption'})

df.plot(label='Natural Gas Consumption', xlabel='Year', ylabel='Billion Cubic Feet', title='Natural Gas Consumption Over Time')
plt.show()

###############################################################################
#   Obtaining Prediction Intervals of ETS Models for Natural Gas Consumption
###############################################################################
# ETS(A,A,N) model setup for natural gas consumption
model = ETSModel(df['NaturalGasConsumption'], error='additive', trend='additive', seasonal=None, initialization_method="estimated")
fit = model.fit()
fcast = fit.forecast(24)  # Forecast for the next 24 months

# Print the summary of the fit
print(fit.summary())

# Plot the forecast
fcast.plot(label='Forecast: Holt’s Linear Method with Additive Error')
plt.xlabel('Year')
plt.ylabel('Forecasted Natural Gas Consumption (Billion Cubic Feet)')
plt.title('24-Month Forecast for Natural Gas Consumption')
plt.legend()
plt.show()

# Display the last time period of your data to confirm the starting point
print(df.tail(1))

# Obtain prediction intervals for the 24-month forecast period
pred = fit.get_prediction(start='2024-01', end='2025-12')  # Adjusted to cover 24 months
pred_intervals = pred.summary_frame(alpha=0.05)

# Plot the actual and predicted values
plt.plot(df.index, df['NaturalGasConsumption'], label='Actual', color='blue')
plt.plot(pred_intervals.index, pred_intervals['mean'], label='Predicted', color='red')
plt.fill_between(pred_intervals.index, pred_intervals['pi_lower'], pred_intervals['pi_upper'], color='grey', alpha=0.3)
plt.title('ETS Forecast with Prediction Intervals for Natural Gas Consumption')
plt.xlabel('Year')
plt.ylabel('Billion Cubic Feet')
plt.legend()
plt.show()

##############################################################################          
# Holt-Winter's Method Forecasts for the next 24-month period
##############################################################################
# Fit the Holt-Winters model with both trend and seasonality components being additive
model_hw = ExponentialSmoothing(df, trend='additive', seasonal='additive', initialization_method="estimated")
fit_hw = model_hw.fit()
fcast_hw = fit_hw.forecast(24) # Forecast for the next 24 periods

# Print the summary of the fit
print(fit_hw.summary())

# Plot the original data, fitted values, and the forecast
plt.plot(df.index, df['NaturalGasConsumption'], label='Original', color='blue')
plt.plot(fit_hw.fittedvalues, label='Fitted', color='orange')
plt.plot(fcast_hw.index, fcast_hw, label='Forecast', color='red')
plt.xlabel('Year')  
plt.ylabel('Billion Cubic Feet')
plt.legend()
plt.title("Holt-Winter's Method Forecasts for Natural Gas Consumption (24 Periods)")
plt.tight_layout()
plt.show()

############################################################################## 
#   VAR method all 3 variables
############################################################################## 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.api import VAR
from datetime import datetime, timedelta

os.chdir('/Users/ivieirivbogbe/Desktop/SPRING 2024/Forecasting Methods - 202410 BANA-7350-1QA/Project')  

# Setting up charting formats:
# Optional plot parameters
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['figure.figsize'] = [16.0, 8.0]
plt.rcParams['font.size']= 20
plt.style.available     # Check what are the styles available for Chart formats
plt.style.use('fivethirtyeight')       # Assign FiveThirtyEight 

#Load the CSV file in Python as a DataFrame
df2 = pd.read_csv('CombinedData.csv', index_col=0, parse_dates=True)

# Plot the individual series
df2['NaturalGasConsumption'].plot(label='Natural Gas Consumption', xlabel='Year', ylabel='Billion Cubic Feet', title='Natural Gas Consumption')
plt.legend()
plt.show()

df2['CPITransportation'].plot(label= 'CPI Transportation', xlabel= 'Year', ylabel= 'Index 1982=100', title= 'CPI Transportation')
plt.legend()
plt.show()

df2['VehicleMilesTraveled'].plot(label='Vehicle Miles Traveled', xlabel= 'Year', ylabel= 'Millions of Miles', title='Vehicle Miles Traveled')
plt.legend()
plt.show()

# Fit the VAR model
model_var = VAR(df2)
fit_var= model_var.fit(2)
fit_var.summary()

fit_aic = model_var.fit(maxlags=15, ic='aic')

best_lag = fit_aic.k_ar

print(f"The best lag value chosen by AIC is: {best_lag}")

fit_aic.summary()

# Forecast for the next 24 periods (months)
fcast_var = fit_aic.forecast(df2.values[-best_lag:], 24)
fcast_var

# Plotting the fits and forecast
fit_var.plot()

fit_var.plot_forecast(24)

# Detailed plots with confidence intervals require obtaining the mid, lower and upper forecasts for the variables 
mid, lower, upper = fit_var.forecast_interval(df2.values[-best_lag:], 24, alpha=0.05)

mid
lower
upper 

# Creating dataframes with time index to house the mid, lower, upper values
start_forecast = datetime(year=2023,month=1,day=1)

horizon = pd.date_range(start=start_forecast, periods=24, freq='M')

fcast_mid = pd.DataFrame(fcast_var, index=horizon, columns=['NaturalGasConsumption', 'CPITransportation', 'VehicleMilesTraveled'])

fcast_lower = pd.DataFrame(fcast_var, index=horizon, columns=['NaturalGasConsumption', 'CPITransportation', 'VehicleMilesTraveled'])

fcast_upper = pd.DataFrame(fcast_var, index=horizon, columns=['NaturalGasConsumption', 'CPITransportation', 'VehicleMilesTraveled'])

# Now do a forecast plot with all the prediction intervals one at a time
plt.figure(figsize=(16,8))
plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=2))
plt.plot(df2.index, df2['NaturalGasConsumption'], label='Training', color='blue')
plt.plot(horizon, fcast_mid['NaturalGasConsumption'], label='Predicted', color='red')
plt.fill_between(horizon, fcast_lower.iloc[:,1], fcast_upper.iloc[:,1], alpha=0.3)
plt.title('Natural Gas Consumption Forecast')
plt.xlabel('Year')
plt.ylabel('Billion Cubic Feet')
plt.legend()
plt.show()

plt.figure(figsize=(16,8))
plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=2))
plt.plot(df2.index, df2['CPITransportation'], label= 'Training', color='blue')
plt.plot(horizon, fcast_mid['CPITransportation'], label='Predicted', color='red')
plt.fill_between(horizon, fcast_lower.iloc[:,1], fcast_upper.iloc[:,1], alpha=0.3)
plt.title('Consumer Price Index-Transportation Forecast')
plt.xlabel('Year')
plt.ylabel('Index')
plt.legend()
plt.show()

plt.figure(figsize=(16,8))
plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=2))
plt.plot(df2.index, df2['VehicleMilesTraveled'], label='Training', color='blue')
plt.plot(horizon, fcast_mid['VehicleMilesTraveled'], label='Predicted', color='red')
plt.fill_between(horizon, fcast_lower.iloc[:,1], fcast_upper.iloc[:,1], alpha=0.3)
plt.title('Vehicle Miles Traveled Forecast')
plt.xlabel('Year')
plt.ylabel('Millions of Miles')
plt.legend()
plt.show()