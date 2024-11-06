### Name:Karnan K
### Reg No:212222230062
### Date: 
# EX.NO.09        A project on Time series analysis on project for forecasting using ARIMA model 

### AIM:
To Create a project on Time series analysis on project for forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of Baggage complaints
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima

df = pd.read_csv('/content/baggagecomplaints.csv', parse_dates=['Date'], index_col='Date')

date_range = pd.date_range(start='2020-01-01', periods=100, freq='M')
complaints = np.random.poisson(lam=20, size=100)  # Simulated complaint data
df = pd.DataFrame({'Date': date_range, 'Baggage': complaints}).set_index('Date')

# Plot the time series
plt.figure(figsize=(10, 6))
plt.plot(df, label="Baggage Complaints")
plt.title("Airline Baggage Complaints Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Complaints")
plt.legend()
plt.show()

# Check for stationarity using ADF test
result = adfuller(df['Baggage'])
print("ADF Statistic:", result[0])
print("p-value:", result[1])

#Differencing 
if result[1] > 0.05: 
    df_diff = df.diff().dropna()
else:
    df_diff = df

# Plot ACF and PACF to determine initial p, q values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_acf(df_diff, ax=plt.gca(), lags=20)
plt.title("ACF Plot")
plt.subplot(1, 2, 2)
plot_pacf(df_diff, ax=plt.gca(), lags=20)
plt.title("PACF Plot")
plt.tight_layout()
plt.show()

auto_model = auto_arima(df_diff, seasonal=False, trace=True)
p, d, q = auto_model.order
print(f"Selected ARIMA order: p={p}, d={d}, q={q}")

model = ARIMA(df, order=(p, d, q))
model_fit = model.fit()
print(model_fit.summary())
forecast = model_fit.get_forecast(steps=12)
forecast_index = pd.date_range(df.index[-1] + pd.DateOffset(1), periods=12, freq='M')
forecast_series = pd.Series(forecast.predicted_mean.values, index=forecast_index)

# Plot the actual and forecasted values
plt.figure(figsize=(10, 6))
plt.plot(df, label="Actual Complaints")
plt.plot(forecast_series, label="Forecasted Complaints", color="orange")
plt.title("Airline Baggage Complaints Forecast")
plt.xlabel("Date")
plt.ylabel("Number of Complaints")
plt.legend()
plt.show()

y_train = df.iloc[:-20]
y_test = df.iloc[-20:]
model_fit = ARIMA(y_train, order=(p, d, q)).fit()
y_pred = model_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test, label="Actual Complaints")
plt.plot(y_pred, label="Predicted Complaints", color="red")
plt.title("Model Evaluation on Test Data")
plt.xlabel("Date")
plt.ylabel("Number of Complaints")
plt.legend()
plt.show()
```

### OUTPUT:

![download](https://github.com/user-attachments/assets/8a5b7ae3-07c3-4ed9-8c2e-60a943c5f6a7)

## ACF AND PACF

![download](https://github.com/user-attachments/assets/f71f2dc8-9ee7-47fc-b2ff-e876b709eb5a)

## actual and forecasted values

![download](https://github.com/user-attachments/assets/4259d927-ddc2-485e-8dba-5a4951db3fc3)

## actual vs predicted

![download](https://github.com/user-attachments/assets/d03e4fd3-e2a8-4bf4-a602-879bc5468eab)

### RESULT:
Thus the program run successfully based on the ARIMA model using python.
