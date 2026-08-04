import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 1. Load and Preprocess the Dataset
df = pd.read_csv('Bike_sharing_dataset.csv')

# Combine 'dteday' and 'hr' into a single Datetime column
df['datetime'] = pd.to_datetime(df['dteday'] + ' ' + df['hr'].astype(str).str.zfill(2) + ':00:00')

# Set datetime as the index and sort chronologically
df = df.set_index('datetime').sort_index()

# Define Target (Endogenous) and Features (Exogenous)
target_col = 'cnt'
exog_cols = ['holiday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']

# For classical models, processing 17,000 hourly rows is computationally heavy.
# We slice the last 30 days (720 hours) for this pipeline demonstration.
df_subset = df.iloc[-720:]

y = df_subset[target_col].astype('float64')
X = df_subset[exog_cols].astype('float64')

# 2. Train / Test Split
# Hold out the last 7 days (168 hours) for testing, the same testset is used for deep learning models(LSTMs)
test_horizon = 168
y_train, y_test = y.iloc[:-test_horizon], y.iloc[-test_horizon:]
X_train, X_test = X.iloc[:-test_horizon], X.iloc[-test_horizon:]

# 3. Fit ARIMAX Model
# Using SARIMAX but setting seasonal_order=(0,0,0,0) makes it an ARIMAX model.
# order=(p, d, q) captures the autoregressive and moving average properties.
model_order = (2, 1, 2) 

print("Fitting the ARIMAX model. This may take a moment depending on the lag order...")
arimax_model = SARIMAX(
    endog=y_train,
    exog=X_train,
    order=model_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = arimax_model.fit(disp=False)
print(results.summary())

# 4. Out-of-Sample Forecasting
# To forecast ARIMAX, we MUST provide the exogenous variables for the test period
forecast_object = results.get_forecast(steps=test_horizon, exog=X_test)
forecast = forecast_object.predicted_mean
conf_int = forecast_object.conf_int()

forecast.index = y_test.index
conf_int.index = y_test.index

# 5. Evaluate Metrics (MAE & RMSE)
def compute_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

mae_score = compute_mae(actuals, forecast)
rmse_score = compute_rmse(actuals, forecast)

print(f"MAE  : {mae_score:.2f} bikes")
print(f"RMSE : {rmse_score:.2f} bikes")

# 6. Visualization
plt.figure(figsize=(14, 6))

# Plot the last 7 days of the training set for visual context
plot_train_subset = y_train.iloc[-168:] 

plt.plot(plot_train_subset.index, plot_train_subset, label='Train (Last 7 Days)', color='gray')
plt.plot(y_test.index, y_test, label='Actual Test Data', color='blue', linewidth=2)
plt.plot(y_test.index, forecast, label='ARIMAX Forecast', color='red', linestyle='--', linewidth=2)

plt.fill_between(
    y_test.index,
    conf_int.iloc[:, 0],
    conf_int.iloc[:, 1],
    color='red',
    alpha=0.15,
    label='95% Confidence Interval'
)

plt.title('ARIMAX Forecast: Hourly Bike Sharing Demand')
plt.xlabel('Date & Time')
plt.ylabel('Total Rentals (cnt)')
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()