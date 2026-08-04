import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
'''
Applying sARIMA model on a seasonal datatset
'''
# 1. Load and Preprocess the Dataset
df = pd.read_csv('perrin-freres-monthly-champagne.csv')

# Clean up column names for easier access
df.columns = ['Month', 'Sales']

# Drop the trailing garbage rows containing NaN values
df = df.dropna()

# Convert the 'Month' column to datetime objects
df['Month'] = pd.to_datetime(df['Month'])

# Set the datetime column as the index
df.set_index('Month', inplace=True)

# Ensure the frequency is set to Monthly Start ('MS') for the model
df = df.asfreq('MS')

# 2. Train / Test Split
test_horizon = 24
train = df.iloc[:-test_horizon]
test = df.iloc[-test_horizon:]

# 3. Fit SARIMA Model
# order=(p, d, q) for non-seasonal components
# seasonal_order=(P, D, Q, s) for seasonal components
# s=12 defines the annual (12-month) seasonality
model_order = (1, 1, 1)
seasonal_order = (0, 1, 1, 12)

sarima_model = SARIMAX(
    train['Sales'],
    order=model_order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

print("Fitting the SARIMA model...")
results = sarima_model.fit(disp=False)
print(results.summary())

# 4. Out-of-Sample Forecasting
# Forecast the exact number of steps as our test set
forecast_object = results.get_forecast(steps=test_horizon)
forecast = forecast_object.predicted_mean
conf_int = forecast_object.conf_int()

# Align the index of the forecast with the test set for comparison
forecast.index = test.index
conf_int.index = test.index

# 5. Evaluate Metrics (MAPE & sMAPE)
def compute_smape(y_true, y_pred):
    return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100

def compute_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_score = compute_mape(test['Sales'].values, forecast.values)
smape_score = compute_smape(test['Sales'].values, forecast.values)

print("       EVALUATION METRICS       ")
print(f"MAPE  : {mape_score:.2f}%")
print(f"sMAPE : {smape_score:.2f}%")

# 6. Visualization
plt.figure(figsize=(12, 6))

plt.plot(train.index, train['Sales'], label='Training Data', color='gray')
plt.plot(test.index, test['Sales'], label='Actual Test Data', color='blue', linewidth=2)
plt.plot(test.index, forecast, label='SARIMA Forecast', color='red', linestyle='--', linewidth=2)

plt.fill_between(
    test.index,
    conf_int.iloc[:, 0],
    conf_int.iloc[:, 1],
    color='red',
    alpha=0.15,
    label='95% Confidence Interval'
)

plt.title('SARIMA Forecast: Perrin Freres Monthly Champagne Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()