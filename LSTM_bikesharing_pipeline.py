import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

'''
Here we are applying neural netwroks on comparitively more contextual 
and non-stationary dataset
'''

# 1. Load and Preprocess the Dataset
print("Loading data...")
df = pd.read_csv('Bike_sharing_dataset.csv', parse_dates={'datetime': ['dteday', 'hr']}, index_col='datetime')
df = df.sort_index()

# Define features (exogenous) and target (endogenous)
# Notice we include 'cnt' in the features so the model can learn from past rentals
features = ['holiday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt']
target_col_index = features.index('cnt')

data = df[features].astype('float64')
# 2. Scale the Data, applied minmax scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. Build 3D Tensors [samples, time_steps, features]
def create_sequences(dataset, lookback_window):
    """
    Slides a window over the dataset to create 3D tensors.
    """
    X, y = [], []
    # Loop stops early enough to ensure we have a target for the last window
    for i in range(len(dataset) - lookback_window):
        # Extract the window of data (all features)
        X.append(dataset[i : (i + lookback_window), :])
        # The target is the 'cnt' value immediately following the window
        y.append(dataset[i + lookback_window, target_col_index])
    return np.array(X), np.array(y)

# We will use the past 24 hours to predict the next hour
lookback = 24 
X, y = create_sequences(scaled_data, lookback)

print(f"3D Tensor Shape (X): {X.shape}") 
print(f"Target Shape (y): {y.shape}")

# 4. Train / Test Split
# Hold out the exact same 168 hours (7 days) for testing as the ARIMAX model, we have choosen the same test set for 
#comparison between both models
test_horizon = 168
X_train, X_test = X[:-test_horizon], X[-test_horizon:]
y_train, y_test = y[:-test_horizon], y[-test_horizon:]

# Extract the dates corresponding to the test set for plotting
test_dates = data.index[-test_horizon:]

# 5. Build and Compile the LSTM
print("\nBuilding LSTM Model...")
model = Sequential()

# The input_shape requires only (time_steps, features)
model.add(LSTM(units=64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2)) # Prevent overfitting, randomly dropes 20% of weights
model.add(Dense(units=1)) # Output a single prediction (cnt)

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# 6. Train the Model
# Unlike ARIMAX, we can comfortably train on the entire 17,000+ row history
print("\nCommencing Training...")
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)


# 7. Out-of-Sample Forecasting & Inverse Scaling
print("\nGenerating Forecast...")
predictions_scaled = model.predict(X_test)

# matching the original number of features (8 columns)
dummy_pred = np.zeros((len(predictions_scaled), len(features)))
dummy_pred[:, target_col_index] = predictions_scaled.flatten()
forecast = scaler.inverse_transform(dummy_pred)[:, target_col_index]

dummy_actual = np.zeros((len(y_test), len(features)))
dummy_actual[:, target_col_index] = y_test
actuals = scaler.inverse_transform(dummy_actual)[:, target_col_index]

# 8. Evaluate Metrics (MAE &  RMSE)
def compute_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

mae_score = compute_mae(actuals, forecast)
rmse_score = compute_rmse(actuals, forecast)

print(f"MAE  : {mae_score:.2f} bikes")
print(f"RMSE : {rmse_score:.2f} bikes")

# 9. Visualization
plt.figure(figsize=(14, 6))

plt.plot(test_dates, actuals, label='Actual Test Data', color='blue', linewidth=2)
plt.plot(test_dates, forecast, label='LSTM Forecast', color='green', linestyle='--', linewidth=2)

plt.title('LSTM Forecast: Hourly Bike Sharing Demand')
plt.xlabel('Date & Time')
plt.ylabel('Total Rentals (cnt)')
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()