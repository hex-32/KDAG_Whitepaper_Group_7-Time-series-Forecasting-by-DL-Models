# 2. Fundamentals of Time Series Forecasting

## 2.1 What is a Time Series?

A **time series** is a sequence of observations recorded at successive points in time, typically at regular intervals. Unlike conventional datasets where observations are assumed to be independent, time series data exhibits temporal dependence, meaning that current observations are often influenced by past values. Examples include daily stock prices, hourly electricity demand, monthly sales, and annual rainfall measurements.

## 2.2 Components of a Time Series

A time series is generally composed of four primary components:

* **Trend:** The long-term upward or downward movement in the data.
* **Seasonality:** Repeating patterns occurring at fixed intervals, such as daily, monthly, or yearly cycles(as handled in **SARIMA**).
* **Cyclic Variation:** Long-term fluctuations with irregular durations, often influenced by economic or environmental factors.
* **Noise (Residual):** Random variations that cannot be explained by the underlying structure.

These components are essential for selecting an appropriate forecasting model.

## 2.3 Types of Forecasting Tasks

Time series forecasting problems can be categorized based on the prediction objective:

* **Univariate Forecasting:** Predicting future values using only the historical values of a single variable(Eg: **ARIMA** & **SARIMA**).
* **Multivariate Forecasting:** Incorporating multiple related variables to improve prediction accuracy(Eg: **Vector Autoregression (VAR)**).
* **Single-Step Forecasting:** Predicting only the next time step(Eg: **Normal(Recurrent) LSTM**).
* **Multi-Step Forecasting:** Predicting multiple future time steps simultaneously or recursively(Eg: **Recursive LSTM**).

## 2.4 Forecasting Horizons

The prediction horizon defines how far into the future a model forecasts.

* **Short-Term:** Minutes to days.
* **Medium-Term:** Weeks to months.
* **Long-Term:** Several months to years.

As the forecasting horizon increases, uncertainty and prediction complexity generally increase.

## 2.5 Data Preprocessing

Effective forecasting depends heavily on data quality. Common preprocessing steps include:

* Handling missing values and outliers.
* Normalization or standardization of features.
* Feature engineering, such as lag variables and rolling statistics.
* Train-validation-test splitting while preserving chronological order.

## 2.6 Evaluation Metrics

Forecasting models are evaluated using quantitative error metrics, including:

* **Mean Absolute Error (MAE)**
* **Mean Squared Error (MSE)**
* **Root Mean Squared Error (RMSE)**
* **Mean Absolute Percentage Error (MAPE)**
* **Symmetric Mean Absolute Percentage Error (sMAPE)**
* **Mean Absolute Scaled Error (MASE)**

The choice of metric depends on the application and the characteristics of the dataset.

## Key Points

* Time series data differs from conventional datasets due to its temporal dependency.
* Trend, seasonality, cyclic behavior, and noise are the primary components of a time series.
* Forecasting tasks may be univariate or multivariate, and single-step or multi-step.
* Appropriate preprocessing and evaluation are crucial for developing reliable forecasting models.

[Goto Evolution of Time Series Forecasting](./Evolution.md)