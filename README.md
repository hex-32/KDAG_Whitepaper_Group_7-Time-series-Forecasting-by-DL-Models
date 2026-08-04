# Time Series Forecasting: A Comparative Study of Statistical, Recurrent, and Modern Deep Learning Architectures

In this project we attempt to compare classical statistical models (ARIMA/SARIMAX) against recurrent deep learning models (LSTM) across two data regimes — **stationary/cyclical** and **non-stationary/contextual** — to understand *when and why* each paradigm succeeds or fails, rather than simply ranking models by accuracy.

---

## Objective
Our main objectives were:
  1. On exactly what type of dataset do classical models fail or show relatively poor performance?
  2. How do deep learning models understand non-stationary and highly contextual datasets?
  3. Does using both paradigms together, via a hybrid model, help?

## Experimentation
We first implemented a SARIMA model on a univariate, linear dataset. As expected, the forecasts were accurate, since the model's core assumption of linearity held for this data.

We then tested a more complex dataset exhibiting a non-linear relationship between historical and current values. As expected, the classical model was unable to fully capture this relationship.

We then applied deep learning models, specifically LSTMs. Since an LSTM is capable of learning long-term context and non-linear dependencies, we observed significantly improved scores.
## Datasets
For the comparison we used two datasets
  1. **Champange sales** : A relatively less complex, linear and stationary dataset
  2. **UCI Bike Sharing (Hourly)** : More contextual, non-linear multivariate dataset

## Pipelines

  1. ARIMAX pipeline
  2. LSTM pipeline
  3. SRIMA pipeline

## Evaluation Metrics

| Metric | Purpose |
|---|---|
| **MAE** (Mean Absolute Error) | Scale-dependent average error magnitude |
| **RMSE** (Root Mean Squared Error) | Penalizes larger errors more heavily |
| **MAPE** (Mean Absolute Percentage Error) | Scale-independent; noted to distort heavily near zero-valued actuals (see findings) |
| **sMAPE** (Symmetric MAPE) | More robust to near-zero actuals than plain MAPE, though not immune |


## Hybrid Modeling

Given that each paradigm fails in a distinct, mechanistically explainable way (ARIMA: seasonal/computational scalability; LSTM: interpretability, data/compute requirements), a natural extension is a **residual hybrid** approach:

1. Fit ARIMA/SARIMAX to capture linear trend and seasonal structure
2. Train an LSTM on the *residuals* to capture remaining nonlinear structure
3. Combine both components for the final forecast

