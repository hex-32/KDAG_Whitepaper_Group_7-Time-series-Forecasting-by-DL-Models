# Time Series Forecasting: A Comparative Study of Statistical, Recurrent, and Modern Deep Learning Architectures

In this project we attempt to compare classical statistical models (ARIMA/SARIMAX) against recurrent deep learning models (LSTM) across two data regimes — **stationary/cyclical** and **non-stationary/contextual** — to understand *when and why* each paradigm succeeds or fails, rather than simply ranking models by accuracy.

>[The Whitepaper](whitepaper.pdf)

---

## Motivation
Traditional statistical models such as ARIMA and SARIMA have long been used for forecasting but often struggle with the nonlinear, high-dimensional, and large-scale nature of modern time series data. Recent advances in deep learning have enabled models to learn complex temporal patterns directly from data, leading to significant improvements in forecasting performance across a wide range of applications.

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

## Datasets
The datasets used in this project are:
1. [Bike Sharing Dataset](Bike_sharing_dataset.csv)
2. [Champagne Sales Dataset](perrin-freres-monthly-champagne.csv)

## Installation
Clone the repository:
```bash
git clone https://github.com/hex-32/KDAG_Whitepaper_Group_7-Time-series-Forecasting-by-DL-Models.git
```
To run the code, you will need to install the following dependencies:
```bash
pip install -r requirements.txt
```
## Run the Pipelines
Then run the main script:
```bash
python ARIMAX_bikesharing_pipeline.py # Run ARIMAX pipeline
python LSTM_bikesharing_pipeline.py   # Run LSTM pipeline
python SARIMA_sales_pipeline.py      # Run SARIMA pipeline
```
