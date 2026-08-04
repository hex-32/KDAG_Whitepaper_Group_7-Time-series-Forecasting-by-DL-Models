# 3. Evolution of Time Series Forecasting

## 3.1 Statistical Methods

Time series forecasting began with statistical models that describe relationships between historical observations using mathematical formulations. Methods such as **Autoregressive (AR)**, **Moving Average (MA)**, **Autoregressive Moving Average (ARMA)**, **Autoregressive Integrated Moving Average (ARIMA)**, and **Seasonal ARIMA (SARIMA)** became the foundation of forecasting for several decades. These models perform well on linear and stationary time series but require manual parameter selection and often struggle with nonlinear or high-dimensional data. More recently, **Facebook Prophet** simplified forecasting by explicitly modeling trend, seasonality, and holiday effects while requiring minimal parameter tuning.

## 3.2 Machine Learning Methods

As larger datasets became available, machine learning methods were introduced to capture nonlinear relationships beyond the capabilities of statistical models. Algorithms such as **Linear Regression**, **Support Vector Regression (SVR)**, **Decision Trees**, **Random Forests**, **XGBoost**, and **LightGBM** improved forecasting performance by learning complex feature interactions. However, these approaches rely heavily on feature engineering and do not inherently model long-term temporal dependencies.

## 3.3 Emergence of Deep Learning

The limitations of traditional methods led to the adoption of deep learning for sequential data. **Recurrent Neural Networks (RNNs)** introduced the ability to learn temporal dependencies directly from historical observations, while **Long Short-Term Memory (LSTM)** and **Gated Recurrent Units (GRUs)** addressed the shortcomings of conventional RNNs in modeling long-range dependencies. Subsequent architectures, including **Temporal Convolutional Networks (TCNs)** and **Transformer-based models**, further improved scalability, parallelization, and long-horizon forecasting performance.

Recent advances have introduced specialized architectures such as **Informer**, **Autoformer**, **FEDformer**, **PatchTST**, **Temporal Fusion Transformer (TFT)**, and foundation models including **Chronos**, **TimeGPT**, and **Moirai**, which leverage large-scale pretraining to generalize across diverse forecasting tasks.

## Key Takeaways

* Statistical models established the foundation of time series forecasting but are limited by assumptions of linearity and stationarity.
* Machine learning methods improved nonlinear modeling but depend on engineered features.
* Deep learning has become the dominant paradigm by learning temporal representations directly from data, with Transformer-based and foundation models representing the current state of the art.

[]