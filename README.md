# Predicting S&P 500 Closing Prices

This repository contains code and analysis for predicting the closing price of the S&P 500 index using various machine learning models.

## Abstract

Predicting stock market trends, especially the closing price, presents challenges and potential rewards [1]. Despite the Efficient Market Hypothesis (EMH) suggesting market efficiency and unpredictability, research indicates that accurate prediction is feasible with suitable models and variables [2]. This project aims to forecast the S&P 500 closing price by leveraging historical data such as opening, closing, high, and low prices. Python and R programming languages, alongside data analysis libraries, were employed for systematic data mining, visualization, statistical analysis, and model selection.

## Overview

The project evaluates Autoregressive Integrated Moving Average (ARIMA), Long Short-Term Memory (LSTM), and linear regression models over three years and 60 days of S&P 500 data, with forecast windows of 3, 7, and 14 days. The analysis found that using a narrow forecast window of 60 days provided superior predictions for the closing price. Among the models assessed, LSTM demonstrated the highest effectiveness, exhibiting lower Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and Root Mean Squared Error (RMSE) compared to ARIMA and linear regression.

## Conclusion

The study highlights the predictive capability of utilizing shorter forecast windows tailored to the dynamic nature of stock market data. This adaptation enhances model responsiveness to evolving trends, thereby improving forecast accuracy. The choice of dataset granularity—whether daily, weekly, or longer—should align with the trader or investor’s timeframe for optimal predictive insights.

Future research could explore international market correlations with the S&P 500 and incorporate economic indicators like inflation, GDP, and geopolitical events for deeper market analysis. Such enhancements promise to refine forecasting accuracy and support informed investment decisions.

## Files

- `ARIMA_EDA_Code.ipynb`: Exploratory Data Analysis and ARIMA modeling notebook.
- `LSTM_3_years.R` and `LSTM_60days.R`: LSTM model scripts for three years and 60 days forecast windows.
- `linear_new_three_years.R` and `linear_new_60_days.R`: Linear regression model scripts.
- `Report.pdf`: Detailed report on methodologies, findings, and conclusions.
- `Codes.zip`: Additional codes and scripts used in the analysis.
- `LICENSE`: License information for the repository.

For further details, refer to the detailed report (`Report.pdf`) and explore individual model scripts and notebooks provided.

