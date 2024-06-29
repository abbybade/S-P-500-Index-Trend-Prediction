library(keras)
library(tidyverse)
library(tensorflow)
library(lubridate)
library(rsample)
library(knitr)
library(readr)
library(dplyr)

# Load and preprocess the dataset
stocks <- read_csv("sp500_index.csv") %>%
  rename(Index = `S&P500`) %>% 
  mutate(Date = ymd(Date)) %>%
  filter((Date >= as.Date("2024-01-10") & Date <= as.Date("2024-04-05")) | 
           (Date >= Sys.Date() - days(60)))#Filter last 60 days

# Normalize data function to scale the 'S&P500' index to 0 and 1
normalize_data <- function(x) {
  min_val <- min(x, na.rm = TRUE)
  max_val <- max(x, na.rm = TRUE)
  if (min_val == max_val) {
    return(rep(0, length(x)))
  } else {
    return ((x - min_val) / (max_val - min_val))
  }
}

# Apply normalization function to data
stocks_clean <- stocks %>%
  mutate(Index = normalize_data(Index))

# Data splitting
set.seed(1994)  # Set seed for reproducibility
split_index <- floor(0.7 * nrow(stocks_clean))  # 70% for training
train_data_index <- stocks_clean[1:split_index, ]
test_data_index <- stocks_clean[(split_index + 1):nrow(stocks_clean), ]
features <- c("Index")

# Prepare LSTM data function
prepare_lstm_data <- function(data, features) {
  x <- as.matrix(data[ , features, drop = FALSE])
  y <- data$Index
  return(list(x = array_reshape(x, c(nrow(data), 1, length(features))), y = y))
}

train_data <- prepare_lstm_data(train_data_index, features)
test_data <- prepare_lstm_data(test_data_index, features)


# Model setup and training
model <- keras_model_sequential() %>%
  layer_lstm(units = 128, return_sequences = TRUE, input_shape = c(1, length(features))) %>%
  layer_lstm(units = 64)%>%
  layer_dense(units = 1)


# Compile and train model
model %>% compile(loss = 'mean_squared_error', optimizer = optimizer_adam(learning_rate = 0.01))
history <- model %>% fit(train_data$x, train_data$y, epochs = 100, batch_size = 235,  validation_split = 0.1,  shuffle = FALSE)
summary(model)

# Predict and evaluate
predicted_values <- model %>% predict(test_data$x)
actual_values <- test_data$y

# Evaluation Metrics
RMSE <- sqrt(mean((predicted_values - actual_values)^2))
MAE <- mean(abs(predicted_values - actual_values))
MAPE <- mean(abs((predicted_values - actual_values) / actual_values)) * 100

# Create a data frame for the metrics
metrics <- data.frame(
  Metric = c("MAPE", "RMSE", "MAE"),
  Value = c(MAPE, RMSE, MAE)
)

# Display the metrics using a table
kable(metrics, caption = "Model Performance Metrics")

# Visualize Predictions
predictions_with_dates <- data.frame(
  Date = test_data_index$Date,
  Actual = actual_values,
  Predicted = as.vector(predicted_values)
)

ggplot(predictions_with_dates, aes(x = Date)) +
  geom_line(aes(y = Actual, colour = "Actual")) +
  geom_line(aes(y = Predicted, colour = "Predicted")) +
  labs(
    title = "Actual vs Predicted S&P 500 Index",
    x = "Date",
    y = "Index",  # Sets the y-axis label to "Index"
    colour = "Legend"  # Correct legend title
  ) +
  scale_colour_manual(
    values = c("Actual" = "blue", "Predicted" = "red"),  # Define custom colors for the lines
    labels = c("Actual", "Predicted")  # Properly label the legend for clarity
  ) +
  theme_minimal()



# Function to calculate forecast error metrics for different forecasting windows
calculate_forecast_errors <- function(actual, predicted, forecast_window) {
  error <- actual[forecast_window:length(actual)] - predicted[1:(length(actual)-forecast_window+1)]
  return(data.frame(
    Forecast_Window = forecast_window,
    MAE = mean(abs(error)),
    MAPE = mean(abs(error / actual[forecast_window:length(actual)])),
    RMSE = sqrt(mean(error^2))
  ))
}

# Generate forecast error metrics for various windows
forecast_errors <- rbind(
  calculate_forecast_errors(actual_values, predicted_values, forecast_window = 3),   
  calculate_forecast_errors(actual_values, predicted_values, forecast_window = 7),  
  calculate_forecast_errors(actual_values, predicted_values, forecast_window = 14)  
)

# Display the forecast errors in a table format
kable(forecast_errors, caption = "Forecast Error Metrics for Different Time Windows")

