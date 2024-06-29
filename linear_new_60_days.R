library(quantmod)
library(dplyr)
library(lubridate)
library(ggplot2)
library(knitr)

# Define the ticker symbol for the S&P 500 Index
ticker_symbol <- "^GSPC"

# Define the start and end dates for the historical data
start_date <- "2013-05-13"
end_date <- "2024-03-31"

# Download S&P data
getSymbols(ticker_symbol, from = start_date, to = end_date)

# Convert the data to a data frame
data <- as.data.frame(`GSPC`) %>%
  mutate(Date = index(`GSPC`))

# Filter out the last 60 days of data
data_last_60_days <- tail(data, 60)

#Rename columns in dataframe
data <- data_last_60_days %>%
  rename(
    Open = GSPC.Open,
    High = GSPC.High,
    Low = GSPC.Low,
    Close = GSPC.Close,
    Volume = GSPC.Volume,
    AdjClose = GSPC.Adjusted
  )

# Visualize the data
ggplot(data, aes(x = Date, y = Close)) + 
  geom_line() +
  labs(title = 'S&P 500 Index', x = 'Date', y = 'Closing Price') +
  theme_minimal() +
  scale_x_date(date_labels = "%Y", date_breaks = "1 year")

# Calculate the correlation matrix among the independent variables
cor_matrix <- cor(data %>% select(Open, High, Low, Close, Volume, AdjClose))

# Modify the correlation matrix as a table
cor_matrix_table <- round(cor_matrix, 2)  # Round the correlation coefficients for better readability

# Use kable to create a formatted table
kable(cor_matrix_table, caption = "Correlation Matrix Among Independent Variables", digits = 2, align = 'c')

# Create a sample of indices for the training set
set.seed(123) 
train_indices <- sample(1:nrow(data), size = 0.7 * nrow(data))

# Split the data into training and testing sets
train <- data[train_indices, ]
test <- data[-train_indices, ]

# Fit the linear regression model
lm.model <- lm(Close ~  Open + High + Low, data = train)
summary(lm.model)
AIC <- AIC(lm.model)
BIC <- BIC(lm.model)
AIC
BIC

# Check for multicollinearity
vif_values <- vif(lm.model) 
print(vif_values)


# Make predictions on the test set if not already done
test$Predictions <- predict(lm.model, newdata = test)

# Compute the residuals for the test set
test$Residuals <- test$Close - test$Predictions

# Compute the evaluation metrics
RMSE <- sqrt(mean(test$Residuals^2))
MAE <- mean(abs(test$Residuals))
MAPE <- mean(abs(test$Residuals / test$Close)) * 100
R_squared <- summary(lm.model)$r.squared

# Create a data frame for the metrics
metrics <- data.frame(
  Metric = c("RMSE", "MAE", "MAPE", "R_squared"),
  Value = c(RMSE, MAE, MAPE, R_squared)
)

# Display the metrics using kable
kable(metrics, caption = "Model Performance Metrics")

# Visualization of Actual vs Predicted values
test <- mutate(test, Data_Set = "Test")
test_plot_data <- test %>%
  select(Date, Close, Predictions) %>%
  pivot_longer(cols = c(Close, Predictions), names_to = "Legend", values_to = "Value")

ggplot(test_plot_data, aes(x = Date, y = Value, color = Legend)) +
  geom_line() +
  labs(title = "Actual vs Predicted S&P 500 Index", x = "Date", y = "Index Value", color = "Legend") +
  scale_color_manual(values = c("Close" = "blue", "Predictions" = "red"),
                     labels = c("Close" = "Actual", "Predictions" = "Predicted")) +
  theme_minimal()

# Function to perform rolling forecast for different windows
rolling_forecast <- function(data, forecast_horizon) {
  forecasts <- vector("numeric", length = nrow(data) - forecast_horizon)
  actuals <- vector("numeric", length = nrow(data) - forecast_horizon)
  
  for (i in 1:(nrow(data) - forecast_horizon)) {
    train_set <- data[1:i, ]
    model <- lm(Close ~ Open + High + Low, data = train_set)
    test_set <- data[(i + 1):(i + forecast_horizon), ]
    prediction <- predict(model, newdata = test_set)
    forecasts[i] <- tail(prediction, 1)
    actuals[i] <- tail(test_set$Close, 1)
  }
  
  return(data.frame(Forecast = forecasts, Actual = actuals))
}

# Define the forecasting horizons
horizons <- c("3-Day" = 3, "7-Day" = 7, "14-Day" = 14)

# Calculate forecasts for each horizon
results <- lapply(horizons, function(h) rolling_forecast(data, h))

# Calculate evaluation metrics for each forecast horizon
evaluation_metrics <- lapply(results, function(result) {
  RMSE <- sqrt(mean((result$Actual - result$Forecast)^2))
  MAE <- mean(abs(result$Actual - result$Forecast))
  MAPE <- mean(abs((result$Actual - result$Forecast) / result$Actual)) * 100
  
  return(data.frame(RMSE = RMSE, MAE = MAE, MAPE = MAPE))
})

# Combine the metrics into one data frame
combined_metrics <- do.call(rbind, evaluation_metrics)
row.names(combined_metrics) <- names(horizons)

# Display the combined metrics using kable
kable(combined_metrics, caption = "Model Performance Metrics for Different Time Windows")
