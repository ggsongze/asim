RUN_FORECAST_TRAINING = True
RUN_NO_FORECAST_TRAINING = True


forecast_results = FORECAST_TUNER.fit() if RUN_FORECAST_TRAINING else None
no_forecast_results = NO_FORECAST_TUNER.fit() if RUN_NO_FORECAST_TRAINING else None


TRAINING_RESULTS = {
    "forecast_window": forecast_results,
    "no_forecast_window": no_forecast_results,
}
TRAINING_RESULTS
