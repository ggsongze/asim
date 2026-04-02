TRAINING_BEST_RESULTS = {}

if forecast_results is not None:
    TRAINING_BEST_RESULTS["forecast_window"] = forecast_results.get_best_result()

if no_forecast_results is not None:
    TRAINING_BEST_RESULTS["no_forecast_window"] = no_forecast_results.get_best_result()

TRAINING_BEST_RESULTS
