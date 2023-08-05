from typing import Dict, Union, Optional, Callable
import os

import pandas as pd
from comet_ml import Experiment
from sklearn.metrics import mean_absolute_error

from src.data.preprocessing import build_features
from src.utils.logger import get_console_logger

logger = get_console_logger()

def get_baseline_model_error(X_test, y_test):
    
    predictions = X_test["price_1_hour_ago"]
    mae = mean_absolute_error(y_test, predictions)

    return mae

def train(X_train, y_train):
    
    experiment = Experiment(
        api_key = os.environ["COMET_ML_API_KEY"],
        workspace = os.environ["COMET_ML_WORKSPACE"],
        project_name = "stock-predictor",
    )

    experiment.add_tag("baseline_model")

    train_sample_size = int(0.9 * len(X_train))
    X_train, X_test = X_train[:train_sample_size], X_train[train_sample_size:]
    y_train, y_test = y_train[:train_sample_size], y_train[train_sample_size:]
    logger.info(f'Train sample size: {len(X_train)}')
    logger.info(f'Test sample size: {len(X_test)}')

    baseline_mae = get_baseline_model_error(X_test, y_test)
    logger.info(f'Test MAE: {baseline_mae}')
    experiment.log_metrics({'Test_MAE': baseline_mae})


if __name__ == '__main__':

    logger.info('Generating features and targets')
    features, target = build_features()
    
    logger.info('Starting training')
    train(features, target)