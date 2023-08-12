from typing import Dict, Union, Optional, Callable
import os

import pandas as pd
from comet_ml import Experiment
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Lasso
import pickle

from src.data.preprocessing import (
    build_features,
    get_preprocessing_pipeline
)
from src.utils.logger import get_console_logger
from src.utils.paths import MODELS_DIR


logger = get_console_logger()


def get_baseline_model_error(X_test, y_test):

    predictions = X_test["price_1_hour_ago"]
    mae = mean_absolute_error(y_test, predictions)

    return mae

def train(X, y):

    experiment = Experiment(
        api_key = os.environ["COMET_ML_API_KEY"],
        workspace=os.environ["COMET_ML_WORKSPACE"],
        project_name = "stock-predictor",
    )

    experiment.add_tag("Lasso")

    train_sample_size = int(0.9 * len(X))
    X_train, X_test = X[:train_sample_size], X[train_sample_size:]
    y_train, y_test = y[:train_sample_size], y[train_sample_size:]
    logger.info(f'Train sample size: {len(X_train)}')
    logger.info(f'Test sample size: {len(X_test)}')

    pipeline = make_pipeline(
            get_preprocessing_pipeline(),
            Lasso()
        )

    # train the model
    logger.info('Fitting model with default hyperparameters')
    pipeline.fit(X_train, y_train)

    # compute test MAE
    predictions = pipeline.predict(X_test)
    test_error = mean_absolute_error(y_test, predictions)
    logger.info(f'Test MAE: {test_error}')
    experiment.log_metrics({'Test_MAE': test_error})

    # save the model to disk
    logger.info('Saving model to disk')
    with open(MODELS_DIR / 'model.pkl', "wb") as f:
        pickle.dump(pipeline, f)

    experiment.log_model("Lasso", str(MODELS_DIR / 'model.pkl'))
    experiment.register_model("Lasso")

if __name__ == '__main__':

    logger.info('Generating features and targets')
    features, target = build_features()

    features = features.head()
    target = target.head()
        
    logger.info('Training model')
    train(features, target)