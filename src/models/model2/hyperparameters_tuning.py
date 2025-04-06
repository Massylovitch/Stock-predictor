import warnings

warnings.filterwarnings("ignore")

import numpy as np
from xgboost import XGBRegressor
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline

from src.data.preprocessing import get_preprocessing_pipeline
from src.utils.logger import get_console_logger

logger = get_console_logger()


def XGB_objective(trial, train, target, STATIC_PARAMS, OPTUNA_FOLDS):

    xgb_params = {
        "max_depth": trial.suggest_int("max_depth", 1, 10, 2),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 1.0, step=0.005),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000, step=100),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.01, 1.0),
        "subsample": trial.suggest_float("subsample", 0.01, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.01, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1.0),
    }

    xgb_params.update(STATIC_PARAMS)

    kf = TimeSeriesSplit(n_splits=OPTUNA_FOLDS)
    scores = []

    logger.info(f"{trial.number=}")
    for split_number, (train_ind, val_ind) in enumerate(kf.split(train, target)):

        X_train, X_val = train.iloc[train_ind], train.iloc[val_ind]
        y_train, y_val = target[train_ind], target[val_ind]

        logger.info(f"{split_number=}")
        logger.info(f"{len(X_train)=}")
        logger.info(f"{len(X_val)=}")

        # train the model
        pipeline = make_pipeline(
            get_preprocessing_pipeline(), XGBRegressor(**xgb_params)
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        scores.append(mae)

        # logger.info(f'{mae=}')

    score = np.array(scores).mean()
    return score


def run_optuna(train, target):

    OPTUNA_FOLDS = 5
    SEED = 13
    OPTUNA_TRIALS = 15
    VERBOSITY = 0

    STATIC_PARAMS = {
        "seed": SEED,
        "eval_metric": "auc",
        "objective": "reg:squarederror",
        "verbosity": VERBOSITY,
    }

    logger.info("Starting hyper-parameter search...")

    func = lambda trial: XGB_objective(
        trial,
        train,
        target,
        STATIC_PARAMS,
        OPTUNA_FOLDS,
    )

    study = optuna.create_study(direction="minimize")
    study.optimize(func, n_trials=OPTUNA_TRIALS)

    logger.info(f"Study Best Value: {study.best_value}")
    logger.info(f"Study Best Params: {study.best_params}")

    return study.best_params, study.best_value
