from typing import List, Tuple, Optional, Union
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer
import fire
from src.utils.paths import DATA_DIR
from src.utils.logger import get_console_logger

logger = get_console_logger()

def build_features(
     path_to_input: Optional[Path] = DATA_DIR / 'external' /'ohlc_data.parquet',
    input_seq_len: Optional[int] = 24,
    step_size: Optional[int] = 1
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Slices and transposes data from time-series format into a (features, target)
    format that we can use to train Supervised ML models
    """
    # load parquet file
    ts_data = pd.read_parquet(path_to_input)
    ts_data = ts_data[['time', 'close']]
    ts_data.sort_values(by=['time'], inplace=True)

    # output features and targets
    features = pd.DataFrame()
    targets = pd.DataFrame()
    
    # pre-compute cutoff indices to split dataframe rows
    indices = get_cutoff_indices_features_and_target(
        ts_data,
        input_seq_len,
        step_size
    )

    # slice and transpose data into numpy arrays for features and targets
    n_examples = len(indices)
    x = np.ndarray(shape=(n_examples, input_seq_len), dtype=np.float32)
    y = np.ndarray(shape=(n_examples), dtype=np.float32)
    times = []
    for i, idx in enumerate(indices):
        x[i, :] = ts_data.iloc[idx[0]:idx[1]]['close'].values
        y[i] = ts_data.iloc[idx[1]:idx[2]]['close'].values
        times.append(ts_data.iloc[idx[1]]['time'])

    # numpy -> pandas
    features = pd.DataFrame(
        x,
        columns=[f'price_{i+1}_hour_ago' for i in reversed(range(input_seq_len))]
    )

    # add back column with the time
    # features['time'] = times

    # numpy -> pandas
    targets = pd.DataFrame(y, columns=[f'target_price_next_hour'])

    return features, targets['target_price_next_hour']

def get_cutoff_indices_features_and_target(
    data: pd.DataFrame,
    input_seq_len: int,
    step_size: int
    ) -> List[Tuple[int, int, int]]:

    stop_position = len(data) - 1
        
    # Start the first sub-sequence at index position 0
    subseq_first_idx = 0
    subseq_mid_idx = input_seq_len
    subseq_last_idx = input_seq_len + 1
    indices = []
    
    while subseq_last_idx <= stop_position:
        indices.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))
        subseq_first_idx += step_size
        subseq_mid_idx += step_size
        subseq_last_idx += step_size

    return indices

def get_subset_of_features(X: pd.DataFrame) -> pd.DataFrame:
    return X[['price_1_hour_ago', 'percentage_return_2_hour', 'percentage_return_24_hour']]

def get_price_percentage_return(X: pd.DataFrame, hours: int) -> pd.DataFrame:
    """Add the price return of the last `hours` to the input DataFrame."""
    X[f'percentage_return_{hours}_hour'] = \
        (X['price_1_hour_ago'] - X[f'price_{hours}_hour_ago'])/ X[f'price_{hours}_hour_ago']
    return X

def get_preprocessing_pipeline():
    return make_pipeline(
        # trends
        FunctionTransformer(get_price_percentage_return, kw_args={'hours': 2}),
        FunctionTransformer(get_price_percentage_return, kw_args={'hours': 24}),

        # select columns
        FunctionTransformer(get_subset_of_features)
    )


if __name__ == '__main__':
    
    features, target = fire.Fire(build_features)
    
    preprocessing_pipeline = get_preprocessing_pipeline()

    preprocessing_pipeline.fit(features)
    X = preprocessing_pipeline.transform(features)
    print(X.head())
