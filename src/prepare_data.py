from typing import Tuple, List

import numpy as np

from data_preprocessing import prepare_training_data


def load_and_prepare_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Backwards-compatible wrapper around the new preprocessing pipeline.

    This preserves the original signature and return values:
    - X: scaled feature matrix
    - y: target array
    - features: list of feature names
    """
    X, y, features, _ = prepare_training_data(csv_path)
    return X, y, features

