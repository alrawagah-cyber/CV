"""K-fold cross-validation helper for tabular annotation datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def make_folds(
    annotations_csv: str | Path,
    n_splits: int = 5,
    stratify_col: str | None = None,
    seed: int = 42,
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Yield (fold_index, train_df, val_df) tuples.

    If `stratify_col` is provided and present, uses StratifiedKFold on that
    column (e.g. 'severity' for layer 3). Otherwise plain KFold.
    """
    df = pd.read_csv(annotations_csv).reset_index(drop=True)
    if stratify_col and stratify_col in df.columns:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        y = df[stratify_col].values
        splits = splitter.split(np.zeros(len(df)), y)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(np.zeros(len(df)))

    for i, (tr_idx, val_idx) in enumerate(splits):
        yield i, df.iloc[tr_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)
