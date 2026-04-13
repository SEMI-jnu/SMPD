from __future__ import annotations
from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from data_loader import COL_TARGET

SPLIT_RANDOM_STATE = 42

def validate_target_column(df: pd.DataFrame, target_col: str) -> None:

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the DataFrame.")


def validate_stratify_columns(df: pd.DataFrame, stratify_columns: list[str] | None, target_col: str) -> list[str]:
    if stratify_columns is None:
        stratify_columns = [target_col]

    missing_columns = [column for column in stratify_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Stratify columns are missing in the DataFrame: {', '.join(missing_columns)}")

    return stratify_columns


def build_stratify_labels(df: pd.DataFrame, stratify_columns: list[str] | None = None, target_col: str = COL_TARGET) -> pd.Series:

    stratify_columns = validate_stratify_columns(df, stratify_columns, target_col)

    if len(stratify_columns) == 1:
        return df[stratify_columns[0]].astype(str)

    return df[stratify_columns].astype(str).agg("__".join, axis=1)


def make_stratified_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    target_col: str = COL_TARGET,
    random_state: int = SPLIT_RANDOM_STATE,
    stratify_columns: list[str] | None = None,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:

    validate_target_column(df, target_col)

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    X = df.drop(columns=[target_col])
    y = build_stratify_labels(df, stratify_columns=stratify_columns, target_col=target_col)

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    split_results = []
    
    for train_index, test_index in splitter.split(X, y):
        train_df = df.iloc[train_index].reset_index(drop=True)
        test_df = df.iloc[test_index].reset_index(drop=True)
        split_results.append((train_df, test_df))

    return split_results
