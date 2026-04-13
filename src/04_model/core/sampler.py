from __future__ import annotations
import pandas as pd
from typing import Tuple

from data_loader import COL_TARGET, COL_REGISTRY

SPLIT_RANDOM_STATE = 42

RQ1_CLASS_BALANCE_OPTIONS = ["original", "balanced"]
RQ2_REGISTRY_BALANCE_OPTIONS = ["original", "balanced"]


def validate_required_columns(df: pd.DataFrame) -> None:

    missing_columns = [col for col in [COL_TARGET, COL_REGISTRY] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in DataFrame: {', '.join(missing_columns)}")


def validate_binary_target(df: pd.DataFrame) -> None:

    unique_values = sorted(df[COL_TARGET].dropna().unique().tolist())
    if not set(unique_values).issubset({0, 1}):
        raise ValueError(f"'{COL_TARGET}' column must be binary (0, 1). Current values: {unique_values}")


def get_class_counts(df: pd.DataFrame) -> Tuple[int, int]:

    validate_binary_target(df)

    value_counts = df[COL_TARGET].value_counts().sort_index()
    negative_count = int(value_counts.get(0, 0))
    positive_count = int(value_counts.get(1, 0))

    return negative_count, positive_count


def sample_balanced_classes(df: pd.DataFrame, random_state: int = SPLIT_RANDOM_STATE) -> pd.DataFrame:

    validate_required_columns(df)
    validate_binary_target(df)

    min_class_count = int(df[COL_TARGET].value_counts().min())
    if min_class_count == 0:
        raise ValueError("Cannot perform class balancing: One of the classes has 0 samples.")

    sampled_groups = []
    for class_value, group_df in df.groupby(COL_TARGET):
        sampled_group_df = group_df.sample(
            n=min_class_count,
            random_state=random_state,
            replace=False,
        )
        sampled_groups.append(sampled_group_df)

    result_df = pd.concat(sampled_groups, axis=0)
    result_df = result_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return result_df


def sample_rq1_dataset(
    df: pd.DataFrame,
    class_balance_setting: str,
    random_state: int = SPLIT_RANDOM_STATE,
) -> pd.DataFrame:

    validate_required_columns(df)

    if class_balance_setting not in RQ1_CLASS_BALANCE_OPTIONS:
        raise ValueError(f"Unsupported balance setting for RQ1. Options: {RQ1_CLASS_BALANCE_OPTIONS}")

    if class_balance_setting == "original":
        return df.copy().reset_index(drop=True)

    if class_balance_setting == "balanced":
        return sample_balanced_classes(df, random_state=random_state)


def get_registry_counts(df: pd.DataFrame) -> pd.Series:
    validate_required_columns(df)
    return df[COL_REGISTRY].value_counts().sort_index()


def sample_balanced_registries(
    df: pd.DataFrame,
    random_state: int = SPLIT_RANDOM_STATE,
) -> pd.DataFrame:

    validate_required_columns(df)

    registry_counts = get_registry_counts(df)
    if registry_counts.empty:
        return df.copy()

    min_registry_count = int(registry_counts.min())
    sampled_registry_groups = []

    for registry_value, registry_df in df.groupby(COL_REGISTRY):
        total_count = len(registry_df)
        pos_df = registry_df[registry_df[COL_TARGET] == 1]
        neg_df = registry_df[registry_df[COL_TARGET] == 0]
        
        pos_ratio = len(pos_df) / total_count if total_count > 0 else 0
        
        target_pos_count = int(round(min_registry_count * pos_ratio))
        target_neg_count = min_registry_count - target_pos_count
        
        sampled_pos = pos_df.sample(n=target_pos_count, random_state=random_state)
        sampled_neg = neg_df.sample(n=target_neg_count, random_state=random_state)
        
        sampled_registry_groups.append(pd.concat([sampled_pos, sampled_neg]))

    result_df = pd.concat(sampled_registry_groups, axis=0)
    result_df = result_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return result_df


def sample_rq2_dataset(
    df: pd.DataFrame,
    registry_balance_setting: str,
    random_state: int = SPLIT_RANDOM_STATE,
) -> pd.DataFrame:

    validate_required_columns(df)

    if registry_balance_setting not in RQ2_REGISTRY_BALANCE_OPTIONS:
        raise ValueError(f"Unsupported balance setting for RQ2. Options: {RQ2_REGISTRY_BALANCE_OPTIONS}")

    if registry_balance_setting == "original":
        return df.copy().reset_index(drop=True)

    if registry_balance_setting == "balanced":
        return sample_balanced_registries(df, random_state=random_state)
