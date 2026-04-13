from __future__ import annotations
from typing import Any
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from data_loader import COL_TARGET

TABLE_COLUMNS = [
    "rq", "model_name", "class_balance_setting", "registry_balance_setting",
    "feature_group", "train_registry", "test_registry", "feature_source",
    "precision", "recall", "f1", "accuracy",
    "precision_std", "recall_std", "f1_std", "accuracy_std",
    "tp", "fp", "fn", "tn",
    "n_train", "n_test", "n_train_positive", "n_train_negative",
    "n_test_positive", "n_test_negative", "evaluation_protocol", "experiment_id"
]


def calculate_confusion_counts(y_true: pd.Series, y_pred: pd.Series) -> dict[str, int]:

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:

    confusion_counts = calculate_confusion_counts(y_true, y_pred)

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tp": confusion_counts["tp"],
        "fp": confusion_counts["fp"],
        "fn": confusion_counts["fn"],
        "tn": confusion_counts["tn"],
    }


def summarize_fold_results(fold_result_list: list[dict[str, float | int]]) -> dict[str, float | int]:

    if not fold_result_list:
        raise ValueError("The fold results list is empty.")

    fold_result_df = pd.DataFrame(fold_result_list)

    return {
        "precision": float(fold_result_df["precision"].mean()),
        "precision_std": float(fold_result_df["precision"].std(ddof=0)),
        "recall": float(fold_result_df["recall"].mean()),
        "recall_std": float(fold_result_df["recall"].std(ddof=0)),
        "f1": float(fold_result_df["f1"].mean()),
        "f1_std": float(fold_result_df["f1"].std(ddof=0)),
        "accuracy": float(fold_result_df["accuracy"].mean()),
        "accuracy_std": float(fold_result_df["accuracy"].std(ddof=0)),
        "tp": int(fold_result_df["tp"].sum()),
        "fp": int(fold_result_df["fp"].sum()),
        "fn": int(fold_result_df["fn"].sum()),
        "tn": int(fold_result_df["tn"].sum()),
    }


def calculate_macro_average_across_registries(registry_result_list: list[dict[str, float]]) -> dict[str, float]:

    if not registry_result_list:
        raise ValueError("The registry results list is empty.")

    registry_result_df = pd.DataFrame(registry_result_list)
    required_columns = ["precision", "recall", "f1", "accuracy"]
    
    missing_columns = [col for col in required_columns if col not in registry_result_df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns required for macro-averaging: {', '.join(missing_columns)}")

    return {
        "precision": float(registry_result_df["precision"].mean()),
        "recall": float(registry_result_df["recall"].mean()),
        "f1": float(registry_result_df["f1"].mean()),
        "accuracy": float(registry_result_df["accuracy"].mean()),
    }


def build_experiment_id(result_row: dict[str, Any]) -> str:

    ordered_keys = [
        "rq", "feature_source", "model_name", "train_registry",
        "test_registry", "class_balance_setting", "registry_balance_setting",
        "evaluation_protocol", "feature_group",
    ]
    parts = [str(result_row.get(key, "none")) for key in ordered_keys]
    return "_".join(parts)


def build_result_row(
    meta_info: dict[str, Any],
    fold_result_list: list[dict[str, float | int]],
    n_train: int, n_test: int,
    n_train_positive: int, n_train_negative: int,
    n_test_positive: int, n_test_negative: int,
) -> dict[str, Any]:

    metric_summary = summarize_fold_results(fold_result_list)

    result_row = {
        "rq": meta_info.get("rq"),
        "feature_source": meta_info.get("feature_source"),
        "model_name": meta_info.get("model_name"),
        "train_registry": meta_info.get("train_registry"),
        "test_registry": meta_info.get("test_registry"),
        "class_balance_setting": meta_info.get("class_balance_setting"),
        "registry_balance_setting": meta_info.get("registry_balance_setting"),
        "evaluation_protocol": meta_info.get("evaluation_protocol"),
        "feature_group": meta_info.get("feature_group"),

        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_train_positive": int(n_train_positive),
        "n_train_negative": int(n_train_negative),
        "n_test_positive": int(n_test_positive),
        "n_test_negative": int(n_test_negative),
    }


    result_row.update(metric_summary)

    result_row["experiment_id"] = build_experiment_id(result_row)

    return result_row


def result_row_to_dataframe(result_row: dict[str, Any]) -> pd.DataFrame:

    result_df = pd.DataFrame([result_row])
    
    missing_columns = [col for col in TABLE_COLUMNS if col not in result_df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns in the result row: {', '.join(missing_columns)}")

    return result_df.loc[:, TABLE_COLUMNS].copy()


def calculate_fold_size_summary(fold_splits: list[tuple[pd.DataFrame, pd.DataFrame]]) -> dict[str, int]:

    train_sizes, test_sizes = [], []
    train_pos, train_neg = [], []
    test_pos, test_neg = [], []

    for train_df, test_df in fold_splits:
        train_sizes.append(len(train_df))
        test_sizes.append(len(test_df))
        train_pos.append(int((train_df[COL_TARGET] == 1).sum()))
        train_neg.append(int((train_df[COL_TARGET] == 0).sum()))
        test_pos.append(int((test_df[COL_TARGET] == 1).sum()))
        test_neg.append(int((test_df[COL_TARGET] == 0).sum()))

    if not train_sizes:
         raise ValueError("The fold splits list is empty.")

    L = len(train_sizes)
    return {
        "n_train": int(round(sum(train_sizes) / L)),
        "n_test": int(round(sum(test_sizes) / L)),
        "n_train_positive": int(round(sum(train_pos) / L)),
        "n_train_negative": int(round(sum(train_neg) / L)),
        "n_test_positive": int(round(sum(test_pos) / L)),
        "n_test_negative": int(round(sum(test_neg) / L)),
    }
