import os
import sys
import pandas as pd

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(os.path.join(base_dir, "src/04_model/core"))
sys.path.append(os.path.join(base_dir, "src/config"))

from pipeline_config import PATHS
from data_loader import (
    COL_ID, COL_REGISTRY, COL_TARGET,
    load_features_df, filter_by_registry,
    get_feature_columns_by_group, build_model_input_df, get_available_registries
)
from trainer import create_model, fit_model, predict_labels
from evaluator import (
    evaluate_predictions, build_result_row,
    result_row_to_dataframe, calculate_macro_average_across_registries
)

DEFAULT_MODEL_NAME = "xgboost"
DEFAULT_MODEL_PARAMS = {"random_state": 42}
SETTING_TYPES = ["single", "multi", "unseen"]


def load_followup_features_df() -> pd.DataFrame:
    file_path = os.path.join(base_dir, PATHS["features_followup"])

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Follow-up feature file not found: {file_path}\n"
            f"Please run the feature engineering pipeline on the follow-up data first."
        )

    df = pd.read_csv(file_path, low_memory=False)
    return df.sort_values(by=[COL_REGISTRY, COL_ID]).reset_index(drop=True)


def _build_meta(setting_type: str, train_registry: str, test_registry: str) -> dict:
    return {
        "rq": "RQ5",
        "feature_source": "smpd_sbom",
        "model_name": DEFAULT_MODEL_NAME,
        "train_registry": train_registry,
        "test_registry": test_registry,
        "class_balance_setting": "original",
        "registry_balance_setting": setting_type,
        "evaluation_protocol": "temporal_holdout",
        "feature_group": "all",
    }


def _make_result_row(meta: dict, result: dict, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    return result_row_to_dataframe(
        build_result_row(
            meta_info=meta,
            fold_result_list=[result],
            n_train=len(train_df),
            n_test=len(test_df),
            n_train_positive=int((train_df[COL_TARGET] == 1).sum()),
            n_train_negative=int((train_df[COL_TARGET] == 0).sum()),
            n_test_positive=int((test_df[COL_TARGET] == 1).sum()),
            n_test_negative=int((test_df[COL_TARGET] == 0).sum()),
        )
    )


def _macro_avg_row(results: list[dict], meta_template: dict) -> pd.DataFrame:
    macro = calculate_macro_average_across_registries(results)
    n = len(results)
    for key in ("tp", "fp", "fn", "tn"):
        macro[key] = int(sum(r[key] for r in results) / n)
    avg_meta = {**meta_template, "test_registry": "macro_average", "train_registry": "macro_average"}
    return result_row_to_dataframe(
        build_result_row(avg_meta, [macro], 0, 0, 0, 0, 0, 0)
    )


def run_rq5_single(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    registries = get_available_registries(train_df)
    result_df_list = []
    all_results = []

    for registry in registries:
        reg_train_df = filter_by_registry(train_df, registry)
        reg_test_df = filter_by_registry(test_df, registry)

        model = create_model(DEFAULT_MODEL_NAME, DEFAULT_MODEL_PARAMS)
        trained = fit_model(model, reg_train_df, COL_TARGET, [COL_ID, COL_REGISTRY])
        y_pred = predict_labels(trained, reg_test_df, COL_TARGET, [COL_ID, COL_REGISTRY])

        result = evaluate_predictions(reg_test_df[COL_TARGET], y_pred)
        all_results.append(result)

        meta = _build_meta("single", registry, registry)
        result_df_list.append(_make_result_row(meta, result, reg_train_df, reg_test_df))
        print(f"  [{registry}] Train: {len(reg_train_df)} | Test (follow-up malicious): {len(reg_test_df)}")

    if all_results:
        result_df_list.append(_macro_avg_row(all_results, _build_meta("single", "", "")))

    return pd.concat(result_df_list, axis=0, ignore_index=True)


def run_rq5_multi(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    
    registries = get_available_registries(test_df)
    result_df_list = []
    all_results = []

    model = create_model(DEFAULT_MODEL_NAME, DEFAULT_MODEL_PARAMS)
    trained = fit_model(model, train_df, COL_TARGET, [COL_ID, COL_REGISTRY])

    for registry in registries:
        reg_test_df = filter_by_registry(test_df, registry)

        y_pred = predict_labels(trained, reg_test_df, COL_TARGET, [COL_ID, COL_REGISTRY])
        result = evaluate_predictions(reg_test_df[COL_TARGET], y_pred)
        all_results.append(result)

        meta = _build_meta("multi", "all", registry)
        result_df_list.append(_make_result_row(meta, result, train_df, reg_test_df))
        print(f"  [{registry}] Train: all ({len(train_df)}) | Test (follow-up malicious): {len(reg_test_df)}")

    if all_results:
        result_df_list.append(_macro_avg_row(all_results, _build_meta("multi", "all", "")))

    return pd.concat(result_df_list, axis=0, ignore_index=True)


def run_rq5_unseen(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:

    registries = get_available_registries(train_df)
    result_df_list = []
    all_results = []

    for test_registry in registries:
        train_registries = [r for r in registries if r != test_registry]
        reg_train_df = train_df[train_df[COL_REGISTRY].isin(train_registries)].copy()
        reg_test_df = filter_by_registry(test_df, test_registry)

        model = create_model(DEFAULT_MODEL_NAME, DEFAULT_MODEL_PARAMS)
        trained = fit_model(model, reg_train_df, COL_TARGET, [COL_ID, COL_REGISTRY])
        y_pred = predict_labels(trained, reg_test_df, COL_TARGET, [COL_ID, COL_REGISTRY])

        result = evaluate_predictions(reg_test_df[COL_TARGET], y_pred)
        all_results.append(result)

        train_label = "+".join(sorted(train_registries))
        meta = _build_meta("unseen", train_label, test_registry)
        result_df_list.append(_make_result_row(meta, result, reg_train_df, reg_test_df))
        print(f"  [{train_label}] → [{test_registry}] Test (follow-up malicious): {len(reg_test_df)}")

    if all_results:
        result_df_list.append(_macro_avg_row(all_results, _build_meta("unseen", "all_combinations", "")))

    return pd.concat(result_df_list, axis=0, ignore_index=True)


def run_rq5() -> pd.DataFrame:

    print("Loading initial (train) features...")
    feature_columns = get_feature_columns_by_group("all")

    raw_train_df = load_features_df()
    train_df = build_model_input_df(raw_train_df, feature_columns)

    print("Loading follow-up (test) features...")
    raw_test_df = load_followup_features_df()
    test_df = build_model_input_df(raw_test_df, feature_columns)

    result_df_list = []

    for setting_type in SETTING_TYPES:
        print(f"\nRunning RQ5 -> Model: XGBoost | Setting: {setting_type}")
        try:
            if setting_type == "single":
                result_df = run_rq5_single(train_df, test_df)
            elif setting_type == "multi":
                result_df = run_rq5_multi(train_df, test_df)
            elif setting_type == "unseen":
                result_df = run_rq5_unseen(train_df, test_df)
            result_df_list.append(result_df)
        except Exception as e:
            print(f"  Error in setting '{setting_type}': {e}")

    final_results = pd.concat(result_df_list, axis=0, ignore_index=True)

    final_results["registry_balance_setting"] = pd.Categorical(
        final_results["registry_balance_setting"], categories=SETTING_TYPES, ordered=True
    )
    final_results = final_results.sort_values(
        by=["registry_balance_setting", "test_registry"], kind="stable"
    ).reset_index(drop=True)

    return final_results


if __name__ == "__main__":
    final_results = run_rq5()

    output_dir = os.path.join(base_dir, PATHS["evaluation_results"])
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "rq5_results.csv")

    float_cols = final_results.select_dtypes(include=["float64"]).columns
    final_results[float_cols] = final_results[float_cols].round(4)

    final_results.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[Success] RQ5 execution complete. Results saved to: {out_path}")
