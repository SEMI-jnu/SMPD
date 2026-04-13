import os
import sys
import pandas as pd

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(os.path.join(base_dir, "src/04_model/core"))
sys.path.append(os.path.join(base_dir, "src/config"))

from pipeline_config import PATHS
from data_loader import (
    COL_ID, COL_REGISTRY, COL_TARGET,
    load_features_df, get_feature_columns_by_group,
    build_model_input_df, get_available_registries
)
from sampler import sample_balanced_registries
from trainer import create_model, fit_model, predict_labels
from evaluator import (
    evaluate_predictions, build_result_row,
    result_row_to_dataframe, calculate_macro_average_across_registries
)

DEFAULT_MODEL_NAME = "xgboost"
DEFAULT_MODEL_PARAMS = {"random_state": 42}
REGISTRY_BALANCE_OPTIONS = ["original", "balanced"]


def run_single_rq3_setting(
    df: pd.DataFrame,
    registry_balance_setting: str,
) -> pd.DataFrame:

    feature_columns = get_feature_columns_by_group("all")
    model_input_df = build_model_input_df(df, feature_columns)

    registries = get_available_registries(model_input_df)
    all_results = []
    result_df_list = []

    for test_registry in registries:
        train_registries = [r for r in registries if r != test_registry]
        train_label = "+".join(sorted(train_registries))

        print(f"  Train: [{train_label}] → Test: [{test_registry}]")

        train_df = model_input_df[model_input_df[COL_REGISTRY].isin(train_registries)].copy()
        test_df = model_input_df[model_input_df[COL_REGISTRY] == test_registry].copy()

        if test_df.empty:
            print(f"  Skipping: no test data for registry '{test_registry}'")
            continue

        if registry_balance_setting == "balanced":
            train_df = sample_balanced_registries(train_df)

        model = create_model(model_name=DEFAULT_MODEL_NAME, model_params=DEFAULT_MODEL_PARAMS)
        trained_model = fit_model(
            model=model,
            train_df=train_df,
            target_column=COL_TARGET,
            drop_columns=[COL_ID, COL_REGISTRY],
        )

        y_pred = predict_labels(
            model=trained_model,
            test_df=test_df,
            target_column=COL_TARGET,
            drop_columns=[COL_ID, COL_REGISTRY],
        )

        single_result = evaluate_predictions(
            y_true=test_df[COL_TARGET],
            y_pred=y_pred,
        )
        all_results.append(single_result)

        result_row = build_result_row(
            meta_info={
                "rq": "RQ3",
                "feature_source": "smpd_sbom",
                "model_name": DEFAULT_MODEL_NAME,
                "train_registry": train_label,
                "test_registry": test_registry,
                "class_balance_setting": "original",
                "registry_balance_setting": registry_balance_setting,
                "evaluation_protocol": "unseen_registry",
                "feature_group": "all",
            },
            fold_result_list=[single_result],
            n_train=len(train_df),
            n_test=len(test_df),
            n_train_positive=int((train_df[COL_TARGET] == 1).sum()),
            n_train_negative=int((train_df[COL_TARGET] == 0).sum()),
            n_test_positive=int((test_df[COL_TARGET] == 1).sum()),
            n_test_negative=int((test_df[COL_TARGET] == 0).sum()),
        )
        result_df_list.append(result_row_to_dataframe(result_row))

    if all_results:
        macro_result = calculate_macro_average_across_registries(all_results)
        n = len(all_results)
        macro_result["tp"] = int(sum(r["tp"] for r in all_results) / n)
        macro_result["fp"] = int(sum(r["fp"] for r in all_results) / n)
        macro_result["fn"] = int(sum(r["fn"] for r in all_results) / n)
        macro_result["tn"] = int(sum(r["tn"] for r in all_results) / n)

        avg_row = build_result_row(
            meta_info={
                "rq": "RQ3",
                "feature_source": "smpd_sbom",
                "model_name": DEFAULT_MODEL_NAME,
                "train_registry": "all_combinations",
                "test_registry": "macro_average",
                "class_balance_setting": "original",
                "registry_balance_setting": registry_balance_setting,
                "evaluation_protocol": "unseen_registry",
                "feature_group": "all",
            },
            fold_result_list=[macro_result],
            n_train=0, n_test=0,
            n_train_positive=0, n_train_negative=0,
            n_test_positive=0, n_test_negative=0,
        )
        result_df_list.append(result_row_to_dataframe(avg_row))

    return pd.concat(result_df_list, axis=0, ignore_index=True)


def run_rq3() -> pd.DataFrame:

    print("Loading engineered features...")
    full_df = load_features_df()
    registries = get_available_registries(full_df)

    result_df_list = []

    for balance_setting in REGISTRY_BALANCE_OPTIONS:
        print(f"\nRunning RQ3 -> Model: XGBoost | Registry Balance: {balance_setting}")
        try:
            result_df = run_single_rq3_setting(full_df, balance_setting)
            result_df_list.append(result_df)
        except Exception as e:
            print(f"Error in RQ3 setting '{balance_setting}': {e}")

    final_results = pd.concat(result_df_list, axis=0, ignore_index=True)

    final_results["registry_balance_setting"] = pd.Categorical(
        final_results["registry_balance_setting"],
        categories=REGISTRY_BALANCE_OPTIONS,
        ordered=True,
    )
    registry_order = registries + ["macro_average"]
    final_results["test_registry"] = pd.Categorical(
        final_results["test_registry"],
        categories=registry_order,
        ordered=True,
    )
    final_results = final_results.sort_values(
        by=["registry_balance_setting", "test_registry"],
        kind="stable",
    ).reset_index(drop=True)

    return final_results


if __name__ == "__main__":
    final_results = run_rq3()

    output_dir = os.path.join(base_dir, PATHS["evaluation_results"])
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "rq3_results.csv")

    float_cols = final_results.select_dtypes(include=["float64"]).columns
    final_results[float_cols] = final_results[float_cols].round(4)

    final_results.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[Success] RQ3 execution complete. Results saved to: {out_path}")
