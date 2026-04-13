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
from sampler import sample_rq2_dataset
from splitter import make_stratified_kfold_splits
from trainer import create_model, fit_model, predict_labels
from evaluator import (
    evaluate_predictions, calculate_fold_size_summary, build_result_row,
    result_row_to_dataframe, calculate_macro_average_across_registries
)

DEFAULT_MODEL_NAME = "xgboost"
DEFAULT_MODEL_PARAMS = {"random_state": 42}
N_SPLITS = 5
REGISTRY_BALANCE_OPTIONS = ["original", "balanced"]


def run_single_rq2_setting(
    df: pd.DataFrame,
    registry_balance_setting: str,
) -> pd.DataFrame:

    feature_columns = get_feature_columns_by_group("all")
    model_input_df = build_model_input_df(df, feature_columns)

    sampled_df = sample_rq2_dataset(model_input_df, registry_balance_setting)

    fold_splits = make_stratified_kfold_splits(
        df=sampled_df,
        n_splits=N_SPLITS,
        target_col=COL_TARGET,
        stratify_columns=[COL_REGISTRY, COL_TARGET],
    )
    fold_size_summary = calculate_fold_size_summary(fold_splits)

    registries = get_available_registries(sampled_df)

    fold_result_dict: dict[str, list] = {reg: [] for reg in registries}
    fold_result_dict["macro_average"] = []

    for train_df, test_df in fold_splits:
        model = create_model(model_name=DEFAULT_MODEL_NAME, model_params=DEFAULT_MODEL_PARAMS)

        trained_model = fit_model(
            model=model,
            train_df=train_df,
            target_column=COL_TARGET,
            drop_columns=[COL_ID, COL_REGISTRY],
        )

        fold_registry_results = []

        for test_registry in registries:
            test_registry_df = filter_by_registry(test_df, test_registry)

            y_pred = predict_labels(
                model=trained_model,
                test_df=test_registry_df,
                target_column=COL_TARGET,
                drop_columns=[COL_ID, COL_REGISTRY],
            )

            fold_result = evaluate_predictions(
                y_true=test_registry_df[COL_TARGET],
                y_pred=y_pred,
            )
            fold_registry_results.append(fold_result)
            fold_result_dict[test_registry].append(fold_result)

        if fold_registry_results:
            macro_result = calculate_macro_average_across_registries(fold_registry_results)
            n = len(fold_registry_results)
            macro_result["tp"] = int(sum(r["tp"] for r in fold_registry_results) / n)
            macro_result["fp"] = int(sum(r["fp"] for r in fold_registry_results) / n)
            macro_result["fn"] = int(sum(r["fn"] for r in fold_registry_results) / n)
            macro_result["tn"] = int(sum(r["tn"] for r in fold_registry_results) / n)
            fold_result_dict["macro_average"].append(macro_result)

    result_df_list = []
    all_targets = registries + ["macro_average"]

    for target_registry in all_targets:
        target_fold_results = fold_result_dict[target_registry]
        if not target_fold_results:
            continue

        meta_info = {
            "rq": "RQ2",
            "feature_source": "smpd_sbom",
            "model_name": DEFAULT_MODEL_NAME,
            "train_registry": "all",           
            "test_registry": target_registry,
            "class_balance_setting": "original",  
            "registry_balance_setting": registry_balance_setting,
            "evaluation_protocol": f"stratified_{N_SPLITS}fold",
            "feature_group": "all",
        }

        n_train = fold_size_summary["n_train"] if target_registry != "macro_average" else 0
        n_test = fold_size_summary["n_test"] if target_registry != "macro_average" else 0

        result_row = build_result_row(
            meta_info=meta_info,
            fold_result_list=target_fold_results,
            n_train=n_train,
            n_test=n_test,
            n_train_positive=fold_size_summary["n_train_positive"],
            n_train_negative=fold_size_summary["n_train_negative"],
            n_test_positive=fold_size_summary["n_test_positive"],
            n_test_negative=fold_size_summary["n_test_negative"],
        )
        result_df_list.append(result_row_to_dataframe(result_row))

    return pd.concat(result_df_list, axis=0, ignore_index=True)


def run_rq2() -> pd.DataFrame:

    print("Loading engineered features...")
    full_df = load_features_df()

    result_df_list = []

    for balance_setting in REGISTRY_BALANCE_OPTIONS:
        print(f"\nRunning RQ2 -> Model: XGBoost | Registry Balance: {balance_setting}")
        try:
            result_df = run_single_rq2_setting(full_df, balance_setting)
            result_df_list.append(result_df)
        except Exception as e:
            print(f"Error in RQ2 setting '{balance_setting}': {e}")

    final_results = pd.concat(result_df_list, axis=0, ignore_index=True)
    
    final_results["registry_balance_setting"] = pd.Categorical(
        final_results["registry_balance_setting"],
        categories=REGISTRY_BALANCE_OPTIONS,
        ordered=True,
    )
    registry_order = get_available_registries(load_features_df()) + ["macro_average"]
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
    final_results = run_rq2()

    output_dir = os.path.join(base_dir, PATHS["evaluation_results"])
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "rq2_results.csv")

    float_cols = final_results.select_dtypes(include=["float64"]).columns
    final_results[float_cols] = final_results[float_cols].round(4)

    final_results.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[Success] RQ2 execution complete. Results saved to: {out_path}")
