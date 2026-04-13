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
from sampler import sample_rq1_dataset
from splitter import make_stratified_kfold_splits
from trainer import train_and_predict
from evaluator import (
    evaluate_predictions, calculate_fold_size_summary, build_result_row,
    result_row_to_dataframe, calculate_macro_average_across_registries
)

DEFAULT_MODEL_NAME = "xgboost"
DEFAULT_MODEL_PARAMS = {"random_state": 42}
N_SPLITS = 5
CLASS_BALANCE_OPTIONS = ["original", "balanced"]


def run_single_rq1_setting(
    df: pd.DataFrame,
    registry_name: str,
    class_balance_setting: str,
) -> pd.DataFrame:

    feature_group_name = "all"
    feature_columns = get_feature_columns_by_group(feature_group_name)

    registry_df = filter_by_registry(df, registry_name)
    model_input_df = build_model_input_df(registry_df, feature_columns)

    sampled_df = sample_rq1_dataset(model_input_df, class_balance_setting)

    fold_splits = make_stratified_kfold_splits(sampled_df, n_splits=N_SPLITS, target_col=COL_TARGET)
    fold_size_summary = calculate_fold_size_summary(fold_splits)

    fold_result_list = []

    for train_df, test_df in fold_splits:
        trained_model, y_pred = train_and_predict(
            model_name=DEFAULT_MODEL_NAME,
            train_df=train_df,
            test_df=test_df,
            target_column=COL_TARGET,
            model_params=DEFAULT_MODEL_PARAMS,
            drop_columns=[COL_ID, COL_REGISTRY]
        )
        
        fold_result = evaluate_predictions(y_true=test_df[COL_TARGET], y_pred=y_pred)
        fold_result_list.append(fold_result)

    meta_info = {
        "rq": "RQ1",
        "feature_source": "smpd_sbom",
        "model_name": DEFAULT_MODEL_NAME,
        "train_registry": registry_name,
        "test_registry": registry_name,
        "class_balance_setting": class_balance_setting,
        "registry_balance_setting": "single_registry",
        "evaluation_protocol": f"stratified_{N_SPLITS}fold",
        "feature_group": feature_group_name,
    }
    
    result_row = build_result_row(
        meta_info=meta_info,
        fold_result_list=fold_result_list,
        n_train=fold_size_summary["n_train"],
        n_test=fold_size_summary["n_test"],
        n_train_positive=fold_size_summary["n_train_positive"],
        n_train_negative=fold_size_summary["n_train_negative"],
        n_test_positive=fold_size_summary["n_test_positive"],
        n_test_negative=fold_size_summary["n_test_negative"],
    )

    return result_row_to_dataframe(result_row)


def run_rq1() -> pd.DataFrame:
    print("Loading engineered features...")
    full_df = load_features_df()
    registries = get_available_registries(full_df)
    
    result_df_list = []
    
    for balance_setting in CLASS_BALANCE_OPTIONS:
        current_setting_results = []
        
        for registry in registries:
            print(f"Running RQ1 -> Model: XGBoost | Domain: {registry} | Balance: {balance_setting}")
            try:
                result_row_df = run_single_rq1_setting(full_df, registry, balance_setting)
                result_df_list.append(result_row_df)
                current_setting_results.append(result_row_df.iloc[0].to_dict())
            except Exception as e:
                print(f"Skipping {registry} due to error: {e}")
        
        if current_setting_results:
            macro_avg = calculate_macro_average_across_registries(current_setting_results)
            macro_avg["tp"] = int(sum(r["tp"] for r in current_setting_results) / len(registries))
            macro_avg["fp"] = int(sum(r["fp"] for r in current_setting_results) / len(registries))
            macro_avg["fn"] = int(sum(r["fn"] for r in current_setting_results) / len(registries))
            macro_avg["tn"] = int(sum(r["tn"] for r in current_setting_results) / len(registries))
            
            macro_meta = {
                "rq": "RQ1",
                "feature_source": "smpd_sbom",
                "model_name": DEFAULT_MODEL_NAME,
                "train_registry": "macro_average",
                "test_registry": "macro_average",
                "class_balance_setting": balance_setting,
                "registry_balance_setting": "single_registry",
                "evaluation_protocol": f"stratified_{N_SPLITS}fold",
                "feature_group": "all",
            }
            
            avg_row = build_result_row(
                meta_info=macro_meta, fold_result_list=[macro_avg],
                n_train=0, n_test=0, n_train_positive=0, n_train_negative=0, n_test_positive=0, n_test_negative=0
            )
            result_df_list.append(result_row_to_dataframe(avg_row))

    return pd.concat(result_df_list, axis=0, ignore_index=True)


if __name__ == "__main__":
    final_results = run_rq1()
    
    output_dir = os.path.join(base_dir, PATHS["evaluation_results"])
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "rq1_results.csv")
    
    float_cols = final_results.select_dtypes(include=['float64']).columns
    final_results[float_cols] = final_results[float_cols].round(4)
    
    final_results.to_csv(out_path, index=False, encoding='utf-8')
    print(f"\n[Success] RQ1 execution complete. Results saved to: {out_path}")
