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
from splitter import make_stratified_kfold_splits
from trainer import create_model, fit_model, predict_labels
from evaluator import (
    evaluate_predictions, calculate_fold_size_summary, build_result_row,
    result_row_to_dataframe, calculate_macro_average_across_registries
)

DEFAULT_MODEL_NAME = "xgboost"
DEFAULT_MODEL_PARAMS = {"random_state": 42}
N_SPLITS = 5

FEATURE_GROUPS = ["general", "people", "license", "dependency", "url", "all"]
SETTING_TYPES = ["single", "multi", "unseen"]


def _build_meta(setting_type: str, feature_group: str, train_registry: str, test_registry: str) -> dict:
    return {
        "rq": "RQ4",
        "feature_source": "smpd_sbom",
        "model_name": DEFAULT_MODEL_NAME,
        "train_registry": train_registry,
        "test_registry": test_registry,
        "class_balance_setting": "original",
        "registry_balance_setting": setting_type,
        "evaluation_protocol": f"stratified_{N_SPLITS}fold" if setting_type in ("single", "multi") else "unseen_registry",
        "feature_group": feature_group,
    }


def _macro_avg_row(results: list[dict], meta: dict) -> pd.DataFrame:

    macro = calculate_macro_average_across_registries(results)
    n = len(results)
    for key in ("tp", "fp", "fn", "tn"):
        macro[key] = int(sum(r[key] for r in results) / n)
    return result_row_to_dataframe(
        build_result_row(meta, [macro], 0, 0, 0, 0, 0, 0)
    )


def run_rq4_single(df: pd.DataFrame, feature_group: str) -> pd.DataFrame:

    feature_columns = get_feature_columns_by_group(feature_group)
    model_input_df = build_model_input_df(df, feature_columns)
    registries = get_available_registries(model_input_df)

    result_df_list = []

    for registry in registries:
        registry_df = filter_by_registry(model_input_df, registry)
        fold_splits = make_stratified_kfold_splits(
            df=registry_df, n_splits=N_SPLITS, target_col=COL_TARGET
        )
        fold_size = calculate_fold_size_summary(fold_splits)
        fold_results = []

        for train_df, test_df in fold_splits:
            model = create_model(DEFAULT_MODEL_NAME, DEFAULT_MODEL_PARAMS)
            trained = fit_model(model, train_df, COL_TARGET, [COL_ID, COL_REGISTRY])
            y_pred = predict_labels(trained, test_df, COL_TARGET, [COL_ID, COL_REGISTRY])
            fold_results.append(evaluate_predictions(test_df[COL_TARGET], y_pred))

        meta = _build_meta("single", feature_group, registry, registry)
        result_df_list.append(result_row_to_dataframe(
            build_result_row(meta, fold_results,
                             fold_size["n_train"], fold_size["n_test"],
                             fold_size["n_train_positive"], fold_size["n_train_negative"],
                             fold_size["n_test_positive"], fold_size["n_test_negative"])
        ))

    return pd.concat(result_df_list, axis=0, ignore_index=True)


def run_rq4_multi(df: pd.DataFrame, feature_group: str) -> pd.DataFrame:

    feature_columns = get_feature_columns_by_group(feature_group)
    model_input_df = build_model_input_df(df, feature_columns)
    registries = get_available_registries(model_input_df)

    fold_splits = make_stratified_kfold_splits(
        df=model_input_df, n_splits=N_SPLITS, target_col=COL_TARGET,
        stratify_columns=[COL_REGISTRY, COL_TARGET]
    )
    fold_size = calculate_fold_size_summary(fold_splits)
    fold_result_dict: dict[str, list] = {reg: [] for reg in registries}
    fold_result_dict["macro_average"] = []

    for train_df, test_df in fold_splits:
        model = create_model(DEFAULT_MODEL_NAME, DEFAULT_MODEL_PARAMS)
        trained = fit_model(model, train_df, COL_TARGET, [COL_ID, COL_REGISTRY])

        fold_reg_results = []
        for registry in registries:
            reg_test_df = filter_by_registry(test_df, registry)
            if reg_test_df.empty:
                continue
            y_pred = predict_labels(trained, reg_test_df, COL_TARGET, [COL_ID, COL_REGISTRY])
            result = evaluate_predictions(reg_test_df[COL_TARGET], y_pred)
            fold_reg_results.append(result)
            fold_result_dict[registry].append(result)

        if fold_reg_results:
            macro = calculate_macro_average_across_registries(fold_reg_results)
            n = len(fold_reg_results)
            for key in ("tp", "fp", "fn", "tn"):
                macro[key] = int(sum(r[key] for r in fold_reg_results) / n)
            fold_result_dict["macro_average"].append(macro)

    result_df_list = []
    for target in registries + ["macro_average"]:
        target_results = fold_result_dict[target]
        if not target_results:
            continue
        meta = _build_meta("multi", feature_group, "all", target)
        n_train = fold_size["n_train"] if target != "macro_average" else 0
        n_test = fold_size["n_test"] if target != "macro_average" else 0
        result_df_list.append(result_row_to_dataframe(
            build_result_row(meta, target_results,
                             n_train, n_test,
                             fold_size["n_train_positive"], fold_size["n_train_negative"],
                             fold_size["n_test_positive"], fold_size["n_test_negative"])
        ))

    return pd.concat(result_df_list, axis=0, ignore_index=True)


def run_rq4_unseen(df: pd.DataFrame, feature_group: str) -> pd.DataFrame:

    feature_columns = get_feature_columns_by_group(feature_group)
    model_input_df = build_model_input_df(df, feature_columns)
    registries = get_available_registries(model_input_df)

    result_df_list = []
    all_results = []

    for test_registry in registries:
        train_registries = [r for r in registries if r != test_registry]
        train_df = model_input_df[model_input_df[COL_REGISTRY].isin(train_registries)].copy()
        test_df = filter_by_registry(model_input_df, test_registry)

        model = create_model(DEFAULT_MODEL_NAME, DEFAULT_MODEL_PARAMS)
        trained = fit_model(model, train_df, COL_TARGET, [COL_ID, COL_REGISTRY])
        y_pred = predict_labels(trained, test_df, COL_TARGET, [COL_ID, COL_REGISTRY])

        single_result = evaluate_predictions(test_df[COL_TARGET], y_pred)
        all_results.append(single_result)

        meta = _build_meta("unseen", feature_group, "+".join(sorted(train_registries)), test_registry)
        result_df_list.append(result_row_to_dataframe(
            build_result_row(meta, [single_result],
                             len(train_df), len(test_df),
                             int((train_df[COL_TARGET] == 1).sum()),
                             int((train_df[COL_TARGET] == 0).sum()),
                             int((test_df[COL_TARGET] == 1).sum()),
                             int((test_df[COL_TARGET] == 0).sum()))
        ))

    if all_results:
        macro_meta = _build_meta("unseen", feature_group, "all_combinations", "macro_average")
        result_df_list.append(_macro_avg_row(all_results, macro_meta))

    return pd.concat(result_df_list, axis=0, ignore_index=True)


def run_rq4() -> pd.DataFrame:

    print("Loading engineered features...")
    full_df = load_features_df()

    total = len(SETTING_TYPES) * len(FEATURE_GROUPS)
    step = 0
    result_df_list = []

    for setting_type in SETTING_TYPES:
        for feature_group in FEATURE_GROUPS:
            step += 1
            print(f"[{step}/{total}] Running RQ4 -> Setting: {setting_type} | Group: {feature_group}")
            try:
                if setting_type == "single":
                    result_df = run_rq4_single(full_df, feature_group)
                elif setting_type == "multi":
                    result_df = run_rq4_multi(full_df, feature_group)
                elif setting_type == "unseen":
                    result_df = run_rq4_unseen(full_df, feature_group)
                result_df_list.append(result_df)
            except Exception as e:
                print(f"  Error: {e}")

    final_results = pd.concat(result_df_list, axis=0, ignore_index=True)

    final_results["registry_balance_setting"] = pd.Categorical(
        final_results["registry_balance_setting"], categories=SETTING_TYPES, ordered=True
    )
    final_results["feature_group"] = pd.Categorical(
        final_results["feature_group"], categories=FEATURE_GROUPS, ordered=True
    )
    final_results = final_results.sort_values(
        by=["registry_balance_setting", "test_registry", "feature_group"],
        kind="stable",
    ).reset_index(drop=True)

    return final_results


if __name__ == "__main__":
    final_results = run_rq4()

    output_dir = os.path.join(base_dir, PATHS["evaluation_results"])
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "rq4_results.csv")

    float_cols = final_results.select_dtypes(include=["float64"]).columns
    final_results[float_cols] = final_results[float_cols].round(4)

    final_results.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[Success] RQ4 execution complete. Results saved to: {out_path}")
