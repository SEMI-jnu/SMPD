import os
import sys
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

MODEL_NAMES = ["xgboost", "gradient_boosting", "random_forest", "decision_tree", "svm"]
MODEL_PARAMS = {"random_state": 42}

SETTING_TYPES = ["single", "multi", "unseen", "temporal"]
N_SPLITS = 5


def _create_model(model_name: str) -> object:

    if model_name == "svm":
        return make_pipeline(StandardScaler(), SVC(**MODEL_PARAMS))
    return create_model(model_name=model_name, model_params=MODEL_PARAMS)


def _build_meta(model_name: str, setting_type: str, train_registry: str,
                test_registry: str, eval_protocol: str) -> dict:
    return {
        "rq": "RQ6",
        "feature_source": "smpd_sbom",
        "model_name": model_name,
        "train_registry": train_registry,
        "test_registry": test_registry,
        "class_balance_setting": "original",
        "registry_balance_setting": setting_type,
        "evaluation_protocol": eval_protocol,
        "feature_group": "all",
    }


def _macro_avg_row(results: list[dict], meta: dict) -> pd.DataFrame:
    macro = calculate_macro_average_across_registries(results)
    n = len(results)
    for key in ("tp", "fp", "fn", "tn"):
        macro[key] = int(sum(r[key] for r in results) / n)
    return result_row_to_dataframe(build_result_row(meta, [macro], 0, 0, 0, 0, 0, 0))


# ── Singl ──────────────────────────────────────────────────────────
def run_rq6_single(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    registries = get_available_registries(df)
    result_df_list = []

    for registry in registries:
        reg_df = filter_by_registry(df, registry)
        fold_splits = make_stratified_kfold_splits(reg_df, n_splits=N_SPLITS, target_col=COL_TARGET)
        fold_size = calculate_fold_size_summary(fold_splits)
        fold_results = []

        for train_df, test_df in fold_splits:
            model = _create_model(model_name)
            trained = fit_model(model, train_df, COL_TARGET, [COL_ID, COL_REGISTRY])
            y_pred = predict_labels(trained, test_df, COL_TARGET, [COL_ID, COL_REGISTRY])
            fold_results.append(evaluate_predictions(test_df[COL_TARGET], y_pred))

        meta = _build_meta(model_name, "single", registry, registry,
                           f"stratified_{N_SPLITS}fold")
        result_df_list.append(result_row_to_dataframe(
            build_result_row(meta, fold_results,
                             fold_size["n_train"], fold_size["n_test"],
                             fold_size["n_train_positive"], fold_size["n_train_negative"],
                             fold_size["n_test_positive"], fold_size["n_test_negative"])
        ))

    return pd.concat(result_df_list, axis=0, ignore_index=True)


# ── Multi ──────────────────────────────────────────────────────────
def run_rq6_multi(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    registries = get_available_registries(df)
    fold_splits = make_stratified_kfold_splits(
        df, n_splits=N_SPLITS, target_col=COL_TARGET,
        stratify_columns=[COL_REGISTRY, COL_TARGET]
    )
    fold_size = calculate_fold_size_summary(fold_splits)
    fold_result_dict: dict[str, list] = {reg: [] for reg in registries}
    fold_result_dict["macro_average"] = []

    for train_df, test_df in fold_splits:
        model = _create_model(model_name)
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
        meta = _build_meta(model_name, "multi", "all", target,
                           f"stratified_{N_SPLITS}fold")
        n_train = fold_size["n_train"] if target != "macro_average" else 0
        n_test = fold_size["n_test"] if target != "macro_average" else 0
        result_df_list.append(result_row_to_dataframe(
            build_result_row(meta, target_results,
                             n_train, n_test,
                             fold_size["n_train_positive"], fold_size["n_train_negative"],
                             fold_size["n_test_positive"], fold_size["n_test_negative"])
        ))

    return pd.concat(result_df_list, axis=0, ignore_index=True)


# ── Unseen ─────────────────────────────────────────────────────────
def run_rq6_unseen(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    registries = get_available_registries(df)
    result_df_list = []
    all_results = []

    for test_registry in registries:
        train_registries = [r for r in registries if r != test_registry]
        train_df = df[df[COL_REGISTRY].isin(train_registries)].copy()
        test_df = filter_by_registry(df, test_registry)

        model = _create_model(model_name)
        trained = fit_model(model, train_df, COL_TARGET, [COL_ID, COL_REGISTRY])
        y_pred = predict_labels(trained, test_df, COL_TARGET, [COL_ID, COL_REGISTRY])

        result = evaluate_predictions(test_df[COL_TARGET], y_pred)
        all_results.append(result)

        train_label = "+".join(sorted(train_registries))
        meta = _build_meta(model_name, "unseen", train_label, test_registry,
                           "unseen_registry")
        result_df_list.append(result_row_to_dataframe(
            build_result_row(meta, [result],
                             len(train_df), len(test_df),
                             int((train_df[COL_TARGET] == 1).sum()),
                             int((train_df[COL_TARGET] == 0).sum()),
                             int((test_df[COL_TARGET] == 1).sum()),
                             int((test_df[COL_TARGET] == 0).sum()))
        ))

    if all_results:
        macro_meta = _build_meta(model_name, "unseen",
                                 "all_combinations", "macro_average", "unseen_registry")
        result_df_list.append(_macro_avg_row(all_results, macro_meta))

    return pd.concat(result_df_list, axis=0, ignore_index=True)


# ── Temporal ──────────────────────
def load_followup_df(feature_columns: list[str]) -> pd.DataFrame:
    file_path = os.path.join(base_dir, PATHS["features_followup"])
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Follow-up feature file not found: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    return build_model_input_df(df, feature_columns)


def run_rq6_temporal(train_df: pd.DataFrame, test_df: pd.DataFrame,
                     model_name: str, sub_setting: str) -> pd.DataFrame:

    registries = get_available_registries(train_df)
    result_df_list = []
    all_results = []
    setting_label = f"temporal_{sub_setting}"

    for test_registry in registries:
        if sub_setting == "single":
            reg_train_df = filter_by_registry(train_df, test_registry)
            train_label = test_registry
        elif sub_setting == "multi":
            reg_train_df = train_df.copy()
            train_label = "all"
        elif sub_setting == "unseen":
            train_registries = [r for r in registries if r != test_registry]
            reg_train_df = train_df[train_df[COL_REGISTRY].isin(train_registries)].copy()
            train_label = "+".join(sorted(train_registries))
        else:
            raise ValueError(f"Unknown temporal sub_setting: {sub_setting}")

        reg_test_df = filter_by_registry(test_df, test_registry)
        if reg_train_df.empty or reg_test_df.empty:
            continue

        model = _create_model(model_name)
        trained = fit_model(model, reg_train_df, COL_TARGET, [COL_ID, COL_REGISTRY])
        y_pred = predict_labels(trained, reg_test_df, COL_TARGET, [COL_ID, COL_REGISTRY])

        result = evaluate_predictions(reg_test_df[COL_TARGET], y_pred)
        all_results.append(result)

        meta = _build_meta(model_name, setting_label, train_label,
                           test_registry, "temporal_holdout")
        result_df_list.append(result_row_to_dataframe(
            build_result_row(meta, [result],
                             len(reg_train_df), len(reg_test_df),
                             int((reg_train_df[COL_TARGET] == 1).sum()),
                             int((reg_train_df[COL_TARGET] == 0).sum()),
                             int((reg_test_df[COL_TARGET] == 1).sum()),
                             int((reg_test_df[COL_TARGET] == 0).sum()))
        ))

    if all_results:
        avg_train = "all_combinations" if sub_setting == "unseen" else \
                    ("all" if sub_setting == "multi" else "macro_average")
        macro_meta = _build_meta(model_name, setting_label,
                                 avg_train, "macro_average", "temporal_holdout")
        result_df_list.append(_macro_avg_row(all_results, macro_meta))

    return pd.concat(result_df_list, axis=0, ignore_index=True)


def run_rq6() -> pd.DataFrame:
    print("Loading engineered features...")
    feature_columns = get_feature_columns_by_group("all")

    raw_train_df = load_features_df()
    train_df = build_model_input_df(raw_train_df, feature_columns)

    print("Loading follow-up features (for temporal settings)...")
    raw_test_df_for_temporal = pd.read_csv(
        os.path.join(base_dir, PATHS["features_followup"]), low_memory=False
    )
    temporal_test_df = build_model_input_df(raw_test_df_for_temporal, feature_columns)

    total = len(MODEL_NAMES) * len(SETTING_TYPES)
    step = 0
    result_df_list = []

    for model_name in MODEL_NAMES:
        for setting_type in SETTING_TYPES:
            step += 1
            print(f"[{step}/{total}] Running RQ6 -> Model: {model_name} | Setting: {setting_type}")
            try:
                if setting_type == "single":
                    result_df = run_rq6_single(train_df, model_name)
                elif setting_type == "multi":
                    result_df = run_rq6_multi(train_df, model_name)
                elif setting_type == "unseen":
                    result_df = run_rq6_unseen(train_df, model_name)
                elif setting_type == "temporal":

                    sub_results = []
                    for sub in ("single", "multi", "unseen"):
                        sub_results.append(
                            run_rq6_temporal(train_df, temporal_test_df, model_name, sub)
                        )
                    result_df = pd.concat(sub_results, axis=0, ignore_index=True)
                result_df_list.append(result_df)
            except Exception as e:
                print(f"  Error: {e}")

    final_results = pd.concat(result_df_list, axis=0, ignore_index=True)

    final_results["registry_balance_setting"] = pd.Categorical(
        final_results["registry_balance_setting"],
        categories=["single", "multi", "unseen",
                    "temporal_single", "temporal_multi", "temporal_unseen"],
        ordered=True,
    )
    final_results["model_name"] = pd.Categorical(
        final_results["model_name"], categories=MODEL_NAMES, ordered=True
    )
    final_results = final_results.sort_values(
        by=["registry_balance_setting", "model_name", "test_registry"],
        kind="stable",
    ).reset_index(drop=True)

    return final_results


if __name__ == "__main__":
    final_results = run_rq6()

    output_dir = os.path.join(base_dir, PATHS["evaluation_results"])
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "rq6_results.csv")

    float_cols = final_results.select_dtypes(include=["float64"]).columns
    final_results[float_cols] = final_results[float_cols].round(4)

    final_results.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[Success] RQ6 execution complete. Results saved to: {out_path}")
