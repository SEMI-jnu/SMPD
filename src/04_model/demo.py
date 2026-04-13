import os
import sys
import joblib
import pandas as pd

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(base_dir, "src/04_model/core"))
sys.path.append(os.path.join(base_dir, "src/config"))

from pipeline_config import PATHS
from data_loader import (
    COL_ID, COL_REGISTRY, COL_TARGET,
    get_feature_columns_by_group, build_model_input_df
)

MODELS_DIR   = os.path.join(base_dir, PATHS["trained_models"])
SAMPLES_PATH = os.path.join(MODELS_DIR, "demo_samples.csv")

DEMO_MODELS = {
    "Single (npm only)": "model_single_npm.pkl",
    "Multi  (all reg.)": "model_multi_all.pkl",
}


def print_banner():
    print("=" * 60)
    print("  SMPD — Supply-chain Malicious Package Detector")
    print("  Demo: Malicious Package Detection via SBOM Features")
    print("=" * 60)


def load_sample_df() -> pd.DataFrame:
    if not os.path.exists(SAMPLES_PATH):
        raise FileNotFoundError(
            f"Demo sample file not found: {SAMPLES_PATH}\n"
            "Please run:  python tmp/make_demo_samples.py  first."
        )
    return pd.read_csv(SAMPLES_PATH, low_memory=False)


def run_demo():
    print_banner()

    print("\n[1/3] Loading demo samples...")
    sample_df = load_sample_df()

    feature_columns = get_feature_columns_by_group("all")
    model_input_df = build_model_input_df(sample_df, feature_columns)

    X = model_input_df.drop(columns=[COL_TARGET, COL_ID, COL_REGISTRY], errors="ignore")
    y_true = model_input_df[COL_TARGET].values

    pkg_ids = sample_df[COL_ID].values if COL_ID in sample_df.columns else [f"pkg_{i}" for i in range(len(sample_df))]

    print(f"  Samples : {len(sample_df)} (npm  benign={int((y_true==0).sum())}, malicious={int((y_true==1).sum())})")

    print("\n[2/3] Running predictions...\n")

    for model_label, model_file in DEMO_MODELS.items():
        model_path = os.path.join(MODELS_DIR, model_file)
        if not os.path.exists(model_path):
            print(f"  [{model_label}] Model file not found: {model_path}")
            continue

        model = joblib.load(model_path)
        y_pred = model.predict(X)

        correct = int((y_pred == y_true).sum())
        total   = len(y_true)

        print(f"  ┌─ Model: {model_label}")
        print(f"  │  Accuracy on demo samples: {correct}/{total}")
        print(f"  │")
        print(f"  │  {'Package ID':<45} {'Ground Truth':<15} {'Prediction'}")
        print(f"  │  {'-'*80}")

        for pkg_id, truth, pred in zip(pkg_ids, y_true, y_pred):
            truth_label = "Malicious" if truth == 1 else "Benign   "
            pred_label  = "Malicious" if pred  == 1 else "Benign   "
            match_mark  = "✓" if truth == pred else "✗"
            pkg_display = str(pkg_id)[:43]
            print(f"  │  {pkg_display:<45} {truth_label:<15} {pred_label}  {match_mark}")

        print(f"  └{'─'*80}\n")

    print("[3/3] Demo complete.")
    print("  Both models above were trained using SBOM-derived metadata features.")
    print("  See src/04_model/experiments/ for full RQ1-RQ6 evaluation pipelines.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
