"""
SMPD Preprocessing Pipeline (Data Preparation)
==============================================
Runs the entire preprocessing flow: Step 1 (Scan) -> Step 2 (Extract) -> Step 3 (Engineer).
This script prepares the feature CSV files required for model training and evaluation.

Usage:
    python src/run_preprocessing.py --target initial
    python src/run_preprocessing.py --target followup
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run the full SMPD pipeline (Step 1 to 3).")
    parser.add_argument(
        "--target", 
        choices=["initial", "followup"], 
        default="initial",
        help="Target dataset to process (initial or followup)"
    )
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    sys.path.append(os.path.join(base_dir, "src/config"))
    from pipeline_config import PATHS

    target = args.target
    print(f"\n{'='*60}")
    print(f"  Starting SMPD Pipeline for: [{target.upper()}]")
    print(f"{'='*60}\n")

    if target == "initial":
        raw_dir      = os.path.join(base_dir, PATHS["raw_packages"])
        sbom_dir     = os.path.join(base_dir, PATHS["sbom_output"])
        meta_dir     = os.path.join(base_dir, PATHS["metadata_output"])
        feature_path = os.path.join(base_dir, PATHS["features_output"], "features.csv")
    else:
        raw_dir      = os.path.join(base_dir, PATHS["raw_packages_followup"])
        sbom_dir     = os.path.join(base_dir, PATHS["sbom_followup"])
        meta_dir     = os.path.join(base_dir, PATHS["metadata_followup"])
        feature_path = os.path.join(base_dir, PATHS["features_followup"])

    scripts = {
        "step1": os.path.join(base_dir, "src/01_sbom_generator/scancode.py"),
        "step2": os.path.join(base_dir, "src/02_metadata_extractor/extractor.py"),
        "step3": os.path.join(base_dir, "src/03_feature_engineer/feature_engineer.py")
    }

    def run_step(name, cmd):
        print(f"\n>>> Running {name}...")
        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"\n[Error] {name} failed with return code {result.returncode}")
            sys.exit(1)

    # Step 1: SBOM Generation
    run_step("Step 1 (SBOM Generation)", [
        sys.executable, scripts["step1"],
        "--input", raw_dir,
        "--output", sbom_dir
    ])

    # Step 2: Metadata Extraction
    run_step("Step 2 (Metadata Extraction)", [
        sys.executable, scripts["step2"],
        "--input", sbom_dir,
        "--output", meta_dir
    ])

    # Step 3: Feature Engineering
    meta_csv = os.path.join(meta_dir, "extracted_metadata.csv")
    run_step("Step 3 (Feature Engineering)", [
        sys.executable, scripts["step3"],
        "--input", meta_csv,
        "--output", feature_path
    ])

    print(f"\n{'='*60}")
    print(f"  Pipeline successfully completed for [{target}]")
    print(f"  Final features saved to: {feature_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
