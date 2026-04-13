import sys
import os
import json
import pandas as pd
import argparse

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(base_dir, "src/03_feature_engineer"))
sys.path.append(os.path.join(base_dir, "src/config"))

from preprocess_functions import *
from pipeline_config import PATHS
from features_config import sbom_all_features

def run_feature_engineering(input_path=None, output_path=None):
    if input_path is None:
        input_path = os.path.join(base_dir, PATHS["metadata_output"], "extracted_metadata.csv")
    
    if output_path is None:
        output_dir = os.path.join(base_dir, PATHS["features_output"])
        output_file = os.path.join(output_dir, "features.csv")
    elif output_path.endswith(".csv"):
        output_file = output_path
        output_dir = os.path.dirname(output_file)
    else:
        output_dir = output_path
        output_file = os.path.join(output_dir, "features.csv")

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_path):
        print(f"Error: Metadata input not found: {input_path}")
        return

    print(f"Loading metadata from: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
 
    
    concatenate_columns(df, "name", "namespace") 
    calculate_ratio(df, "duration", "files_count", "duration_per_file")

    target_column = "name"
    binary_classification(df, target_column)
    count_characters(df, target_column)
    count_special_characters_for_name(df, target_column, category_column="registry", category_skip_map={"npm": 1}, default_skip_n=0)
    calculate_ratio(df, f"{target_column}_special_count", f"{target_column}_length", f"{target_column}_special_count_ratio")

    target_column = "version"
    binary_classification(df, target_column)
    split_version(df, target_column)

    target_column = "description"
    binary_classification(df, target_column)
    count_characters(df, target_column)
    count_special_characters(df, target_column)
    calculate_ratio(df, f"{target_column}_special_count", f"{target_column}_length", f"{target_column}_special_count_ratio")

    target_column = "parties_name"
    binary_classification_for_list(df, target_column)
    count_list_values(df, target_column)
    calculate_ratio(df, f"{target_column}_count", "parties_count", "parties_name_exist_ratio")
    analyze_list_string_lengths(df, target_column)

    target_column = "parties_role"
    count_list_unique_values(df, target_column)

    target_column = "parties_email"
    analyze_list_string_lengths(df, target_column)

    target_column = "declared_holder"
    binary_classification(df, target_column)
    count_characters(df, target_column)
    count_special_characters(df, target_column)
    calculate_ratio(df, f"{target_column}_special_count", f"{target_column}_length", f"{target_column}_special_count_ratio")

    target_column = "other_holders"
    binary_classification_for_list(df, target_column)
    count_list_values(df, target_column)
    analyze_list_string_lengths(df, target_column)

    target_column = "other_holders_nullcount"
    calculate_ratio(df, f"{target_column}", "files_count", f"{target_column}_ratio")

    target_column = "declared_license_expression"
    binary_classification(df, target_column)
    count_characters(df, target_column)
    target_column = process_license_expression(df, target_column)
    binary_classification_for_list(df, target_column)
    count_list_values(df, target_column)
    analyze_list_string_lengths(df, target_column)
    target_column = map_license_groups(df, target_column)
    binary_classify_license_groups(df, target_column)

    target_column = "other_license_expressions"
    binary_classification_for_list(df, target_column)
    count_list_values(df, target_column)
    analyze_list_string_lengths(df, target_column)
    target_column = process_license_expression(df, target_column)
    binary_classification_for_list(df, target_column)
    count_list_values(df, target_column)
    analyze_list_string_lengths(df, target_column)
    target_column = map_license_groups(df, target_column)
    binary_classify_license_groups(df, target_column)

    target_column = "other_license_expressions_nullcount"
    calculate_ratio(df, f"{target_column}", "files_count", f"{target_column}_ratio")

    target_column = "other_languages"
    binary_classification_for_list(df, target_column)
    count_list_values(df, target_column)

    target_column = "other_languages_file_count"
    calculate_ratio(df, f"{target_column}", "files_count", f"{target_column}_ratio")

    target_column = "keywords"
    binary_classification_for_list(df, target_column)
    count_list_values(df, target_column)
    analyze_list_string_lengths(df, target_column)

    url_columns = [
        "homepage_url", "download_url", "bug_tracking_url", "code_view_url", 
        "vcs_url", "repository_homepage_url"
    ]
    for url_col in url_columns:
        binary_classification(df, url_col)
        count_characters(df, url_col)

    target_column = "dependencies_purl"
    binary_classification_for_list(df, target_column)
    count_list_values(df, target_column)
    analyze_list_string_lengths(df, target_column)

    target_column = "dependencies_scope"
    binary_classification_for_list(df, target_column)
    count_list_values(df, target_column)
    count_list_unique_values(df, target_column)
    
    target_column = "dependencies_extracted_requirement"
    binary_classification_for_list(df, target_column)
    count_list_values(df, target_column)
    calculate_ratio(df, f"{target_column}_count", "dependencies_count", f"{target_column}_ratio")

    target_column = "dependencies_is_runtime"
    binary_classification_for_list(df, target_column)
    count_list_values(df, target_column)
    calculate_ratio(df, f"{target_column}_count", "dependencies_count", f"{target_column}_ratio")

    target_column = "dependencies_is_optional"
    binary_classification_for_list(df, target_column)
    count_list_values(df, target_column)
    calculate_ratio(df, f"{target_column}_count", "dependencies_count", f"{target_column}_ratio")


    print(f"Applying sbom_all_features selection... (Expected {len(sbom_all_features)} features)")
    
    missing_cols = [col for col in sbom_all_features if col not in df.columns]
    if missing_cols:
         print(f"[Warning] Missing {len(missing_cols)} features. Filling them with NaN. Lists: {missing_cols}")
         for m in missing_cols:
              df[m] = pd.NA

    processed_df = df[sbom_all_features].copy()

    print(f"Saving structured feature matrix {processed_df.shape} to {output_file}")
    processed_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print("Success! (Feature Engineering Pipeline Completed)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Feature Engineering on extracted metadata.")
    parser.add_argument("--input", help="Path to extracted_metadata.csv")
    parser.add_argument("--output", help="Path to save features.csv (file or directory)")
    args = parser.parse_args()

    run_feature_engineering(input_path=args.input, output_path=args.output)
