from __future__ import annotations
import os
import sys
import pandas as pd
from pathlib import Path

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(os.path.join(base_dir, "src/config"))

from pipeline_config import PATHS
from features_config import (
    sbom_general_information,
    sbom_people,
    sbom_license,
    sbom_dependency,
    sbom_url,
    sbom_all_features,
    META_COLUMNS,
    COL_ID,
    COL_REGISTRY,
    COL_TARGET
)

def validate_required_meta_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in META_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required meta columns in DataFrame: {', '.join(missing_columns)}")


def load_features_df() -> pd.DataFrame:

    file_path = os.path.join(base_dir, PATHS["features_output"], "features.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Feature file not found: {file_path}")

    df = pd.read_csv(file_path, low_memory=False)
    validate_required_meta_columns(df)

    return df.sort_values(by=[COL_TARGET, COL_REGISTRY, COL_ID]).reset_index(drop=True)


def filter_by_registry(df: pd.DataFrame, registry_name: str) -> pd.DataFrame:

    validate_required_meta_columns(df)
    
    filtered_df = df[df[COL_REGISTRY] == registry_name].copy().reset_index(drop=True)
    if filtered_df.empty:
        raise ValueError(f"No data found for registry: '{registry_name}'")
        
    return filtered_df


def get_feature_columns_by_group(group_name: str) -> list[str]:

    mapping = {
        "all": [c for c in sbom_all_features if c not in META_COLUMNS],
        "general": sbom_general_information,
        "people": sbom_people,
        "license": sbom_license,
        "dependency": sbom_dependency,
        "url": sbom_url
    }
    
    if group_name not in mapping:
        raise ValueError(f"Unsupported feature group: '{group_name}'. \nAvailable groups: {list(mapping.keys())}")
        
    return mapping[group_name]


def build_model_input_df(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:

    validate_required_meta_columns(df)
    
    selected_columns = []
    for column in META_COLUMNS + feature_columns:
        if column not in selected_columns:
            selected_columns.append(column)
            
    missing_columns = [column for column in selected_columns if column not in df.columns]
    if missing_columns:
         raise ValueError(f"Requested feature columns missing in DataFrame: {', '.join(missing_columns)}")
         
    return df[selected_columns].copy()


def get_available_registries(df: pd.DataFrame) -> list[str]:
    
    validate_required_meta_columns(df)
    return sorted(df[COL_REGISTRY].dropna().unique().tolist())
