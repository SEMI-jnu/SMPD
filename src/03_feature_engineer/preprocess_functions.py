import os
import pandas as pd
import re
import ast

def concatenate_columns(df, target_column, prefix_column):
    df[target_column] = df[prefix_column].fillna('') + df[target_column]

def binary_classification(df, target_column, invalid_values=None):

    if invalid_values is None:
        invalid_values = {"unknown", "", " ", "none", "null"}

    binary_column = f"{target_column}_exist"

    df[binary_column] = df[target_column].apply(
        lambda x: 1 if (pd.notna(x) and (x == 0 or (isinstance(x, str) and x.strip().lower() not in invalid_values))) else 0
    )

def count_characters(df, target_column, invalid_values=None):

    if invalid_values is None:
        invalid_values = {"unknown", "", " ", "none", "null", "0"}

    processed_column_name = f"{target_column}_length"

    df[processed_column_name] = df[target_column].apply(
        lambda x: len(str(x).strip()) if pd.notna(x) and str(x).strip().lower() not in invalid_values else 0
    )

def count_special_characters(df, target_column):

    special_count_column = f"{target_column}_special_count"
    special_binary_column = f"{target_column}_special_exist"

    df[special_count_column] = df[target_column].astype(str).apply(
        lambda x: len(re.findall(r"[^a-zA-Z0-9\s]", x)) if pd.notna(x) else 0
    )

    df[special_binary_column] = (df[special_count_column] > 0).astype(int)

def count_special_characters_for_name(df, target_column, category_column, category_skip_map, default_skip_n=0):

    special_count_column = f"{target_column}_special_count"
    special_binary_column = f"{target_column}_special_exist"

    def count_special_chars(row):
        skip_n = category_skip_map.get(row[category_column], default_skip_n)
        text = str(row[target_column])
        return len(re.findall(r"[^a-zA-Z0-9\s]", text[skip_n:])) if pd.notna(text) and len(text) > skip_n else 0

    df[special_count_column] = df.apply(count_special_chars, axis=1)

    df[special_binary_column] = (df[special_count_column] > 0).astype(int)

def calculate_ratio(df, numerator_column, denominator_column, result_column, zero_denominator_value=-1, decimal_places=4):

    df[numerator_column] = pd.to_numeric(df[numerator_column], errors='coerce')
    df[denominator_column] = pd.to_numeric(df[denominator_column], errors='coerce')

    ratio = df[numerator_column] / df[denominator_column]

    ratio[(df[numerator_column] == 0) & (df[denominator_column] == 0)] = 0

    ratio[(df[numerator_column] != 0) & (df[denominator_column] == 0)] = zero_denominator_value

    df[result_column] = ratio.round(decimal_places)

def split_version(df, target_column, invalid_values=None, max_limits=None):

    num_parts = 3 
    column_suffixes = ["_major", "_minor", "_patch"]
    
    if invalid_values is None:
        invalid_values = {"unknown", "", " ", "none", "null"}

    if max_limits is None:
        max_limits = [100] * num_parts

    split_columns = {f"{target_column}{suffix}": [] for suffix in column_suffixes}

    for value in df[target_column]:
        if pd.isna(value) or (isinstance(value, str) and value.strip().lower() in invalid_values):
            split_values = [0] * num_parts
        else:
            parts = str(value).split(".")
            split_values = [int(part) if part.isdigit() else 0 for part in parts[:num_parts]]
            split_values += [0] * (num_parts - len(parts))
            split_values = [min(value, max_limits[i]) for i, value in enumerate(split_values)]

        for i, suffix in enumerate(column_suffixes):
            split_columns[f"{target_column}{suffix}"].append(split_values[i])

    for col_name, col_values in split_columns.items():
        df[col_name] = col_values

def binary_classification_for_list(df, target_column):

    result_column = f"{target_column}_exist"
    
    invalid_values = {None, "", " ", "none", "null", "NaN", "nan", "NAN"}

    def process_list(x):
        if isinstance(x, str):
            try:
                x = ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return 0
        if isinstance(x, list):
            return 1 if any(str(item).strip() not in invalid_values for item in x) else 0
        return 0

    df[result_column] = df[target_column].apply(process_list)

def count_list_values(df, target_column):

    result_column = f"{target_column}_count"
    
    invalid_values = {None, "", " ", "none", "null", "NaN", "nan", "NAN"}

    def process_list(x):
        if isinstance(x, str):
            try:
                x = ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return 0
        if isinstance(x, list):
            return sum(1 for item in x if str(item).strip() not in invalid_values)
        return 0

    df[result_column] = df[target_column].apply(process_list)

def count_list_unique_values(df, target_column):

    result_column = f"{target_column}_unique_count"
    
    invalid_values = {None, "", " ", "none", "null", "NaN", "nan", "NAN"}

    def process_list(x):
        if isinstance(x, str):
            try:
                x = ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return 0
        if isinstance(x, list):
            return len(set(item for item in x if str(item).strip() not in invalid_values))
        return 0

    df[result_column] = df[target_column].apply(process_list)

def analyze_list_string_lengths(df, target_column, decimal_places=4):

    result_avg = f"{target_column}_length_avg"
    result_max = f"{target_column}_length_max"
    result_min = f"{target_column}_length_min"
    

    invalid_values = {None, "", " ", "none", "null", "NaN", "nan", "NAN"}

    def compute_lengths(x):

        if isinstance(x, str):
            try:
                x = ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return 0, 0, 0
        if isinstance(x, list):
            valid_lengths = [
                len(str(item)) if str(item).strip() not in invalid_values else 0
                for item in x
            ]
            if valid_lengths:
                avg_length = round(sum(valid_lengths) / len(valid_lengths), decimal_places)
                max_length = round(max(valid_lengths), decimal_places)
                min_length = round(min(valid_lengths), decimal_places)
                return avg_length, max_length, min_length
        return 0, 0, 0

    df[[result_avg, result_max, result_min]] = df[target_column].apply(lambda x: pd.Series(compute_lengths(x)))

def process_license_expression(df, target_column):

    single_column = f"{target_column}_single"
    multiple_column = f"{target_column}_multiple"
    split_column = f"{target_column}_split"
    unknown_column = f"{target_column}_unknown_exist"

    unknown_values = {"unknown", "unknown-license-reference"}

    def clean_license(license_str):
        return re.sub(r"[\(\)]", "", license_str).strip().lower()

    def convert_to_list(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            try:
                parsed_value = ast.literal_eval(x)
                if isinstance(parsed_value, list):
                    return parsed_value
            except (ValueError, SyntaxError):
                return [x]
        return [x] if pd.notna(x) else []

    temp_license_data = df[target_column].apply(convert_to_list)

    def classify_licenses(license_list):
        single_licenses = []
        multiple_licenses = []
        split_licenses = set()
        unknown_flag = 0

        if not isinstance(license_list, list):
            return [], [], [], 0

        for license_str in license_list:
            if isinstance(license_str, str):
                if any(op in license_str for op in [" AND ", " OR "]):
                    multiple_licenses.append(license_str)
                else:
                    single_licenses.append(license_str)

                cleaned_parts = [clean_license(part) for part in re.split(r" AND | OR ", license_str)]
                split_licenses.update(cleaned_parts)

                if any(part in unknown_values for part in cleaned_parts):
                    unknown_flag = 1

        split_licenses = list(filter(lambda x: x not in unknown_values, split_licenses))

        return single_licenses, multiple_licenses, split_licenses, unknown_flag

    df[[single_column, multiple_column, split_column, unknown_column]] = temp_license_data.apply(
        lambda x: pd.Series(classify_licenses(x))
    )
    return f"{target_column}_split"

def map_license_groups(df, target_column, prefix_mapping=None, result_column=None):

    if result_column is None:
        result_column = f"{target_column}_mapped"

    if prefix_mapping is None:
        prefix_mapping = {
            "lgpl": ["lgpl"],
            "gpl": ["gpl"],
            "mit": ["mit"],
            "apache": ["apache"],
            "bsd": ["bsd"],
            "cc": ["cc"],
            "mpl": ["mpl"],
            "isc": ["isc"],
            "zlib": ["zlib"],
            "public-domain": ["public-domain", "unlicense"]
        }

    def convert_to_list(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            try:
                parsed = ast.literal_eval(x)
                return parsed if isinstance(parsed, list) else [parsed]
            except (ValueError, SyntaxError):
                return [x]
        return []

    def map_licenses(license_list):
        mapped = set()
        for license_str in license_list:
            license_str = str(license_str).lower().strip()
            for group, prefixes in prefix_mapping.items():
                if any(license_str.startswith(pfx) for pfx in prefixes):
                    mapped.add(group)
        return list(mapped)

    df[result_column] = df[target_column].apply(convert_to_list).apply(map_licenses)

    return result_column

def binary_classify_license_groups(df, target_column):

    all_groups = set(group for sublist in df[target_column] if isinstance(sublist, list) for group in sublist)

    for group in sorted(all_groups):
        exist_column = f"{target_column}_{group}_exist"
        df[exist_column] = df[target_column].apply(
            lambda x: 1 if isinstance(x, list) and group in x else 0
        )

def classify_author_existence(df, target_column):

    result_column = f"{target_column}_author_exist"

    def has_author(x):
        if isinstance(x, str):
            try:
                x = ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return 0
        if isinstance(x, list):
            return 1 if 'author' in x else 0
        return 0

    df[result_column] = df[target_column].apply(has_author)