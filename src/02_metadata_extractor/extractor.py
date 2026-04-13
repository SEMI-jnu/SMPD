import os
import json
import pandas as pd
import argparse
import hashlib

def extract_headers_duration(data):
    try:
        headers = data.get("headers", [])
        if not isinstance(headers, list) or not headers:
            return {"duration": None}
        return {"duration": headers[0].get("duration", None)}
    except (KeyError, TypeError, IndexError):
        return {"duration": None}

def extract_headers_files_count(data):
    try:
        headers = data.get("headers", [])
        if not isinstance(headers, list) or not headers:
            return {"files_count": None}
        return {"files_count": headers[0].get("extra_data", {}).get("files_count", None)}
    except (KeyError, TypeError, IndexError):
        return {"files_count": None}

def extract_summary_declared_holder(data):
    try:
        summary = data.get("summary", {})
        if not isinstance(summary, dict):
            return {"declared_holder": None}
        return {"declared_holder": summary.get("declared_holder", None)}
    except (KeyError, TypeError):
        return {"declared_holder": None}

def extract_summary_other_holders(data):
    try:
        holders = data.get("summary", {}).get("other_holders", [])
        if not isinstance(holders, list):
            holders = []
        non_null_values = [entry.get("value") for entry in holders if entry.get("value") is not None]
        non_null_count = sum(entry.get("count", 0) for entry in holders if entry.get("value") is not None)
        null_count = sum(entry.get("count", 0) for entry in holders if entry.get("value") is None)
        return {
            "other_holders": non_null_values,
            "other_holders_count": non_null_count,
            "other_holders_nullcount": null_count,
        }
    except (KeyError, TypeError, IndexError):
        return {"other_holders": [], "other_holders_count": 0, "other_holders_nullcount": 0}

def extract_summary_other_license_expressions(data):
    try:
        expressions = data.get("summary", {}).get("other_license_expressions", [])
        if not isinstance(expressions, list):
            expressions = []
        non_null_values = [entry.get("value") for entry in expressions if entry.get("value") is not None]
        non_null_count = sum(entry.get("count", 0) for entry in expressions if entry.get("value") is not None)
        null_count = sum(entry.get("count", 0) for entry in expressions if entry.get("value") is None)
        return {
            "other_license_expressions": non_null_values,
            "other_license_expressions_count": non_null_count,
            "other_license_expressions_nullcount": null_count,
        }
    except (KeyError, TypeError, IndexError):
        return {"other_license_expressions": [], "other_license_expressions_count": 0, "other_license_expressions_nullcount": 0}

def extract_summary_other_languages(data):
    try:
        languages = data.get("summary", {}).get("other_languages", [])
        if not isinstance(languages, list):
            languages = []
        language_list = [entry.get("value") for entry in languages if entry.get("value") is not None]
        total_count = sum(entry.get("count", 0) for entry in languages)
        return {
            "other_languages": language_list,
            "other_languages_file_count": total_count,
        }
    except (KeyError, TypeError, IndexError):
        return {"other_languages": [], "other_languages_file_count": 0}

def extract_packages_basic_info(data):
    try:
        packages = data.get("packages", [])
        if not isinstance(packages, list) or not packages:
            return {"type": None, "primary_language": None, "namespace": None, "name": None, "version": None, "description": None}
        
        package = packages[0]
        namespace = package.get("namespace", None)
        name = package.get("name", None)
        version = package.get("version", None)
        
        return {
            "registry": package.get("type", None),
            "primary_language": package.get("primary_language", None),
            "namespace": namespace,
            "name": name,
            "version": version,
            "description": package.get("description", None),
        }
    except (KeyError, TypeError, IndexError):
        return {"registry": None, "primary_language": None, "namespace": None, "name": None, "version": None, "description": None}

def extract_packages_parties(data):
    try:
        packages = data.get("packages", [])
        if not isinstance(packages, list) or not packages:
            return {"parties_count": 0, "parties_name": [], "parties_type": [], "parties_role": [], "parties_email": [], "parties_url": []}

        package = packages[0]
        parties = package.get("parties", [])
        if not isinstance(parties, list):
            parties = []

        return {
            "parties_count": len(parties),
            "parties_name": [entry.get("name", None) for entry in parties],
            "parties_type": [entry.get("type", None) for entry in parties],
            "parties_role": [entry.get("role", None) for entry in parties],
            "parties_email": [entry.get("email", None) for entry in parties],
            "parties_url": [entry.get("url", None) for entry in parties],
        }
    except (KeyError, TypeError, IndexError):
        return {"parties_count": 0, "parties_name": [], "parties_type": [], "parties_role": [], "parties_email": [], "parties_url": []}

def extract_packages_detail_info(data):
    fields = ["qualifiers", "subpath", "release_date", "keywords", "homepage_url", "download_url", 
              "bug_tracking_url", "code_view_url", "vcs_url", "declared_license_expression", 
              "declared_license_expression_spdx", "is_private", "is_virtual", "repository_homepage_url", 
              "repository_download_url", "api_data_url", "package_uid", "purl"]
    try:
        packages = data.get("packages", [])
        if not isinstance(packages, list) or not packages:
            return {field: None for field in fields}
        package = packages[0]
        return {field: package.get(field, None) for field in fields}
    except (KeyError, TypeError, IndexError):
        return {field: None for field in fields}

def extract_dependencies_count(data):
    try:
        dependencies = data.get("dependencies", [])
        if not isinstance(dependencies, list):
            dependencies = []

        return {
            "dependencies_count": len(dependencies),
            "dependencies_purl_count": sum(1 for entry in dependencies if entry.get("purl")),
            "dependencies_extracted_requirement_count": sum(1 for entry in dependencies if entry.get("extracted_requirement")),
            "dependencies_scope_count": sum(1 for entry in dependencies if entry.get("scope")),
            "dependencies_scope_unique_count": len(set(entry.get("scope") for entry in dependencies if entry.get("scope"))),
            "dependencies_is_runtime_count": sum(1 for entry in dependencies if entry.get("is_runtime", False)),
            "dependencies_is_optional_count": sum(1 for entry in dependencies if entry.get("is_optional", False)),
            "dependencies_is_pinned_count": sum(1 for entry in dependencies if entry.get("is_pinned", False)),
            "dependencies_is_direct_count": sum(1 for entry in dependencies if entry.get("is_direct", False)),
        }
    except (KeyError, TypeError, IndexError):
        return {
            "dependencies_count": 0, "dependencies_purl_count": 0, "dependencies_extracted_requirement_count": 0,
            "dependencies_scope_count": 0, "dependencies_scope_unique_count": 0, "dependencies_is_runtime_count": 0,
            "dependencies_is_optional_count": 0, "dependencies_is_pinned_count": 0, "dependencies_is_direct_count": 0,
        }

def extract_dependencies_list(data):
    try:
        dependencies = data.get("dependencies", [])
        if not isinstance(dependencies, list):
            dependencies = []
        return {
            "dependencies_purl": [entry.get("purl", None) for entry in dependencies],
            "dependencies_scope": [entry.get("scope", None) for entry in dependencies],
            "dependencies_extracted_requirement": [entry.get("extracted_requirement", None) for entry in dependencies],
            "dependencies_is_runtime": [entry.get("purl", None) for entry in dependencies if entry.get("is_runtime", False)],
            "dependencies_is_optional": [entry.get("purl", None) for entry in dependencies if entry.get("is_optional", False)],
            "dependencies_is_pinned": [entry.get("purl", None) for entry in dependencies if entry.get("is_pinned", False)],
            "dependencies_is_direct": [entry.get("purl", None) for entry in dependencies if entry.get("is_direct", False)],
        }
    except (KeyError, TypeError, IndexError):
        return {
            "dependencies_purl": [], "dependencies_scope": [], "dependencies_extracted_requirement": [],
            "dependencies_is_runtime": [], "dependencies_is_optional": [], "dependencies_is_pinned": [], "dependencies_is_direct": []
        }

import sys

def main():
    parser = argparse.ArgumentParser(description="Extract metadata from SBOM JSON files.")
    parser.add_argument("--input", help="Directory containing SBOM JSON files")
    parser.add_argument("--output", help="Directory to save extracted metadata CSV")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    sys.path.append(os.path.join(base_dir, "src/config"))
    from pipeline_config import PATHS

    input_dir = args.input if args.input else os.path.join(base_dir, PATHS["sbom_output"])
    output_dir = args.output if args.output else os.path.join(base_dir, PATHS["metadata_output"])
    output_file = os.path.join(output_dir, "extracted_metadata.csv")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Input directory: {input_dir}")
    print(f"Output metadata file: {output_file}")

    if not os.path.exists(input_dir):
        print("Error: Input SBOM directory does not exist.")
        return

    data_list = []
    error_logs = []

    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    print(f"Found {len(json_files)} JSON files to process.")

    for filename in json_files:
        file_path = os.path.join(input_dir, filename)
        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            extracted_data = {
                "id": file_hash,
                "malicious": -1
            }

            extracted_data.update(extract_packages_basic_info(data))
            extracted_data.update(extract_packages_parties(data))
            extracted_data.update(extract_headers_duration(data))
            extracted_data.update(extract_headers_files_count(data))
            extracted_data.update(extract_summary_declared_holder(data))
            extracted_data.update(extract_summary_other_holders(data))
            extracted_data.update(extract_summary_other_license_expressions(data))
            extracted_data.update(extract_summary_other_languages(data))
            extracted_data.update(extract_packages_detail_info(data))
            extracted_data.update(extract_dependencies_count(data))
            extracted_data.update(extract_dependencies_list(data))

            data_list.append(extracted_data)

        except Exception as e:
            error_msg = f"Error processing {filename}: {str(e)}"
            print(f" -> {error_msg}")
            error_logs.append(error_msg)

    if data_list:
        df = pd.DataFrame(data_list)
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\nSuccessfully extracted metadata for {len(data_list)} packages.")
        print(f"Saved to: {output_file}")
    else:
        print("\nNo metadata was extracted.")

    if error_logs:
        log_file = os.path.join(output_dir, "extractor_error_log.txt")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(error_logs))
        print(f"Check {log_file} for error details.")

if __name__ == "__main__":
    main()
