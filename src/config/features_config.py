"""
SBOM Features Configuration
"""

sbom_general_information = [
    'name_exist', 'name_length', 'name_special_exist', 'name_special_count', 'name_special_count_ratio',
    'version_exist', 'version_major', 'version_minor', 'version_patch',
    'description_exist', 'description_length', 'description_special_exist', 'description_special_count', 'description_special_count_ratio',
    'keywords_exist', 'keywords_count', 'keywords_length_avg',
    'duration', 'duration_per_file',
    'files_count', 'other_languages_exist', 'other_languages_count', 'other_languages_file_count', 'other_languages_file_count_ratio'
]

sbom_people = [
    'parties_count', 'parties_name_exist', 'parties_name_exist_ratio', 'parties_name_length_avg', 'parties_role_unique_count',
    'parties_email_length_avg',
    'declared_holder_exist', 'declared_holder_length', 'declared_holder_special_exist', 'declared_holder_special_count', 'declared_holder_special_count_ratio',
    'other_holders_exist', 'other_holders_count', 'other_holders_length_avg', 'other_holders_nullcount', 'other_holders_nullcount_ratio'
]

sbom_license = [
    'declared_license_expression_exist', 'declared_license_expression_length',
    'declared_license_expression_split_count', 'declared_license_expression_split_length_avg',
    'declared_license_expression_split_mapped_mit_exist', 'declared_license_expression_split_mapped_apache_exist', 'declared_license_expression_split_mapped_isc_exist', 'declared_license_expression_split_mapped_bsd_exist',
    'other_license_expressions_exist', 'other_license_expressions_count', 'other_license_expressions_length_avg', 'other_license_expressions_unknown_exist', 'other_license_expressions_nullcount', 'other_license_expressions_nullcount_ratio',
    'other_license_expressions_split_exist', 'other_license_expressions_split_count', 'other_license_expressions_split_length_avg',
    'other_license_expressions_split_mapped_mit_exist', 'other_license_expressions_split_mapped_bsd_exist', 'other_license_expressions_split_mapped_apache_exist', 'other_license_expressions_split_mapped_gpl_exist'
]

sbom_dependency = [
    'dependencies_count', 'dependencies_purl_exist', 'dependencies_purl_length_avg', 'dependencies_scope_unique_count',
    'dependencies_is_runtime_exist', 'dependencies_is_runtime_count', 'dependencies_is_runtime_ratio',
    'dependencies_is_optional_exist', 'dependencies_is_optional_count', 'dependencies_is_optional_ratio',
    'dependencies_extracted_requirement_exist', 'dependencies_extracted_requirement_count', 'dependencies_extracted_requirement_ratio'
]

sbom_url = [
    'homepage_url_exist', 'homepage_url_length',
    'download_url_exist', 'download_url_length',
    'bug_tracking_url_exist', 'bug_tracking_url_length',
    'repository_homepage_url_exist', 'repository_homepage_url_length',
    'code_view_url_exist', 'code_view_url_length', 'vcs_url_exist', 'vcs_url_length'
]

# Metadata Columns
COL_ID = "id"
COL_REGISTRY = "registry"
COL_TARGET = "malicious"

META_COLUMNS = [COL_ID, COL_REGISTRY, COL_TARGET]

sbom_other = [COL_TARGET, COL_REGISTRY, COL_ID]

sbom_all_features = (
    sbom_general_information + 
    sbom_people + 
    sbom_license + 
    sbom_dependency + 
    sbom_url + 
    sbom_other
)

# Total Valid Target Features: 89
