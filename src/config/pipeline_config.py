"""
Pipeline Configuration Settings
"""
SCANCODE_EXECUTABLE = r"C:\Projects\ScanCode\scancode.bat"
SCANCODE_TIMEOUT = 60

PATHS = {
    # Initial Dataset
    "raw_packages":       "data/01_raw_packages/initial",
    "sbom_output":        "data/02_sbom/initial",
    "metadata_output":    "data/03_metadata/initial",
    "features_output":    "data/04_features",
    "trained_models":     "data/05_trained_models",
    "evaluation_results": "data/06_evaluation_results",
    # Follow-up Dataset
    "raw_packages_followup":  "data/01_raw_packages/followup",
    "sbom_followup":          "data/02_sbom/followup",
    "metadata_followup":      "data/03_metadata/followup",
    "features_followup":      "data/04_features/features_followup.csv",
}
