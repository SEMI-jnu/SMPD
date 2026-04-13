# SMPD: Experimental Code for SBOM-Based Malicious Package Detection

This repository provides the core experimental code and a simple demo for the paper **"Bridging Registry Fragmentation: SBOM-Based Metadata for Generalized Malicious Package Detection"**.

This study investigates a method for detecting malicious packages in a multi-registry environment using **SBOM (Software Bill of Materials)**-based metadata, instead of relying on registry-specific manifest metadata. This repository is intended to help readers inspect the experimental setup and run a few example cases directly.

## Contents

- SBOM generation code
- SBOM parsing and metadata processing code
- feature engineering code
- experimental scripts for RQ1–RQ6
- sample files for quick inspection
- two representative pretrained models
  - `model_single_npm.pkl`
  - `model_multi_all.pkl`

Considering repository size and distribution scope, only a subset of sample files is provided instead of the full experimental dataset. Access to newly collected malicious package data is not provided in this repository. For research inquiries, please contact 98limgbo@gmail.com.

## Repository Structure

### Data

- `data/01_raw_packages/`: input packages
- `data/02_sbom/`: generated SBOM files
- `data/03_metadata/`: parsed metadata
- `data/04_features/`: feature files for experiments
- `data/05_trained_models/`: demo models and sample files
- `data/06_evaluation_results/`: output directory for experimental results

The following directories contain both `initial/` and `followup/` subsets:
- `data/01_raw_packages/`
- `data/02_sbom/`
- `data/03_metadata/`

Here:
- `initial/`: initial collected dataset
- `followup/`: later collected dataset used for temporal evaluation

### Source Code

- `src/01_sbom_generator/`: generates SBOM files from raw packages
- `src/02_metadata_extractor/`: parses SBOM files and processes metadata
- `src/03_feature_engineer/`: feature generation code
- `src/run_preprocessing.py`: data preprocessing pipeline (Step 1-3 runner)
- `src/04_model/demo.py`: demo script
- `src/04_model/experiments/`: experimental scripts for RQ1–RQ6
- `src/config/`: path and configuration files

## Environment

- Windows
- Python 3.10 or higher

Required packages are listed in `requirements.txt`.

    pip install -r requirements.txt

## ScanCode Setup

This project uses **ScanCode Toolkit** for SBOM generation.

- Official repository: https://github.com/aboutcode-org/scancode-toolkit

Before running the pipeline, install ScanCode Toolkit separately and update the following path in `src/config/pipeline_config.py` to match your local environment.

    SCANCODE_EXECUTABLE = r"YOUR_PATH_TO_SCANCODE\scancode.bat"

For detailed installation and usage instructions, please refer to the official ScanCode Toolkit repository.

## Data Preprocessing

Before running experiments or training new models, you need to generate feature matrices from raw packages.  
You can run the entire preprocessing pipeline (Steps 1 to 3) using a single command.

**To process the initial dataset:**
```bash
python src/run_preprocessing.py --target initial
```

**To process the follow-up dataset (for temporal evaluation):**
```bash
python src/run_preprocessing.py --target followup
```

This script sequentially executes SBOM generation, metadata extraction, and feature engineering.

## Demo

You can run a simple demo using the provided sample files and pretrained models.  
**The demo does not require ScanCode Toolkit to be installed.**

    python src/04_model/demo.py

The demo uses the following two representative models:

- **Single**: a model trained only on npm data
- **Multi**: a model trained on integrated data from npm, PyPI, and RubyGems

## Experimental Scripts

This repository includes experimental scripts corresponding to the main research questions in the paper.

- `rq1.py`: RQ1. Single-Registry Detection
- `rq2.py`: RQ2. Multi-Registry Detection
- `rq3.py`: RQ3. Unseen Registry Generalization
- `rq4.py`: RQ4. Feature Contribution
- `rq5.py`: RQ5. Temporal Robustness
- `rq6.py`: RQ6. Classifier Comparison

Example:

    python src/04_model/experiments/rq1.py

Experimental results are saved to `data/06_evaluation_results/`.

### Contact
If you have questions about this repository or would like to request access to newly collected malicious package data for research purposes, please contact 98limgbo@gmail.com.