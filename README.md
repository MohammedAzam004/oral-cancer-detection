# Oral Cancer Detection System

A Streamlit-based oral cancer screening app built around an EfficientNetV2S classifier. The project accepts a mouth image, predicts `Cancer detected` or `No cancer detected`, and shows a Grad-CAM attention map to explain which regions influenced the result.

## Project Summary

- Web app for single-image oral cancer screening
- Binary classifier using TensorFlow/Keras
- Grad-CAM visualization for explainability
- Local evaluation and retraining utilities included
- GitHub-ready repository with datasets, secrets, virtual environments, and local experiment outputs ignored

## Current Serving Model

The app currently serves the bundled model at `models/EfficientNetV2S_mouth_cancer.keras`.

Validated inference setup:

- Preprocessing: legacy normalized RGB
- Threshold: `0.46`
- Label meaning: lower score means `cancer`, higher score means `no cancer`

## Measured Accuracy

The current bundled model was evaluated on a cleaned holdout split created from the local dataset added to this workspace.

- Accuracy: `82.9%`
- Precision: `83.3%`
- Recall: `86.2%`
- Specificity: `78.9%`
- F1 score: `84.7%`
- Test samples: `158`

Evaluation notes:

- Clean unique images used: `1052`
- Conflicting duplicate hashes excluded: `1`
- Unreadable images excluded: `10`
- The fine-tuned checkpoint underperformed the bundled model, so the bundled model remains the serving model

The saved metrics report is at `reports/training_report.json`.

## Features

- Clean Streamlit UI for image upload and screening
- Binary prediction with confidence display
- Grad-CAM attention map
- Fixed validated inference pipeline
- Dataset evaluation helpers
- Reproducible training and holdout evaluation script
- Unit tests for predictor and evaluation utilities

## Project Structure

```text
oral-cancer/
  app.py
  train_model.py
  run_app.ps1
  run_app.bat
  requirements.txt
  requirements-dev.txt
  README.md
  CONTRIBUTING.md
  LICENSE
  .env.example
  .gitignore
  .gitattributes
  models/
    EfficientNetV2S_mouth_cancer.keras
  reports/
    training_report.json
  tests/
    test_evaluation.py
    test_predictor.py
  utils/
    evaluation.py
    gradcam.py
    predictor.py
```

## Installation

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

Optional development tools:

```powershell
pip install -r requirements-dev.txt
```

## Run the App

Recommended on Windows:

```powershell
.\run_app.ps1
```

Alternative:

```powershell
venv\Scripts\python.exe -m streamlit run app.py
```

Then open `http://127.0.0.1:8501`.

## Run Tests

```powershell
venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Configuration

The app works with the bundled model by default. You can optionally override settings with environment variables or Streamlit secrets.

Supported settings:

- `MODEL_PATH`
- `METRICS_REPORT_PATH`
- `EVAL_DATASET_DIR`

By default the app uses:

- model: `models/EfficientNetV2S_mouth_cancer.keras`
- metrics report: `reports/training_report.json`

Example configuration files:

- `.env.example`
- `.streamlit/secrets.toml.example`

## Training and Evaluation

The script `train_model.py` can:

- deduplicate images by file hash
- exclude conflicting duplicates
- filter unreadable files
- create stratified train, validation, and test splits
- evaluate the bundled model
- fine-tune the model on the local dataset

Example:

```powershell
venv\Scripts\python.exe train_model.py --epochs 3 --batch-size 8 --learning-rate 0.0001
```

## Repository Notes

This repository is prepared for GitHub publishing:

- local datasets are ignored
- virtual environments are ignored
- Streamlit secrets are ignored
- local training artifacts are ignored
- only the useful shipped model and the final metrics report are kept

## Disclaimer

This project is for research and educational use only. It is not a medical diagnosis system. Any positive result or suspicious image should be reviewed by a qualified clinician.
