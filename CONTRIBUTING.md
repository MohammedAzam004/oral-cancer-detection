# Contributing

Thank you for contributing to this project.

## Development Setup

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Run the App

```powershell
.\run_app.ps1
```

Alternative:

```powershell
venv\Scripts\python.exe -m streamlit run app.py
```

## Run Tests

```powershell
venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Contribution Guidelines

- Keep changes focused and easy to review
- Follow the existing project structure and naming style
- Add or update tests when behavior changes
- Update `README.md` when user-facing behavior changes
- Keep the UI simple and clinically readable

## Do Not Commit

- local datasets
- virtual environments
- Streamlit secrets
- temporary files and caches
- local training artifacts that are not part of the shipped app

## Pull Request Checklist

- the app starts successfully
- tests pass locally
- no secrets or datasets are included
- documentation is updated when needed

## Notes

This repository is for research and educational use. Model outputs are screening signals only and must not be presented as a clinical diagnosis.
