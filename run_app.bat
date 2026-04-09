@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "PYTHON_EXE=%PROJECT_ROOT%venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%PROJECT_ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo Could not find a project virtual environment.
    echo Expected venv\Scripts\python.exe or .venv\Scripts\python.exe
    exit /b 1
)

"%PYTHON_EXE%" -m streamlit run "%PROJECT_ROOT%app.py"
