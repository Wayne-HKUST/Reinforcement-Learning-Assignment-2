@echo off

set SCRIPTS_DIR=%~dp0
set ROOT_DIR=%~dp0..
set CONDA_DIR=%ROOT_DIR%\.conda
set VENV_DIR=%ROOT_DIR%\.venv
set CONDA=%CONDA_DIR%\Scripts\conda.exe
set PIP=%VENV_DIR%\Scripts\pip.exe
set PIP_PATH=%VENV_DIR%\Scripts\pip.exe
set PATH=%PATH%;%VENV_DIR%\Scripts
set MINICONDA_VERSION=py312_24.11.1-0
set MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-%MINICONDA_VERSION%-Windows-x86_64.exe
set PYTHON_VERSION=3.9
set PATH=%PATH%;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x64

if not exist "%CONDA_DIR%" (
    mkdir "%CONDA_DIR%"
    echo Downloading Miniconda installer version: %MINICONDA_VERSION%...
    powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%MINICONDA_URL%', '%TEMP%\Miniconda3-%MINICONDA_VERSION%-Windows-x86_64.exe')"
    if not exist "%TEMP%\Miniconda3-%MINICONDA_VERSION%-Windows-x86_64.exe" (
        echo Failed to download the Miniconda installer. Please check your network connection.
        pause
        exit /b 1
    )
    echo Installing Miniconda, please wait...
    start /wait "" "%TEMP%\Miniconda3-%MINICONDA_VERSION%-Windows-x86_64.exe" /InstallationType=JustMe /RegisterPython=0 /S /D=%CONDA_DIR%
    if not exist "%CONDA_DIR%\Scripts\conda.exe" (
        echo Miniconda installation failed. Please check the error information.
        pause
        exit /b 1
    )
    @REM %CONDA% install -y mpi4py
    echo Miniconda has been successfully installed in %CONDA_DIR%, and the environment variables have been updated.
) 

if not exist %VENV_DIR% (
    echo creating .venv
    %CONDA% create --prefix %VENV_DIR% python=%PYTHON_VERSION% -y
)

@REM %PIP% install --only-binary=:all: -r %SCRIPTS_DIR%../requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
@REM %PIP% install --only-binary=:all: -r %SCRIPTS_DIR%../requirements.txt -i https://pypi.org/simple
@REM %PIP% install --only-binary=:all: -r %SCRIPTS_DIR%../requirements.txt -i https://mirrors.aliyun.com/pypi/simple
%PIP% install -r %SCRIPTS_DIR%../requirements.txt -i https://mirrors.aliyun.com/pypi/simple

