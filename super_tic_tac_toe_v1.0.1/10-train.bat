@echo off

cls


set ROOT_DIR=%~dp0
set CODE_DIR=%ROOT_DIR%code
set PYTHON=%ROOT_DIR%.venv\python.exe
set PATH=%ROOT_DIR%.MPI\Bin;%PATH%
set PATH=%ROOT_DIR%.venv\Scripts;%PATH%
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=3

cd %ROOT_DIR%
cmd /c "call %ROOT_DIR%scripts\setup.bat"
%PYTHON% %CODE_DIR%\train.py

pause
