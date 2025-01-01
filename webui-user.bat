@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=

@REM Uncomment following code to reference an existing A1111 checkout.
@REM set A1111_HOME=
@REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% ^
@REM --models-dir  %A1111_HOME%/models ^
@REM --ckpt-dir %A1111_HOME%/models/Stable-diffusion ^
@REM --hypernetwork-dir %A1111_HOME%/models/hypernetworks ^
@REM --embeddings-dir %A1111_HOME%/embeddings ^
@REM --lora-dir %A1111_HOME%/models/Lora ^

@REM Uncomment following code to set venv to the old checkout.
@REM set VENV_DIR=%A1111_HOME%\\venv

call webui.bat
