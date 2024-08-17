@echo off
echo [stable-diffusion webui]
echo [https://github.com/AUTOMATIC1111/stable-diffusion-webui/ ]

set COMMANDLINE_ARGS=%1

set NVCC_FLAGS=-allow-unsupported-compiler
set TF_ENABLE_ONEDNN_OPTS=0
set PYTHON=
set GIT=
set VENV_DIR=
REM set SADTALKER_CHECKPOINTS=...
set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --api

REM #### CUSTOM ARGUMENTS ####
REM #------------------------#
REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --disable-all-extensions 
REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --disable-extra-extensions
REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --reinstall-torch --reinstall-xformers
set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --xformers
set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --allow-code
REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --disable-safe-unpickle
set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --upcast-sampling --opt-split-attention
REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --gradio-img2img-tool color-sketch
REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --listen
REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --nowebui
REM #------------------------#

echo [command line arguments:"%COMMANDLINE_ARGS%"]

timeout /T 3 /NOBREAK

call webui.bat
