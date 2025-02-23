@echo off
echo ^[ stable-diffusion webui ^]
echo ^[ https://github.com/AUTOMATIC1111/stable-diffusion-webui/ ^]

set COMMANDLINE_ARGS= %*

if "%*"=="--server" (
	echo ^[ Starting as Server ^]
	set COMMANDLINE_ARGS= --listen --enable-insecure-extension-access
) else (
	echo ^[ Hint: use "--server" to start as server ^]
)

echo ^.^.^.

REM set NVCC_FLAGS=-allow-unsupported-compiler
set TF_ENABLE_ONEDNN_OPTS=0
set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS= %COMMANDLINE_ARGS%

REM #### CUSTOM ARGUMENTS ####
REM #------------------------#
set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --api --allow-code
set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --medvram
set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --upcast-sampling --opt-split-attention
set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --no-half
set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --disable-safe-unpickle
set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --xformers
REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --enable-insecure-extension-access
REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --listen
REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --nowebui
REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --disable-all-extensions 
REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --disable-extra-extensions
REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --reinstall-torch --reinstall-xformers
REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --no-download-sd-model
REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --gradio-img2img-tool color-sketch
REM #
REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --log-startup

set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512
REM set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:768

REM #
REM #------------------------#

echo ^[ command line arguments ^]
echo ^| ^"%COMMANDLINE_ARGS%^" ^]
echo ^.^.^.

echo ^[ STARTE SD-WEBUI ^]

call webui.bat