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

set NVCC_FLAGS=-allow-unsupported-compiler
set TF_ENABLE_ONEDNN_OPTS=0
set PYTHON=
set GIT=git
set VENV_DIR=
set SADTALKER_CHECKPOINTS=G:\stable-diffusion-webui\models\SadTalker\checkpoints
set COMMANDLINE_ARGS= %COMMANDLINE_ARGS%

REM #### CUSTOM ARGUMENTS ####
REM #------------------------#
set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --api --allow-code
set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --upcast-sampling 
set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --enable-insecure-extension-access
REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --listen
REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --nowebui
set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --opt-split-attention
REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --disable-all-extensions 
REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --disable-extra-extensions
REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --reinstall-torch --reinstall-xformers
set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --xformers
set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --no-download-sd-model
REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --gradio-img2img-tool color-sketch
set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --disable-safe-unpickle
set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --medvram --log-startup
REM #
REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --log-startup

REM set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512
REM set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:768

REM #
REM #------------------------#

echo ^[ command line arguments ^]
echo ^| ^"%COMMANDLINE_ARGS%^" ^]
echo ^.^.^.

echo ^[ STARTE SD-WEBUI ^]

call webui.bat
