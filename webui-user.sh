#!/bin/bash
#########################################################
# Uncomment and change the variables below to your need:#
#########################################################

# Install directory without trailing slash
install_dir="/home/$(whoami)"

# Name of the subdirectory
clone_dir="stable-diffusion-webui"

# Full installation path
install_path="${install_dir}/${clone_dir}"

echo "Initialisiere MacMilling Skript (www.macmilling.de)"
echo "sd-webui path: $install_path"

export SADTALKER_CHECKPOINTS="${install_path}/models/SadTalker/checkpoints"
export COMMANDLINE_GENERAL_ARGS="--api --api-log --allow-code --skip-torch-cuda-test"
export COMMANDLINE_SERVER_ARGS="--listen --port 7969"
export COMMANDLINE_OPT_ARGS="--use-cpu all --no-half --no-half-vae"
export COMMANDLINE_CUSTOM_ARGS="--update-check --enable-insecure-extension-access"
export COMMANDLINE_AUTH_ARGS="--gradio-allowed-path ${install_path} --gradio-auth macmilling:litlithammer"
export COMMANDLINE_DEBUG_ARGS="" #--reinstall-torch --reinstall-xformers"

# Commandline arguments for webui.py, for example: export COMMANDLINE_ARGS="--medvram --opt-split-attention"
export COMMANDLINE_ARGS="${COMMANDLINE_DEBUG_ARGS} ${COMMANDLINE_GENERAL_ARGS} ${COMMANDLINE_SERVER_ARGS} ${COMMANDLINE_OPT_ARGS} ${COMMANDLINE_AUTH_ARGS} ${COMMANDLINE_CUSTOM_ARGS}"
echo "start-parameter: $COMMANDLINE_ARGS"

export IIB_CACHE_DIR ="${install_path}/cache/ibb/"
export IIB_SECRET_KEY=none

# python3 executable
#python_cmd="python3"

# git executable
#export GIT="git"

# python3 venv without trailing slash (defaults to ${install_dir}/${clone_dir}/venv)
venv_dir="venv"

# script to launch to start the app
#export LAUNCH_SCRIPT="launch.py"

# install command for torch
#export TORCH_COMMAND="pip install -U torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113"

# Requirements file to use for stable-diffusion-webui
#export REQS_FILE="requirements_versions.txt"

# Fixed git repos
#export K_DIFFUSION_PACKAGE=""
#export GFPGAN_PACKAGE=""

# Fixed git commits
#export STABLE_DIFFUSION_COMMIT_HASH=""
#export CODEFORMER_COMMIT_HASH=""
#export BLIP_COMMIT_HASH=""

# Uncomment to enable accelerated launch
#export ACCELERATE="True"

# Uncomment to disable TCMalloc
#export NO_TCMALLOC="True"

###########################################
