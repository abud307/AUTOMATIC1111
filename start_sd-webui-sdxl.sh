# rem CALL conda.bat activate ai-stable-webui && python launch.py --xformers --api --no-half --disable-nan-check --port 7861
# 添加代理
# export http_proxy="http://ngrok.samuelzuuka.com:15081"
# export https_proxy="http://ngrok.samuelzuuka.com:15081"
# wget "https://www.google.com"
# exit 1
# /home/samuel/dev-tools/miniconda3/bin/activate activate stable-diffusion && 
python launch.py --xformers --api --no-half --disable-nan-check --port 17860 --gradio-auth zukai:zukai_1804
