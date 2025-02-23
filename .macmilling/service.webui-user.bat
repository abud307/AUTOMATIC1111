@echo off
:loop
echo ^[ STARTING SD WEBUI ^]
Start webui-user.bat | set /P "="
goto loop