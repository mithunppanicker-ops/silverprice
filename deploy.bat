@echo off
echo Creating virtual environment...
py -m venv venv
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated successfully!
cmd /k
