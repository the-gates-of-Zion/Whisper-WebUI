@echo off
echo Installing Miniconda...

:: Download the Miniconda installer
:: Replace the URL below with the actual Miniconda installer download link
:: You can choose either the 32-bit or 64-bit version based on your system
:: start /wait "" "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"

:: Install Miniconda silently
:: Adjust the installation path as needed
:: /S: Silent mode
:: /D: Installation directory
start /wait "" "Miniconda3-latest-Windows-x86_64.exe" /S /D=C:\Miniconda3

:: Add Miniconda to system PATH
:: Adjust the path as needed
setx /M PATH "C:\Miniconda3\Scripts;C:\Miniconda3\Library\bin;%PATH%"

echo Miniconda installation complete!
echo You may need to restart your command prompt or terminal to use conda.
pause

@echo off
echo Installing CUDA 11.8, Git, Python 3.10.0, and FFmpeg...

:: Install CUDA 11.8
echo Downloading CUDA 11.8 installer...
:: Replace the URL below with the actual CUDA 11.8 installer download link
:: start /wait "" "[1](https://developer.nvidia.com/cuda-downloads)"

:: Install Git
echo Downloading Git installer...
:: Replace the URL below with the actual Git installer download link
:: start /wait "" "[2](https://git-scm.com/downloads)"

:: Install Python 3.10.0
echo Downloading Python 3.10.0 installer...
:: Replace the URL below with the actual Python 3.10.0 installer download link
:: start /wait "" "[3](https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe)"

:: Install FFmpeg
echo Downloading FFmpeg...
:: Replace the URL below with the actual FFmpeg installer download link
:: start /wait "" "[4](https://ffmpeg.org/download.html)"

:: Add FFmpeg/bin to system PATH
echo Adding FFmpeg to system PATH...
:: Replace the path below with the actual installation path of FFmpeg
:: setx /M PATH "C:\path\to\FFmpeg\bin;%PATH%"

echo Installation complete!
pause