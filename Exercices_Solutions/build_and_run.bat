@echo off

set QTDIR=C:\lib\Qt\5.14.2\msvc2017_64\bin
set VCDIR=C:\Program Files\Microsoft Visual Studio\2022\Community
set CUDA_DIR=C:\PROGRA~1\NVIDIA~2\CUDA\v12.1
set CUDA_ARCH=86
set WIN_SDK=10.0.22621.0

REM NE PLUS RIEN MODIFIER

REM Converti les noms longs en noms courts... OMG !!!

echo %~s1 "%QTDIR%" > DIRtempName.txt
set /p QTDIR=<DIRtempName.txt

echo %~s1 "%VCDIR%" > DIRtempName.txt
set /p VCDIR=<DIRtempName.txt

echo %~s1 "%CUDA_DIR%" > DIRtempName.txt
REM set /p CUDA_DIR=<DIRtempName.txt

del DIRtempName.txt

set CUDA_BIN_DIR=%CUDA_DIR%\bin
set CUDA_LIB_DIR=%CUDA_DIR%\Lib\x64
set CUDA_INC_DIR=%CUDA_DIR%\include

echo %~s1 "%CUDA_INC_DIR%" 

set PATH=%QTDIR%\bin;%CUDA_BIN_DIR%;%MPI_EXEC_DIR%
echo %PATH%

REM Mise en place du path pour Visual
pushd %VCDIR%\VC\Auxiliary\Build
	call vcvarsall.bat x64
popd

start /WAIT /B %QTDIR% qmake.exe -tp vc -r Exercices_Solutions.pro

start /WAIT "Visual" devenv Exercices_Solutions.sln

@echo off
echo.
set /p id="press ENTER to quit"

goto:eof

REM Functions here...

:convertToShortPath 
echo %~s1
EXIT /B 0

goto:eof