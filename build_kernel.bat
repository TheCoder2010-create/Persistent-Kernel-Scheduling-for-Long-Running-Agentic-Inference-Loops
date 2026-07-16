@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
nvcc -O3 -arch=sm_89 persistent_kernel.cu -o persistent_kernel.exe