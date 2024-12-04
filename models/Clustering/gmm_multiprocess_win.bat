@echo off
:: Variables
set total_start=2
set total_end=10000
set max_processes=32
set max_chunk_size=100

:: Function to run a chunk of experiments
:run_chunk
set start=%1
set end=%2
start /b python models/Clustering/gmm_train.py %start% %end%
exit /b

:: Main loop for dividing experiments into chunks
setlocal enabledelayedexpansion
set count=0

for /l %%i in (%total_start%, %max_chunk_size%, %total_end%) do (
    set /a chunk_start=%%i
    set /a chunk_end=%%i + %max_chunk_size% - 1
    if !chunk_end! gtr %total_end% set /a chunk_end=%total_end%
    
    :: Run in parallel, respecting max_processes
    set /a count+=1
    call :run_chunk !chunk_start! !chunk_end!
    if !count! geq %max_processes% (
        timeout /t 1 /nobreak >nul
        set /a count=0
    )
)

:: Wait for all jobs to finish
echo "All experiments completed."
pause
