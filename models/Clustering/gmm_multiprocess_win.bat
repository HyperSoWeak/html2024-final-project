@echo off
:: Enable delayed expansion for runtime variable substitution
setlocal enabledelayedexpansion

:: Variables
set total_start=2
set total_end=10000
set max_processes=32
set max_chunk_size=100

:: Initialize the job count
set /a count=0

:: Main loop for dividing experiments into chunks
for /l %%i in (%total_start%, %max_chunk_size%, %total_end%) do (
    set /a chunk_start=%%i
    set /a chunk_end=%%i + %max_chunk_size% - 1
    if !chunk_end! gtr %total_end% set /a chunk_end=%total_end%

    :: Start the Python script with proper arguments in parallel
    start /b cmd /c python models/Clustering/gmm_train.py !chunk_start! !chunk_end!

    :: Increment the active process counter
    set /a count+=1

    :: If the max_processes limit is reached, wait before starting more
    if !count! geq %max_processes% (
        echo "Waiting for processes to finish..."
        call :wait_for_processes
        set /a count=0
    )
)

:: Ensure all processes finish before exiting
call :wait_for_processes

echo "All experiments completed."
pause
exit /b

:: Subroutine to wait for active processes
:wait_for_processes
for /f "tokens=*" %%A in ('tasklist /FI "IMAGENAME eq cmd.exe" /NH') do (
    timeout /t 1 /nobreak >nul
    goto :wait_for_processes
)
exit /b
