@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

REM Sync code/config/handbook to USB stick. Additive only: copy/overwrite, never delete.
REM Does NOT sync: docs data datasets runs .venv .env resources *.parquet
REM Usage: run from project root, or double-click this script.
REM Dest fixed to F:\audio-data-engine

set "SRC=%~dp0.."
for %%I in ("%SRC%") do set "SRC=%%~fI"

set "DEST=F:\audio-data-engine"
set "LOG=%TEMP%\audio-data-engine-usb-sync.log"
set "FAIL=0"

REM ---------- check USB ----------
if not exist "F:\" (
    echo.
    echo [ERROR] Drive F: not found. Insert USB and retry.
    echo.
    pause
    exit /b 1
)

REM safety: only allow fixed dest
if /I not "%DEST%"=="F:\audio-data-engine" (
    echo.
    echo [ERROR] Unexpected DEST, abort: %DEST%
    echo.
    pause
    exit /b 1
)

if not exist "%DEST%" (
    echo Dest missing, creating: %DEST%
    mkdir "%DEST%"
)

echo.
echo ============================================
echo   USB additive sync [copy/overwrite, no delete]
echo ============================================
echo   SRC : %SRC%
echo   DEST: %DEST%
echo   LOG : %LOG%
echo ============================================
echo.
echo Whitelist dirs:
echo   src  pipelines  configs  tests  scripts  tasks  handbook-dir
echo Whitelist root files:
echo   pyproject.toml  README.md  .gitignore  .env.example
echo   and local *.txt handbooks at repo root
echo.
echo Never synced:
echo   docs  data  datasets  runs  .venv  .env  resources  *.parquet
echo.
echo Starting...
echo.

if exist "%LOG%" del /f /q "%LOG%" >nul 2>&1
echo audio-data-engine USB additive sync > "%LOG%"
echo SRC=%SRC% >> "%LOG%"
echo DEST=%DEST% >> "%LOG%"
echo. >> "%LOG%"

REM ---------- 1) whitelist dirs ----------
call :sync_dir "src"
call :sync_dir "pipelines"
call :sync_dir "configs"
call :sync_dir "tests"
call :sync_dir "scripts"
call :sync_dir "tasks"
call :sync_dir "手册"

REM ---------- 2) whitelist root files ----------
call :sync_file "pyproject.toml"
call :sync_file "README.md"
call :sync_file ".gitignore"
call :sync_file ".env.example"
call :sync_file "文档.txt"
call :sync_file "单条流水线执行命令.txt"
call :sync_file "三条流水线执行手册.txt"
call :sync_file "全自动训练评测闭环执行手册-dev.txt"
call :sync_file "全自动训练评测闭环执行手册-local.txt"
call :sync_file "目录.txt"

REM optional resources - keep disabled by default
REM call :sync_dir "resources"

echo.
if "!FAIL!"=="0" (
    echo [OK] Additive sync finished. No files deleted on DEST.
) else (
    echo [WARN] Finished with !FAIL! failures. See: %LOG%
)
echo DEST: %DEST%
echo LOG : %LOG%
echo.
echo Next on server:
echo   1. Copy USB tree to /data2/data-cp/lizi/tmp/audio-data-engine
echo   2. Run handbook under 手册/dev for server hot update
echo.
pause
if "!FAIL!"=="0" (exit /b 0) else (exit /b 1)


REM ========== subroutines ==========

:sync_dir
set "REL=%~1"
if not exist "%SRC%\%REL%" (
    echo   [skip] dir missing: %REL%
    echo SKIP DIR %REL% >> "%LOG%"
    exit /b 0
)
echo   [dir] %REL%
if not exist "%DEST%\%REL%" mkdir "%DEST%\%REL%"
robocopy "%SRC%\%REL%" "%DEST%\%REL%" /E /MT:8 /R:2 /W:3 ^
    /XD "__pycache__" ".pytest_cache" ".ruff_cache" ".mypy_cache" ".eggs" "htmlcov" "dist" "build" ^
    /XF "*.pyc" "*.pyo" "*.pyd" "*.parquet" "*.log" "*.bak" "*.bak_*" "Thumbs.db" ".DS_Store" ".coverage" ^
    /NFL /NDL /NP /LOG+:"%LOG%"
set "RC=!ERRORLEVEL!"
if !RC! GEQ 8 (
    echo   [ERROR] dir sync failed: %REL%  robocopy=!RC!
    set /a FAIL+=1
)
exit /b 0

:sync_file
set "REL=%~1"
if not exist "%SRC%\%REL%" (
    echo   [skip] file missing: %REL%
    echo SKIP FILE %REL% >> "%LOG%"
    exit /b 0
)
echo   [file] %REL%
copy /Y "%SRC%\%REL%" "%DEST%\%REL%" >nul
if errorlevel 1 (
    echo   [ERROR] file copy failed: %REL%
    echo FAIL FILE %REL% >> "%LOG%"
    set /a FAIL+=1
) else (
    echo OK FILE %REL% >> "%LOG%"
)
exit /b 0
