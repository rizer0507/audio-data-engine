@echo off
setlocal EnableDelayedExpansion

:: 将本项目增量同步到 U 盘（E:\audio-data-engine）
:: 仅拷贝源码、配置、文档等必要文件，跳过缓存/数据集/运行产物等大体积目录
:: 用法：双击运行，或在项目根目录执行  scripts\sync_to_usb.bat

chcp 65001 >nul 2>&1

set "SRC=%~dp0.."
for %%I in ("%SRC%") do set "SRC=%%~fI"

set "DEST=E:\audio-data-engine"
set "LOG=%TEMP%\audio-data-engine-usb-sync.log"

:: ---------- 检查 U 盘 ----------
if not exist "E:\" (
    echo.
    echo [错误] 未检测到 E: 盘，请先插入 U 盘后重试。
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   同步 audio-data-engine 到 U 盘
echo ============================================
echo   源目录 : %SRC%
echo   目标   : %DEST%
echo   日志   : %LOG%
echo ============================================
echo.
echo 排除项：.git / 虚拟环境 / __pycache__ / data 下音频与缓存
echo         runs / datasets / 本地数据集 / 模型资源 / parquet 等
echo.
echo 开始同步（仅传输有变化的文件）...
echo.

:: ---------- robocopy 增量镜像 ----------
:: /MIR  : 目标与源保持一致（删除目标中已移除的文件）
:: /MT:8 : 8 线程加速
:: /R:2 /W:3 : 失败重试 2 次，间隔 3 秒
:: /XD   : 排除目录
:: /XF   : 排除文件类型
robocopy "%SRC%" "%DEST%" /MIR /MT:8 /R:2 /W:3 ^
    /XD ".git" ".venv" "venv" "__pycache__" ".pytest_cache" ".ruff_cache" ".mypy_cache" ^
        ".idea" ".vscode" "htmlcov" "dist" "build" ".eggs" "node_modules" ^
        "data\cache" "data\derived" "data\raw" "data\exports" ^
        "runs" "datasets" "qwen_vs_sensevoice" "数据集" "resources\source" ^
    /XF "*.pyc" "*.pyo" "*.pyd" "*.egg-info" ".env" "*.parquet" "*.log" "*.bak" "*.bak_*" ^
        "Thumbs.db" ".DS_Store" ".coverage" ^
    /NFL /NDL /NP /LOG:"%LOG%"

set "RC=%ERRORLEVEL%"

:: robocopy 退出码 0-7 均表示成功（含"无文件需复制"）
if %RC% GEQ 8 (
    echo.
    echo [错误] 同步失败，robocopy 退出码: %RC%
    echo 详见日志: %LOG%
    echo.
    pause
    exit /b %RC%
)

:: ---------- 恢复空目录占位（.gitkeep）----------
call :ensure_gitkeep "data\cache"
call :ensure_gitkeep "data\derived"
call :ensure_gitkeep "data\raw"
call :ensure_gitkeep "data\exports"
call :ensure_gitkeep "configs\augmentation"
call :ensure_gitkeep "configs\denoise"
call :ensure_gitkeep "tests"

echo.
if %RC% EQU 0 (
    echo [完成] 目标已是最新，无需复制新文件。
) else (
    echo [完成] 同步成功，部分文件已更新。
)
echo 目标路径: %DEST%
echo 详细日志: %LOG%
echo.
pause
exit /b 0

:: 子例程：确保排除目录在 U 盘上保留 .gitkeep 占位
:ensure_gitkeep
set "REL=%~1"
if exist "%SRC%\%REL%\.gitkeep" (
    if not exist "%DEST%\%REL%" mkdir "%DEST%\%REL%"
    copy /Y "%SRC%\%REL%\.gitkeep" "%DEST%\%REL%\" >nul 2>&1
)
exit /b 0
