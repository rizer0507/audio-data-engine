@echo off
setlocal EnableDelayedExpansion

:: 将本项目全量同步到 U 盘（E:\audio-data-engine）
:: 每次同步前会彻底删除 U 盘上的旧项目目录，再重新拷贝
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

:: 安全校验：仅允许删除固定目标路径，避免误删
if /I not "%DEST%"=="E:\audio-data-engine" (
    echo.
    echo [错误] 目标路径异常，已中止: %DEST%
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
echo 排除项：.git / 虚拟环境 / __pycache__ / data / runs / datasets
echo         本地数据集 / 模型资源 / parquet 等
echo.
if exist "%DEST%" (
    echo 正在删除 U 盘上的旧项目: %DEST%
    rmdir /s /q "%DEST%"
    if exist "%DEST%" (
        echo.
        echo [错误] 无法删除旧目录，请关闭占用该目录的程序后重试。
        echo.
        pause
        exit /b 1
    )
    echo 旧项目已删除。
    echo.
)
echo 开始全量拷贝...
echo.

:: ---------- robocopy 全量拷贝 ----------
:: /E    : 复制子目录（含空目录）
:: /MT:8 : 8 线程加速
:: /R:2 /W:3 : 失败重试 2 次，间隔 3 秒
:: /XD   : 排除目录
:: /XF   : 排除文件类型
robocopy "%SRC%" "%DEST%" /E /MT:8 /R:2 /W:3 ^
    /XD ".git" ".venv" "venv" "__pycache__" ".pytest_cache" ".ruff_cache" ".mypy_cache" ^
        ".idea" ".vscode" "htmlcov" "dist" "build" ".eggs" "node_modules" ^
        "data" "runs" "datasets" "qwen_vs_sensevoice" "数据集" "resources\source" ^
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
    echo [完成] 全量拷贝成功。
) else (
    echo [完成] 全量拷贝成功，部分文件已更新。
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
