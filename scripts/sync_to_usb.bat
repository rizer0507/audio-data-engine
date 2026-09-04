@echo off
setlocal EnableDelayedExpansion

:: =============================================================================
:: 同步「需要上服务器」的代码/配置/文档 → U 盘（增量，只加/覆盖，不删）
::
:: 原则（保护服务器环境）：
::   1. 不删除 U 盘上的目标目录，也不使用 /MIR /PURGE（目标多出来的文件一律保留）
::   2. 只拷贝下方白名单；服务器独有资产永不通过本脚本带走/覆盖：
::        data/  datasets/  runs/  数据集/  .venv/  .env  resources/  *.parquet
::   3. 同名文件会用本机版本覆盖（加/更新）；不会减少目标侧文件
::
:: 用法：双击运行，或在项目根目录执行  scripts\sync_to_usb.bat
:: U 盘拷到服务器后，执行 手册/dev/04-服务器热更新-tmp到正式工程.txt
::   落到 /data2/data-cp/lizi/audio-data-engine（同样只覆盖白名单，不碰资产）
:: =============================================================================

chcp 65001 >nul 2>&1

set "SRC=%~dp0.."
for %%I in ("%SRC%") do set "SRC=%%~fI"

set "DEST=F:\audio-data-engine"
set "LOG=%TEMP%\audio-data-engine-usb-sync.log"
set "FAIL=0"

:: ---------- 检查 U 盘 ----------
if not exist "F:\" (
    echo.
    echo [错误] 未检测到 F: 盘，请先插入 U 盘后重试。
    echo.
    pause
    exit /b 1
)

:: 安全校验：仅允许写到固定目标路径
if /I not "%DEST%"=="F:\audio-data-engine" (
    echo.
    echo [错误] 目标路径异常，已中止: %DEST%
    echo.
    pause
    exit /b 1
)

if not exist "%DEST%" (
    echo 目标目录不存在，正在创建: %DEST%
    mkdir "%DEST%"
)

echo.
echo ============================================
echo   增量同步到 U 盘（只加/覆盖，不删）
echo ============================================
echo   源目录 : %SRC%
echo   目标   : %DEST%
echo   日志   : %LOG%
echo ============================================
echo.
echo 白名单目录:
echo   src  docs  pipelines  configs  tests  scripts  tasks  手册
echo 白名单根文件:
echo   pyproject.toml  README.md  .gitignore  .env.example
echo   文档.txt  单条流水线执行命令.txt  三条流水线执行手册.txt
echo   全自动训练评测闭环执行手册-dev.txt
echo   全自动训练评测闭环执行手册-local.txt
echo   目录.txt
echo.
echo 明确不同步（保护服务器）:
echo   data  datasets  runs  数据集  .venv  .env  resources  *.parquet
echo   （不删除 U 盘已有内容，不 purge）
echo.
echo 开始增量拷贝...
echo.

:: 清空旧日志
if exist "%LOG%" del /f /q "%LOG%" >nul 2>&1
echo audio-data-engine USB additive sync > "%LOG%"
echo SRC=%SRC% >> "%LOG%"
echo DEST=%DEST% >> "%LOG%"
echo. >> "%LOG%"

:: ---------- 1) 白名单目录（robocopy /E，无 /PURGE /MIR = 不删目标多出文件）----------
:: /XO 不加：允许用本机较新代码覆盖同名文件
call :sync_dir "src"
call :sync_dir "docs"
call :sync_dir "pipelines"
call :sync_dir "configs"
call :sync_dir "tests"
call :sync_dir "scripts"
call :sync_dir "tasks"
call :sync_dir "手册"

:: ---------- 2) 白名单根文件 ----------
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

:: ---------- 可选：resources（默认关闭；含正式机 source path，确认后再开）----------
:: call :sync_dir "resources"

echo.
if "!FAIL!"=="0" (
    echo [完成] 增量同步成功（只加/覆盖，未删除目标文件）。
) else (
    echo [警告] 同步结束，但有 !FAIL! 项失败，详见: %LOG%
)
echo 目标路径: %DEST%
echo 详细日志: %LOG%
echo.
echo 下一步（服务器）:
echo   1. 将 U 盘内容拷到 /data2/data-cp/lizi/tmp/audio-data-engine
echo   2. 执行 手册/dev/04-服务器热更新-tmp到正式工程.txt
echo      （与本脚本白名单一致：只覆盖代码/配置/手册，不碰 data/datasets/runs/.venv）
echo.
pause
if "!FAIL!"=="0" (exit /b 0) else (exit /b 1)


:: =============================================================================
:: 子例程
:: =============================================================================

:sync_dir
set "REL=%~1"
if not exist "%SRC%\%REL%" (
    echo   [跳过] 目录不存在: %REL%
    echo SKIP DIR %REL% >> "%LOG%"
    exit /b 0
)
echo   [目录] %REL%
if not exist "%DEST%\%REL%" mkdir "%DEST%\%REL%"
:: /E 含子目录；排除缓存；绝不加 /PURGE /MIR
robocopy "%SRC%\%REL%" "%DEST%\%REL%" /E /MT:8 /R:2 /W:3 ^
    /XD "__pycache__" ".pytest_cache" ".ruff_cache" ".mypy_cache" ".eggs" "htmlcov" "dist" "build" ^
    /XF "*.pyc" "*.pyo" "*.pyd" "*.parquet" "*.log" "*.bak" "*.bak_*" "Thumbs.db" ".DS_Store" ".coverage" ^
    /NFL /NDL /NP /LOG+:"%LOG%"
set "RC=!ERRORLEVEL!"
if !RC! GEQ 8 (
    echo   [错误] 目录同步失败: %REL%  robocopy=!RC!
    set /a FAIL+=1
)
exit /b 0

:sync_file
set "REL=%~1"
if not exist "%SRC%\%REL%" (
    echo   [跳过] 文件不存在: %REL%
    echo SKIP FILE %REL% >> "%LOG%"
    exit /b 0
)
echo   [文件] %REL%
copy /Y "%SRC%\%REL%" "%DEST%\%REL%" >nul
if errorlevel 1 (
    echo   [错误] 文件拷贝失败: %REL%
    echo FAIL FILE %REL% >> "%LOG%"
    set /a FAIL+=1
) else (
    echo OK FILE %REL% >> "%LOG%"
)
exit /b 0
