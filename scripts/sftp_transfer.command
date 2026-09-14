#!/bin/bash
# Mac 上双击本文件会打开 Terminal 并启动传输。
# 首次若提示无法打开，在终端执行一次：
#   chmod +x scripts/sftp_transfer.command

cd "$(dirname "$0")" || exit 1
exec bash "./sftp_transfer.sh"
