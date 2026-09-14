#!/bin/bash
# 在 Mac 上双击同目录的 sftp_transfer.command 即可启动。
# 本文件也可以在终端运行：bash scripts/sftp_transfer.sh

set -u

DIR="$(cd "$(dirname "$0")" && pwd)"
PY_SCRIPT="$DIR/sftp_transfer.py"
CONF="${HOME}/.sftp_transfer.conf"

# Finder 双击时 PATH 很短，补上 Homebrew / 用户安装路径。
export PATH="/opt/homebrew/bin:/usr/local/bin:${HOME}/.local/bin:${PATH}"
export LANG="${LANG:-zh_CN.UTF-8}"
export LC_ALL="${LC_ALL:-zh_CN.UTF-8}"

pause_close() {
  echo
  read -r -p "按回车关闭窗口..." _
}

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "找不到传输脚本: $PY_SCRIPT"
  pause_close
  exit 1
fi

find_python() {
  local candidate
  local candidates=(
    /opt/homebrew/bin/python3
    /usr/local/bin/python3
    /usr/bin/python3
    "${HOME}/.local/bin/python3"
    "${HOME}/miniconda3/bin/python3"
    "${HOME}/anaconda3/bin/python3"
    "${HOME}/.pyenv/shims/python3"
    python3
    python
  )
  for candidate in "${candidates[@]}"; do
    if [[ "$candidate" == /* ]]; then
      if [[ -x "$candidate" ]]; then
        echo "$candidate"
        return 0
      fi
    elif command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

PY="$(find_python || true)"
if [[ -z "$PY" ]]; then
  echo "未找到 python3。请先安装 Python 3（可用 Homebrew: brew install python）。"
  pause_close
  exit 1
fi

if ! "$PY" -c "import paramiko" >/dev/null 2>&1; then
  echo "当前 Python: $PY"
  echo "缺少 paramiko，SFTP 传输需要它。"
  read -r -p "现在安装？[Y/n] " install_now
  if [[ -z "$install_now" || "$install_now" == [Yy] || "$install_now" == [Yy][Ee][Ss] ]]; then
    "$PY" -m pip install paramiko || {
      echo "安装失败。可手动执行: $PY -m pip install paramiko"
      pause_close
      exit 1
    }
  else
    echo "已取消。请先执行: $PY -m pip install paramiko"
    pause_close
    exit 1
  fi
fi

LAST_HOST=""
LAST_USER=""
LAST_PORT="22"
if [[ -f "$CONF" ]]; then
  while IFS='=' read -r key value; do
    case "$key" in
      HOST) LAST_HOST="$value" ;;
      USER) LAST_USER="$value" ;;
      PORT) LAST_PORT="$value" ;;
    esac
  done < "$CONF"
fi

prompt() {
  local label="$1"
  local default="$2"
  local __out="$3"
  local input
  if [[ -n "$default" ]]; then
    read -r -p "${label} [${default}]: " input
    printf -v "$__out" '%s' "${input:-$default}"
  else
    while true; do
      read -r -p "${label}: " input
      if [[ -n "$input" ]]; then
        printf -v "$__out" '%s' "$input"
        return
      fi
      echo "不能为空。"
    done
  fi
}

clear
echo "================================"
echo "  SFTP 文件传输"
echo "================================"
echo "回车沿用括号里的上次值。密码稍后输入，不会保存。"
echo

prompt "服务器 IP" "$LAST_HOST" HOST
prompt "用户名" "$LAST_USER" USER
prompt "端口" "${LAST_PORT:-22}" PORT

if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "端口无效: $PORT"
  pause_close
  exit 1
fi

umask 077
{
  echo "HOST=${HOST}"
  echo "USER=${USER}"
  echo "PORT=${PORT}"
} > "$CONF"

RESUME_ARGS=()
read -r -p "跳过大小相同的文件（中断后继续）？[y/N] " resume_choice
if [[ "$resume_choice" == [Yy] || "$resume_choice" == [Yy][Ee][Ss] ]]; then
  RESUME_ARGS+=(--resume)
fi

# 双击启动时工作目录是用户主目录。切到仓库根，方便写相对路径。
if [[ "$(basename "$DIR")" == "scripts" && -d "$DIR/.." ]]; then
  cd "$DIR/.." || exit 1
else
  cd "$DIR" || exit 1
fi

echo
echo "当前目录: $(pwd)"
echo "本地路径可写绝对路径，也可把文件夹拖进这个窗口。"
echo "连上后: put <本地> <远端> | get <远端> <本地> | ls [远端] | quit"
echo
echo "正在连接 ${USER}@${HOST}:${PORT} …"
echo

"$PY" "$PY_SCRIPT" shell --host "$HOST" --user "$USER" --port "$PORT" "${RESUME_ARGS[@]}"
status=$?

echo
if [[ "$status" -ne 0 ]]; then
  echo "已退出（代码 ${status}）。"
fi
pause_close
exit "$status"
