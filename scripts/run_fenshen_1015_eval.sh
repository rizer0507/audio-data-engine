#!/usr/bin/env bash
# 评测 fenshen_1015 的 train / dev，转写结束后写出字准 xlsx。
# 第三个参数省略时 train 和 dev 都跑。第四个参数省略时 best、over、base 都跑。
# 先把 gold 放到 datasets/stage1/derived，再自行拉起要用的端口。
#   full: best 5560, over 5561
#   lora: best 5565, over 5566
#   qwen-base: 5555。第四个参数为 all 时总是转写。
#   只转写 best/over 时，本次切分已有 qwen-base parquet 就跳过 5555；没有则调用 5555 补转写。
# 字准 xlsx 始终和 qwen-base 比较。
#
# 选中的端口同时跑。qwen_asr_batch 单次已是 8 个分片打同一个 vLLM，
# 同一端口不再并行叠 train 和 dev。两边都跑时，端口内部 dev 结束后立刻接下一条。
# 5555 的 --asr-run：dev=qwen-base-dev，train=qwen-base-full。
#
# 用法:
#   bash scripts/run_fenshen_1015_eval.sh full v014
#   bash scripts/run_fenshen_1015_eval.sh full v014 train best
#   bash scripts/run_fenshen_1015_eval.sh full v014 dev over
#   bash scripts/run_fenshen_1015_eval.sh full v014 both best-over
#   bash scripts/run_fenshen_1015_eval.sh lora v014 both all

set -euo pipefail

MODE="${1:-}"
VER="${2:-}"
SPLIT="${3:-both}"
CKPT="${4:-all}"

usage() {
  echo "用法: bash scripts/run_fenshen_1015_eval.sh <lora|full> <v0xx> [train|dev|both] [all|best|over|best-over]"
  echo "示例: bash scripts/run_fenshen_1015_eval.sh full v014"
  echo "示例: bash scripts/run_fenshen_1015_eval.sh full v014 train best"
  echo "示例: bash scripts/run_fenshen_1015_eval.sh full v014 dev over"
  echo "示例: bash scripts/run_fenshen_1015_eval.sh full v014 both best-over"
  echo "full 的 best 在 5560，over 在 5561。lora 的 best 在 5565，over 在 5566。"
  echo "字准表始终和 qwen-base 比较。只转写 best/over 时，没有 base parquet 才会调用 5555。"
}

if [[ "$MODE" != "lora" && "$MODE" != "full" ]]; then
  usage
  exit 1
fi

if [[ ! "$VER" =~ ^v[0-9]{3}$ ]]; then
  echo "版本号须为 v 加三位数字，例如 v014，当前: ${VER:-<空>}"
  usage
  exit 1
fi

if [[ "$SPLIT" != "train" && "$SPLIT" != "dev" && "$SPLIT" != "both" ]]; then
  echo "切分须为 train、dev 或 both，当前: ${SPLIT}"
  usage
  exit 1
fi

if [[ "$CKPT" != "all" && "$CKPT" != "best" && "$CKPT" != "over" && "$CKPT" != "best-over" ]]; then
  echo "权重须为 all、best、over 或 best-over，当前: ${CKPT}"
  usage
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PORT_BASE=5555
if [[ "$MODE" == "full" ]]; then
  PORT_BEST=5560
  PORT_OVER=5561
else
  PORT_BEST=5565
  PORT_OVER=5566
fi

DERIVED="datasets/stage1/derived"
TRAIN_GOLD="${DERIVED}/gold_fenshen_1015_train_${VER}.parquet"
DEV_GOLD="${DERIVED}/gold_fenshen_1015_dev_${VER}.parquet"
TRAIN_EVAL="eval_fenshen_1015_train_${VER}"
DEV_EVAL="eval_fenshen_1015_dev_${VER}"
ASR_DIR="datasets/stage3/asr"

want_train=0
want_dev=0
want_best=0
want_over=0
want_base=0
if [[ "$SPLIT" == "train" || "$SPLIT" == "both" ]]; then
  want_train=1
fi
if [[ "$SPLIT" == "dev" || "$SPLIT" == "both" ]]; then
  want_dev=1
fi
if [[ "$CKPT" == "all" || "$CKPT" == "best" || "$CKPT" == "best-over" ]]; then
  want_best=1
fi
if [[ "$CKPT" == "all" || "$CKPT" == "over" || "$CKPT" == "best-over" ]]; then
  want_over=1
fi

BASE_DEV_PARQUET="${ASR_DIR}/${DEV_EVAL}/qwen-base-dev_asr_${DEV_EVAL}.parquet"
BASE_TRAIN_PARQUET="${ASR_DIR}/${TRAIN_EVAL}/qwen-base-full_asr_${TRAIN_EVAL}.parquet"
need_base_dev=0
need_base_train=0
if [[ "$want_dev" -eq 1 ]]; then
  if [[ "$CKPT" == "all" || ! -f "$BASE_DEV_PARQUET" ]]; then
    need_base_dev=1
    want_base=1
    if [[ "$CKPT" != "all" ]]; then
      echo "缺少 qwen-base dev 转写，调用 ${PORT_BASE}: ${BASE_DEV_PARQUET}"
    fi
  else
    echo "已有 qwen-base dev，不调用 ${PORT_BASE}: ${BASE_DEV_PARQUET}"
  fi
fi
if [[ "$want_train" -eq 1 ]]; then
  if [[ "$CKPT" == "all" || ! -f "$BASE_TRAIN_PARQUET" ]]; then
    need_base_train=1
    want_base=1
    if [[ "$CKPT" != "all" ]]; then
      echo "缺少 qwen-base train 转写，调用 ${PORT_BASE}: ${BASE_TRAIN_PARQUET}"
    fi
  else
    echo "已有 qwen-base train，不调用 ${PORT_BASE}: ${BASE_TRAIN_PARQUET}"
  fi
fi

if [[ "$want_train" -eq 1 && ! -f "$TRAIN_GOLD" ]]; then
  echo "缺少 gold: $TRAIN_GOLD"
  exit 1
fi
if [[ "$want_dev" -eq 1 && ! -f "$DEV_GOLD" ]]; then
  echo "缺少 gold: $DEV_GOLD"
  exit 1
fi

echo "======== 条数 ${VER} ${SPLIT} ========"
python - "$SPLIT" "$TRAIN_GOLD" "$DEV_GOLD" <<'PY'
import sys
import pyarrow.parquet as pq

split, train_path, dev_path = sys.argv[1], sys.argv[2], sys.argv[3]
if split in ("train", "both"):
    train_rows = pq.read_table(train_path).num_rows
    print(f"train {train_rows}  {train_path}")
if split in ("dev", "both"):
    dev_rows = pq.read_table(dev_path).num_rows
    print(f"dev   {dev_rows}  {dev_path}")
PY

echo "======== 注册评测集 ========"
if [[ "$want_train" -eq 1 ]]; then
  audio-data eval register "$TRAIN_GOLD" --name "$TRAIN_EVAL" --force
fi
if [[ "$want_dev" -eq 1 ]]; then
  audio-data eval register "$DEV_GOLD" --name "$DEV_EVAL" --force
fi

echo "======== 检查模型端口 ========"
ports=()
if [[ "$want_best" -eq 1 ]]; then
  ports+=("$PORT_BEST")
fi
if [[ "$want_over" -eq 1 ]]; then
  ports+=("$PORT_OVER")
fi
if [[ "$want_base" -eq 1 ]]; then
  ports+=("$PORT_BASE")
fi
python - "${ports[@]}" <<'PY'
import json
import sys
import urllib.request

for port in sys.argv[1:]:
    url = f"http://127.0.0.1:{port}/v1/models"
    print(f"GET {url}")
    with urllib.request.urlopen(url, timeout=15) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    payload = json.loads(body)
    names = [item.get("id") for item in payload.get("data") or []]
    print(f"port {port} models: {names}")
PY

run_one() {
  local split="$1"
  local port="$2"
  local asr_run="$3"
  local eval_name
  if [[ "$split" == "train" ]]; then
    eval_name="$TRAIN_EVAL"
  else
    eval_name="$DEV_EVAL"
  fi
  local out="${ASR_DIR}/${eval_name}/${asr_run}_asr_${eval_name}.parquet"
  echo "======== ${asr_run}  http://127.0.0.1:${port} ========"
  echo "输出: ${out}"
  unset QWEN_ASR_API_BASE QWEN_ASR_API_BASES
  export QWEN_ASR_API_KEY=dummy
  export QWEN_ASR_API_BASE="http://127.0.0.1:${port}"
  audio-data pipeline run pipelines/qwen_asr_batch.yaml \
    --eval-name "$eval_name" \
    --asr-run "$asr_run" \
    --force
}

run_port() {
  local port="$1"
  shift
  export PYTHONUNBUFFERED=1
  echo "======== 端口 ${port} 开始 ========"
  while [[ $# -ge 2 ]]; do
    run_one "$1" "$port" "$2"
    shift 2
  done
  echo "======== 端口 ${port} 结束 ========"
}

best_jobs=()
over_jobs=()
base_jobs=()
if [[ "$want_dev" -eq 1 ]]; then
  if [[ "$want_best" -eq 1 ]]; then
    best_jobs+=(dev "qwen-${MODE}-dev-best")
  fi
  if [[ "$want_over" -eq 1 ]]; then
    over_jobs+=(dev "qwen-${MODE}-dev-over")
  fi
fi
if [[ "$need_base_dev" -eq 1 ]]; then
  base_jobs+=(dev qwen-base-dev)
fi
if [[ "$want_train" -eq 1 ]]; then
  if [[ "$want_best" -eq 1 ]]; then
    best_jobs+=(train "qwen-${MODE}-train-best")
  fi
  if [[ "$want_over" -eq 1 ]]; then
    over_jobs+=(train "qwen-${MODE}-train-over")
  fi
fi
if [[ "$need_base_train" -eq 1 ]]; then
  base_jobs+=(train qwen-base-full)
fi

LOG_DIR="runs/fenshen_1015_eval_${MODE}_${VER}_${SPLIT}_${CKPT}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
BEST_LOG="${LOG_DIR}/best.log"
OVER_LOG="${LOG_DIR}/over.log"
BASE_LOG="${LOG_DIR}/base.log"

running=()
if [[ "$want_best" -eq 1 ]]; then
  running+=("best :${PORT_BEST}")
fi
if [[ "$want_over" -eq 1 ]]; then
  running+=("over :${PORT_OVER}")
fi
if [[ "$want_base" -eq 1 ]]; then
  running+=("base :${PORT_BASE}")
fi
echo "并行: ${running[*]}"
if [[ "$SPLIT" == "both" ]]; then
  echo "同端口内先 dev 再 train"
elif [[ "$SPLIT" == "dev" ]]; then
  echo "只跑 dev"
else
  echo "只跑 train"
fi
if [[ "$want_base" -eq 1 ]]; then
  base_runs=()
  if [[ "$need_base_dev" -eq 1 ]]; then
    base_runs+=("qwen-base-dev")
  fi
  if [[ "$need_base_train" -eq 1 ]]; then
    base_runs+=("qwen-base-full")
  fi
  echo "base 的 asr-run: ${base_runs[*]}"
fi

pid_best=""
pid_over=""
pid_base=""
tail_logs=()
if [[ "$want_best" -eq 1 ]]; then
  : >"$BEST_LOG"
  echo "日志: ${BEST_LOG}"
  run_port "$PORT_BEST" "${best_jobs[@]}" >>"$BEST_LOG" 2>&1 &
  pid_best=$!
  tail_logs+=("$BEST_LOG")
fi
if [[ "$want_over" -eq 1 ]]; then
  : >"$OVER_LOG"
  echo "日志: ${OVER_LOG}"
  run_port "$PORT_OVER" "${over_jobs[@]}" >>"$OVER_LOG" 2>&1 &
  pid_over=$!
  tail_logs+=("$OVER_LOG")
fi
if [[ "$want_base" -eq 1 ]]; then
  : >"$BASE_LOG"
  echo "日志: ${BASE_LOG}"
  run_port "$PORT_BASE" "${base_jobs[@]}" >>"$BASE_LOG" 2>&1 &
  pid_base=$!
  tail_logs+=("$BASE_LOG")
fi
tail -n +1 -f "${tail_logs[@]}" &
pid_tail=$!

cleanup_jobs() {
  local pids=()
  [[ -n "$pid_best" ]] && pids+=("$pid_best")
  [[ -n "$pid_over" ]] && pids+=("$pid_over")
  [[ -n "$pid_base" ]] && pids+=("$pid_base")
  [[ -n "${pid_tail:-}" ]] && pids+=("$pid_tail")
  if [[ ${#pids[@]} -gt 0 ]]; then
    kill "${pids[@]}" 2>/dev/null || true
  fi
  exit 130
}
trap cleanup_jobs INT TERM

code_best=0
code_over=0
code_base=0
if [[ -n "$pid_best" ]]; then
  wait "$pid_best" || code_best=$?
fi
if [[ -n "$pid_over" ]]; then
  wait "$pid_over" || code_over=$?
fi
if [[ -n "$pid_base" ]]; then
  wait "$pid_base" || code_base=$?
fi
trap - INT TERM
sleep 1
kill "$pid_tail" 2>/dev/null || true
wait "$pid_tail" 2>/dev/null || true

if [[ -n "$pid_best" ]]; then
  echo "best 退出码 ${code_best}  日志 ${BEST_LOG}"
fi
if [[ -n "$pid_over" ]]; then
  echo "over 退出码 ${code_over}  日志 ${OVER_LOG}"
fi
if [[ -n "$pid_base" ]]; then
  echo "base 退出码 ${code_base}  日志 ${BASE_LOG}"
fi

echo "======== 产物 ${ASR_DIR} ========"
python - "$MODE" "$VER" "$ASR_DIR" "$SPLIT" "$CKPT" <<'PY'
import sys
from pathlib import Path
import pyarrow.parquet as pq

mode, ver, asr_dir, split, ckpt = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4], sys.argv[5]
missing = []
jobs = []
run_best = ckpt in ("all", "best", "best-over")
run_over = ckpt in ("all", "over", "best-over")
run_base = True
if split in ("dev", "both"):
    if run_best:
        jobs.append((f"qwen-{mode}-dev-best", f"eval_fenshen_1015_dev_{ver}"))
    if run_over:
        jobs.append((f"qwen-{mode}-dev-over", f"eval_fenshen_1015_dev_{ver}"))
    if run_base:
        jobs.append(("qwen-base-dev", f"eval_fenshen_1015_dev_{ver}"))
if split in ("train", "both"):
    if run_best:
        jobs.append((f"qwen-{mode}-train-best", f"eval_fenshen_1015_train_{ver}"))
    if run_over:
        jobs.append((f"qwen-{mode}-train-over", f"eval_fenshen_1015_train_{ver}"))
    if run_base:
        jobs.append(("qwen-base-full", f"eval_fenshen_1015_train_{ver}"))
for asr_run, eval_name in jobs:
    path = asr_dir / eval_name / f"{asr_run}_asr_{eval_name}.parquet"
    if not path.is_file():
        missing.append(str(path))
        print(f"MISSING {path}")
        continue
    rows = pq.read_table(path).num_rows
    print(f"OK {rows}  {path.as_posix()}")
if missing:
    raise SystemExit(1)
print(f"完成: {mode} {ver} {split} {ckpt} 共 {len(jobs)} 个 parquet")
PY
if [[ "$code_best" -ne 0 || "$code_over" -ne 0 || "$code_base" -ne 0 ]]; then
  exit 1
fi

echo "======== 字准 xlsx ========"
python scripts/eval_fenshen_1015_xlsx.py "$MODE" "$VER" --split "$SPLIT" --ckpt "$CKPT"
