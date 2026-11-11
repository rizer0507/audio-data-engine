#!/usr/bin/env bash
# 评测 fenshen_1015 的 train / dev，转写结束后写出字准 xlsx。
# 第三个参数默认 dev，第四个参数默认 best。
# 先把 gold 放到 datasets/stage1/derived，再自行拉起要用的端口。
#   full: best 5560, over 5561
#   lora: best 5565, over 5566
#   rl: 固定 5580；sophon: 固定 6179。两个模式都只提供 best。
#   qwen-base: 5555。第四个参数为 all 时总是转写。
#   只转写 best/over 时，本次切分已有 qwen-base parquet 就跳过 5555；没有则调用 5555 补转写。
# 字准 xlsx 始终和 qwen-base 比较；lora-full 将 LoRA 和 Full 放进同一份报表。
#
# 选中的端口同时跑。qwen_asr_batch 单次已是 8 个分片打同一个 vLLM，
# 同一端口不再并行叠 train 和 dev。两边都跑时，端口内部 dev 结束后立刻接下一条。
# 5555 的 --asr-run：dev=qwen-base-dev，train=qwen-base-full。
#
# 用法:
#   bash scripts/run_fenshen_1015_eval.sh full v014
#   bash scripts/run_fenshen_1015_eval.sh lora-full v030
#   bash scripts/run_fenshen_1015_eval.sh full v014 train best
#   bash scripts/run_fenshen_1015_eval.sh full v014 dev over
#   bash scripts/run_fenshen_1015_eval.sh full v014 both best-over
#   bash scripts/run_fenshen_1015_eval.sh lora v014 both all
#   bash scripts/run_fenshen_1015_eval.sh rl v025
#   bash scripts/run_fenshen_1015_eval.sh sophon v032
#   SOPHON_PROFILE=legacy bash scripts/run_fenshen_1015_eval.sh sophon v032

set -euo pipefail

MODE="${1:-}"
VER="${2:-}"
SPLIT="${3:-dev}"
CKPT="${4:-best}"

usage() {
  echo "用法: bash scripts/run_fenshen_1015_eval.sh <lora|full|lora-full|rl|sophon> <v0xx> [train|dev|both] [all|best|over|best-over]"
  echo "示例: bash scripts/run_fenshen_1015_eval.sh lora-full v030（默认 dev best）"
  echo "示例: bash scripts/run_fenshen_1015_eval.sh full v014 train best"
  echo "示例: bash scripts/run_fenshen_1015_eval.sh full v014 dev over"
  echo "示例: bash scripts/run_fenshen_1015_eval.sh full v014 both best-over"
  echo "示例: bash scripts/run_fenshen_1015_eval.sh rl v025"
  echo "示例: bash scripts/run_fenshen_1015_eval.sh sophon v032（默认 dev best）"
  echo "Sophon 默认 SOPHON_PROFILE=v1，回退可设 SOPHON_PROFILE=legacy。"
  echo "full 的 best 在 5560，over 在 5561。lora 的 best 在 5565，over 在 5566。rl 固定在 5580，sophon 固定在 6179，两者无 over。"
  echo "字准表始终和 qwen-base 比较。只转写 best/over 时，没有 base parquet 才会调用 5555。"
}

case "$MODE" in
  lora-full) models=(lora full) ;;
  lora|full|rl|sophon) models=("$MODE") ;;
  *) usage; exit 1 ;;
esac

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
PORT_SOPHON=6179
SOPHON_RUN_TAG=""
QWEN_PIPELINE="pipelines/qwen_asr_batch.yaml"
if [[ "$MODE" == "sophon" ]]; then
  SOPHON_PROFILE="${SOPHON_PROFILE:-v1}"
  if [[ ! "$SOPHON_PROFILE" =~ ^(legacy|v[0-9]+(-[a-z0-9]+)*)$ ]]; then
    echo "无效的 SOPHON_PROFILE: $SOPHON_PROFILE" >&2
    exit 1
  fi
  SOPHON_PIPELINE="pipelines/sophon_asr_${SOPHON_PROFILE}.yaml"
  if [[ "$SOPHON_PROFILE" == "legacy" ]]; then
    SOPHON_PIPELINE="$QWEN_PIPELINE"
  fi
  if [[ ! -f "$SOPHON_PIPELINE" ]]; then
    echo "缺少 Sophon 流水线配置: $SOPHON_PIPELINE" >&2
    exit 1
  fi
fi
single_port=0
if [[ "$MODE" == "rl" || "$MODE" == "sophon" ]]; then
  single_port=1
  if [[ "$CKPT" == "over" || "$CKPT" == "best-over" ]]; then
    echo "${MODE} 只有 best 端口，请用 all 或 best"
    usage
    exit 1
  fi
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
if [[ "$single_port" -eq 0 && ( "$CKPT" == "all" || "$CKPT" == "over" || "$CKPT" == "best-over" ) ]]; then
  want_over=1
fi
# 单端口模式的 all 对应 base+best 字准表。
XLSX_CKPT="$CKPT"
if [[ "$single_port" -eq 1 && "$CKPT" == "all" ]]; then
  XLSX_CKPT="best"
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
port_models=()
port_ckpts=()
for model in "${models[@]}"; do
  case "$model" in
    full) best_port=5560; over_port=5561 ;;
    lora) best_port=5565; over_port=5566 ;;
    rl) best_port=5580 ;;
    sophon) best_port="$PORT_SOPHON" ;;
  esac
  if [[ "$want_best" -eq 1 ]]; then
    ports+=("$best_port")
    port_models+=("$model")
    port_ckpts+=(best)
  fi
  if [[ "$want_over" -eq 1 ]]; then
    ports+=("$over_port")
    port_models+=("$model")
    port_ckpts+=(over)
  fi
done
if [[ "$want_base" -eq 1 ]]; then
  ports+=("$PORT_BASE")
  port_models+=(base)
  port_ckpts+=(base)
fi
SOPHON_MODEL="$(python - "$MODE" "$PORT_SOPHON" "${ports[@]}" <<'PY'
import json
import sys
import urllib.request

mode, sophon_port, *ports = sys.argv[1:]
for port in ports:
    url = f"http://127.0.0.1:{port}/v1/models"
    print(f"GET {url}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=15) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    payload = json.loads(body)
    names = [item.get("id") for item in payload.get("data") or []]
    print(f"port {port} models: {names}", file=sys.stderr)
    if mode == "sophon" and port == sophon_port:
        model = next((name for name in names if name), None)
        if model is None:
            raise SystemExit(f"端口 {port} 没有返回可用的 Sophon 模型 ID")
        print(model)
PY
)"

run_one() {
  local split="$1"
  local port="$2"
  local asr_run="$3"
  local eval_name
  local pipeline="$QWEN_PIPELINE"
  local run_args=()
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
  if [[ "$MODE" == "sophon" && "$port" == "$PORT_SOPHON" ]]; then
    pipeline="$SOPHON_PIPELINE_SNAPSHOT"
    # Bind each split's shards to this run directory. --force disables checkpoint reuse.
    run_args+=(--resume "$LOG_DIR/$split")
    export QWEN_ASR_CHAT_FORMAT=sophon
    export QWEN_ASR_MODEL="$SOPHON_MODEL"
  fi
  audio-data pipeline run "$pipeline" \
    --eval-name "$eval_name" \
    --asr-run "$asr_run" \
    "${run_args[@]}" \
    --force
}

run_port() {
  local port="$1"
  local model="$2"
  local ckpt="$3"
  export PYTHONUNBUFFERED=1
  echo "======== 端口 ${port} 开始 ========"
  if [[ "$model" == "base" ]]; then
    if [[ "$need_base_dev" -eq 1 ]]; then
      run_one dev "$port" qwen-base-dev
    fi
    if [[ "$need_base_train" -eq 1 ]]; then
      run_one train "$port" qwen-base-full
    fi
  else
    if [[ "$model" == "sophon" ]]; then
      ckpt="${ckpt}-${SOPHON_RUN_TAG}"
    fi
    if [[ "$want_dev" -eq 1 ]]; then
      run_one dev "$port" "qwen-${model}-dev-${ckpt}"
    fi
    if [[ "$want_train" -eq 1 ]]; then
      run_one train "$port" "qwen-${model}-train-${ckpt}"
    fi
  fi
  echo "======== 端口 ${port} 结束 ========"
}

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="runs/fenshen_1015_eval_${MODE}_${VER}_${SPLIT}_${CKPT}_${RUN_STAMP}"
if [[ "$MODE" == "sophon" ]]; then
  mkdir -p runs
  LOG_DIR="$(mktemp -d "${LOG_DIR}_${SOPHON_PROFILE}_XXXXXX")"
  SOPHON_RUN_TAG="${SOPHON_PROFILE}-${RUN_STAMP}-${LOG_DIR##*_}"
  SOPHON_PIPELINE_SNAPSHOT="$LOG_DIR/sophon.pipeline.yaml"
  python - "$SOPHON_PIPELINE" "$SOPHON_PIPELINE_SNAPSHOT" <<'PY'
from pathlib import Path
import shutil
import sys

import yaml

source, snapshot = map(Path, sys.argv[1:])
original = source.read_bytes()
snapshot.with_name("sophon.profile.yaml").write_bytes(original)
pipeline = yaml.safe_load(original)
for index, step in enumerate(pipeline["pipeline"]):
    params = step.get("params", {})
    for key in ("config_path", "normalization_path", "blank_exact_hotwords_path"):
        if params.get(key):
            dependency = snapshot.with_name(f"{index:02d}-{key}.yaml")
            shutil.copyfile(params[key], dependency)
            params[key] = dependency.as_posix()
snapshot.write_text(yaml.safe_dump(pipeline, allow_unicode=True, sort_keys=False), encoding="utf-8")
PY
  printf 'profile=%s\nrun_tag=%s\npipeline=%s\napi_base=http://127.0.0.1:%s\nmodel=%s\n' \
    "$SOPHON_PROFILE" "$SOPHON_RUN_TAG" "$SOPHON_PIPELINE" "$PORT_SOPHON" "$SOPHON_MODEL" \
    >"$LOG_DIR/sophon.run.txt"
  echo "Sophon 配置快照: $SOPHON_PIPELINE_SNAPSHOT  run_tag: $SOPHON_RUN_TAG"
else
  mkdir -p "$LOG_DIR"
fi
job_names=()
running=()
for i in "${!ports[@]}"; do
  name="${port_ckpts[i]}"
  if [[ "$MODE" == "lora-full" && "${port_models[i]}" != "base" ]]; then
    name="${port_models[i]}-${name}"
  fi
  job_names+=("$name")
  running+=("$name :${ports[i]}")
done
echo "并行: ${running[*]}"
if [[ "$SPLIT" == "both" ]]; then
  echo "同端口内先 dev 再 train"
else
  echo "只跑 ${SPLIT}"
fi

job_pids=()
tail_logs=()
for i in "${!ports[@]}"; do
  log="${LOG_DIR}/${job_names[i]}.log"
  echo "日志: ${log}"
  run_port "${ports[i]}" "${port_models[i]}" "${port_ckpts[i]}" >"$log" 2>&1 &
  job_pids+=("$!")
  tail_logs+=("$log")
done
tail -n +1 -f "${tail_logs[@]}" &
pid_tail=$!

cleanup_jobs() {
  kill "${job_pids[@]}" "$pid_tail" 2>/dev/null || true
  exit 130
}
trap cleanup_jobs INT TERM

failed=0
for i in "${!job_pids[@]}"; do
  code=0
  wait "${job_pids[i]}" || code=$?
  echo "${job_names[i]} 退出码 ${code}  日志 ${tail_logs[i]}"
  if [[ "$code" -ne 0 ]]; then
    failed=1
  fi
done
trap - INT TERM
sleep 1
kill "$pid_tail" 2>/dev/null || true
wait "$pid_tail" 2>/dev/null || true

echo "======== 产物 ${ASR_DIR} ========"
python - "$MODE" "$VER" "$ASR_DIR" "$SPLIT" "$XLSX_CKPT" "$SOPHON_RUN_TAG" <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, "scripts")
from eval_fenshen_1015_xlsx import asr_jobs

mode, ver, asr_dir, split, ckpt = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4], sys.argv[5]
run_tag = sys.argv[6] or None
missing = []
jobs = []
for selected_split in ("dev", "train") if split == "both" else (split,):
    eval_name = f"eval_fenshen_1015_{selected_split}_{ver}"
    jobs.extend((run, eval_name) for _, run in asr_jobs(mode, selected_split, ckpt, sophon_run_tag=run_tag))
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
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

echo "======== 字准 xlsx ========"
xlsx_args=("$MODE" "$VER" --split "$SPLIT" --ckpt "$XLSX_CKPT")
if [[ "$MODE" == "sophon" ]]; then
  xlsx_args+=(--sophon-run-tag "$SOPHON_RUN_TAG")
fi
python scripts/eval_fenshen_1015_xlsx.py "${xlsx_args[@]}"
