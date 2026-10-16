# VAD 双 batch 的共享 CPU 预算

适用：Linux、128 个逻辑 CPU，同时运行两个 `vad_segment_batch`。
两个 batch 各 8 个分片、每片 2 个父源 worker，共保留 32 路音频并发。
模型 `ncpu` 默认改为 1；新启动时限制 OpenMP、MKL、OpenBLAS、NumExpr 线程为 1，
并设置被动等待以减少小块推理的线程空转。

## 正在运行：立即限制两个任务合计占用

将 `scripts/run_vad_cpu_budget.py` 同步到服务器后，找出两个 batch 的顶层 PID：

```bash
ps -eo pid,ppid,args | grep '[a]udio-data pipeline run pipelines/vad_segment_batch.yaml'
python scripts/run_vad_cpu_budget.py --attach-pids PID_A PID_B
```

`PID_A PID_B` 替换为两个顶层 pipeline 进程，不能只选其中的分片子进程。
脚本会扫描它们的后代，并设置每个已有线程的 CPU 亲和性，两个任务共用同一组 CPU。
不需要停止任务。同一用户通常不需要 root；若权限不足，脚本会报错。
它不修改运行中模型的 `ncpu`，因此当前任务先获得硬限制，线程优化在下一次模型加载时生效。

## 新启动：使用完整的线程优化和 CPU 上限

同步以下文件到服务器，确保实际 Python 环境加载的是更新后的工程代码：

- `src/audio_engine/core/fsmn_vad/adapter.py`
- `src/audio_engine/operators/audio/fsmn_vad_segment.py`
- `pipelines/vad_segment_batch.yaml`
- `scripts/run_vad_cpu_budget.py`

激活原有 Python 环境。服务器原始权重为 `/data2/data-cp/xijujun/models/fsmn-vad`；
运行前设置已 prepare 的 `FSMN_VAD_RUNTIME_DIR`（通常为工程内 `var/models/fsmn-vad-fenshen-v1`），在工程根目录执行：

```bash
python scripts/run_vad_cpu_budget.py BATCH_A BATCH_B
```

将 `BATCH_A BATCH_B` 替换为实际批次名。同一批次不要同时保留旧任务再启动新任务。
需要恢复旧 run 时，先正常停止旧进程，然后指定各自原有 run 目录：

```bash
python scripts/run_vad_cpu_budget.py BATCH_A BATCH_B --resume runs/RUN_A runs/RUN_B
```

`--resume` 使用现有 checkpoint；VAD 的 checkpoint 以分片算子完成为单位，
未完成分片可能重新计算。不要承诺每条已处理父源都可免重算。

## 预算含义和效果验收

默认 `--cpu-percent 30`：向下取整选择 38 个逻辑 CPU，两个 batch 共享，
合计可用计算容量为全机的 `38 / 128 = 29.6875%`。
已有 cpuset 限制时仅使用当前允许的 CPU。脚本启动时会打印实际集合和比例。
这限制的是这两个 VAD 任务及其后代；其他进程、内核工作和监控统计差异不在此预算内。
程序自行重新设置亲和性会绕过此限制；当前引擎不这样做。

不要为两个 batch 分别挑选两组 38 个 CPU，也不要认为仅设环境变量就有 30% 硬上限。
新启动的日志应显示 `source_workers=2`、`ncpu=1`。

以相同数据阶段比较连续 3–5 分钟的两个 batch **合计父源完成条数/秒**，
同时看整机 CPU、各任务 CPU 和错误数；不要混用父源数与输出片段数。
此前约 50 条/秒、98% 的观测不能保证优化后依然 50 条/秒。
若完全没有线程浪费，仅按 CPU 时间缩减估算约为 `50 × 30 / 98 ≈ 15 条/秒`；
这只是预算估算，不是实测。若新方案明显高于该值，说明削减线程开销带来了收益。
若仍达不到目标，保持 CPU 上限，后续评估 ONNX CPU 推理或 GPU 合批。
