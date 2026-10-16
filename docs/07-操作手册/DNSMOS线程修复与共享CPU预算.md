# DNSMOS 线程修复与共享 CPU 预算

## 已确认的问题与速度口径

用户提供的 `shard-000.log` 中，11:37:35.142 开始当前块，11:40:05.204 完成 400 条，
约 150 秒完成 400 条，单片约 2.7 条/秒。`400/500` 是 checkpoint 块的累计完成数，
不是每秒 400 条。日志没有其他分片，因此不能据此确认整批总吞吐。

原配置单 batch 同时运行 4 个分片，每片 4 个线程，每个线程持有独立 ONNX 会话。
原会话没有设置内部线程数和忙等待，每个会话都可能创建按物理核心数扩张的线程池。
现在保留原来的 16 路评分并发，默认改为每会话 2 个计算线程、图顺序执行、关闭忙等待。
官方模型支持动态 batch，同一片段最多 4 个窗口一起推理，保留逐窗校准后平均的计算顺序。
固定 batch=1 的模型继续逐窗调用。
模型摘要按路径和文件状态缓存，避免每条音频重复读取整个模型；模型替换/修改会失效。
会话缓存包含线程配置，避免误复用旧设置。

ONNX 官方线程说明：<https://onnxruntime.ai/docs/performance/tune-performance/threading.html>。

## 将所有目标任务限制在一个共享组

必须将 VAD 和 DNSMOS 的所有目标 batch 纳入同一个组。仅更新 YAML、减少内部线程或
设置环境变量，不能保证整机占用低于 30%。旧的 `run_vad_cpu_budget.py` 亲和性方式
不作为 ONNX 的严格保证：默认 ONNX 会自行设置线程亲和性。

新工具使用 Linux cgroup v2 的 `cpu.max`。128 逻辑 CPU 时，共享配置为
`3800000 100000`，即 38 个 CPU 的等效计算时间，约为全机的 29.6875%。
配额按调度周期执行；非常短的采样窗口仍可能看到瞬时尖峰。
该比例约束组内任务，其他服务和内核额外开销另算，不能保证整机全部进程总计小于 30%。
组内重新设置线程亲和性不会突破 cgroup 限额。

在服务器激活原有 Python 环境，在项目根目录先配置一次：

```bash
sudo "$(command -v python)" scripts/audio_cpu_budget.py configure
```

需要 sudo 是因为创建和写入系统 cgroup 需要管理员权限。工具不会修改或关闭其他资源控制器。
无 cgroup v2、CPU controller 或写权限时明确报错，不会悄悄退回无硬限额执行。
重启服务器后需重新配置该临时组。

### 已经运行的任务

查出每个目标 batch 的顶层 pipeline PID，将全部 PID 放到同一次命令中：

```bash
sudo "$(command -v python)" scripts/audio_cpu_budget.py attach PID_VAD_A PID_VAD_B PID_DNSMOS_A
```

工具迁移这些父进程及所有已有后代；迁移一个 PID 会同时迁移它的所有线程，后续子进程
继承同一个组。迁移无需中断任务，但当前 ONNX 会话内部线程设置不会热更新，需重新加载模型。
若只迁移 DNSMOS，其他仍在组外的 VAD 不受此共享限额约束。

### 后续启动或恢复

更新下述文件后，使用带预算的入口启动 DNSMOS：

```bash
sudo -E "$(command -v python)" scripts/audio_cpu_budget.py run -- \
  "$(command -v audio-data)" pipeline run pipelines/dnsmos_score_batch.yaml \
  --source-name "$BATCH"
```

恢复用户日志中的原 run，可以附加：

```bash
sudo -E "$(command -v python)" scripts/audio_cpu_budget.py run -- \
  "$(command -v audio-data)" pipeline run pipelines/dnsmos_score_batch.yaml \
  --source-name 20260813-1-n \
  --resume runs/20261016_113543_dnsmos_score_batch_20260813-1-n_shards
```

先正常停止同批次旧任务，避免同时写同一个 run。已有 checkpoint 会复用；未完成的
checkpoint 块可能重算。VAD 同样使用这个入口，仅将 YAML 换成 `vad_segment_batch.yaml`。
所有这些独立启动的任务都加入固定的 `/sys/fs/cgroup/audio-engine-cpu`，共用一份配额。
`run` 在进入 cgroup 后降回执行 sudo 的原用户，再执行评分命令，不以 root 处理音频。
同时显式限制 OpenMP/MKL/OpenBLAS/NumExpr 线程为 1，设置被动等待。

普通 `audio-data pipeline run ...` 会采用新的 ONNX 线程设置，但不会自动进入共享 cgroup。

## 部署文件

- `src/audio_engine/core/quality/dnsmos_p835.py`
- `src/audio_engine/operators/quality/dnsmos.py`
- `src/audio_engine/core/evidence/scoring.py`
- `configs/quality/dnsmos_p835.yaml`（保留服务器原有 `model_path`）
- `pipelines/dnsmos_score_batch.yaml`
- `scripts/audio_cpu_budget.py`
- `scripts/run_vad_cpu_budget.py`（新工具复用其中的进程树和线程环境辅助代码）
- `scripts/benchmark_dnsmos_threads.py`

VAD 原有的 `ncpu=1` 修复也需保留。更新配置文件不影响已加载会话，需重新加载才能生效。
新 DNSMOS 日志应出现 `intra_threads=2 inter_threads=1 spinning=False`。

## 速度不下降的验收

本地 16 逻辑核、真实官方 ONNX 模型的短测试发现：单线程减少 CPU 时间但会降低吞吐，
因此没有将单线程作为 DNSMOS 最终默认。选择 2 个计算线程是待服务器验收的方案，
并不构成“128 核机器限额后一定不降速”的承诺。

最终双线程 + 每片段 4 窗口合批，在本地 16 逻辑核 Windows、真实模型与 40 条真实 WAV 上，
一次对比得到原版 2.03 条/秒、修复后 2.18 条/秒（约 +7.1%），进程 CPU 时间从 273.03 秒
降至 133.70 秒（约 -51.0%），最大分数差 2.05e-7。未将服务器 cgroup 配额应用到该本地测试。
原始结果保存在 `var/dnsmos_thread_benchmark_real_wav_batched.json`。
这只证明该次本地测试没有降速，不代表服务器吞吐保证。

先在服务器用同一批实际片段比较模型推理，绕过结果缓存：

```bash
python scripts/benchmark_dnsmos_threads.py \
  --model-path /实际路径/sig_bak_ovr.onnx \
  --audio-dir /实际片段目录 --limit 500 --workers 4 \
  --output var/dnsmos_thread_benchmark_server.json
```

测试顺序为旧版默认线程池、修复后线程池；输入、模型、worker 数相同，分别预热。
输出包含条数/秒、进程 CPU 时间、最大分数差，速度下降或分数差超过 1e-4 时退出码为 1。
避免同时运行其他负载干扰比较；该工具只比较一个分片的推理，不替代整条流水线验收。

最终验收需在同样 batch 数和同一组片段上，比较多个完整 checkpoint 块的合计完成数/秒，
排除冷启动和缓存命中比例差异，同时检查错误数、分数一致性，以及共享组的 CPU 使用。
单片约 2.7 条/秒仅为给定日志的参考，不能拿 VAD 的 50 条/秒作 DNSMOS 基线。
若不降速未通过，保留硬上限并继续测线程/worker 组合或 GPU；不要通过放宽 CPU 配额伪造达标。

可以读取共享组的 `cpu.stat`，用两个采样点的 `usage_usec` 差值计算：
`CPU% = Δusage_usec / (采样秒数 × 1e6 × 128) × 100`。
`nr_throttled` / `throttled_usec` 的变化能帮助判断是否正在碰到配额。
Linux 控制组说明：<https://docs.kernel.org/admin-guide/cgroup-v2.html>。

## 本次验证记录

- DNSMOS 线程/窗口合批/摘要缓存、cgroup 预算、生产编排、VAD 调度和分片相关测试共 141 项通过。
- 修改文件的 Python 未定义名称与导入检查通过；新增脚本和测试的完整 Ruff 检查通过。
- 额外运行人审回归时，`test_adjudication_cannot_replace_missing_second_review` 失败；
  本次未修改该人审实现，保留该失败供相关工作排查。
- 本地真实模型和真实 WAV 的评分一致性/速度对比通过。
- 未接入 Linux 生产服务器，cgroup 系统调用按测试替身验证；服务器真实限额和吞吐验收尚未完成。
