# Tag 并发执行与日志

Tag 的共享证据引擎承担吞吐，规则保持独立判断。不同规则继续使用各自声明的词表，本轮没有加入跨规则词表、条件或文本比较结果缓存。

## 并发配置

生产 YAML 的 tag 算子参数已配置：

```yaml
executor: auto
workers: 8
batch_size: 32
checkpoint_every: 100
retain_results: false
engine_workers: {vad: 4, dnsmos: 4, energy: 8}
engine_max_in_flight: {vad: 8, dnsmos: 8, energy: 16}
engine_retries: 0
```

- `workers` 控制文本证据及规则评估调度。`auto` 对小于 4096 条的批次使用线程，较大批次使用进程；声学补算使用线程调度。
- `batch_size` 是每个调度任务的样本数，实际大小会根据输入量、worker 数量和在途上限缩小，避免少量大块占满队列却只启用一个 worker。
- `max_in_flight` 始终以样本数计数，默认 `2 × workers × 实际 batch_size`；显式指定时严格遵守上限。
- VAD、DNSMOS、energy 共用一次运行级声学调度，各自有独立的 worker 和在途上限。即使文本调度只有一个 worker，不同声学引擎也可以同时执行。
- 同一条样本中多个 tag 需要同一种声学事实时，沿用现有依赖合并，只请求一次。已有成功证据直接复用。
- 各声学工作线程独立持有模型会话；完成样本后释放该样本的事实和调用历史，保留模型会话。不同引擎使用隔离的样本副本，再合并输出字段。
- `engine_retries` 是声学引擎失败后最多额外尝试的次数，默认 0。规则异常会结束运行，已采集的成功结果仍保存。
- 调用方注入的声学适配器默认对 `compute` 加锁；声明 `thread_safe=True` 的适配器可自行保证线程安全。默认适配器自动使用运行级引擎调度。

原生产命令继续生效：

```bash
audio-data pipeline run pipelines/tag_dataset_free_tags_v2.yaml --source-name "$BATCH" --config configs/tagging/zh_asr_free_tags_v2.yaml
```

流水线的 `--workers`、`--executor`、`--max-in-flight`、`--checkpoint-every` 会传到 tag 服务。`--max-in-flight 0` 表示自动上限；`--checkpoint-every 0` 只关闭外层流水线检查点，保留 tag 自身恢复状态。

服务入口也支持分块和引擎配置：

```bash
audio-data tags run --batch "$BATCH" --config configs/tagging/zh_asr_free_tags_v2.yaml --workers 8 --batch-size 32 --engine-workers '{"vad":4,"dnsmos":8,"energy":8}' --engine-max-in-flight '{"vad":8,"dnsmos":16,"energy":16}' --engine-retries 1
```

强制进程模式仅用于已有证据模式；在 Windows Python 脚本中直接调用时，入口放在 `if __name__ == "__main__":` 内。

## runs 与执行过程

流水线运行时，tag 的运行视图位于 `runs/<pipeline-run>/tagging/<step>_<hash>/`；独立运行则位于 `runs/tagging/<batch>/<policy>/work/<tag-run-id>/`。两边通过 `pipeline_run.json` 关联，历史 tag 子运行保留各自的状态和指标快照。

- `run.json`：父流水线 ID、tag 子运行 ID、步骤名、PID、配置及输入指纹、实际并发配置、当前阶段、开始/更新时间、最终状态、产物和错误。
- `metrics.json`：各引擎提交、在途、完成、失败、取消、复用、尝试与重试数量，累计/平均/最大耗时，以及整体吞吐。
- `progress_view.json`：处理数、恢复数、已提交数、已完成块数、吞吐和 ETA。
- `tagging.log.jsonl`：结构化事件，含父 run、tag run、步骤、样本和引擎信息；每 50 MB 轮转。
- `definition_snapshot.json`：仍保存在 tag 工作目录，`run.json` 给出完整路径。

阶段包含 initializing、evaluating、checkpointing、exporting、publishing、finalizing，最终为 completed、failed 或 interrupted。运行中每 15 秒刷新指标并输出进度，即使主线程正在等待模型，也能看到引擎在途数量。进程池无法精确观察各进程内部的瞬时运行/排队数量，这两项显示为 null，在途和完成数量仍准确。

使用原有进度入口查看或持续跟随：

```bash
audio-data pipeline progress "runs/<pipeline-run>" --follow
audio-data pipeline progress "runs/tagging/<batch>/<policy>/work/<tag-run-id>" --follow
```

INFO 输出阶段和汇总；逐条样本、块和模型调用写入 TRACE 文件日志；重试及声学失败输出 WARNING；规则或运行异常记录 ERROR 和堆栈。指标中的 completed 包含已结束的失败调用，failed 是其中的失败数量；retries 和 attempts 单独计数。

## 分块持久化与恢复

- 样本结果按检查点写入不可变 JSONL 块，`sample_results.sqlite3` 索引样本位置和提交标记，避免生成百万个单样本 JSON 文件。旧单样本 JSON 仍能读取，并在恢复时导入索引。
- 批次状态使用 `tag_state.sqlite3` 增量事务更新；运行收尾导出兼容的 `tag_state.json` 快照。已有 JSON 状态自动导入，人工操作通过原 API 访问索引状态，并刷新 JSON 快照。索引存在后以 SQLite 为状态来源。
- 结果块、状态、完成标记依次提交。大批次的 progress.json 只保存数量和索引引用，避免每个检查点重写全部完成 ID。
- 流水线恢复时，输入、定义和规则范围指纹一致会自动接续对应 tag 子运行；输入原文或配置变化会新建子运行。
- 相同 tag run 使用由操作系统释放的 SQLite 锁，防止两个进程同时续跑；异常退出或杀进程后，锁会释放。
- 不完整运行不发布新快照；正常异常退出保存已采集的成功结果，强制终止后重新计算尚未提交的工作。

独立续跑仍使用：

```bash
audio-data tags run --batch "$BATCH" --config configs/tagging/zh_asr_free_tags_v2.yaml --resume-run-id "$RUN_ID" --workers 8
```

生产算子和 CLI 默认惰性读取规则结果，审计 JSONL、Parquet 和 XLSX 均分块/流式导出。直接使用 Python 服务 API 时，`retain_results` 默认 true 以兼容旧调用；大批次应设为 false。现有 Manifest 接口仍持有整批输入与输出样本列表，本轮控制的是任务、规则证据和导出中间数据的内存，并未重写整个流水线的输入接口。
