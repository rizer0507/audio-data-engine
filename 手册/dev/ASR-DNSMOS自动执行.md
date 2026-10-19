# ASR + DNSMOS 服务器自动执行

脚本：`scripts/run_asr_dnsmos.py`，使用 Python 标准库和服务器上的 `tmux`，在 Linux 服务器的 `asr` 环境执行。
配置：`configs/asr_dnsmos_server.json`，按两份手工命令保留模型路径、端口、启动参数、ASR 别名和 GPU。

所有 batch 的 GPU 固定为物理卡 **5**，配置中出现其他 GPU 编号会直接报错。
GPU 队列严格串行：`20260813-1-n` 的 Qwen → SenseVoice → GLM，再执行 `20260814-1-n` 的 Qwen → SenseVoice → GLM。
每次切换模型前停止上一个服务并确认 GPU 5 释放。三路 ASR 均单进程。
两批 DNSMOS 使用 CPU，独立并行监控，不等待 GPU 队列；DNSMOS 保留 YAML 分片配置。
CPU 预算方案见 `docs/07-操作手册/DNSMOS线程修复与共享CPU预算.md`。

## 当前已经手工启动的三路任务

默认配置的 `existing_tasks` 对应目前服务器上已经运行的：

- `20260813-1-n/qwen`：`--asr-run qwen1`，使用 GPU 5 上的 Qwen vLLM。
- `20260813-1-n/dnsmos`。
- `20260814-1-n/dnsmos`。

启动时自动扫描服务器本地进程，精确关联三路任务的 run 目录，冻结 PID 与进程启动身份到任务配置快照。
**不会重新启动这三路任务，也不需要先停止它们。** 原手工任务继续留在原来的终端/tmux 中；后续新任务才创建约定名称的新 tmux。
同时绑定 Qwen 服务，要求物理卡 5、5555 端口、Qwen 模型路径及 `qwen3-asr` 服务名全部匹配且唯一。

调度总日志每 30 秒记录三路任务的实际落盘进度：Qwen `run.log` 的条数、百分比、速率、ETA，以及两批 DNSMOS 的分片聚合进度。
进度也写入 `status.json` 的 `tasks`。日志暂无进度或当前块尚未提交时明确显示记录缺失/滞后，不模拟进度。

确认手工 `qwen1` 结束并发布了本次 run 对应的正式产物后，脚本自动停止之前绑定的 Qwen vLLM 及其子进程。
个别样本失败会保留统计和告警，不会阻止 GPU 释放及后续转写。
先 TERM，超时再 KILL；信号发送前核对 PID 的启动身份，随后检查 GPU 5 无计算进程且显存回落。
之后进入下一个 GPU 任务。按当前队列，下一项是 SenseVoice 本地转写（无 vLLM），然后是 GLM vLLM + 转写。
DNSMOS 在 GPU 切换时继续评分；其进度或完成状态不会被当成 GPU 任务。

如果 `qwen1` 或 DNSMOS 在脚本启动之前已经结束，`run_dir: "auto"` 会自动扫描 `runs/`、
`var/asr_dnsmos/` 及 artifact catalog，按 batch、模型、ASR 别名关联最近一次运行。
核对失败统计、正式产物路径、大小、SHA-256 和分片发布状态后，直接复用已完成结果。
不会仅凭进程不存在判成功，也不会在最近一次运行失败后回退到更早的成功记录。
如果 Qwen vLLM 还在运行，自动清理对应服务；已经退出则直接检查 GPU 释放，再接续任务。
如果是在任务运行中启动脚本，也兼容“正式产物已发布，但客户端尚在收尾退出”的情况。

自定义的 run 根目录可以通过配置 `existing_runs_roots` 增补（默认 `["runs", "var/asr_dnsmos"]`）；
只有没有足够证据或最新运行未通过核验时才会停止。需要明确指定时，仍可把 `run_dir` 填为真实目录。
若显式指定的 run 仍在运行，进程打开的 run 目录也必须与指定路径一致。
已有任务的完成判断要求合法的失败统计、当前 run 的 `artifact.json`、非空正式产物；DNSMOS 还要求所有 shard 都发布完成。
`failed=1` 等样本失败统计允许继续，状态显示 `completed_with_errors`，不会被隐藏或当作零失败。

以后启动完全新的批次时，修改 `batches` 并清空 `existing_tasks`，即可由脚本自行创建全部任务。
`dnsmos_parallel` 默认为 `true`；新 DNSMOS 会在 GPU 队列之前启动，和 GPU 工作并行。
当前手工 DNSMOS 的线程/CPU 限额沿用原进程配置，不会通过监控脚本热修改。

## 检查服务器上已经运行的任务

将 `scripts/run_asr_dnsmos.py` 和 `scripts/inspect_asr_dnsmos.py` 放到服务器项目的 `scripts/` 下，
在服务器本地执行。检查已有任务不需要创建任务目录，也不需要重启现有任务：

```bash
cd /data2/data-cp/lizi/audio-data-engine
conda activate asr
python scripts/run_asr_dnsmos.py inspect \
  --batch 20260813-1-n --batch 20260814-1-n

# 每 30 秒刷新一次（Ctrl+C 只结束查看）
watch -n 30 'python scripts/run_asr_dnsmos.py inspect --batch 20260813-1-n --batch 20260814-1-n'
```

`inspect` 只读扫描当前 Linux `/proc`，精确匹配批次参数及项目工作目录，显示进程 PID、ASR 别名和
`CUDA_VISIBLE_DEVICES`。通过进程打开的 run 日志或 `--resume` 参数关联 run 目录，读取：

- Qwen 等单进程任务：`run.log` 最新 `[PROGRESS]`（条数、百分比、速率、ETA）和 checkpoint。
- DNSMOS 分片任务：父 run 的 `progress.json`、`PROGRESS.log`、各 shard checkpoint。
- 最后更新时间、已提交数量和失败数。未提交的当前块不会算入 checkpoint；已处理数可能包含失败。

主脚本的 `inspect` 也会对配置中已没有进程的已有任务执行完成核验，显示“已核验完成”，
因此 `qwen1` 结束后仍能看到它的最终状态，而不是只显示没有找到进程。独立检测脚本保留纯进程/显式目录检查模式。

没有可靠关联时明确显示“尚未确定 run 目录”，不会猜测历史 run 就是当前任务。
如果任务已经退出，或进程信息读取受限，可以显式指定对应目录：

```bash
python scripts/run_asr_dnsmos.py inspect \
  --run-dir runs/20261016_113543_dnsmos_score_batch_20260813-1-n_shards
```

可加 `--json` 输出完整快照，保存或复制用于排查。也可以仅复制 `inspect_asr_dnsmos.py`，
通过 `python scripts/inspect_asr_dnsmos.py --batch ...` 独立检查。
读取权限与执行脚本的 Linux 用户一致；“没有找到进程”或“进度 100%”都不能单独证明整批成功。
心跳/日志文件时间也不能单独证明任务仍在推进，需观察已处理数量是否增长。

## 启动

先同步新增的脚本和配置到服务器，再运行：

```bash
cd /data2/data-cp/lizi/audio-data-engine
conda activate asr

# 可先查看两批各阶段命令，不启动服务或处理数据
python scripts/run_asr_dnsmos.py plan

# 后台启动。该目录用于保存这次任务；新任务请使用新目录。
python scripts/run_asr_dnsmos.py start --job-dir var/asr_dnsmos/aug13-14
```

`start` 将调度器放到后台，并在执行每个服务/流水线时分别创建独立的 tmux 会话，启动返回后可以退出 SSH。
前台调试可以把 `start` 换成 `run`。不要额外手工启动同端口的 vLLM。
默认执行两批；如仅运行一批，复制配置并删除另一批及其 `existing_tasks` 项，启动时传 `--config 配置路径`。

输入依赖已经完成的入库/VAD 产物。`source_dir` 仅作为环境变量保留，与原命令一致，脚本不会重新扫描源音频或执行 VAD。
服务器须已有 `audio-data`、Qwen 的 `vllm`、`nvidia-smi`、`tmux`，以及配置中的 GLM 独立环境和模型资源。
新开终端续跑前也需要 `conda activate asr`。

## 每个任务独立的 tmux

会话名采用 `{batch}-{task}`。以 `20260813-1-n` 为例，按执行阶段创建：

- `20260813-1-n-qwen-vllm`：Qwen 服务；确认就绪后才启动转写。
- `20260813-1-n-asr-run-qwen1`：Qwen `--asr-run qwen1`。
- `20260813-1-n-asr-run-sensevoice1`：SenseVoice 本地转写。
- `20260813-1-n-glm-vllm`：GLM 服务；确认就绪后才启动转写。
- `20260813-1-n-asr-run-glm1`：GLM `--asr-run glm1`。
- `20260813-1-n-dnsmos`：DNSMOS 流水线，其分片仍由流水线管理。

```bash
tmux ls
tmux attach -t '=20260813-1-n-qwen-vllm'
tmux attach -t '=20260813-1-n-asr-run-qwen1'
```

查看后用 `Ctrl+b`，再按 `d` 脱离会话。任务输出会同时显示在 tmux 窗格、写入原来的 `service.log` / `pipeline.log`。
任务完成或失败后保留窗格（`remain-on-exit`），可查看最后输出；看到 `Pane is dead` 表示窗格内任务已退出。
一个阶段结束时仍会清理客户端和服务进程、确认 GPU 释放，再进入后续阶段。保留窗格不代表服务仍在占卡。

同名会话仍在运行，或属于其他任务目录时，脚本拒绝重复启动。`--resume` 只会自动删除并重建
同一任务目录创建、已退出且确认完成清理的同名窗格。已完成阶段会继续跳过。
新建任务目录若需要复用同一批次名，请先确认旧会话已结束，再手动移除对应旧会话。
tmux 会话及其 worker 状态文件路径也会记录在任务的 `status.json` 中。

tmux server 可能早于当前 conda 环境启动；脚本会将当前任务环境传给窗格内的任务，
GLM 仍使用配置指定的独立 Python 环境。无需在每个会话内手工 `conda activate`。

## 查看日志和状态

```bash
# 总日志：阶段起止、命令、30 秒心跳、错误及清理结果
tail -F var/asr_dnsmos/aug13-14/runner.log

# Qwen 服务启动、模型加载、请求等原始日志
tail -F var/asr_dnsmos/aug13-14/20260813-1-n/qwen/service.log

# 当前阶段批处理进度与错误；替换批次/阶段即可查看其他任务
tail -F var/asr_dnsmos/aug13-14/20260813-1-n/qwen/pipeline.log

python scripts/run_asr_dnsmos.py status --job-dir var/asr_dnsmos/aug13-14
```

状态包含 GPU 队列当前 `batch`、`stage`、`phase`、最后更新时间、已完成阶段、错误、PID 和 `process_alive`。
`tasks` 单独记录并行 DNSMOS 和当前 ASR 的来源、run 目录、完成状态及最近一次实际进度。
样本失败会汇总到任务目录的 `sample_failures.json`，包含任务名、失败统计次数、对应 `metrics.json` 路径及分步骤统计。
统计次数可能包括同一样本在不同步骤的失败，不等于去重后的样本数；具体样本 ID 和错误保留在原 run 日志、checkpoint、manifest 中。
全部任务处理结束但含样本失败时，调度器状态为 `completed_with_errors`，退出码为 0，日志保留告警。
`runner.log` 是调度信息，逐样本进度请看各阶段 `pipeline.log`；后台初始化问题看 `launcher.log`。
心跳表示调度进程仍在监控，不能单独证明模型正在推进；需结合批处理日志判断。
日志续跑时追加保留。

## 失败、停止和恢复

单条或若干样本失败（包括某个 DNSMOS shard 的统计 `failed=1`）会记录、汇总并继续后续任务，不自动重跑整批。
服务启动失败、就绪超时、处理中服务退出、流水线/分片进程非零退出、失败统计文件缺失或非法，或 GPU 清理不成功时，整个队列停止，不自动重试。
并行 DNSMOS 的进程异常、未发布整片产物等任务错误也会阻止后续 GPU 任务启动，清理本脚本启动的当前任务。
失败或主动停止监控时，原来手工启动的任务不会被整体强杀；只有已确认完整发布结果的已有转写才触发绑定 vLLM 的自动清理。
服务通过 `/v1/models` 检查对应模型是否就绪。停止阶段时通知 tmux 窗格内的 worker，
由 worker 清理本任务创建的进程组，先 TERM，超时后 KILL；不会使用全局 `pkill`。
调度器读取窗格 worker 发布的真实任务退出码，不将 `tmux new-session` 成功当作任务成功。
GPU 阶段前后检查指定物理卡：无计算进程且显存不超过 `gpu_idle_memory_mb`（默认 1024 MiB）。不满足则等待，超时停止。可按服务器空闲基线修改配置。
同一仓库的调度任务共用文件锁，防止重复启动；端口被外部服务占用时也会停止。

```bash
# 主动停止：会清理当前客户端和服务，状态最终变为 stopped
python scripts/run_asr_dnsmos.py stop --job-dir var/asr_dnsmos/aug13-14

# 修复失败原因后续跑（使用任务目录内的配置快照）
python scripts/run_asr_dnsmos.py start --job-dir var/asr_dnsmos/aug13-14 --resume
```

续跑保留已成功阶段。本脚本启动的失败阶段使用原 pipeline run 目录传递 `--resume`，由已有流水线 checkpoint/cache 决定恢复粒度。
已有手工任务保留绑定身份。若原任务失败，先修复并手工恢复到原 run，再对调度器执行 `--resume`；
脚本会检测新进程，并要求其关联的仍是原 run，才更新绑定。不会把旧 PID 当作新进程使用。
因少量样本失败而标为 `completed_with_errors` 的阶段，在续跑时会跳过，不会重跑整批。
若需要补跑这些样本，按 `sample_failures.json` 定位原 run，再根据算子的 checkpoint/失败样本恢复策略单独排查和补跑。
失败统计检查会递归覆盖 DNSMOS 的 shard run，不仅判断客户端退出码。

配置快照保存在 `var/asr_dnsmos/aug13-14/config.json`。修改仓库默认配置只影响新任务；要修改续跑配置，应在任务停止后编辑其快照，保留批次名、阶段身份和已有产物对应关系。
若需要完全重跑阶段，建议创建新任务目录，确认 ASR 别名及产物覆盖是否符合预期后执行。

这里的“保活”指 SSH 断开后继续执行。服务器重启、调度进程被 SIGKILL 或系统 OOM 后不会自动重启；恢复前先检查状态、日志和 GPU 残留，再手动 `--resume`。
脚本本地经过不依赖模型/GPU 的调度测试，真实服务器的模型加载、显存释放和实际数据执行仍需在服务器验证。
