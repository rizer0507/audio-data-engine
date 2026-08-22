# SenseVoice 高并发识别流水线构建文档

## 1. 建设目标

在 Qwen-ASR 批量识别完成后，以其输出 manifest 为输入，再执行一次
SenseVoice 识别，最终在同一条样本记录中同时保留两套结构化结果：

```text
清洗后的 manifest
  → Qwen-ASR 高并发识别
  → qwen_asr_source_A.parquet
  → SenseVoice 高并发识别
  → qwen_sensevoice_source_A.parquet
  → 差异计算、聚合统计、badcase 导出（后续阶段）
```

本次建设的重点是 **SenseVoice 模型常驻、批量推理、分片并发、逐样本容错和结构化落盘**。
不在本阶段直接实现比对规则，但输出契约必须能够支撑后续 Qwen-ASR 与 SenseVoice 的
文本一致率、字错率、语言、情感和音频事件等维度的统计。

## 2. 当前实现与必须改造的问题

仓库已有 `asr.sensevoice`，但当前实现不能用于十万级音频的正式批处理：

1. `_call_model()` 每处理一条样本都会执行一次 `AutoModel(...)`，模型无法常驻复用；
2. Operator 是逐样本执行，没有利用 SenseVoice/FunASR 的批量推理能力；
3. 推理异常会返回 mock 文本，真实失败与伪造结果无法区分；
4. 只保存 `text/model/version`，没有规范化 SenseVoice 文本中的语言、情感、事件标签；
5. 现有 `baseline_sensevoice.yaml` 从原始 manifest 开始，不能明确表达“Qwen 完成后再跑”；
6. 缺少高并发 SenseVoice 专用流水线、失败隔离测试和部署参数说明。

因此不能只新增一份 YAML；应仿照 `asr.qwen_batch` 建设
`asr.sensevoice_batch`，并保留原 `asr.sensevoice` 兼容已有流水线。

## 3. 设计原则

### 3.1 两阶段运行，不把两个大模型装入同一进程

Qwen-ASR 和 SenseVoice 使用两个独立的 `run-shards` 作业。SenseVoice 阶段读取已经包含
`transcripts.qwen` 的 manifest，并在其上追加 `transcripts.sensevoice`。这样可以：

- 避免同一进程同时驻留两个模型造成显存竞争；
- 分别调优两个模型的 batch size 与每卡实例数；
- SenseVoice 失败时不必重跑 Qwen-ASR；
- 直接以 Qwen 阶段产物作为可恢复的阶段边界。

### 3.2 一进程一模型，进程内批处理

并发分成两层：

- **进程间并发**：manifest 按时长均衡切分，由 `run-shards` 把进程分配到 GPU；
- **进程内并发**：每个进程只加载一次 SenseVoice，然后按 `batch_size` 批量调用
  `AutoModel.generate()`。

不要在线程之间共享 CUDA 模型。模型缓存只保证同一进程、同一配置复用；跨进程各自拥有
模型实例。

### 3.3 逐样本可追溯、可恢复

每条样本都必须有独立的 cache key、状态、错误信息和 lineage。批次失败时降级为单条重试，
坏音频只能使自身失败，不能丢弃整个 batch 或 shard。中断后依靠 cache 与 checkpoint 续跑。

## 4. 计划新增和修改的文件

| 文件 | 变更 |
| --- | --- |
| `src/audio_engine/operators/asr/sensevoice.py` | 增加配置解析、进程内模型缓存、批量推理、标签解析及 `SenseVoiceBatchASROperator` |
| `src/audio_engine/operators/asr/__init__.py` | 导出新的批量 Operator |
| `configs/asr/sensevoice.yaml` | 补充设备、batch、语言、ITN、模型版本及推理参数 |
| `pipelines/sensevoice_asr_batch.yaml` | 新增以 Qwen 结果 manifest 为输入的 SenseVoice 阶段 |
| `tests/test_sensevoice_asr.py` | 覆盖模型单次加载、批处理、缓存、失败隔离和结构化结果 |

推荐注册名：

```text
类名：SenseVoiceBatchASROperator
name：sensevoice_batch
完整 operator：asr.sensevoice_batch
```

## 5. 模型加载设计

### 5.1 配置优先级

参数按以下优先级解析，便于服务器部署时覆盖模型位置：

```text
流水线 params.model_path
  > 环境变量 SENSEVOICE_MODEL_PATH
  > configs/asr/sensevoice.yaml:model_path
  > iic/SenseVoiceSmall
```

服务端应优先使用已经下载完成的本地模型目录，避免批量任务启动时访问公网。建议配置：

```yaml
model: sensevoice-small
model_version: "<固定的本地模型版本或提交号>"
model_path: /data2/data-cp/zcl/models/SenseVoiceSmall
device: cuda:0
batch_size: 32
language: auto
use_itn: true
disable_update: true
```

`model_version` 不应使用“latest”，应记录可复现的版本号或模型提交号。服务器实际路径确认前，
不得把示例路径当作已验证路径。

### 5.2 进程内缓存

仿照 Qwen-ASR，模块内维护 `_MODEL_CACHE` 和 `_MODEL_LOCK`。缓存键至少包含：

```json
{
  "model_path": "...",
  "device": "cuda:0",
  "model_revision": "...",
  "vad_model": null,
  "punc_model": null
}
```

加载函数 `_load_sensevoice_model(settings)` 在锁内检查缓存，并且只在未命中时创建：

```python
AutoModel(
    model=settings["model_path"],
    device=settings.get("device", "cuda:0"),
    disable_update=settings.get("disable_update", True),
)
```

生产模式缺少 `funasr`、模型加载失败或 CUDA 初始化失败时必须抛出异常，交给 Operator 标记
失败；严禁静默返回 mock。只有显式 `--mock` 或 `params.mock: true` 才能产生 mock 结果。

### 5.3 模型预热

每个 shard worker 在处理首个真实 batch 前完成模型加载。可选增加一条短音频做 warm-up，并把
加载耗时、预热耗时写入 run log。预热失败应直接暴露为该 shard 的启动错误，不能进入正式统计。

## 6. 批量推理与高并发设计

### 6.1 BatchOperator 行为

`SenseVoiceBatchASROperator` 对齐 `QwenBatchASROperator`：

1. 遍历输入样本，跳过已完成项并读取逐样本缓存；
2. 把待推理样本按 `batch_size` 切块；
3. 一次传入一组音频路径进行推理；
4. 校验返回条数与输入条数一致；
5. 批次失败时逐条重试，隔离损坏音频；
6. 成功项写 cache、lineage 和 `status["asr.sensevoice_batch"] = "completed"`；
7. 失败项写 `errors["asr.sensevoice_batch"]`，不得写虚假的 transcript。

FunASR 版本间批量参数可能不同。实现时应以服务器锁定版本实际验证
`AutoModel.generate(input=[...], batch_size=...)` 的调用形式；不得通过捕获所有异常并返回 mock
来掩盖 API 不兼容。

### 6.2 初始并发参数

SenseVoiceSmall 与 Qwen3-ASR 的显存特征不同，不直接照搬“每卡 2 实例、batch=32”的结论。
建议从以下保守基线压测：

```text
4 张 GPU × 每卡 1 个进程 = 4 shards 并发
每进程 batch_size = 32
```

依次测试 `batch_size=16/32/64`，再测试每卡 1/2 个实例。选择满足以下条件的组合：

- 峰值显存保留至少 15% 余量；
- 无 OOM、无长时间 GPU stall；
- samples/s 或 audio-hours/hour 最优；
- 错误率和结果内容不因 batch 改变而异常。

最终生产值应记录在压测报告和流水线配置中，而不是仅写在启动命令里。

### 6.3 分片策略

音频长度差异较大时必须使用 `duration-balanced`，避免最后一个长音频 shard 拖慢整体任务。
分片数应等于实际并发 worker 数；若需要更细粒度的失败重跑，可使用 worker 数的整数倍，
但当前 `run-shards` 的调度与模型重复加载成本需先压测确认。

## 7. 结构化识别结果契约

### 7.1 Manifest 中的结果

Qwen 阶段已有的数据不能被覆盖。SenseVoice 完成后，一条样本的核心结构如下：

```json
{
  "id": "sample-000001",
  "transcripts": {
    "qwen": {
      "text": "客户表示暂时不需要",
      "model": "Qwen3-ASR-1.7B",
      "version": "...",
      "extra": {"language": "Chinese"}
    },
    "sensevoice": {
      "text": "客户表示暂时不需要",
      "model": "sensevoice-small",
      "version": "<固定版本>",
      "confidence": null,
      "extra": {
        "raw_text": "<|zh|><|NEUTRAL|><|Speech|>客户表示暂时不需要",
        "language": "zh",
        "emotion": "NEUTRAL",
        "events": ["Speech"]
      }
    }
  },
  "status": {
    "asr.qwen_batch": "completed",
    "asr.sensevoice_batch": "completed"
  },
  "errors": {},
  "lineage": []
}
```

字段约束：

- `text`：移除 SenseVoice 控制标签后的可比对纯文本；
- `extra.raw_text`：模型原始输出，便于解析逻辑升级后回溯；
- `extra.language/emotion/events`：从原始标签规范化出的结构字段；
- `confidence`：模型未可靠返回时使用 `null`，不能伪造为 `1.0`；
- `model/version`：必须来自解析后的最终配置；
- `qwen_text`、`sensevoice_text`：由现有 manifest 扁平化逻辑派生，便于 DataFrame 查询。

标签解析器应单独写成纯函数并测试。未知标签保留在 `extra.unknown_tags`，不能静默丢失；
纯文本清洗只处理已知控制标签，不应误删用户实际说出的尖括号内容。

### 7.2 为后续比对预留的统计输出

后续 `quality.transcript_diff` 应读取 `transcripts.qwen.text` 与
`transcripts.sensevoice.text`，将结果写入 `quality.asr_comparison`，建议契约为：

```json
{
  "quality": {
    "asr_comparison": {
      "normalizer_version": "v1",
      "qwen_normalized": "客户表示暂时不需要",
      "sensevoice_normalized": "客户表示暂时不需要",
      "exact_match": true,
      "cer": 0.0,
      "edit_distance": 0,
      "qwen_chars": 10,
      "sensevoice_chars": 10
    }
  }
}
```

识别原文永远保留在 `transcripts`，归一化文本和指标只写入 `quality`，避免统计清洗覆盖模型输出。

## 8. 流水线配置

新增 `pipelines/sensevoice_asr_batch.yaml`：

```yaml
name: sensevoice_asr_batch

input:
  manifest: datasets/manifests/qwen_asr_source_A.parquet

output:
  manifest: datasets/manifests/qwen_sensevoice_source_A.parquet

execution:
  executor: sequential
  workers: 1
  fail_fast: false
  checkpoint_every: 500

pipeline:
  - name: sensevoice_asr
    operator: asr.sensevoice_batch
    params:
      input_audio_key: resampled_16k
      config_path: configs/asr/sensevoice.yaml
```

输入验收条件：

- `id` 唯一且稳定；
- `audio.resampled_16k` 存在且文件可读；
- `transcripts.qwen.text` 已存在；
- Qwen 失败样本的处理策略已明确：默认跳过，只对 Qwen 成功样本运行 SenseVoice；若为了统计
  SenseVoice 独立成功率而保留这些样本，应通过显式配置开启并单独统计。

实现跳过策略时，建议增加前置条件参数（例如
`required_status: asr.qwen_batch=completed`），不要依赖空字符串判断成功状态。

## 9. 生产运行流程

以下以 4 个 worker 的保守起始配置为例。确认 Qwen 阶段已完成后执行：

```bash
audio-data manifest shard datasets/manifests/qwen_asr_source_A.parquet \
  --shards 4 \
  --strategy duration-balanced \
  --output-dir datasets/shards/sensevoice_source_A

audio-data pipeline run-shards pipelines/sensevoice_asr_batch.yaml \
  --shard-dir datasets/shards/sensevoice_source_A \
  --parallel-shards 4 \
  --gpus 0,1,2,3 \
  --instances-per-gpu 1 \
  --run-root runs/sensevoice_source_A

audio-data manifest merge "runs/sensevoice_source_A/shard-*.parquet" \
  --output datasets/manifests/qwen_sensevoice_source_A.parquet \
  --expected-shards 4
```

合并前必须检查：

1. 实际 shard 数与 `--expected-shards` 一致；
2. 合并后样本 `id` 无重复、无意外丢失；
3. 成功、失败、跳过数量之和等于输入数量；
4. Qwen transcript 在 SenseVoice 阶段前后数量和内容一致；
5. 非 mock 生产任务中不存在以 `[mock:sensevoice:` 开头的结果。

## 10. 监控与容量指标

每个 run 至少记录：

- 输入、成功、失败、跳过、cache hit 数量；
- 模型加载时间、总运行时间、每 batch 推理耗时；
- 实际音频总时长、samples/s、audio-hours/hour、实时率 RTF；
- batch size、GPU、worker 数、模型路径与版本；
- 峰值显存、OOM 次数、批次降级为单条重试次数；
- 空文本率、语言分布、情感分布、事件标签分布。

告警建议：失败率超过 0.5%、空文本率相对抽样基线突增、出现 mock 文本、某 shard 吞吐显著
低于中位数或合并后样本数不守恒时，禁止进入比对统计阶段。

## 11. 测试计划

### 11.1 单元测试

- 相同解析配置连续调用时 `AutoModel` 只构造一次；
- 不同模型路径或设备使用不同缓存键；
- 7 条样本、`batch_size=3` 时调用批次大小为 `3/3/1`；
- 输出顺序与输入顺序完全一致；
- 批量调用失败后逐条重试，坏文件失败、其余样本成功；
- 返回数量不一致时显式失败；
- cache hit 不进入模型推理；
- mock 只有显式开启时生效；
- `raw_text` 的语言、情感、事件标签正确解析；
- 未知标签被保留；
- 结果写入 `transcripts.sensevoice` 且不覆盖 `transcripts.qwen`。

### 11.2 集成与压测

```bash
pytest -q tests/test_sensevoice_asr.py
pytest -q
audio-data pipeline run pipelines/sensevoice_asr_batch.yaml --mock
```

随后用 100、1,000、10,000 条真实音频逐级压测。每一级都核对结果守恒、内容抽样、失败重跑、
断点续跑以及 GPU 指标；未通过前不得直接提交十万条全量任务。

## 12. 实施顺序与验收标准

### 阶段 A：模型加载与结构解析

完成配置解析、环境变量覆盖、模型单例缓存和 SenseVoice 标签解析。用一条真实 WAV 验证模型
只加载一次且输出同时包含纯文本与原始文本。

### 阶段 B：批量 Operator

实现 `asr.sensevoice_batch`、逐样本缓存、失败隔离、lineage/status/errors，并通过 fake model
单元测试。此阶段不依赖 GPU 即可覆盖主要控制流。

### 阶段 C：分片流水线

新增 YAML，使用 Qwen 输出 manifest 完成 mock 集成测试，再在单卡上跑 100 条真实音频。

### 阶段 D：并发压测与全量运行

在目标服务器测出 batch 与实例数组合，固化配置，先跑 1,000/10,000 条灰度，再运行全量。

最终验收必须同时满足：

- 一个 worker 生命周期内，相同配置的模型只加载一次；
- 十万级任务可分片并行、可断点续跑、可只重跑失败 shard；
- 单个坏音频不会导致整批或整条流水线退出；
- 每条成功样本同时保有互不覆盖的 Qwen 与 SenseVoice 结构化结果；
- 生产结果无 mock 文本，输入输出样本数守恒；
- 固定版本、参数、输入音频和规范化规则均可从 manifest/run 产物追溯；
- 输出无需重新识别即可直接进入后续 ASR 比对统计。
