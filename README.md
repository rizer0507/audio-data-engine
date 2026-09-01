# Audio Data Engine

Manifest 驱动的音频数据处理引擎 — 将 WAV/PCM 原始文件通过可组合的 Operator 流水线处理，最终以 Manifest（JSONL / Parquet）作为数据集的唯一真相源。

## 架构

```
Pipeline YAML → Pipeline Engine → Operators → Manifest
                    ↑
              Raw Audio (data/raw/)
```

- **Sample**：一条音频的逻辑单元，包含 audio 资产路径、转写结果、质量指标、标签和数据血缘
- **Operator**：统一接口的处理能力（PCM 转换、重采样、降噪、ASR、增强、质检）
- **Manifest**：JSONL（交换）/ Parquet（分析）
- **Pipeline**：YAML 配置的 Operator DAG 组合
- **Cache**：基于 `sha256 + operator + version + params` 的幂等缓存，支持断点续跑

## 安装

```bash
pip install -e .
```

## 快速开始

### 1. 导入原始数据

```bash
audio-data ingest data/raw --name raw_20260820
```

### 2. 查看统计

```bash
audio-data stats raw_20260820
```

### 3. 运行单个 Operator

```bash
audio-data run sensevoice --dataset raw_20260820 --mock
audio-data run qwen_asr --dataset raw_20260820 --mock
audio-data run qwen_asr --dataset raw_20260820 --filter "label_badcase == 'noise'"
```

### 4. 运行 Pipeline

```bash
audio-data pipeline run pipelines/baseline_qwen.yaml --mock
audio-data pipeline run pipelines/denoise_qwen.yaml --mock
```

流水线是唯一的生产运行入口。每次运行都会在 `runs/<时间>_<名称>/` 保存实际配置、
manifest、metrics、checkpoint 和 `run.log`，无需再手工串联命令。

### 自由脚本与高并发

临时处理逻辑无需修改 core 或注册新算子。编写一个含
`process(sample, params, context)` 的 Python 文件，然后在 YAML 使用 `script.python`：

```yaml
execution: {executor: process, workers: 16, max_in_flight: 64}
pipeline:
  - name: my_policy
    operator: script.python
    params:
      path: scripts/my_policy.py
      threshold: 0.9
```

`sample` 是可序列化字典，函数返回 `audio`、`transcripts`、`quality`、`labels` 或样本
标量字段的增量更新。业务脚本通过 `context.log("message", key=value)` 写日志；引擎将其
记录为 `runs/.../script_logs/<step>.jsonl`，自动附带 UTC 时间、step 和 sample_id，在线程
和多进程高并发下仍可追溯。`context.artifact_path("result.json")` 可获得该步骤、该样本
独占的产物路径。脚本内容参与缓存键计算，因此修改脚本后会自动重新处理。完整示例见
`pipelines/script_example.yaml` 和 `scripts/examples/label_from_duration.py`。

### 5. 对比两个 ASR 模型

```bash
audio-data compare qwen sensevoice --dataset raw_20260820
```

### 6. 导出训练集

```bash
audio-data export raw_20260820 --format jsonl
audio-data export raw_20260820 --format scp
```

### 7. 清洗并核对已有 ASR 结果

输入包含 Qwen 识别结果的 XLSX，以及 SenseVoice 识别产生的 `fenp`（逐行 JSON）、
JSONL、JSON、Parquet 或 XLSX 文件：

```bash
audio-data reconcile-transcripts \
  --xlsx data/qwen_results.xlsx \
  --sensevoice-result data/sensevoice_results.fenp \
  --output data/exports/asr_reconciled.xlsx \
  --threshold 0.90
```

流水线会按 `id`（也支持 `sample_id`、`audio_id`、`utt_id` 等常见列名）合并两份结果，
删除工作簿所有字符串单元格内的 SenseVoice `<|...|>` 控制字段，再忽略空白、标点、
大小写和全半角差异计算字符级 Levenshtein 相似度。输出包含清洗文本、归一化文本、
`character_similarity`、`asr_consistent` 和判断原因，并在 Excel 同目录写出
`*.summary.json`。默认相似度不低于 `0.90` 才认为一致；无法自动识别列名时可使用
`--id-column`、`--sensevoice-id-column`、`--qwen-column` 和 `--sensevoice-column` 显式指定。

## 目录结构

```
audio-data-engine/
├── configs/          # Operator 默认配置
├── pipelines/        # Pipeline YAML 定义
├── src/audio_engine/ # 核心代码
├── datasets/manifests/
├── data/
│   ├── raw/          # 原始音频（只读）
│   ├── derived/      # 处理产物
│   ├── cache/        # 可删缓存
│   └── exports/      # 最终交付
└── runs/             # 每次运行的日志和产物
```

## 已注册 Operator

```bash
audio-data operators
```

| 类别 | Operator | 说明 |
|------|----------|------|
| audio | pcm_to_wav, resample, vad, denoise | 音频预处理 |
| asr | qwen, sensevoice | ASR 转写 |
| augmentation | add_noise, speed_perturb, volume_perturb | 数据增强 |
| quality | snr, cer, filter, transcript_diff | 质量评估 |
| script | python | 将任意 Python 处理脚本接入流水线 |

## 扩展

新增 Operator 只需：

1. 继承 `BaseOperator`，实现 `_execute()`
2. 用 `@register_operator` 装饰器注册
3. 在 `operators/__init__.py` 中 import

ASR Operator 继承 `BaseASROperator`，实现 `transcribe()` 即可。

## 自动训练与评测闭环规划

当前引擎已覆盖样本级流水线、分片、断点恢复和多 ASR 聚合。关于数据自动分拣、人工审核、
不可变数据集版本、外部训练框架接入、模型注册以及新旧模型回归评测的整体改造，见
[全自动训练与评测闭环改造方案](docs/流水线改进/全自动训练与评测闭环改造方案.md)。

## 技术栈

Python 3.10+ · Pydantic · Typer · PyYAML · Pandas · PyArrow · SoundFile · SciPy · Loguru
