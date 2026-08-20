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

### 5. 对比两个 ASR 模型

```bash
audio-data compare qwen sensevoice --dataset raw_20260820
```

### 6. 导出训练集

```bash
audio-data export raw_20260820 --format jsonl
audio-data export raw_20260820 --format scp
```

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

## 扩展

新增 Operator 只需：

1. 继承 `BaseOperator`，实现 `_execute()`
2. 用 `@register_operator` 装饰器注册
3. 在 `operators/__init__.py` 中 import

ASR Operator 继承 `BaseASROperator`，实现 `transcribe()` 即可。

## 技术栈

Python 3.10+ · Pydantic · Typer · PyYAML · Pandas · PyArrow · SoundFile · SciPy · Loguru
