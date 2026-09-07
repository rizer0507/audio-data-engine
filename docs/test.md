# test

> 测试文件。内容为当前工作区 `D:\Work\audio-data-engine` 的简要描述。

## 项目简介

**Audio Data Engine** —— Manifest 驱动的音频数据处理引擎：将 WAV/PCM 原始文件通过可组合的 Operator 流水线处理，最终以 Manifest（JSONL / Parquet）作为数据集的唯一真相源。

核心概念：Sample（音频逻辑单元）、Operator（统一接口处理能力）、Manifest（JSONL 交换 / Parquet 分析）、Pipeline（YAML 配置的 Operator DAG）、Cache（`sha256 + operator + version + params` 幂等缓存）。

## 目录结构

| 目录/文件 | 用途 |
|-----------|------|
| `src/` | 核心代码（`src/audio_engine/`） |
| `pipelines/` | Pipeline YAML 定义 |
| `configs/` | Operator 默认配置 |
| `tasks/` | 可恢复任务 DAG |
| `scripts/` | 自由脚本（`script.python`） |
| `datasets/` | 数据集 manifest 产物 |
| `data/` | `raw/`、`derived/`、`cache/`、`catalog/` |
| `runs/` | 每次运行的日志与产物 |
| `resources/` | 数据源登记（`manifest.yaml`） |
| `tests/` | 测试 |
| `docs/` | 中文文档体系 |
| `手册/` | 三大工序一键执行分册 |

## 三大工序

1. **工序一** 清洗落库打标
2. **工序二** 训练
3. **工序三** 评测

## 技术栈

Python 3.10+ · Pydantic · Typer · PyYAML · Pandas · PyArrow · SoundFile · SciPy · Loguru
