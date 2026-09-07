# datasets — 可复用业务产物目录

> Manifest（JSONL / Parquet）是数据集的唯一真相源。  
> 本目录存放**跨多次运行复用**的业务产物；单次执行的日志 / checkpoint 在 `runs/`。

## 现行布局（Phase A + B）

通过 `--source-name` / `--asr-run` / `--eval-name` / `eval register` 写出的正式产物按工序落盘：

```text
datasets/
  stage1/
    cleaned/     cleaned_{BATCH}.parquet
    asr/         {alias}_asr_{BATCH}.parquet          # ★ 昂贵
    derived/     multi_asr_* / classified_* / summary_* / gold_* / reviewed_*
  stage3/
    eval_sets/   eval_{BATCH}.parquet
    asr/         {alias}_asr_eval_{BATCH}.parquet     # ★ 昂贵
    derived/     eval_aggregate_* / eval_metrics_*
    reports/     {eval_name}/evaluation.{json,xlsx}   # 权威评测报告
  manifests/     兼容层：旧文件仍可读；新写入不再默认落此
```

命名 stem 不变，代码权威：`src/audio_engine/core/source_naming.py`。  
`resolve_existing_manifest`：**先搜 stage 目录，再回退 `manifests/`**。

评测报告：权威路径为 `stage3/reports/{eval_name}/`；`runs/.../reports/` 仍写副本。  
未传 `eval_name` 时（旧调用）仍只写 `runs/.../reports/`。

## 与 runs / exports 的分工

| 目录 | 职责 |
| --- | --- |
| `datasets/` | 可复用业务真相源（按工序） |
| `runs/` | 单次运行 config / log / metrics / checkpoint（+ 报告副本） |
| `data/exports/` | 金标 / 汇总等 xlsx 交付物 |
| `data/catalog/` | 不可变 artifact 注册索引 |

## 查看与引用

```bash
# 统计（stem 或路径均可，解析兼容新旧根）
audio-data stats cleaned_mt3000

# 产物目录
audio-data artifact list --kind manifest
```

改造需求：[009-datasets模块改造需求](../docs/04-改进需求/进行中/009-datasets模块改造需求.md)  
工序对齐见 [工序总览](../docs/01-项目架构/工序总览.md)。
