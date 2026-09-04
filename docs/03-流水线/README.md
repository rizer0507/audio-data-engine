# 流水线文档

> 三大工序与流水线的权威对齐表见 [工序总览](../01-项目架构/工序总览.md)。  
> 本目录补充各流水线的细节说明。

| 大工序 | 本目录文档 | 主要 Pipeline |
| --- | --- | --- |
| **工序一** 数据清洗落库打标 | [数据源落入流水线](./数据源落入流水线.md)（前置） | `resources/manifest.yaml` 登记 |
| | [数据清洗流水线](./数据清洗流水线.md) | `data_cleaning_source_A.yaml` |
| | （ASR / 聚合 / 分拣见工序总览工序一表） | `*_asr_batch.yaml`、`multi_asr_aggregate.yaml`、`asr_metric_pipeline.yaml`、`classify_dataset.yaml` |
| **工序二** 训练 | （细节见工序总览；引擎未完整集成） | `training run` / 可选 `build_training_set.yaml` |
| **工序三** 评测 | [评测流水线](./评测流水线.md) | 场景1：`eval register` → 跑批 → `eval_aggregate` → `eval_metric_pipeline`；场景2：`classify_external_gold` → `classified_` → 同上（[006](../04-改进需求/已完成/006-工序一清洗引擎拆分需求.md)） |

操作级命令：`手册/dev/`、`手册/local/` 对应 `01` / `02` / `03` 分册。  
模型识别操作细节：`docs/07-操作手册/`。
