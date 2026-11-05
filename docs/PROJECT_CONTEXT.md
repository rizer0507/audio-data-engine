# Audio Data Engine 项目上下文

核对日期：2026-11-05。基于分支 `v4.1`、现有文档和部分相关历史对话整理；不是完整聊天导出。

## VAD 裁切（2026-11-05）

FSMN 切段版本 `fsmn_vad_segment/v9`：应用层裁切与 `ph_ai_model_asr_rt_cpu` 的 `vad_asr_core_new` 同步——split-end 为 `pending-100`～`end+100`，完整端点/EOF 起点 `−100` 且完整端点仍裁到当时缓冲末尾。旧 v8 片段须从父音频重跑。

## 定位与边界

Manifest 驱动的音频数据治理引擎：原始音频入库、VAD/多路 ASR/质量证据、分拣或自由标签、人工标注交接、数据集发布及独立评测。核心实现位于 `src/audio_engine/`，编排与策略位于 `pipelines/`、`configs/`，专项维护脚本在 `scripts/`。

配套仓库 `audio-label-master` 承担人工标注；`audio-qwen3-asr-train` 承担 SFT/LoRA/GSPO 训练。不得仅凭同名目录认为跨仓库输入兼容；核对样本身份、标签、数据版本与音频路径。

## 优先阅读

- [文档导航与状态判定](README.md)：架构、规范、流水线描述当前状态；需求不能默认已实现，历史归档不能直接作为开发依据。
- [工序总览](01-项目架构/工序总览.md)。
- [训练平台对接上下文](01-项目架构/训练平台对接上下文-Qwen3-ASR.md)。
- [标注平台 VAD 分片契约](01-项目架构/标注平台对接上下文-audio-label-master-VAD分片.md)。
- [可组合规则与多标签分流](07-操作手册/可组合规则与多标签分流.md)。
- [Web 服务](web-service.md)。

## 已核实的关键约定

- Manifest（JSONL/Parquet）承载样本、转写、质量、标签与血缘；流水线是生产编排入口。缓存结合内容、算子版本和参数，改处理逻辑时注意缓存与恢复身份。
- 旧五类生产口径是 `selection_five_class_v3`，不要与更早的 `selection_v3.0` 混用。自由标签使用独立的 `selection_multi_tag_v1`：默认复用已有证据，不隐式重跑 ASR/声学，不等于取代旧入口。
- 自由标签库位于 `configs/tagging/library/`，词表在 `configs/tagging/lexicons/`；运行配置必须显式选中 tag。新增库文件不等于自动生效。恢复会核对定义指纹。
- 人工 accepted Gold 与 release 分组拆分/冻结是正式训练数据的必要治理环节；候选文本或三模型一致不能自动当成人工确认。

## fenshen_1015 评测知识

入口 `scripts/run_fenshen_1015_eval.sh` 调用转写和 XLSX 评测链，不是重新给原始音频做生产分拣。示例（先准备金标、模型服务与依赖）：

```bash
bash scripts/run_fenshen_1015_eval.sh full v024 dev best
```

当前脚本默认 base 5555，Full best/over 5560/5561，LoRA best/over 5565/5566；实际部署前重读脚本。输入金标位于 `datasets/stage1/derived/`，按 train/dev 与版本命名。

分类实现见 `scripts/eval_fenshen_1015_xlsx.py`：显式 `label_eval_type` 可指定分类；否则 non_speech/empty 优先，再按候选与金标 plain 文本相等分 auto_gold 或 hardcase。所有行关闭 `label_eval_type_split` 时只出 overall。评测桶不代表当前待评模型识别是否正确，也不等同声学五类。

历史排查发现：极少数超长复读输出可让句级正确率很高而汇总字准很低。比较结果必须确认删语气词、重复折叠、空金标等计分口径；保留原始输出，另存诊断修正版，不能把手工修正后的指标当原始模型成绩。

## 验证与后续接手

安装入口见根 README：`pip install -e .`；开发依赖见 `pyproject.toml`。代码修改后按模块选择 `python -m pytest tests/<相关路径>`，不要为阅读文档启动生产流水线。

本次只建立知识同步入口，未运行数据加工、模型推理或服务器验收。下一项业务任务由用户指定；执行前核对所需批次、数据版本和相关运行手册，不能从旧聊天推断当前服务器任务仍在运行。
