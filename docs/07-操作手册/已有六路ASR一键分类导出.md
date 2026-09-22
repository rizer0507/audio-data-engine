# 已有六路 ASR 一键分类导出

适用于已经用三个模型家族各跑两次、六路使用相同 batch 且分别指定过 `--asr-run` 的场景。脚本复用聚合、音频能量、**Silero VAD（全量）+ EfficientAT（候选）+ 五类 v2.3** 与汇总导出工序，**不启动 ASR，不需要 ASR 模型服务，不伪造 register/reservation/release**。

在项目环境中执行，依赖与项目相同（`pip install -e .`）。v2.3 还需可读的 Silero / EfficientAT 权重（见 034 配置）；失败可审计，**不会伪造** `present`/`absent`。Legacy v2.2 若走 DNSMOS，另需 `pip install -e '.[dnsmos]'`。

依据需求：

- [035 · 工序一后置分类入口同步 v2.3](../04-改进需求/进行中/035-工序一后置分类入口同步v2.3.md)（本入口默认）
- [034 · v2.3 最小声学分拣](../04-改进需求/进行中/034-v2.3最小声学分拣正式需求.md)（规则与引擎）
- [031 · 工序一后置统一分类分拣入口](../04-改进需求/进行中/031-工序一后置统一分类分拣入口.md)（体检与入口形态基线）

服务器执行手册台账：[手册/dev/05-一条命令多步骤.txt](../../手册/dev/05-一条命令多步骤.txt) §2。

## 最短命令（stage1 固定双跑别名）

工序一 `stage1 run` / 手工双跑使用 `qwen_1/2`、`glm_1/2`、`sensevoice_1/2` 时，无需 dataset YAML，也无需 `--family`：

```bash
# 推荐：脚本入口（能量步并发；8 卡服务器建议 16~32）
python scripts/classify_asr_to_xlsx.py --batch "$BATCH" --energy-workers 32

# 等价 CLI（同一实现；--workers 是 --energy-workers 别名）
audio-data stage1 classify --batch "$BATCH" --energy-workers 32

# 显式指定规则版本
# audio-data stage1 classify --batch "$BATCH" --rule-version v2_3_min_acoustic
# audio-data stage1 classify --batch "$BATCH" --rule-version v2_2_auto_noise   # legacy
```

`--source-dir` 可选：仅做目录存在性提示，**不会重扫/重洗音频**。分类实际读取 cleaned 底表的 `audio.resampled_16k`。

默认产物（与 034 正式交付对齐）：

| 产物 | 默认路径 |
|------|----------|
| 完整分类 XLSX | `data/exports/summary_five_class_v2_3_min_acoustic_${BATCH}.xlsx` |
| 正式 classified parquet | `datasets/stage1/derived/classified_five_class_v2_3_min_acoustic_${BATCH}.parquet` |

Legacy `--rule-version v2_2_auto_noise` 使用独立 stem `*_five_class_v2_2_auto_noise_*`，**不会**覆盖 v2.3 默认路径。

仅要 XLSX 时加 `--no-classified-parquet`。可用 `--output` / `--classified-output` 覆盖；默认不覆盖已有文件，替换须 `--overwrite`。

## 别名解析顺序（写死）

1. **显式 `--family qwen=run1,run2`**（可重复三次）
2. **批次 dataset YAML** 的 `model_families`（`--dataset-config`，或自动查找 `configs/datasets/zh_asr_v3_<batch横线换下划线>.yaml`）
3. **stage1 固定别名**：`qwen_1/2`、`glm_1/2`、`sensevoice_1/2`

历史批次（如 `qwen-basr-1`）**不得静默猜测**；没有配置/显式映射时按 stage1 别名体检，缺路则失败并提示如何传入 `--family`。

## 历史 / 手工双跑别名

```bash
python scripts/classify_asr_to_xlsx.py \
  --batch "$BATCH" \
  --family qwen=qwen-basr-1,qwen-basr-2 \
  --family glm=glm-base-1,glm-base-2 \
  --family sensevoice=sv-base-1,sv-base-2
```

显式 `--family` 优先于 dataset 配置与 stage1 默认。当前规则模板以 Qwen 为目标家族，三族中须包含 `qwen`。

## 前置体检（缺任何一路则失败）

启动分类引擎前只读检查 cleaned 底表与六路 ASR。失败时非零退出，报告按家族/路次列出存在与缺失及期望路径，并给出摘要，例如：

```text
错误: batch=demo-batch 分类前置检查失败，缺少以下 ASR 结果：
  [qwen] qwen_1     OK   datasets/stage1/asr/qwen_1_asr_demo-batch.parquet
  [qwen] qwen_2     缺失 datasets/stage1/asr/qwen_2_asr_demo-batch.parquet
  ...
摘要: 缺失家族 sensevoice（两路皆无）；家族 qwen 缺路次 qwen_2。
请确认推理时 --source-name/--batch 与 --asr-run 是否与上表别名逐字一致；
历史别名请用 --family 显式传入，勿依赖静默猜测。
```

缺路时**不会**进入聚合/分类，避免半套结果被当成完整交付。

## 输入位置与可选参数

- `--rule-version`：默认 `v2_3_min_acoustic`；legacy `v2_2_auto_noise`。
- `--source-dir` / `--source_dir`：可选。目录须存在（若传入）；不改写底表路径、不重新扫描。`source_path` 可保留历史 PCM 来源。
- `--cleaned`：默认 `datasets/stage1/cleaned/cleaned_<batch>.parquet`。必须是全量音频底表。
- `--asr-dir`：默认 `datasets/stage1/asr`。读取每个 `<asr-run>_asr_<batch>.parquet`。
- `--dnsmos-config`：仅 legacy v2.2 使用；默认 `configs/quality/dnsmos_p835.yaml`。**v2.3 默认不跑 DNSMOS，且不得用 DNSMOS 冒充 v2.3。**
- `--output` / `--classified-output` / `--no-classified-parquet` / `--overwrite`：见上。
- `--energy-workers` / `--workers`：仅加速**音频能量**步的线程并发，默认 `1`；服务器建议 `16~32`（受 CPU/磁盘限制）。**不**控制 Silero / EfficientAT 并发。

相对命令行路径以执行命令时的目录解释，内部规则配置以项目根目录解释。请在能够读取底表 `resampled_16k` 路径的服务器执行。

## 输出内容

同一个 XLSX 包含：

- **分类统计**：五类及 excluded 数量、总量；v2.3 另含 VAD failed、EfficientAT 候选调用数、`acoustic_state` 分布（不再把「DNSMOS 回退」当主 KPI）。
- **分类明细**：列契约对齐正式 `review export-summary`（六路文本、类别、`noise_kind`、家族四态、能量、以及 `vad_speech_state` / `speech_ratio` / `audio_event_state` / `acoustic_state` / `decision_evidence` / `hardcase_reason` 等）。

业务类别：`voicemail`、`semantic_risk`、`environment_noise`、`gold_candidate`、`hardcase`。`excluded` 保留原因，不是第六业务类。`gold_candidate` 是自动候选，不代表已成人工金标。`noise_kind` 主推 `silence` / `background` / `human_noise`。

### 引擎编排

| `--rule-version` | 编排 | 配置 |
|------------------|------|------|
| `v2_3_min_acoustic`（默认） | `audio_energy → silero_vad → efficientat_candidates → efficientat → classify` | `zh_asr_five_class_v2_3_min_acoustic` + `silero_vad_v1` / `efficientat_v1` / `acoustic_evidence_v1` |
| `v2_2_auto_noise`（legacy） | `audio_energy → dnsmos_v2_2_candidates → classify` | `zh_asr_five_class_v2_2_auto_noise` |

默认路径与 `pipelines/classify_dataset_five_class_v2_3_min_acoustic.yaml` **语义等价**。若 034 依赖未齐，入口会明确失败并提示改用 legacy，**禁止静默回退到 DNSMOS**。

成功对账后原子写入最终 XLSX（及默认 classified parquet）。单批上限 1,048,575 条。

需要正式训练/评测集发布时，继续使用原来的 `register-asr-run → prepare_dataset_v3 → attach_asr_v3` 身份校验流程。

依据：[服务器端五类完整自动分类与 XLSX 导出](服务器端五类完整自动分类与XLSX导出.md)、[034 需求](../04-改进需求/进行中/034-v2.3最小声学分拣正式需求.md)。
