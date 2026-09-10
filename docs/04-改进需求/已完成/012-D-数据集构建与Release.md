# 012-D · 数据集构建与 Release

> 细分来源：[012-四家族双跑分拣与人工金标数据集闭环重构](./012-四家族双跑分拣与人工金标数据集闭环重构.md)（母文档，**不得删改**）
> 索引：[012-细分需求索引](./012-细分需求索引.md)
> 状态：**工程实现完成（2026-09-10）**；人工金标 / 伪标签抽检验收 **未开始**（依赖 012-C 人工听审完成后的真实批）。本文定义目标行为；下列「实现清单」记录已落地能力。
> 版本：`selection_v3.0` / `annotation_v3.0` / `dataset_policy_v3.0` / `business_metrics_v1.0`。
> 输入：Kimi-ASR、GLM-ASR、SenseVoice、Qwen-ASR 四个家族，每家族两份全量识别结果；微调对象为 Qwen-ASR。
> 实施范围（本阶段）：人工校准消费、固定双评测集、dev、训练配额、跨集防泄漏、原子 Release；对应母文档阶段 **D：数据集**。
> 前置：[012-A](./012-A-契约与隔离.md)、[012-B](./012-B-质量与分拣.md)、[012-C](./012-C-人工审核与伪标签门禁.md)。下游：[012-E](./012-E-业务评测与生产切换.md)。
> 本需求统一承接原 007、008、009（007/008 已归档，009 已完成） 的相关目标，取代其与母文档冲突的设计；v1/v2 保留用于复现。不得直接覆盖旧分拣产物。

## 实现清单 / 测试 / 阻塞（工程）

### 已交付

| 项 | 落点 |
| --- | --- |
| 确定性配额抽样（eval_random / eval_core 互斥层 / train 四池 / dev） | `core/dataset_v3/sampling.py` |
| 跨集泄漏校验（组 / 哈希 / 通话 / 源音频 / 近重复；含未入选 eval 保留组） | `core/dataset_v3/release.py` → `validate_cross_split_leakage` |
| 原子 Release（暂存 → 校验 → 发布；同 ID 幂等 / 不同内容 fail-fast） | `core/dataset_v3/release.py` → `publish_release_v3` |
| `dataset_policy_v3` Catalog 契约（train/dev/eval_core/eval_random，不合并为 test） | `core/catalog.py` → `DatasetRelease` |
| 生产入口 | `pipelines/build_dataset_v3.yaml` + `quality.build_dataset_v3` |
| 配额 / 门禁配置 | `configs/datasets/zh_asr_v3.yaml`（`build` / `eval_*` / `train_pools` / `train_cross_constraints`） |
| v1 `release build` 保留；文档与 docstring 指向 v3 pipeline | `cli/main.py` |

### 测试结果

- `tests/test_dataset_v3_build.py`：空金标可进 eval、抽样确定性、eval 保留组禁入 train、配额不足拒发、原子幂等与碰撞、Catalog v3 契约、算子冒烟
- 回归：`test_selection_v3_contract` / `test_annotation_v3` / `test_catalog`（release 契约）通过

### 剩余阻塞 / 边界（不阻塞本阶段工程关闭）

1. **人工听审与伪标签抽检尚未在真实批完成**：工程可按双审金标与 audit 报告冻结 Release，但生产伪标签训练集与正式 eval 仍须等人审验收后才可宣称放行。
2. **`allow_non_speech_train=false`（默认）**：人工 non_speech 空目标进入 `pending_non_speech_train` / excluded，待训练适配器验收空转写后再开。
3. **业务评测指标工程已完成** → [012-E](./012-E-业务评测与生产切换.md)（门禁阈值校准与生产切换仍阻塞）。
4. 近重复指纹算法 / PCM 哈希自动计算仍由上游或 A 边界承接；本阶段只消费已有字段。

### 人工数据验收

本阶段工程可消费 reservation + 人审 revision + 伪标签 audit 报告并原子发布；**人工金标规模与伪标签抽检验收：未开始**（阻塞正式生产 Release 宣称）。

### 可运行命令

```bash
# 前置：prepare → quality → classify → review export/import →（可选）audit-pseudo
audio-data pipeline run pipelines/build_dataset_v3.yaml \
  --config configs/datasets/zh_asr_v3.yaml

# 配置中 build.release_id / train_size 必填（或由算子 params 覆盖）；
# reservation_path / audit_report_path 可写在 pipeline params 或由 Manifest 戳记解析。
# 产物：data/releases/{release_id}/
#   train|dev|eval_core|eval_random|excluded.parquet|jsonl
#   release.json / sampling.json / leakage_report.json / audit_report.json
```

## 0. 本阶段交付目标（母文档 §16.4）

**D：数据集**——人工校准、固定双评测集、dev、训练配额、跨集防泄漏、原子 Release。

每阶段分别提供实现清单、测试结果和剩余阻塞；工程完成与人工数据验收完成分开记录。

## 1. 目标与硬约束（母文档 §1，全局适用）

交付可复现链路：

```text
数据源快照 + 音频分组/查重 + 八路 ASR
  → DNSMOS 独立评分
  → 四家族状态与业务风险分析
  → 候选分类 + Qwen 纠错价值 + 人审队列
  → 人工听审 / 双审 / 仲裁 / 伪标签抽检
  → train / dev / eval_core / eval_random 独立冻结
  → Qwen Base / LoRA / Full SFT 在相同评测集上的 CER + 业务风险指标
```

必须满足：

1. 同家族两次推理用于发现不稳定性，最多计一票；四家族也可能共同出错，共识不等于真实标签。
2. Qwen 不担任金标基准，不享有投票加权；Kimi、GLM、SenseVoice 是三个辅助家族，不预设某一个绝对正确。
3. 任何一路出现业务语义冲突都须保留，禁止用族内 medoid 或多数票掩盖。
4. 正式评测仅用经验证的人工/外源金标；伪标签只允许进入通过抽检门禁的训练子集。
5. 输入已是上游切分段，不新增、不调用、不依赖 VAD；不生产 `speech_ratio` / `vad_edge_risk`。
6. DNSMOS 只衡量音频质量，不能判定无人声、串音、说话人归属或语义正确性。
7. 原始音频与八路原文只读；禁止自动拼接多个转写创造标签，禁止将“不”“嗯”等关键内容清空。
8. 所有阈值、词表、配额、审核规则版本化；校准未通过时允许分拣和送审，禁止自动发布伪标签训练集。

## 2. 本阶段必须复用与修复的现有入口（母文档 §2 相关）

- `release build`：v3 不使用“所有 accepted 混在一起随机拆 train/dev/test”的逻辑；先锁定用途，再按不同集合的标签门禁冻结。
- `core/selection_v2/dataset_builder.py` 当前仅赋候选用途，不能视作已完成配额抽样。

母文档 §2 完整条目（不得遗漏，归属标注）：

- `core/selection_v2/`：借鉴模块拆分，新建 `selection_v3/`；不得把新规则继续堆到 `selection_engine.py`。→ **B**
- `quality.aggregate_manifests`：复用对齐与哈希校验，扩展八路输入完整性报告。→ **A**
- `review export/import`：扩展字段、双审状态和空金标；现有 `accepted` 必须非空的限制不能沿用到 v3。→ **C**
- `release build`：v3 不使用“所有 accepted 混在一起随机拆 train/dev/test”的逻辑；先锁定用途，再按不同集合的标签门禁冻结。→ **D 主责**
- `core/selection_v2/dataset_builder.py` 当前仅赋候选用途，不能视作已完成配额抽样。→ **D 主责**
- `metrics/runner.py` 当前仅接受 CER，扩展同一指标入口，禁止独立脚本再实现一套业务指标。→ **E**
- `core/eval_ready.py` 当前 ID/重复组检查需扩展为同源片段、通话组、音频内容检查。→ **A / E**
- `noise_risk` 当前主要提取和透传，必须新增实际消费判定，不能只增加字段。→ **B**

## 3. 数据隔离消费规则（母文档 §9 全文复述，本阶段执行分配与门禁）

> 分组与 reservation 由 [012-A](./012-A-契约与隔离.md) 产出；本阶段在构建 Release 时强制消费，不得绕过。

### 3.1 泄漏组

按以下关系建无向图并取连通分量为 `leakage_group_id`：

- 同 `call_id/conversation_id`；有可靠客户/说话人标识时按配置并入同组。
- 同 `source_audio_id` 的切片、增强版本。
- 相同文件哈希或规范化解码 PCM 哈希。
- 已确认近重复指纹关系；保存算法、阈值与匹配证据。

分组字段必须从来源索引解析，不能盲猜文件名前缀。缺来源/通话映射的样本可分拣和人审，但进入治理隔离池；首版不得发布到声称无通话泄漏的正式训练/评测组合。元数据补齐后再解锁。

相同内容去重在训练中只保留代表，不删除历史记录；随机评测按原始目标分布采样，允许相同内容保留出现频次，但推断时按泄漏组处理相关性。近重复不确定匹配先隔离确认，不以相同文本判音频重复。

### 3.2 固定分配顺序

1. 冻结原始输入快照、原始 ID 列表、分组映射与目标分布定义。
2. 在查看模型错误和调整分拣规则前，按原始样本 ID 等概率随机选 `eval_random`，锁定其全部泄漏组；种子和完整候补序列固化。
3. 从剩余组预留 `eval_core_reserve`（默认按组哈希 15%）；其余为开发池。专项集只从该 reserve 选取，避免边挑训练边反向挪评测。
4. 开发池中独立预留 calibration 和 dev 组，剩余才是 train pool。默认 dev 占开发池组数 10%，calibration 按目标条数逐组选取。
5. 规则与质量阈值只用 calibration，模型/训练参数只用 dev。专项集构建规则冻结后才在 eval_core_reserve 分拣、抽样、听审。
6. 所有未入选的 eval 保留组仍禁止进入当轮 train/dev/calibration；禁止因难标、空标、Qwen 已识别正确而回流。

如历史模型结果已被查看，记录评测设计形成时间与已使用信息；不宣称历史数据是从未观察的独立测试。新微调结果不得参与修改已冻结评测集。

## 4. 评测集构建与冻结（母文档 §10 全文）

### 4.1 eval_random_v001

- 目标 2000 条，均匀抽自原始片段总体，不能先按 ASR 是否非空、噪声分数、共识、类别或 Qwen 错误筛选。
- 保留真实出现的语气词、噪声、空片段与重复频次；整组锁定但只审核入选片段。
- 解码失败、无法确定真值等样本记录为不可评估，报告原始入选数、有效数、排除原因；不悄悄换成容易样本。
- 如补量，只按冻结候补顺序补，并保留初始样本队列的覆盖率统计；报告指标适用范围是可评估子集，不能冒充全体原始分布。

### 4.2 eval_core_v001

默认 2000 条，按人工最终属性形成互斥主层，按以下顺序归层：

1. 明确否定/拒绝：500。
2. 纯语气词/中性短反馈：300。
3. 人工确认无人声（含环境噪声、忙音、静音）：300。
4. 含可辨目标语音的噪声/串音：300。
5. 明确肯定：300。
6. 其他普通语音和难例：300。

跨层属性仍以多标签保留，例如带噪否定句属于第 1 层，同时进入 noisy 切片。初始候选按 v3 风险标签过采样，人工属性确定后补齐；不得按微调模型表现补样。

每层覆盖来源和原时长；否定层覆盖“不需要/不用/不要/没有/不是”等不同表达，不能全是同一句。teacher 同意/Qwen 不同的难例与四家族一致的普通样本都要覆盖。报告注明专项集非自然分布，不能与 random 简单平均成线上指标。

### 4.3 通用门禁

- 两套 eval 必须 100% 完成双审/仲裁或等价外源验收；有 speech 与 confirmed non_speech 的空真值。
- `unintelligible/ambiguous_target` 单独存弃权集合，报告覆盖率，不作为空金标计算。
- eval_core 与 eval_random 组间无交集；与 train/dev/calibration 也无 ID、文件/PCM 哈希、源音频、通话或已确认近重复组交集。
- 配额不足输出 shortfall，正式构建失败；明确修改配额需新配置版本，不能自动重复样本凑数。
- 冻结音频引用及内容哈希、人工真值、全部标签、抽样名单、分组映射和版本；后续修标生成新版本。
- 训练新增数据也要检查已冻结 eval 的全体保留组，不能只检查最终入选 ID。

## 5. Qwen 训练集与 dev（母文档 §11 全文）

### 5.1 准入与目标文本

- 人工/可信外源 speech：以 gold_text 为训练目标。
- 通过独立抽检的 pseudo_high：以 candidate_text 为训练目标，保留伪标签身份；不写入 human gold。
- pseudo_medium、未审风险样本、缺失推理且未完成人工例外审核、不可辨/目标不明：不进入训练。
- 人工 non_speech：仅训练适配器经测试支持空转写目标后加入，仍包含必要的模型任务模板与结束 token；否则形成独立待接入清单，禁止改成“噪声”或静默丢弃。
- dev 仅使用开发保留组中的人工/可信外源金标，含业务风险和自然样本两种统计视图；不使用正式 eval 选 epoch 或调参数。

### 5.2 默认配额

训练目标条数 `train_size` 必填，首轮建议 10000；采用四个互斥池，按顺序归池：

1. **40% Qwen 已确认错误修正**：人工真值证明 Qwen 任一路关键或普通转写错误，优先语义、语气词正向化、噪声幻觉。
2. **20% 其他人工高风险样本**：不属于第 1 池的短句、否定、噪声、串音等；保留 Qwen 已识别正确的风险样本，避免只学“全部否定”。
3. **20% 人工普通样本**：不属于前两池，覆盖肯定、否定、中性和正常长句。
4. **20% 审计通过的 pseudo_high**：普通能力保持；排除已入前三池者。

全体再施加交叉约束：肯定与否定语音各 ≥15%；可辨带噪/串音语音 ≥10%；non_speech ≤10%；pseudo_high ≤20%。这些是首版可修改实验配方，不是效果保证。各百分比以最终训练条数为分母，辅助报告音频小时分布。

按 source、时长、语义、质量层做确定性配额分配；无放回、每个 exact duplicate 仅一份，默认同通话最多 3 条。优先满足硬隔离与标签门禁，再满足配额；不足输出缺口，不自动复制、降低金标等级或超额填伪标签。

v3 首版通过配额控制，不假定训练框架支持 sample_weight；扩展权重必须由适配器显式声明支持并验收。增广只对 train 执行，继承原组和 split，单独记录增强比例。

## 6. 标准字段与产物（母文档 §12 全文）

Manifest 中分开保存以下维度，禁止一个 type 同时承载可信度、用途和真错误：

```text
classification: type, risk_tags[], reason_codes[], rule_version
family_evidence: 四家族状态、两路引用、代表、支持簇、相似度
target_evidence: Qwen 对 teacher 差异与人工确认错误
quality: DNSMOS 原始分、状态、noise_band、阈值版本
annotation: candidate_text, gold_text, gold_kind, label_tier, label_source,
            annotation_state, 人工属性、人员、审核记录、revision
allocation: leakage_group_id, duplicate_group_id, reservation,
            dataset_role, split, sampling_stratum, sampling_probability
provenance: source_snapshot, run digests, policy digests, artifact parents
```

可放入现有 `labels/quality`，但新增 Pydantic 契约验证，不强制破坏 Sample 兼容性。legacy `label/gold_text` 仅通过明确映射导出，不能把伪标签映射成正式评测金标。

产物落点：

```text
datasets/stage1/quality/{source}/{quality_version}/
datasets/stage1/derived/{source}/selection_v3/{run_id}/
datasets/stage1/review/{queue_id}/{revision}/
data/releases/{release_id}/
  train.parquet/jsonl
  dev.parquet/jsonl
  eval_core.parquet/jsonl
  eval_random.parquet/jsonl
  excluded.parquet
  release.json / sampling.json / leakage_report.json / audit_report.json
datasets/stage3/reports/{eval_release}/{experiment_id}/
runs/                                      # 仅执行痕迹
```

业务 Release 增加 `dataset_policy_v3` 契约，兼容旧 train/dev/test 类型读取；不得把 eval_core 和 eval_random 拼成一个 test 后丢失角色。

Release ID 已存在时先校验身份与内容：相同返回已有引用；不同 fail-fast。所有产物先暂存、校验、原子发布，失败不留下半份正式 Release；禁止先覆盖 parquet 再让 Catalog 拒绝冲突。

## 7. 本阶段开发交付与执行入口（母文档 §14 相关）

建议模块（本阶段相关，**已落地**）：

```text
core/dataset_v3/
  sampling.py / audit.py / release.py
configs/datasets/zh_asr_v3.yaml
```

生产入口继续使用 PipelineRunner；本阶段新建 YAML：

4. `build_dataset_v3.yaml`：消费分类+审核+抽检记录，确定性配额抽样、门禁、冻结 Release。**已落地。**

CLI 示例（可运行）：

```bash
audio-data pipeline run pipelines/build_dataset_v3.yaml --config configs/datasets/zh_asr_v3.yaml
```

`--config` 若现有 pipeline CLI 尚不支持，新增配置注入且保留旧参数语义；八路 artifact、reservation、质量产物、审核 revision、train_size、release_id 必须从已解析配置/Catalog 显式获取，不靠寻找“最新文件”。

不重构 ASR 推理、不训练 DNSMOS、不建设分布式平台、不自动启动模型微调；本需求交付可供外部训练框架消费的正式数据。

母文档 §14 完整模块与入口清单（归属标注，不得遗漏）：

```text
core/selection_v3/
  config.py / types.py / input_contract.py          # A / B
  family_evidence.py / semantic_risk.py / consensus.py  # B
  classifier.py / review_router.py                  # B
core/dataset_v3/
  grouping.py / reservation.py                      # A
  sampling.py / audit.py / release.py               # D 主责（audit 与 C 协同）
operators/quality/dnsmos.py                         # B
configs/selection/zh_asr_v3.yaml                    # B
configs/selection/semantic_lexicon_zh_v3.yaml        # B
configs/quality/dnsmos_p835.yaml                    # B
configs/datasets/zh_asr_v3.yaml                     # A / D
configs/annotation/zh_asr_v3.yaml                   # C
configs/metrics/business_risk_v1.yaml               # E
```

生产入口 YAML：

1. `prepare_dataset_v3.yaml` → **A**
2. `audio_quality_sidecar.yaml` → **B**
3. `classify_dataset_v3.yaml` → **B**
4. `build_dataset_v3.yaml` → **D**

完整 CLI 链路（实现后必须可运行）：

```bash
audio-data pipeline run pipelines/prepare_dataset_v3.yaml --config configs/datasets/zh_asr_v3.yaml
audio-data pipeline run pipelines/audio_quality_sidecar.yaml --source-name "$BATCH"
audio-data pipeline run pipelines/classify_dataset_v3.yaml --source-name "$BATCH"
# review export → 独立听审/复核 → review import；具体参数在实现时同步 CLI --help
audio-data pipeline run pipelines/build_dataset_v3.yaml --config configs/datasets/zh_asr_v3.yaml
# 固定 eval release → 各模型独立跑批 → eval_aggregate → eval_metric_pipeline
```

顶层 Task DAG 在等待人工时保存 waiting_review 状态；可输出队列与缺口，不能模拟人审自动跨过。

## 8. 验收用例：标注与数据安全（母文档 §15.2 全文收录）

- accepted non_speech 空字符串可完整经过审核、Release、支持空目标的训练导出及 eval；pending/null 和 unintelligible 不得通过。
- 一审填完不等于双审完成；同标注员自审、冲突未仲裁、旧 revision 导入均被拒绝。→ **C**
- 人工导入保留原 type/risk_tags/八路原文；人审 gold 与 candidate_text 独立。→ **C**
- 不同路径相同音频、同通话、同源切片、增强、已确认近重复跨集合均阻断发布。
- eval 保留但未入选的同组样本也不得进入训练；缺 group 元数据不能静默退回按 ID 随机拆分。
- 随机集名单不因 ASR、DNSMOS、分类或新模型结果变化；所有排除可追溯。
- 输出配额不足、审计失败、标注未完成时不得宣称生成正式 Release。
- 同输入、配置、版本、seed 在不同顺序/分片/续跑下得到相同分类、采样 ID 与内容摘要。
- 重复发布幂等；同 ID 不同内容先失败，不覆盖既有文件。

本阶段必须通过的验收重点：

- eval_random / eval_core 配额、互斥层、100% 双审门禁、跨集防泄漏。
- 训练四池配额与交叉约束；不足输出缺口，不自动复制或降级。
- 标准字段分维、产物落点、`dataset_policy_v3`、原子发布与幂等。
- 配额不足、审计失败、标注未完成时不得宣称生成正式 Release。
