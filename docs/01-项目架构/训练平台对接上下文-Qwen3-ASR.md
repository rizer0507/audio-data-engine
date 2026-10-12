# Qwen3-ASR 训练平台：数据治理与标注对接上下文

> 核对日期：2026-10-12。用途：交给新训练平台的产品、后端和训练开发人员，作为跨平台联调与开发输入。
>
> 依据：当前 `audio-data-engine` 工作区的代码及仓库内 `Qwen3-ASR-main/`；基线 HEAD 为 `8cfed5bf415c15a0e80b19a7f5f4a0e73260d9bd`。工作区包含未提交的 061 人工分类与训练排除实现，本文包含这些行为，不能仅检出上述提交就认为契约完全一致。
>
> 本文区分“已有能力”和“建议补齐”。未运行真实标注平台、GPU 训练或服务器端到端验收；仓库内 Qwen 目录包含本地 LoRA 扩展，本文不声称它与最新官方上游完全一致。

## 1. 先明确三个平台的职责

**数据治理平台（本仓库）**负责原始数据登记、音频清洗、多模型转写、质量诊断、分类、标注结果回填、数据血缘及数据产物管理。现状是 Python 引擎、CLI、脚本和文件目录，并非已经提供完整跨平台 HTTP 服务。

**标注平台**负责消费待标池、播放音频、分配任务、听写与复核，输出带原样身份字段的人工标注结果。本仓库能核对的是其输入输出契约及回填实现，不能据此确认另一仓库的实际部署和 API。

**新训练平台**负责选择数据版本与训练配方、形成可训练输入、调度 Qwen3-ASR 任务、管理日志/checkpoint/实验、调用评测并展示结果。分类规则和人工标注事实仍由上游负责；训练平台记录消费的确切版本。

推荐数据流：

```text
原始音频 → 清洗、多 ASR、五类分拣 → classified Manifest
                                    │
                         待标池 → 标注平台 → annotations
                                    │              │
                                    └── 治理侧校验回填
                                             │
                             with_gold / 正式 batch 仓库
                                             │
                       按配方汇总 → 准入、分组拆分、冻结 Release
                                             │
                            Qwen JSONL 适配与音频可访问性校验
                                             │
                                 训练平台 → checkpoint / model
                                             │
                      固定评测集独立推理 → 治理评测 → 实验报告
```

这是一条目标集成链路，当前各组件并未全部自动串通。尤其是 `with_gold → assemble → release build → Qwen`，不能直接顺序调用就认定完成。

## 2. 当前数据目录与版本概念

所有下述路径相对本仓库根目录；Linux 训练节点需要自己的路径映射。

- `resources/manifest.yaml`：原始数据源登记。
- `data/raw/`：原始音频；`data/derived/`：处理后的音频等资产。
- `datasets/stage1/cleaned/`、`asr/`、`derived/`：工序一清洗、识别和分类 Manifest。
- `datasets/stage1/derived/classified_five_class_v3_<batch>.parquet`：现行五类 v3 分类结果。
- 同目录 `classified_five_class_v3_<batch>_with_gold.parquet`：听写回填结果。
- `datasets/stage1/review/semantic_risk_transcripts/<batch>/`：`queue.parquet`、`playlist.md`、可选 `wav/` 和标注文件。
- `datasets/stage1/warehouses/<batch>/`：正式 batch 仓库，含 `manifest.parquet`、`warehouse.json`。
- `datasets/stage1/trainsets/<trainset_version>/`：036 多批次训练数据汇总产物。
- `data/catalog/`：文件型注册中心，包含 `artifacts/`、`releases/`、`models/`、`warehouses/`。
- `datasets/stage3/`：独立评测数据、ASR、汇总和报告。
- `runs/`：运行配置、进度、日志等执行证据，不能代替业务数据版本。

几个容易混淆的对象必须分别建模：

1. **batch**：来源批次，不是训练版本。
2. **BatchWarehouse**：一批次一个正式仓库，包含整批治理结果；不是 train/dev/test Release。同 batch 内容变化会触发不可变注册约束，不应直接覆盖正式仓库。
3. **trainset_version**：036 汇总版本，含入选样本和配方；尚未完成正式拆分、训练标签构造。
4. **DatasetRelease / release_id**：冻结的数据分区与策略、金标、归一化版本。`outputs` 的值是 artifact ID，不是文件路径。
5. **ArtifactRecord / artifact_id**：一个已登记文件的 URI、SHA-256、大小、生产者和元数据。校验只覆盖该文件本身，不自动校验其中每条音频。
6. **ModelVersion / model_id**：基座、训练 Release、配方与 checkpoint 的关联记录。

036 默认拒绝覆盖同版本，但提供 `--overwrite`；因此不能把版本目录名当成不可变性的充分证明。新平台应冻结输入摘要，已使用版本禁止覆盖，修订生成新版本。

## 3. Manifest 与样本身份契约

实现依据：[sample.py](../../src/audio_engine/core/sample.py)、[manifest.py](../../src/audio_engine/core/manifest.py)、[input_contract.py](../../src/audio_engine/core/selection_v3/input_contract.py)。

### 3.1 基础字段

- `id`：Manifest 样本 ID；标注交换表中名为 `sample_id`。
- `source_path`、`sha256`：来源路径及样本摘要。不要假定任意阶段的 `sha256` 都是训练实际读取的 WAV 摘要。
- `sample_rate`、`channels`、`duration`：音频元数据，可能缺失。
- `audio`：音频变体路径字典，如 `resampled_16k`、`raw`。
- `transcripts`：各识别运行的转写结果，属于模型输出。
- `quality`：相似度、声学指标等证据。
- `labels`：分类、人工标签、训练排除、来源等业务字段。
- `lineage`、`status`、`errors`：处理血缘、步骤状态及错误。

Parquet 中 `audio/transcripts/quality/labels/lineage/status/errors` 可能是 JSON 字符串列。优先用 `Manifest.load()`；新语言服务需显式 JSON 解码，不能假定它们都是 Parquet struct。扁平化 `label_*` 列不要取代规范 `labels` 字典。

### 3.2 身份与摘要

批次内回填契约为 `(sample_id, original_audio_sha256)`；跨平台建议使用 `(batch, sample_id, original_audio_sha256)`，避免跨批次同名 ID 冲突。

原始音频摘要由 `original_audio_sha256(sample)` 统一解析：优先 labels/quality 中的 `original_audio_sha256`、`source_audio_sha256`、`raw_audio_sha256`，再退回 `sample.sha256`。不要重算重采样 WAV 的摘要来替换此身份字段。

建议训练输入另外保存实际使用的音频变体、内容摘要与 URI。这样既能追溯原始身份，又能确认训练读取的具体内容。

### 3.3 训练平台必须保留的治理字段

至少保存：原始身份、来源 Manifest 摘要、`source_batch`、有效类别、`source_category`、`gold_text`、`gold_kind`、`annotation_state`、`annotation_type`、`exclude_from_training`、`training_exclusion_reason`、标注协议、标注人和时间、标签来源、音频变体、split 及分组依据。

不能把 `gold_text`、`candidate_text`、`label` 简化为一个不带来源的字符串；三者的准入含义不同。

## 4. 五类分拣与人工修订规则

现行规则依据 [数据挑选规则_v3](../02-规范规则/数据挑选规则_v3.md)：`selection_five_class_v3`，入口 `classify_dataset_five_class_v3` / `stage1 classify`。它与旧 `selection_v3.0`、v2.2、v2.3 不是同一个协议。记录真实输入的规则版本，不仅凭文件名前缀判断。

基础五类为 `gold_candidate`、`semantic_risk`、`hardcase`、`environment_noise`、`voicemail`。分类桶表达治理用途，不直接等于训练许可：

- `gold_candidate` 是候选，不能仅凭类别认定为正式人工金标。
- `semantic_risk` 需要处理转写冲突；机转参考文不能自动冒充听写结果。
- `hardcase` 可能包含“不要训练”的人工标记，不能默认纳入。
- `environment_noise` 只有在明确设计无语音训练目标时才纳入。
- 自动 `voicemail` 只是规则隔离结果，不等于人工确认文本。

当前 061 实现允许 Web v2 人工分类扩展，权威入口是 [human_category.py](../../src/audio_engine/core/human_category.py)：

- 有效类别读取优先级：`reviewed_category → category → type → training_category`，这些是 labels 内字段；`source_category` 只作溯源。
- `silence / background_suspected / human_noise` 映射到 `environment_noise`，`gold_kind=non_speech`，`gold_text=""`。
- `multi_speaker` 映射到同名类别，`gold_kind=ambiguous_target`，`gold_text=null`，`exclude_from_training=true`。
- 有效类别、`annotation_type`、类别别名冲突中的任何多人说话证据，或独立排除标记为真，均触发训练硬排除。

**排除必须早于文本归一化、候选文回退、类别过滤、配额和写文件。** `null` 不是空字符串：多人说话不能转换成静音目标，也不能回退机转文再进入训练。没有 `annotation_type` 的历史 empty 不能推断为 silence。

## 5. 标注平台交接与回填

### 5.1 当前轻量听写通道

[annotate_semantic_risk_transcripts.py](../../scripts/annotate_semantic_risk_transcripts.py) 提供 `prepare / label / merge`。Web 平台替代人工 `label` 环节；治理侧继续负责 prepare 与 merge。

待标表核心字段：`sample_id`、`original_audio_sha256`、`batch`、`queue_index`、`category`、`wav_path`、`candidate_text`；可附 `source_path`、`selected_family`、`duration`。`queue_index` 仅是播放顺序，不能作回填键。

听音解析优先 `audio.resampled_16k`，再到 `pcm_wav / pcm_to_wav / raw / pcm_wav_path` 等可用来源。部署后路径必须可达；软链或 Windows 绝对路径不构成跨机器交付保证。

标注输出 Parquet / JSONL 应保留身份，并提供 `status`、`gold_text`、`gold_kind`、标注协议；建议附 `annotator_id`、`annotated_at`、`label_source`、`decision`。Web v2 还需保留人工分类及训练排除字段。

- `status=annotated / empty` 才进入现有 merge。
- `skipped` 等状态不写入 gold。
- v2 使用 `semantic_risk_web_v2`，代码读取 `protocol` 或 `transcript_annotation_protocol`。
- merge 写入 `annotation_state=annotated` 与 `is_human_verified=true`；**这不等于正式双审 `human_accepted`**。
- v1 听写通常不改分类；v2 会写入人工有效类别并保存原 `source_category`。旧 039 文档的“类别不变”只适用于旧通道，不能覆盖 v2 行为。

### 5.2 当前 merge 的边界与平台前置校验

代码先按 `sample_id` 构建字典，再核对双方非空摘要；同 ID 多条标注会发生后行覆盖，空摘要并不总会被拒绝，CLI 的 batch 参数也不能替代逐行批次验证。

因此接入服务应在调用前验证：batch 一致、ID 唯一、身份摘要非空且一致、协议可识别、状态与文本语义相符。同一身份的不同修订应显式选择版本，不能依赖文件行序。

建议始终使用 `merge --strict`，并检查生成的 `.parquet.merge_report.json`：`missing_in_classified=0`、`sha256_mismatches=0`，`applied` 与有效唯一标注数一致。该 mismatches 计数也可能包含捕获到的其他标注校验错误，应结合错误日志判断。非 strict 模式可能有错误仍返回成功，不能只看退出码。

### 5.3 正式仓库通道是另一条协议

[batch 唯一数据仓库手册](../07-操作手册/batch唯一数据仓库-人工标注入仓.md) 使用整批 annotation_v3 包、`review import --protocol v3` 与 warehouse freeze，覆盖全部分类样本，可带 `reviewed_category`。该通道以 `__EMPTY__` 表达确认无语音，空白表示未完成。

它与轻量听写 annotations 的 `status/empty` 不是同一种表结构。不能把轻量 `_with_gold` 的存在当成整批正式仓库已验收，也不能把普通分类 XLSX 直接当正式 v3 包。

## 6. 当前训练汇总能做什么、不能做什么

入口：[assemble_training_parquet.py](../../scripts/assemble_training_parquet.py)。从多个 batch 的指定 Parquet 按显式类别汇总，先执行 061 硬排除，再处理人工排除标记、类别过滤及摘要去重。

产物位于 `datasets/stage1/trainsets/<version>/`：

- `manifest.parquet`：入选样本。
- `recipe.yaml`：配方快照及输入文件 SHA-256。
- `summary.json`、`summary.md`：数量、时长、类别、标签缺失与排除统计。
- `lineage.json`：样本来源及清洗前后类别。
- `details.csv`：包含未入选行的明细。

配方必须显式给出 categories；重复音频策略默认 `fail`，可选 `keep_first`。人工 semantic_risk 文本中的“不做训练”等标记按配置洗成 hardcase 并清空目标，之后才过滤类别。

**汇总不会自动保证每条样本有可训练目标，不生成 Qwen JSONL，不做训练/验证拆分，也不启动训练。** 音频引用与缺失标签统计不等于训练节点真实解码通过。包含 hardcase 的配方可能保留无有效文本行，必须再过训练准入。

仓库已有示例配方还使用 v2.3 路径；新批次复制时必须明确更新 `rule_stem` 和每个 `parquet`，不能原样拿来代表五类 v3。

## 7. Release、标签准入与防泄漏

已有 [catalog.py](../../src/audio_engine/core/catalog.py) 定义 DatasetRelease：记录 `source_artifact_id`、`outputs`、`policy_version`、`normalization_version`、`gold_revision`、`split_seed`、`group_key`、counts 等。

现有拆分路径有两种，选择时不可混用语义：

1. `audio-data release build` 是 legacy v1/v2 路径，消费 `human_accepted / auto_accepted` 且非空的 gold，产出 train/dev/test；当前已加入 061 排除。听写的 `annotated` 状态和无语音空目标不能直接通过这个入口。
2. `core/dataset_v3/` 有分组、保留集、采样、审计和 Release 能力。`dataset_policy_v3` 要求 train/dev/eval_core/eval_random；不能合并两个评测角色为一个 test。存在代码不代表当前 batch 已补齐元数据并通过门禁。

建议新平台先定义训练标签策略：人工语音使用已通过约定审核等级的 `gold_text`；伪标签如获准使用，另记来源、审核/置信条件和配比；待标 semantic_risk 默认禁止。不要为了通过 legacy build 直接批量把 `annotated` 改成 `human_accepted`。

无语音训练需单独定义准入及 Qwen 目标编码，验证后再开放；不能用 legacy 非空 gold 门禁冒充已支持。

分组应覆盖同一原音频、同一通话/会话、可靠说话人及确认的近重复关系。参考 [grouping.py](../../src/audio_engine/core/dataset_v3/grouping.py)：可读 `call_id/conversation_id`、`source_audio_id`、原始/PCM 摘要、近重复组等，不能猜文件名前缀当会话 ID。缺失分组元数据应进入待治理或明确阻断。

正式训练发布前建议断言：

- train/dev/正式评测之间身份、原始摘要、泄漏组交集均为空；跨 batch 同源和音频增强变体也应同组。
- 训练样本与多人说话身份集合交集为零，且逐行再次检查硬排除。
- 标签来源及审核等级满足当前配方；正式评测不使用未经认可的自动伪金标。
- Release、Manifest、训练 JSONL 与音频清单均有摘要，重跑能解释数据变化。

## 8. Qwen3-ASR 训练输入适配

本地依据：[finetuning/README.md](../../Qwen3-ASR-main/finetuning/README.md)、[qwen3_asr_sft.py](../../Qwen3-ASR-main/finetuning/qwen3_asr_sft.py)、[qwen3_asr_lora.py](../../Qwen3-ASR-main/finetuning/qwen3_asr_lora.py)。以下描述的是当前本地代码。

训练 JSONL 每行要求 `audio` 和 `text`，支持可选 `prompt`。示意：

```json
{"audio":"/mnt/asr/audio/utt0001.wav","text":"language Chinese<asr_text>您好，请问有什么可以帮您。"}
```

本地 SFT 通过 `--train_file` 和 `--eval_file` 读取 JSON 数据，默认 `--sr 16000`。治理 Manifest 的 `audio` 是字典，训练器的 `audio` 是文件路径；治理 JSONL 不能直接作为 Qwen JSONL。

**建议新增训练适配器，当前不是已接通接口：**

1. 读取已冻结并通过准入的 train/dev Manifest，保留样本身份和 split。
2. 先执行硬排除，再按固定策略产生目标文本；不从未知来源自动回退。
3. 解析真实训练音频，映射到 worker 可访问路径；检查存在、可解码、声道、采样率、时长与实际摘要。时长上限等以选定模型和实测资源约束配置，本文不杜撰阈值。
4. 有明确语言时加入相应 `language ...<asr_text>` 前缀；无语言信息可按本地说明使用 `language None<asr_text>`。只在适配层编码，治理 `gold_text` 保持人工正文，避免重复加前缀。
5. 固定生成 `train.jsonl`、`dev.jsonl` 及独立身份索引，记录行号到 `(batch, sample_id, original_audio_sha256)`、目标来源、音频变体及摘要的映射。
6. 冻结 adapter 版本、规则摘要、数据摘要、条数和拒绝原因。训练器会丢弃未使用列，不能仅依靠训练 JSONL 的额外列保存治理血缘。

建议适配产物作为 `kind=training_input` 注册，并在 metadata 关联 Release。该 artifact kind 已存在，但完整适配器仍需实现。

训练控制参数已有 `--model_path / --train_file / --eval_file / --output_dir`、batch_size、grad_acc、lr、epochs、save_steps、`--resume_from / --resume`。本地 LoRA 输出 adapter 权重；部署与评测需明确 base+adapter 或合并模型，本地 README 提供 merge 工具。不要将裸 adapter 目录视为完整可推理模型。

## 9. 已有外部训练调用接口

[training.py](../../src/audio_engine/core/training.py) 的 `run_training_job()` 通过同步子进程调用外部命令，CLI 为 `audio-data training run`。它提供以下环境变量：

```text
AUDIO_DATA_RELEASE_ID
AUDIO_DATA_TRAIN_MANIFEST
AUDIO_DATA_DEV_MANIFEST
AUDIO_DATA_RECIPE
AUDIO_DATA_CHECKPOINT
```

train/dev 是从 catalog 验证并解析出的治理 Manifest URI，不是 Qwen JSONL；Qwen 脚本也不会自动读取这些环境变量。桥接脚本需要消费环境变量、执行适配并构造 Qwen 参数，或由新平台统一执行同等逻辑。

已有任务记录 `state.json`、`input.json`、`stdout.log`、`stderr.log`；状态包含 running/succeeded/failed。任务 ID 由 release、配方摘要、命令、checkpoint、model_id、base_model 等计算；相同成功任务会返回已登记模型，running 会拒绝重复执行。

只有子进程退出码为零且 checkpoint 路径存在才登记 `ModelVersion(status=ready)`。这仅是最小门禁：未证明权重完整、可加载或评测达标。模型记录包含 base_model、training_release_id、training_recipe、checkpoint_uri、git_commit 和 training_job_id。

新平台仍需补：持久化队列、GPU 资源与互斥、worker 心跳、取消/超时、进程失联恢复、实时日志、checkpoint 完整性校验、模型可加载验证。已有 `running` 记录不能当作 worker 心跳；训练启动异常和服务重启场景需额外收敛任务状态。

若训练平台自己调度，应由单一调度者拥有任务状态，治理侧保留数据与模型关联记录，避免双方各启动一次训练。

## 10. 评测回流

现有评测与训练解耦：输入是已登记评测集和按 ID 对齐的模型推理 Manifest，不要求经过已有 training run。详见 [评测流水线](../03-流水线/评测流水线.md)。

可复用 `eval register / eval check`、Qwen 独立 ASR 流水线、`eval_aggregate.yaml`、`eval_metric_pipeline.yaml`。主要报告在 `datasets/stage3/reports/<eval_name>/evaluation.json` 与 `.xlsx`。

新平台建议记录并展示：模型/adapter/基座版本、评测集版本与摘要、推理配置、归一化与指标配置、CER/字准率、分类切片、无语音误识别等业务指标、`eval_trust`、`publish_status`、报告 artifact 与 URI。

训练 loss 或 dev loss 不能替代业务评测；`ready` 不能替代发布通过。缺少门禁校准时可能得到 `needs_review`，不得由平台改写成 pass。固定评测集用于比较 base 与新 checkpoint，评测失败样本经治理分流后再送标注，避免直接回灌并污染保留评测集。

## 11. 建议的跨平台集成边界（待实现）

第一期可以采用“文件产物 + catalog + 受控 worker 命令”联通，无需先假设存在 REST API。后续服务化建议封装以下操作；这里不是已部署路由或现成 OpenAPI：

- 查询批次、分类策略与产物摘要；导出待标池。
- 接收标注版本，完成身份与协议校验，执行严格回填并返回报告。
- 按配方构建候选训练集，执行准入/拆分/泄漏检查，发布不可覆盖的数据 Release。
- 为指定 Release 生成版本化 Qwen training_input。
- 提交、查询、取消训练任务；登记经验证的模型产物。
- 提交固定评测任务并关联报告与发布状态。

每次写操作建议包含 `request_id`/幂等键、输入版本及摘要、协议版本、操作者或服务身份；返回 job ID、状态、产物引用与结构化错误。身份冲突、重复结果、输入已变化、硬排除命中、泄漏、音频不可读应有明确错误码。

所有新平台引用都应指向具体版本，避免运行中追踪一个可变的 latest。跨机器部署须确定共享盘或对象存储、路径映射和访问权限；当前本地 catalog 的 URI 不会自动变为远程下载地址。

## 12. 联调顺序与验收用例

1. 固定三方代码版本/工作区补丁和一个小批次；包括正常语音、人工无语音、多人说话、未完成标注、同 ID 冲突、错误摘要、跨批次重复和不可读音频。
2. 导出待标池，标注平台导回；核验身份守恒、v1/v2 协议、null/空串不失真。重复 ID、空摘要与错 batch 应由接入校验拒绝。
3. strict merge 后检查报告与抽样 labels；确认 v2 人工类别生效，source_category 可追溯，未命中行保持原结果。
4. 汇总候选集；验证硬排除发生在目标文本回退前，清洗前后类别与丢弃原因可追溯，缺失 gold 不被误判为可训练。
5. 完成约定审核与分组拆分，固定评测保留集；交集检查通过后冻结 Release。
6. 适配 JSONL 并在训练 worker 上逐条预检音频；校验条数、文本编码与身份索引一致。
7. 运行小规模训练冒烟，检查取消/失败/恢复、checkpoint 可加载和 LoRA 产物类型；不以 GPU 进程启动作为成功。
8. 同一固定评测版本比较 base 与新模型，回收报告和门禁状态，验证从实验能反查到标注、分类、源音频。

仓库已有相关测试入口（本文未执行，供实施时复用）：

```bash
python -m pytest tests/test_assemble_training_parquet.py tests/test_training_and_evaluation.py -q
python -m pytest tests/unit/test_multi_speaker_training_exclude_061.py tests/unit/test_training_outlets_exclude_061.py -q
python -m pytest tests/test_annotation_v3.py tests/test_warehouse_033.py tests/test_dataset_v3_build.py -q
```

正式开发前仍需三方确定：标注平台实际导出版本及地址、允许的标签审核等级与伪标签比例、是否训练无语音目标、可靠分组元数据、固定评测版本、存储路径映射，以及采用本地 Qwen 扩展还是另行固定的官方源码版本。这些是配置与验收输入，不应在平台里隐式猜测。

## 13. 开发阅读入口

- [工序总览](工序总览.md)：治理、训练、评测整体划分。
- [039 标注交接契约](../04-改进需求/进行中/039-语音标注平台构建先验.md)：理解旧听写通道；v2 差异以本文及当前代码为准。
- [标注后训练数据汇总](../07-操作手册/标注后训练数据汇总.md)：036 操作与 061 排除。
- [032 完整流水线设计](../04-改进需求/进行中/032-工序一标注到Qwen3-ASR训练数据流水线设计.md)：目标设计参考，不能视为全部已实现。
- [annotation_v3](../../src/audio_engine/core/annotation_v3/)：正式标注协议与金标状态机。
- [dataset_v3](../../src/audio_engine/core/dataset_v3/)：分组、采样、保留与审计。
- [warehouse](../../src/audio_engine/core/warehouse/)：整批完成校验与正式仓库能力。
- [CLI](../../src/audio_engine/cli/main.py)：release、training、review、eval 的真实参数与门禁。

历史 README 与部分手册仍含 v2.2/v2.3 默认值、旧分类入口及旧文档路径。开发时以选定版本的代码、现行规则及实际产物为准；本文新增的集成建议不表示已经存在服务接口。
