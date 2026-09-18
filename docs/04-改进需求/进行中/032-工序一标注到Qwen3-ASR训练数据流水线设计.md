# 032 工序一标注到 Qwen3-ASR 训练数据流水线设计

日期：2026-09-17。状态：设计，尚未实现。范围：工序一清洗、五分类、人工标注之后的数据组织、配比、冻结及工序二训练接入；不改变现行五分类规则。

## 1. 目标与核心决定

拿到 XLSX / Parquet / JSONL 后，先汇入带来源、审核证据和音频身份的统一 Manifest，再完成分组隔离、训练配额选择、Release 冻结，最后导出 Qwen 专用 JSONL 并自动启动训练。

**XLSX 是人工交付界面，Manifest 是业务真相源，Qwen JSONL 是训练消费格式。三者不能互相替代。** 不把多个 Excel 直接拼接后随机切分，也不把自动分类成功等同于标签可训练。

```text
classified + 人工回填 / 外源金标 + 来源索引
  → 导入校验与审核证据核验
  → reviewed 统一 Manifest
  → 全局去重与泄漏分组、继承 reservation
  → 固定 dev / 独立 eval；剩余组形成 train_pool
  → 按类别与训练价值选择 train
  → 质量/配额/泄漏门禁 → 不可变 Dataset Release
  → Qwen train.jsonl + val.jsonl + 行级追溯索引
  → 音频与 processor 预检 → training run → checkpoint 验证与登记
  → 固定评测集上对比 base / candidate
```

本文把用户口径 `val` 映射到引擎现有 `dev`，传给训练器 `--eval_file`。`eval_random` / `eval_core` 是独立测试用途，不送给训练器选 checkpoint。

## 2. 仓库已有能力与缺口

本设计基于合入仓库的本地代码，训练框架版本以 Release 所记录的提交和文件摘要为准。

- `src/audio_engine/core/annotation_v3/`：已有人工标注包、导入、复审与金标证据协议。优先复用，不另造“填了文本就审核通过”的快捷通道。
- `core/dataset_v3/{grouping,reservation,sampling,quota_solver,release}.py`：已有泄漏分组、用途预留、四类训练池、配额与原子 Release；默认输出在 `data/releases/`，含 train/dev/eval_core/eval_random/excluded。
- `configs/datasets/zh_asr_v3.yaml`：已有训练池 40/20/20/20、交叉约束、每通话上限 3 等配置；**尚不能视为本文五业务类可配置配比已实现**。
- `core/training.py`：已有外部训练执行和 Registry；传入的是引擎 train/dev Manifest，不会自动转成 Qwen `audio/text`。
- `Qwen3-ASR-main/finetuning/qwen3_asr_sft.py`：实际读取 `audio`、`text`、可选 `prompt`，支持 `--train_file` 和 `--eval_file`；训练中验证是 loss，不能替代生成式 CER / 业务评测。
- `Qwen3-ASR-main/scripts/tools/xlsx_to_jsonl.py`：当前按 `ID`、`IT标注文本`、`wav_dir/ID.wav` 转换，过滤空标签并按行随机切分。仅适合已有数据的临时转换，正式入口不能复用它的拆分逻辑。
- `quality.split_dataset`：按组哈希分配，所谓 stratify 参与哈希，不保证精确分层配额。正式主流程复用 reservation，不能再运行它重分一次。
- `pipelines/build_training_set.yaml` 是旧流程示例，没有实现本文的标注导入、配比及 Qwen 导出。

需补齐：多格式入口适配、五分类到训练选择的明确契约、冻结验证集管理、类别联合配额、Qwen 导出器与启动适配器。现有 v3 的评测预留/双审/配额门禁继续生效，不为先跑通训练而绕开。

## 3. 输入如何组织

### 3.1 一个批次一个源快照

每批保存原始交付文件、分类 Manifest、来源索引、标注协议版本及文件 SHA256。相同音频在多个文件出现时先归并身份，不把三个格式当三份数据。

- **XLSX**：显式配置工作表、列映射。优先按 `sample_id` 回连 classified；历史 `ID` 必须连同 source/batch 命名空间使用。ID 按字符串读取并保留前导零，禁止依靠行号 join。要求保存原始表，禁止原地改写。
- **Parquet / 引擎 JSONL**：通过现有 Manifest 读取，保留 `audio/transcripts/labels/lineage`。扁平表字段必须显式映射回嵌套结构，不能认为顶层 `gold_text` 会自动进入 `labels`。
- **仅有 Qwen audio/text JSONL**：只能当外源文本与音频候选。必须补来源、类别、审核、分组及文本版本证据后才能正式构建；语言前缀须按格式解析一次，不能重复加前缀。
- **只剩 XLSX、缺 classified**：用稳定 ID + 音频索引建立候选 Manifest，再补全来源和审核。已有人工文本不丢弃，但不得伪造复审记录。

列映射建议：`ID → id`、`IT标注文本 → labels.gold_text`、音频路径 → `audio[选定音频键]`；原分类导出的 `type` 保留为 `labels.type`。业务五类使用 `labels.category`，**type 是细分桶，不能直接当五类**。所有映射记录在导入报告中。

### 3.2 统一数据契约

沿用 `Sample`，新增字段放入 labels 或独立版本化元数据；以下包含拟新增字段，不表示现有导入器均已支持。

- 身份：`id`、source/batch/snapshot 标识、`source_path`、`sha256`、原音频/标准 PCM 摘要。
- 音频：`audio` 键路径、`duration`、`sample_rate`、`channels`、本次选用的音频键及摘要。
- 来源关系：source 命名空间内的 `source_audio_id`、`call_id/conversation_id`、可靠 `speaker_id`、`leakage_group_id`。不能从文件名前缀猜通话。
- 分类：原始 `category/type/rule_version/risk_tags` 保留；若人工纠正类别，另存 `reviewed_category` 和依据。拟新增 `training_category` 为审核后的配额口径，无纠正时用原 category。
- 标签：`gold_text`、`gold_kind`、`label_source`、`label_tier`、`annotation_state`、标注/复审/裁决身份、队列版本及文本修订号。
- 派生：训练池、split、reservation 摘要、采样种子、配置摘要、入选/排除原因。

`gold_text=null` 表示未知，`gold_text=""` 仅表示经确认无语音。复用 XLSX 的 `__EMPTY__` / `__NULL__` 约定；空白单元格绝不自动当静音。公式单元格、重复 ID 文本冲突、未知枚举、找不到音频、缺失来源均写入错误清单并阻断相关数据发布。

### 3.3 文本准入

第一版只用有证据的人工/可信外源标签；单审训练样本沿用当前独立抽检规则，val 采用双审或裁决证据。`pending/conflict/rejected`、听不清、目标说话人不确定、无效音频不进入监督训练。

多模型共识保留为候选。只有启用伪标签实验并满足现有审计契约时，才能进入 `pseudo_high_audited`，绝不进入 val/正式测试。

训练正文使用审核后的完整逐字转写；分类用的中文筛除、语义比较文本不能替代训练文本。数字、英文、标点按版本化转写规范保留或统一处理，不为提高一致率删除正文。背景/混合说话人的转写范围必须统一，无法确定目标时回流裁决。

## 4. train / val 的拆分与防泄漏

### 4.1 先隔离，再配比

1. 在所有批次候选与已有冻结集合间建立全局关系图。相同原录音、通话、可靠说话人、文件/PCM hash、已确认近重复任一关联，合并为同一泄漏组；不同来源的局部 ID 加命名空间，内容 hash 跨来源比较。
2. 继承已有 reservation，包括未最终入选的评测预留组；calibration 和 governance_hold 不回流训练。
3. 对首次接入、没有 reservation 的历史批次，在采样之前显式创建并冻结 reservation，记录数据/模型结果此前是否被查看。不得重新包装成“从未观察过的盲测”。现有 v3 的 eval 目标与审核要求必须完成；复用历史 eval 时新增引用和一致性验证支持，不能仅把目标改成 0。
4. 在未被上述用途占用的合格组中固定 dev。初次可用约 10% 的样本数为目标、90% 留训练池，但必须以整组分配；比例是目标而非拆组要求。已有 dev 永远不因训练配方变化而重新随机抽取。
5. 只在 train_pool 中进行类别配比和下采样；未选中样本保留后续利用原因。

val 保留目标业务自然分布，不照搬训练配比。少数关键类别不足时，单独建诊断切片或使用 eval_core，不能偷偷混入主 val 改变整体指标。每次记录各类条数、时长、独立组数及覆盖缺口；组太少时标记指标证据不足。

分组哈希提供稳定初始顺序，首次分配用带类别/时长偏差惩罚的整组选择；冻结后以分配表为准。大组导致比例无法满足时报告偏差或阻断，不拆组。缺分组元数据进入 hold，不能回退到每条 ID 独立分组并宣称防泄漏完成。

跨批次新增样本若连上已有 val/test 组，继承保留用途；新增关系把历史 train 与 val/test 连通时阻断发布，标记受影响版本并重新评估污染范围，不能静默挪动已训练样本。

### 4.2 三种评测角色

- `dev → val.jsonl`：训练过程验证与 checkpoint 选择，固定人工金标。
- `eval_random`：独立、近业务自然分布的效果判断。
- `eval_core`：否定反转、短句、噪声、串音等风险覆盖。

验证集可被用于调参，不能把它的最佳指标当独立测试结果。错误挖掘只在训练池做；不得将 val/test 的错误样本补回同一轮训练。

## 5. 如何控制类别占比并保证训练价值

### 5.1 五业务类与训练池分开

现行五类是 `gold_candidate / semantic_risk / hardcase / voicemail / environment_noise`。它们描述业务情况；分类后人工补出正确文本，才能成为训练样本。

第一轮提出一个**待实验验证的起始配方**，不是已证明最优比例：gold_candidate 50%、semantic_risk 25%、hardcase 20%、voicemail 5%、environment_noise 0%。例如最终 train 10,000 条，对应 5,000 / 2,500 / 2,000 / 500 / 0 条；这些配额在 val/test 隔离之后计算。

- gold_candidate：提供普通语音覆盖，仍要求人工/可信外源文本，不因名字含 gold 而自动成为金标。
- semantic_risk：优先人工确认的否定、数字、短句及意义反转纠错，不训练分类器猜出的文本。
- hardcase：只选人能可靠听清并完成审核的困难音频；无法转写不是“更有价值的难例”。
- voicemail：文本准确且业务确实需要时纳入。按录音/近重复和模板限制重复，避免大量相同提示音占据训练。
- environment_noise：第一版关闭空转写训练。该类若人工确认有可转写语音，应记录类别纠正并进入对应语音类；确认无语音的样本保留作幻觉诊断。后续先验收空目标的 collator、loss、生成停止行为，再独立试验少量负样本。

另一条轴复用现有四训练池：`qwen_error_fix / human_high_risk / human_ordinary / pseudo_high_audited`。第一版建议 40% / 20% / 40% / 0%，与现有默认 40/20/20/20 区分；伪标签审计完成后另开版本实验。

同一条样本只占一个主业务类和一个主训练池。先将符合伪标签审计的非人工标签归入 pseudo；人工标签按“冻结 base Qwen 在训练池上可确认的错误 → 其他人工高风险 → 普通人工”顺序互斥归池。错误定义、CER 归一化、base 模型及解码摘要必须固定，不能用 val 结果决定样本池。

### 5.2 联合配额，而非重复拼接

对每个候选定义二元入选变量 x；限制总量 N、每业务类数量、每训练池数量，以及来源/通话/时长/噪声覆盖。类别与训练池交叉满足：不能先各自抽一份再拼起来，否则会重复计数和破坏比例。

实施时扩展现有 `quota_solver` 与 sampling，支持版本化的 `category × pool` 联合选择；不能声称现有四池求解器已支持任意五类约束。同一输入、稳定 ID 排序、种子及配置必须得到相同结果。

- 主配额按**唯一条数**计算，比例总和必须为 1；使用最大余数法把比例转换为整数目标，平局按固定类名顺序。
- 同时报告音频小时数及占比。短句过多会导致条数达标但有效语音不足，因此配置总时长范围、时长桶覆盖及按来源上限；这些值由候选盘点确定，不能凭空承诺。
- 保留 `max_per_call=3` 起步值，增加同模板/重复上限及 source/speaker 集中度报告。去重后不放回抽样，不靠复制少数类凑配额。
- 质量、审核、防泄漏、唯一性是不可放松的硬约束；类别/池比例的允许误差必须显式配置，不能内置静默降级。
- 第一版配额不足默认失败，报告目标、合格供给、约束后容量和缺口。补标、调小总量或发布新配方后再构建。

只考虑业务类时，给定可用量 A_c、比例 p_c，总量上界为 `min(floor(A_c / p_c))`（仅 p_c > 0）；它只是上界，交叉池、去重与每通话限制还会降低容量。例如 voicemail 仅 200 条且比例 5%，总量最多 4,000，而不是复制成 500 条完成 10,000。

样本均衡与训练器重采样只能选一个主控制点。第一版导出唯一行、训练器常规 shuffle；不再叠加 weighted sampler。若后续启用重复曝光，分别报告唯一条数、曝光次数和等效时长。

### 5.3 配方草案（拟新增 schema，不可直接交给当前 CLI）

```yaml
schema_version: training_dataset_recipe_v1
release_id: zh_asr_human_mix_v001
seed: 42
inputs:
  reviewed_manifest: datasets/stage1/derived/reviewed_batch001.parquet
  reservation: <现有冻结 reservation 文件>
  source_index: <不可变来源索引>
train:
  target_count: 10000
  category_field: training_category
  category_ratios:
    gold_candidate: 0.50
    semantic_risk: 0.25
    hardcase: 0.20
    voicemail: 0.05
    environment_noise: 0.00
  pool_ratios:
    qwen_error_fix: 0.40
    human_high_risk: 0.20
    human_ordinary: 0.40
    pseudo_high_audited: 0.00
  max_per_call: 3
  replacement: false
  on_shortfall: fail
validation:
  use_frozen_dev: true
  require_dual_review: true
export:
  format: qwen3_asr_jsonl_v1
  audio_key: resampled_16k
  language: Chinese
  allow_non_speech: false
```

该配置需编译为现有 reservation/build 配置及新增类别约束；未写出的 v3 门禁继续继承并在最终快照展开，不能因草案省略而取消。

## 6. 目录与版本组织

沿用现有 `data/releases` 和 catalog，不把 Release 擅自迁移到新根目录。以下 stage2 和 export 子目录为新增设计。

```text
data/exports/annotations/<batch>/<revision>/    # 原始人工交付 XLSX
datasets/stage1/derived/reviewed_<batch>.parquet
datasets/stage2/candidates/<snapshot>/          # 合并候选、导入问题与盘点
data/releases/<release_id>/
  train.parquet / dev.parquet
  train.jsonl / dev.jsonl                      # 引擎 Manifest，不是 Qwen 格式
  eval_random.* / eval_core.* / excluded.*
  release.json / sampling.json / leakage_report.json / audit_report.json
datasets/stage2/exports/<release_id>/<export_digest>/
  train.jsonl / val.jsonl                      # Qwen audio/text
  train.index.parquet / val.index.parquet      # line_no → sample_id / 审核 / 类别
  export.json / preflight.json                 # 摘要、路径映射及预检结果
runs/training/<job_id>/                        # 日志、状态、checkpoint
configs/training_sets/<recipe>.yaml            # 拟新增数据配方
configs/training/<recipe>.yaml                 # 拟新增训练超参
```

音频保留在已登记外部存储，Manifest 引用它；不把原音频复制进 Git。服务器路径通过版本化根目录映射解析，导出时固化训练端可见绝对路径及音频摘要。Windows 的 D 盘路径不能直接写给 Linux 训练机。

Release 不可覆盖；改标签、split、配比、输入音频或规范须生成新版本。仅部署路径变化可生成同 Release 的新 export，但摘要仍校验相同音频内容。保存输入摘要、分类/标注/归一化版本、reservation、种子、代码/依赖版本，确保可重建。

## 7. Qwen 导出和训练衔接

### 7.1 导出契约

每行形如：

```json
{"audio":"/mnt/audio/batch001/000123.wav","text":"language Chinese<asr_text>我现在不需要，谢谢。","prompt":""}
```

只从 Release 中的 gold_text 生成正文；类别、风险等级、复审备注不混入 text。语言未知使用框架支持的 `language None`，多语数据逐条使用已核验语言，不一律伪装中文。第一版 prompt 固定空字符串，避免把答案或标注提示泄漏进模型输入。

导出时验证行数、唯一 ID、正文和语言前缀、音频可解码且时长为正、训练端路径可访问；选定音频应与人工听音版本语义一致，禁止误用 Kimi 推理 padding 音频。16 kHz 与本地 collator 默认对齐；时长/文本长度上限依据实际 processor 和训练资源预检设定，超限先隔离或另行对齐切分，禁止静默截断造成音文错位。引擎继续不隐式新增 VAD。

行号索引从 1 开始，与 JSONL 一一对应；记录 sample_id、audio digest、gold revision、category、pool、split。门禁失败不发布半成品；先临时写出、全量校验，再原子冻结 export 并登记 catalog。

### 7.2 自动执行状态机

`annotation_imported → eligible → split_locked → sampled → release_published → exported → preflight_passed → training_running → checkpoint_verified → evaluated`。

人工审核完成并非一定整批都进训：未合格项继续留队列；是否允许以已完成子集构建由配方明确，必须重新满足完整配额和门禁。自动任务只消费已冻结、通过门禁的产物；不监听“文件刚出现”就开始训练。

现有 `training run` 的环境变量为：

- `AUDIO_DATA_RELEASE_ID`
- `AUDIO_DATA_TRAIN_MANIFEST`
- `AUDIO_DATA_DEV_MANIFEST`
- `AUDIO_DATA_RECIPE`
- `AUDIO_DATA_CHECKPOINT`

新增 Qwen adapter 读取这些变量，核验 release/export 摘要，解析训练配方，构造参数数组启动本地 finetuning 脚本，不依赖 shell 字符串拼接。adapter 负责 Manifest → Qwen export 的引用衔接，不修改上游 SFT 的数据职责。

导出完成后的训练命令形态如下；路径为占位示例，batch/学习率仅冒烟参数，需按实际资源验收：

```bash
torchrun --nproc_per_node=2 Qwen3-ASR-main/finetuning/qwen3_asr_sft.py \
  --model_path /models/Qwen3-ASR-1.7B \
  --train_file /exports/release001/train.jsonl \
  --eval_file /exports/release001/val.jsonl \
  --output_dir /runs/training/job001 \
  --batch_size 1 --grad_acc 16 --epochs 1 --lr 2e-5 \
  --save_steps 200
```

训练环境独立于数据处理环境，锁定框架、torch/transformers 等依赖和启动工作目录。启动前执行实际 processor/collator 小批预检，并确认工序一 GPU 已释放。

当前 runner 仅以退出码和 checkpoint 路径存在判成功，需要增强：核验实际权重/config/processor、可加载性及少量推理。SFT 当前按保存步写 checkpoint，极短冒烟可能未到保存步，adapter 必须要求至少一次有效保存或补齐明确最终保存契约，不能把空 output_dir 注册成模型。LoRA 必须区分 adapter 与合并权重；需要现有推理入口时通过合并与加载验证再登记对应可用产物。

任务幂等键涵盖 release、export、基础模型摘要、训练配方和代码版本。重复成功事件不重训，失败重试只复用相同输入；新增 attempt 和日志，记录恢复 checkpoint。超时/进程中断需转为明确失败或可恢复状态，不能永久停在 running。自动训练不意味着自动替换线上模型。

## 8. 如何判断“有训练价值”

先产出候选盘点：每类/池的原始量、审核合格量、去重量、隔离后可用量、时长、来源/说话人覆盖、排除原因与配额缺口。供给不足时先减少训练规模或补标，不通过降标签标准追求条数。

做三组可比实验：base；自然分布的人工训练集；本文配比的人工训练集。保持相同基础权重、文本规范、训练预算与固定 val/test；若时长或曝光预算不同，在报告中明确，避免把多训练带来的收益归因于配方。

评估至少包含整体 CER、按五类/语音时长/噪声的 CER、否定与关键数字错误、无语音幻觉率、普通样本回退；无语音参考为空不计算逐条 CER，用非空输出率等独立指标。记录样本数、时长及按泄漏组重采样的置信区间，少量样本不能据此断言改进。

配比调整只根据训练诊断与 val；独立 test 用于确定候选的最终验收。伪标签、空转写、噪声增强分别做新增实验，不能一次引入多个变量后无法解释收益。

## 9. 实施拆分与验收

1. **导入与盘点**：增加 XLSX/扁平表适配，复用 annotation_v3，输出 reviewed Manifest 和错误清单。验收多格式同源结果一致、ID 前导零、空值/确认空文本、冲突修订和审核证据。
2. **分组与冻结**：复用 grouping/reservation，支持跨批次冻结集合核验和 dev 固定引用。验收同通话/重复音频不跨集、缺元数据阻断、旧 reservation 不漂移、新关系冲突可检出。
3. **联合配比**：扩展 sampling/quota_solver，保留 release 原子发布。验收整数配额、交叉约束不可行、少数类不足、每通话上限、确定性与不放回。
4. **导出与适配器**：新增 `core/training_export/` 与 `scripts/train_qwen3_from_release.py`（拟议路径），以现有 training run 为入口。验收 JSONL/索引/Manifest 一致、Linux 路径、音频摘要、前缀单次写入、val 参数正确。
5. **小规模闭环**：先用满足 reservation/审核要求的小配置构建 Release，在实际训练服务器完成一次训练、保存、加载、推理和 base/candidate 评测；再提高规模。小规模配置另存，不假称完成默认 2,000+2,000 评测配额。

完成标准：一份数据配方引用已完成标注的快照后，可自动生成可追溯且防泄漏的 train/val，按配置控制比例，遇到缺口明确失败，并用冻结产物启动 Qwen 训练、登记可加载模型与评测报告。本文交付只完成设计，尚未执行上述开发或 GPU 验收。

## 10. 关联资料

- [工序总览](../../01-项目架构/工序总览.md)
- [现行五分类规则](../../02-规范规则/数据挑选规则_v2_2_auto_noise.md)
- [002 全自动训练评测闭环](002-全自动训练评测闭环.md)：本文细化其中的数据到训练接入部分。
- [031 后置分类入口](031-工序一后置统一分类分拣入口.md)：本文消费其 classified/XLSX 产物。
- [Qwen 训练说明](../../../Qwen3-ASR-main/finetuning/README.md)
- [现有数据策略](../../../configs/datasets/zh_asr_v3.yaml)
