# audio-data-engine → audio-label-master：VAD 分片后对接上下文

> 核对日期：2026-10-14。提供方：`audio-data-engine`；接收方：`audio-label-master`。
>
> 本文是生成标注平台改动文档之前的上下文知识文档，说明上游数据单位的变化、现有平台契约以及由此产生的影响。本文不代表标注平台已完成适配。
>
> 依据两个项目当前本地工作区核对，包含未提交改动；引擎基线 HEAD 为 `8ae239dec1c9a6b5fbb19028ef7f91beca1973ef`。本次只做代码与文档核查，未执行真实模型、标注平台或服务器端到端验证。

## 1. 已确认的业务口径

本次变化的核心是：**以子片段为标注单位，以父 PCM 为共享上下文关联对象，播放器只播放当前子音频。**

用户已经明确三个要求：

1. **每个 VAD 分片独立标注。** 保持整体页面和现有操作方式，任务分包、待标条数和进度改按子片段统计；一条父 PCM 可以产生多条待标记录，整体数据量通常会增大。
2. **沿用原来的上下文导入。** 一条父 PCM 切出的多个子片段可以共用同一份上下文，不要求用户为每个子片段重新整理或重复导入上下文。
3. **只播放对应子音频。** 本次无需父音频播放器，也无需父子播放切换。父音频的路径和身份用于上下文关联及血缘追溯。

历史 041 文档中“父录音可供上下文回听”的表述，不构成本次标注平台的功能要求；以这里已确认的播放范围为准。

## 2. 上游发生了什么变化

当前工作区的新链路处理顺序为：

```text
父 PCM 登记
  → 规范化为 16 kHz 单声道 PCM16 音频
  → FSMN VAD 切段并过滤短段
  → 每个保留片段成为独立 Sample
  → 片段自己的 ASR、聚合、分类／标签分流
  → 按分流结果交付片段待标池
  → 标注平台逐片段标注
  → 引擎按片段身份回填
```

原先“一条待标记录对应一个父 PCM”的假设不再适用于新批次。现在一条父 PCM 可以对应零个、一个或多个保留片段；同一父 PCM 的不同片段也可能进入不同分流结果。

当前切段规则中，小于 1 秒的片段丢弃，至少 1 秒的有效片段保留。具有有效起点的文件尾补段也可进入下游；无语音、全短段、解码或 VAD 失败的父源不会直接整条兜底送标。因此，平台条数取决于**实际交付的子片段集合**，不能简单按父 PCM 数量乘固定系数推算。

`cleaned_<batch>` 在新链路中是片段 Manifest。分类和标签处理的是片段自己的音频及 ASR 证据，父 PCM 的旧转写、分类、金标和完成状态不直接继承给子片段。

本次上下文适用于 VAD 后的各类分流，不限于旧 `semantic_risk` 听写池。待标范围由上游交付池决定，平台无需重新执行 VAD 或分流。旧分类和自由标签的模式接线是另一项工程，不因本文而自动视为已经完成。

## 3. 需要分清的三种关联

### 3.1 标注与回填：关联当前子片段

引擎内的片段身份为：

```text
Sample.id = labels.segment_id
Sample.sha256 = labels.original_audio_sha256 = labels.segment_pcm_sha256
```

导成交接表后，`Sample.id` 对应 `sample_id`。当前片段 ID 形式是 `seg_` 加 28 位摘要，根据父源、规范音频、声道、裁切采样边界和处理版本稳定生成。

原有回填键的字段名继续使用：

```text
批次内：(sample_id, original_audio_sha256)
跨批次管理：(batch, sample_id, original_audio_sha256)
```

这里的 `original_audio_sha256` 在新链路里表示**当前子片段未加 padding 的裸 PCM 内容摘要**，不是父 PCM 的摘要，也不是整份 WAV 文件的摘要。平台应原样保留，不能对下载或转码后的 WAV 重新算摘要并替换此字段。

每个片段有自己的标注记录和状态。内容不同的片段摘要不同；内容完全相同也可能有相同摘要，所以片段主键不能缩减为单一摘要。

### 3.2 血缘：关联父源

片段 Manifest 的 `labels` 中已有：

- `source_audio_id`：父源身份，通常来自切段前的父 Sample ID。
- `source_pcm_sha256` / `parent_pcm_sha256`：父源摘要，用于追溯，不能替代片段回填摘要。对非裸 PCM 输入，现实现可能继承源文件摘要，不能一概解释为去掉文件头后的 PCM 体摘要。
- `segment_index`：父源内片段序号，从 0 开始；仅辅助展示。
- `vad_start_ms` / `vad_end_ms`：在规范化父音频上的全局毫秒边界。
- `clip_start_sample` / `clip_end_sample`：实际裁切的采样边界，区间为左闭右开。
- `vad_processing_version`、`is_vad_segment`：处理版本与片段标识。

文件尾补段可能 `vad_end_ms=null`，真实补尾终点保存在 `vad_inferred_end_ms`，并标记 `vad_end_reason=eof_fallback`。平台不能仅因 `vad_end_ms` 为空就拒绝一个已经发布的有效子片段；播放使用现成子文件即可。

### 3.3 上下文：关联父 PCM 原来的业务记录

`audio-label-master` 现有上下文匹配关系是：

```text
待标条目 PoolItem.business_record_id
  ↔ 上下文 Excel 的 file_id
  → 读取该记录的 AI话术
```

**`sample_id` 是片段身份，`business_record_id` 是上下文关联身份，两者在新链路中可以不同。** 同一父 PCM 的多个片段应保留相同的父业务记录 ID，同时具有各自的 `sample_id`。

`source_audio_id` 只有在确认与原上下文 `file_id` 完全一致时，才可以直接映射成 `business_record_id`。如果原父 PCM 的业务 ID 与其 Sample ID 不同，上游需要保留原业务 ID 或提供明确映射，平台不能根据 `seg_...`、文件名截取或片段序号猜测。

## 4. 原上下文导入如何继续使用

平台当前上下文 Excel 的必需列是 **`file_id` 和 `AI话术`**。本次保留这个文件格式、导入入口和展示位置。

例如，父 PCM 对应业务记录 `record_A`，切出三个子片段。上下文文件仍只需原来的这一条记录：

```text
上下文：file_id=record_A → AI话术=这条父 PCM 原来对应的话术

子片段 A1：sample_id=seg_...A1，business_record_id=record_A
子片段 A2：sample_id=seg_...A2，business_record_id=record_A
子片段 A3：sample_id=seg_...A3，business_record_id=record_A
```

三个子片段都展示同一份父级上下文，但各自保存听写结果、人工类别、跳过和完成状态。**共享的是上下文，不是标注答案或完成状态。** 当前子片段的候选转写也不能用父录音整段转写代替。

对接时需要注意现有平台的实际行为：

- 标准队列导入优先读取显式 `business_record_id`；缺失时，仅在同时存在 `id` 和 `sample_id` 的情况下尝试取 `id`。新片段包应明确传入父业务 ID，不能依赖默认行为。
- 原始 Parquet 转池后端支持配置 `business_record_id_field`，默认通常取 `id`，但目前只读取顶层列。父业务键需显式投影为顶层列再配置；若只存在于 `labels.source_audio_id` 等嵌套结构，还需补充取值适配。无论采用哪种来源，都必须验证值等于原上下文 `file_id`。当前前端尚无父业务键配置项，不能当作用户在现有页面已经可以完成的配置。
- 现有匹配不会自动从 `source_audio_id` 推导父业务 ID；旧 legacy 模式的前缀处理也不能作为 VAD 片段到父记录的通用转换规则。
- 原业务 ID 应按字符串保留，避免数值转换丢失前导零；不能擅自截断 UUID、改大小写或添加片段后缀。

同父片段即使分到不同任务包，也应能使用同一份上下文。上下文缺失时保留原有缺失提示和补导能力，不要求因此合并片段或回退父音频播放。父级上下文不自动变成“前一个片段的听写文本”，本次无需额外推导片段间上下文。

现有上下文展示还会优先使用条目原带的 `previous_text`，然后才使用 Excel 话术；需要核对交付池中的原有值，避免将旧整段转写误放进该字段。此处沿用既有展示规则，不新增上下文文件格式。

## 5. 音频播放路径必须明确指向子片段

业务上说的“子 PCM”，当前引擎实际落盘为 **WAV 封装的 PCM16 片段**：16 kHz、单声道，平台可以继续使用现有 WAV 播放能力。

当前片段 Sample 中的路径含义如下：

- **`audio.resampled_16k`：子片段 WAV，作为本次播放来源。**
- `audio.normalized_16k`：规范化父音频。
- `audio.parent_resampled_16k`：规范化父音频。
- `audio.parent_raw`：未切分的父源音频。
- **`source_path`：当前实现也指向规范化父音频，不能作为子片段播放来源。**

平台标准队列的播放字段是 `wav_path`。交接适配应将其绑定到当前片段的 `audio.resampled_16k`，或该子文件拷贝／路径映射后的可访问位置：

```text
子片段 Sample.audio.resampled_16k
  → 交接记录 wav_path
  → 平台媒体服务
  → 现有标注页播放器
```

采用原始 Parquet 转池时，当前平台的自动候选顺序是 `source_path` 优先、`audio.resampled_16k` 次之。这会在新 VAD 数据上优先选中父 WAV。对片段池应明确使用以下映射并核对实际解析结果，而不能依赖自动推荐：

```text
wav_primary         = audio.resampled_16k
wav_fallback        = null
wav_fallback_policy = none
```

片段文件缺失或不可达时，应报告当前片段音频异常，不能自动播放父音频。父音频路径只作追溯资料，无需出现在播放控件中。播放时长、进度和重播范围都以实际子文件为准，不再以父音频时长为准，也不需要浏览器现场裁切父文件。

跨机器交付仍需保证子文件可访问；引擎本机绝对路径或只在源机器有效的软链，不能直接等同于平台可播放的媒体地址。

## 6. 对分包、标注和页面的影响

保持现有待标池、任务分包、认领、标注页、上下文区、播放器、保存和导出流程，变化主要落在数据关联和统计口径。

- **分包按片段条数。** 原先每包 500 条的含义，变为每包 500 个待标子片段。同一父 PCM 的多个片段不能因父 ID 或父摘要相同而被合并或当成重复导入。
- **进度按片段状态。** 一个父 PCM 有三个片段，完成其中一个就只完成一条；不能将另外两个自动置为完成。任务包数和总条数按实际待标片段重新计算。
- **标注只覆盖当前片段。** 现有听写、无语音、人工类别、多人说话、跳过、复核等动作继续沿用，保存对象为当前子片段。
- **参考文本对应当前片段。** `candidate_text` 如需展示，应来自当前片段的识别结果；若既有模式为盲标，继续遵守原有显示规则。
- **上下文可以重复展示。** 切到同父的另一片段时，可看到同一份 AI 话术，但必须加载另一片段自己的音频、答案和状态。
- **页面无需增加父源工作台。** 父源 ID、片段序号或区间可以作为现有信息区域的辅助信息，不改变整体布局，也不要求增加时间轴或父音频播放器。

多标签可能同时命中同一个片段。一个池内统计应区分“标签命中次数”和“待标片段数”，不能仅因命中多个 tag 就把同一片段重复算作多条独立标注任务；确需不同审核轮次时沿用原有任务／版本区分。

## 7. 标注结果回传的语义

平台继续按原协议导出标注结果，新增的核心要求是：**回传身份必须属于被播放、被标注的那个子片段。**

必须保持片段 `sample_id` 和 `original_audio_sha256` 原样对应，并保留适用的 batch、池／任务和包版本关联。`business_record_id` 仅用于上下文关联，父 `source_audio_id` 和父摘要仅用于血缘，均不能取代片段回填键。

现有平台的人工标注语义仍然适用：有语音时保存该片段文本；确认无语音时按原协议表达空目标；多人说话等已有分类和训练排除语义继续按原协议保存。VAD 切出了片段，不代表片段一定是可正常听写的单人语音。

两个既有回流通道需要继续区分：

- 轻量听写 annotations 使用 `status`、`gold_kind`、`gold_text` 等字段，Web v2 可以带人工分类及训练排除信息。
- 正式 `annotation_v3` / warehouse 包有自己的不可变字段、包版本和审核流程，表格中的空白、`__NULL__`、`__EMPTY__` 要按该协议解释。

本次不能借片段适配混用两套协议，也不能把旧 039 文档中 v1 的“只标 semantic_risk、类别不变”套到所有现有平台操作上。

父源与片段血缘应能从交接输入或关联的源 Manifest 追溯。若使用正式 v3 包，已有不可变血缘和边界字段须保持原样；普通 annotations 目前不保证携带这些扩展列，不能仅凭普通导出文件就声称血缘已完整贯穿。

## 8. 当前已经具备与尚需对齐的边界

### 已在当前工作区看到的能力

1. 引擎已生成独立片段 Sample、片段 WAV、父源血缘、片段摘要及边界，后续入口继续消费片段 Manifest。
2. 引擎 `annotation_v3` 和 warehouse 导出行已经增加 `source_audio_id`、`parent_pcm_sha256`、`segment_id`、边界、处理版本、片段标识和 `parent_raw_audio` 等信息。
3. 平台已有 `business_record_id`、标准队列 `wav_path`、原始 Parquet 字段映射能力，以及按 `file_id` 导入 AI 话术的上下文流程。

### 不能当作已完成的对接

1. **引擎正式 v3 包目前没有导出子音频 `wav_path`。** 包里的 `source_path` 和 `parent_raw_audio` 仍是父音频；平台原始 Parquet 自动映射又优先选择 `source_path`。需补充明确的子播放引用并正确映射，不能直接拿当前包就宣称播放适配完成。
2. **父业务 ID 尚不能靠默认映射保证。** 切段时新建片段 labels，并不自动继承任意父级业务字段。交付池需明确给出与原上下文 `file_id` 对应的 `business_record_id`，或提供经过核实的父源映射。平台后端有业务键映射能力，前端配置入口尚未贯通。
3. **轻量 `prepare` 队列与正式 v3 包字段并不相同。** 轻量队列已有 `wav_path`，但当前未完整带出父源／片段血缘；其现有音频解析还有缺子文件后回退到父 `source_path` 的可能，交付前需要核对实际目标，不能假定现有解析已经满足只播子音频。
4. **平台会忽略部分未知列。** 当前标准池导入会报告忽略的额外列，PoolItem 及普通 annotations 导出未完整贯穿 VAD 父子字段。原始 Parquet 路线可保留源行关联，但这不等于所有平台对象和普通导出都已经支持新字段。
5. **EOF 信息在导出包中还不完整。** 原始片段 Manifest 有 `vad_end_reason`、`vad_inferred_end_ms`，当前正式包未完整带出。若后续展示区间，需保留真实补尾语义；只播放现成子音频不依赖重新推断终点。
6. **所有分流以片段为单位，不等于现有平台已支持所有分流导入。** 平台标准队列导入当前仍限制 `category=semantic_risk`，原始 Parquet 缺 category 时也默认填该值。其他分流池的类别／标签交接需另行适配，不能把自动补出的默认类别当作上游真实分流结果。

这些差异是后续改动文档的输入，本文不指定数据库迁移、接口路由或实现排期，也不在本次修改两个项目的业务代码。

## 9. 一个完整业务示例

假设父 PCM `record_A.pcm` 原来匹配上下文 `file_id=record_A`，上游切出两个待标片段。以下为关系示意，ID 和摘要均为占位值，并非可直接导入的真实包：

```text
父源：source_audio_id=record_A
父上下文：file_id=record_A → AI话术=您好，请问您需要办理什么业务？

待标条目 A1
  sample_id / segment_id = seg_<片段一的稳定ID>
  original_audio_sha256  = <片段一裸PCM摘要>
  business_record_id     = record_A
  wav_path               = <片段一WAV的可访问路径>
  gold_text              = <只填写片段一听到的内容>

待标条目 A2
  sample_id / segment_id = seg_<片段二的稳定ID>
  original_audio_sha256  = <片段二裸PCM摘要>
  business_record_id     = record_A
  wav_path               = <片段二WAV的可访问路径>
  gold_text              = <只填写片段二听到的内容>
```

平台出现两条待标记录，两个条目展示相同 AI 话术；打开 A1 只播放 A1，打开 A2 只播放 A2。保存 A1 不影响 A2；回传后引擎分别回填两条片段记录，不汇总成一条父 PCM 答案。

## 10. 历史数据与后续联调的判断依据

历史未切段池按原有身份和音频路径继续处理，不能仅因日期较早就推断它是父源模式；新片段池应通过明确身份、片段标识或来源快照识别。

父 PCM 的历史上下文可以在业务 ID 一致时复用。父 PCM 的历史整段听写答案、完成状态及旧包身份不能自动拆分或复制给新片段；重新切段导致身份变化时，也不能仅凭父源相同复用旧片段答案。

后续联调至少应能证明：

1. 一父多子导入后保持多条独立待标记录，父源共享不会导致误去重。
2. 一条原格式上下文 Excel 记录可匹配同父的全部子片段，跨任务包也可复用。
3. 各子片段实际播放的是自己的 WAV，播放时长与内容正确，缺文件不会退回父音频。
4. 保存一个片段不会改变兄弟片段的文本、类别或完成状态，分包和进度按片段计数。
5. 导回身份与输入逐条一致，回填不会串片段，旧整段标注不能误回填到新片段。
6. 不同父业务 ID 的上下文不会串用；原 ID 的前导零及 UUID 等字符串保持不变。
7. 已发布的 EOF 补尾片段可正常导入播放，不因模型终点为空而误拒绝。
8. 普通 annotations 与正式 v3 包分别遵循原协议，父源关联及忽略字段的处理有据可查。

以上是供后续改动方案与验证使用的判断依据，本次未执行这些联调用例。

## 11. 开发阅读入口

`audio-data-engine`：

- [清洗流水线](../../pipelines/data_cleaning_source_A.yaml)：VAD 接入位置与片段输出。
- [FSMN VAD 算子](../../src/audio_engine/operators/audio/fsmn_vad_segment.py)：子文件、父子路径和 Sample 的实际生成。
- [片段身份与字段映射](../../src/audio_engine/core/fsmn_vad/identity.py)：片段 ID、父源摘要与子 PCM 摘要。
- [回填身份解析](../../src/audio_engine/core/selection_v3/input_contract.py)：`original_audio_sha256` 的片段语义。
- [正式标注包](../../src/audio_engine/core/annotation_v3/package.py)、[不可变字段](../../src/audio_engine/core/annotation_v3/types.py)、[warehouse 导出](../../src/audio_engine/core/warehouse/export.py)：已有扩展列及包契约。
- [轻量听写工具](../../scripts/annotate_semantic_risk_transcripts.py)：`prepare / merge` 与 `wav_path`。
- [041 父源／片段兼容说明](../04-改进需求/进行中/041-兼容说明-父源与片段契约.md)：上游当前字段说明。
- [039 旧交接契约](../04-改进需求/进行中/039-语音标注平台构建先验.md)：理解历史听写通道，适用范围见本文。

`audio-label-master`，以下路径相对于该项目根目录：

- `backend/app/services/script_context.py`：上下文 Excel 列与 `business_record_id ↔ file_id` 匹配。
- `backend/app/services/import_pool.py`：标准队列字段、业务记录 ID 默认值和未知列处理。
- `backend/app/services/field_mapping.py`、`backend/app/services/convert_pool.py`：原始 Parquet 转池的身份与音频映射。
- `backend/app/domain/contracts.py`、`backend/app/services/export_annotations.py`：队列／标注交换字段及普通导出。
- `backend/app/services/previous_text.py`：上下文展示优先级和有效预标注的来源规则。
- `frontend/src/pages/AnnotationPage.tsx`、`frontend/src/components/audio/AudioPlayer.tsx`：现有标注页与单音频播放器。

接收项目开发时应复核这些入口的最新实现。本文确认的是数据语义与用户要求，具体修改清单另行生成。
