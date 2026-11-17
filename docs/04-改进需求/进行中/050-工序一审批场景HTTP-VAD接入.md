# 工序一统一场景识别、混合采样率入库与审批 HTTP VAD

状态：调用方已实现（2026-11-16）。对照 `v4.1`、工序一分步手册及 FireRedVAD HTTP 服务源码落地本仓入库/双路 VAD/手册；实际服务地址、部署版本和服务器联调结果待补充。

## 1. 业务目标和范围

工序一用一个入口扫描原始文件，根据文件名写入场景标记并按正确采样率解码。同目录可混放数字分身、审批面谈和审批外呼，生成一份父 Manifest。两条 VAD 流水线继续并列，操作员可以运行任一路，混合批次要处理全部已识别音频时分别运行两路。审批流水线负责调用已交付的服务、按返回区间本地切分和生成片段 Manifest。服务部署、权重、推理代码和配置算法归服务侧。本期交付范围是 `手册/dev/01-工序一-数据清洗落库打标.txt` 对应的分步 CLI 与下游兼容性。

- 入库统一走 `pipelines/audio_ingest.yaml`，保留现有命令和 `ingested_<batch>.parquet` 命名。用一个场景字段标记三类来源，已识别样本在整个工序一保留同一场景值。
- 数字分身走 `pipelines/vad_segment_batch.yaml`，按场景选中分身父源，使用内嵌 FSMN 和当前 `fsmn_vad_segment/v9` 行为。保留现有配置、片段 ID 和输出命名。
- 审批新增 `pipelines/vad_segment_shenpi_batch.yaml`，从同一份父 Manifest 选中审批面谈和外呼，逐条按场景取 HTTP 配置。沿用项目的算子注册、Manifest 和 `audio-data pipeline run` 入口。建议注册 `audio.shenpi_vad_segment`，一条父样本产出零到多条子样本。
- 两路片段具备同一结构和场景字段语义，直接进入已有 Qwen、GLM、DNSMOS、证据合并和自由打标。现有 XLSX 话术剔除仍为可选，匹配用父源 `source_audio_id`。

## 2. 当前实现审查与入库改造

### 2.1 审查结果

- `Manifest.ingest` 用 `path.stem` 生成 `Sample.id`，保留 `audio.raw` 并计算原始文件 SHA-256。裸 `.pcm` / `.raw` 无文件头，扫描阶段没有采样率，`pipelines/audio_ingest.yaml` 在 `audio.pcm_to_wav` 固定传 `source_sample_rate: 16000`。`audio/pcm.py` 的 `_load_mono` 只据配置解释裸 PCM。按旧入口处理面谈 8 kHz PCM 时，产物仍是可读的 16 kHz 文件，因此现有 `quality.probe` 和 `duration > 0` 检查可能照常通过，留下半时长、倍速音频。
- `audio/pcm.py` 实际负责 ch0 提取和第一次重采样，后面的 `audio.resample` 对已是 16 kHz mono 的 `pcm_wav` 通常只是建立 `resampled_16k` 路径别名。裸 PCM 经 `resample_audio` 后得到仍在 int16 数值尺度上的浮点数组，直接写成 PCM16 会削波。本机用一秒、峰值约 9800 的 8 kHz 合成波通过这两步复现：声明 16000 得到 8000 帧、0.5 秒；声明 8000 得到正确的 16000 帧、1 秒，但有 15920 帧饱和。第二步还把 `quality.source_sample_rate` 覆盖成 16000，此时 `labels.source_sample_rate` 仍是 8000。
- `BaseOperator.compute_cache_key` 默认用源 SHA、算子版本和共享参数，忽略逐样本 ID、解释采样率和实际派生输入。用两个不同 ID、完全相同的合成 PCM 字节复现了第二条命中第一条的转换缓存，直接拿到第一条的 WAV 路径。改造时还要检查后续 `audio.resample`、`quality.probe` 的缓存和 checkpoint，以及 `derived_audio_path` 仅含原始 SHA 和 ID 时的旧、新配方覆盖风险。`audio.normalize_report` 的现有直方图基于告警行，不能作为混合批次的完整场景计数。
- 现有 FSMN 切片直接构造新的 `Sample`，只继承列举的采样来源字段。单纯在入库写入标签会在 VAD 时丢失。两路须由共享片段构造显式传递场景标记。现有 `Sample.to_flat_dict` 会自动把标签投影为 `label_*`，证据合并以片段底表的深拷贝为基础，可以沿用这条链路。

### 2.2 ID 路由规则

从原始文件的 stem（`Manifest.ingest` 生成的 `Sample.id`）判定。按用户确认，数字分身文件名均以 `fenshen` 开头。审批匹配已知前缀 `shenpi_ai-` 或 `shepi_ai-`。保留原始父源 ID 的拼写与大小写。

- 以 `fenshen` 开头 → `fenshen`（数字分身）。例如 `fenshen-276cc46a-bf3387be`。分身匹配只看开头的 `fenshen`，不额外要求紧跟 `-`。裸 PCM 按现行 16000 Hz 配置解释，容器音频按文件头采样率处理。
- `shenpi_ai-` 或 `shepi_ai-` → 按前两个 ASCII `-` 切分，检查第二个 `-` 后的**整个后缀**。含大写 `V` 或 `P` → `shenpi_interview`（审批_面谈），8000 Hz。例如 `shenpi_ai-c24b27s-V2610091`、`shepi_ai-h726h9999-Ph829888`、`shepi_ai-V726-8jP29887`。
- 已确认审批前缀，且后缀无大写 `V/P` → `shenpi_outbound`（审批_外呼），16000 Hz。例如 `shepi_ai-7jsn7288-8j22j987`、`shepi_ai-V726-8j229887`。仅在前两段出现 `V/P` 不触发面谈。
- 审批中段或后缀为空、缺第二个分隔符时记录命名不合约，进入入库拒绝记录。新入库中，未匹配已知前缀的文件可以继续按通用规则规范化，标记技术状态 `unknown`、计入未识别，两个场景 VAD 都不选中。历史 Manifest 缺少场景字段时，分身入口仍兼容 `fenshen` 前缀，已知审批前缀必须提示重入库，其他历史样本的通用 FSMN 调用维持原有兼容行为，场景保持缺省。

### 2.3 入库实现要求

1. 在现有入库 YAML 接入逐文件规则。复用 `ingest.scan → audio.pcm_to_wav → audio.resample → audio.normalize_report → quality.probe → filter/select`。在裸 PCM 解码和转换缓存查询前确定场景和有效源采样率。让场景解析覆盖分片与非分片正常入口，存储并复用同一份已解析结果。规则、版本和场景键集中定义，其他复用通用算子的旧 YAML 保留其显式采样率配置。
2. 裸 PCM 按 PCM16LE 单声道解析，检查非空且字节数为偶数，以 `源样本数 / 源采样率` 计算时长。首版审批入口接受 `.pcm` / `.raw` 的裸 PCM。审批文件遇到容器头，包括后缀为 `.pcm` 的 RIFF/WAVE，记录输入不合约，避免把容器头上传成裸 PCM。分身及通用输入继续按真实文件头处理容器。
3. 复用现有本地重采样与 ch0 规范化策略。修复裸 int16 数据重采样后的尺度处理，例如先转换到标准浮点幅度域，再重采样、按 PCM16 写入，或显式舍入/限幅后以 int16 写入。用实际采样内容证实未发生大面积饱和。16 kHz 裸 PCM 直接封装 WAV，验证 PCM payload 逐样本相同。最终 `audio.resampled_16k`、顶层 `sample_rate=16000`、`channels=1` 和 `duration` 均对应规范化音频，8 kHz 面谈时长正确且 2 倍样本数。后续 `resample` 直通和 `probe` 不得覆盖原始采样语义或用旧缓存恢复旧时长。
4. 父 Manifest 保留 `audio.raw`、原始 `sha256`，通过 `labels.audio_scene` 存 `fenshen`、`shenpi_interview` 或 `shenpi_outbound`，未识别用 `unknown`，由现有序列化展开为字符串列 `label_audio_scene`。源采样率、`normalized_sample_rate=16000`、采样率来源继续各用原有字段，规则版本写入 lineage 或运行快照。规范化质量字段或运行记录保留原始和规范化时长，`labels.source_sample_rate` 与 `quality.source_sample_rate` 均表达原始值。统计父源数、有效数、三类场景、未识别数、真实源采样率和拒绝原因。保护父源 ID 的唯一性，避免后续父源结果文件按 ID 覆盖。
5. 转换缓存要区分不同父 ID、场景、有效原始采样率和规范化配方。依赖派生音频的后续缓存须区分实际规范化输入及对应配方，不能只靠原始 SHA 或共用 YAML 名。新版规范化产物路径应与旧配方隔离，同一原文件在新旧链运行时保留各自 WAV。更新相关算子/配方版本并检查分片及 checkpoint 指纹。历史审批若曾用固定 16 kHz 入库，从 `audio.raw` 对应原始目录以新批次名重新入库，再重做 VAD 和下游。已误解码或削波的父 WAV 不能靠修改标签恢复。
6. 已识别审批前缀但命名不合约、原文件不可读、PCM 格式不合约的记录须有可定位的父源拒绝原因并退出有效 Manifest。校验有效审批父源已经正确规范化。审批 VAD 以入库 `labels.audio_scene` 选 profile，并核验规则版本、ID、场景与源采样率一致，检查本地父音频是对应的 16 kHz WAV、时轴与原始文件相符。对有审批前缀但缺场景标签的旧父源，给出重入库提示，两路均不得静默消费该记录。审批原文件缺失或内容与登记哈希不一致时报失败。

## 3. 执行命令和产物

审批默认把片段产物命名为 `segments_<batch>-shenpi.parquet`，后续用 `<batch>-shenpi` 作为 `--source-name`。这是下游产物命名用的批次别名，片段的父源 ID 仍取原入库记录。这样同一份入库可以跑两路，片段、ASR、DNSMOS、合并和打标产物各自可追溯。

按 `src/audio_engine/core/source_naming.py` 的 `manifest_path` 和 `resolve_existing_manifest` 接入。当前新写默认走批次子目录，读取支持旧的扁平布局。两路 VAD 的默认输入按原 `<batch>` 查找，数字分身写 `<batch>`，审批写 `<batch>-shenpi`。VAD 支持现有 `--input-manifest` 和 `--output-manifest` 显式指定，并确认与 `--source-name` 合用时的优先级。显式路径用于隔离试跑和自定义快照。

两条 VAD 入口按 `labels.audio_scene` 选择各自父源。使用全批输入验证并统计选中、其他场景、未识别与历史资料缺标记等情况。分片与非分片一致，错误分支不能触发模型调用。选中数为零时报告该分支无可处理父源，产出可读取的零行结果，跳过模型或 HTTP 初始化。历史或显式传入的审批父源若缺标记，或已有标记与已知文件名前缀矛盾，应报告需修复的父源，防止误走分身。

以下为目标执行方式。`BATCH` 为本次入库批次，`SOURCE_DIR` 指向可混放三类音频的原始目录，`SHENPI_VAD_API_BASE` 填实际部署地址及端口。已有符合本版契约的父 Manifest 可从工序 2 开始。

```bash
# 工序 1，统一入库并标记场景
audio-data pipeline run pipelines/audio_ingest.yaml --source-name "$BATCH" --source-dir "$SOURCE_DIR"

# 工序 2a，选择数字分身，仅处理分身父源
audio-data pipeline run pipelines/vad_segment_batch.yaml --source-name "$BATCH"

# 工序 2b，选择审批，同时处理面谈和外呼父源
export SHENPI_VAD_API_BASE='http://your-vad-host:8090'
audio-data pipeline run pipelines/vad_segment_shenpi_batch.yaml --source-name "$BATCH"

# 工序 3–5，消费审批片段。分身使用 VAD_BATCH="$BATCH" 跑相同命令。
export VAD_BATCH="${BATCH}-shenpi"
audio-data pipeline run pipelines/qwen_asr_batch.yaml --source-name "$VAD_BATCH" --asr-run qwen1
audio-data pipeline run pipelines/glm_asr_batch.yaml --source-name "$VAD_BATCH" --asr-run glm1
audio-data pipeline run pipelines/dnsmos_score_batch.yaml --source-name "$VAD_BATCH"
audio-data pipeline run pipelines/evidence_join.yaml --source-name "$VAD_BATCH" --join-manifest qwen1 --join-manifest glm1
audio-data pipeline run pipelines/tag_dataset_free_tags_v2.yaml --source-name "$VAD_BATCH" --config configs/tagging/zh_asr_free_tags_v2.yaml
```

已有 ASR/DNSMOS 的部署配置继续从工序一手册读取。若环境设置了 `AUDIO_ENGINE_DNSMOS_MANIFEST`，合并时须指向相同分支同次片段产生的 DNSMOS 清单，或取消覆盖以按 `VAD_BATCH` 解析。相同批次别名下重新切分后，应基于当前片段快照重跑对应下游。现有一步到位总入口只编排已有 FSMN 分支，混合批次要处理审批父源时按本节分步执行。

## 4. HTTP 调用契约和两套配置

同事交付包为 `vad_http_service`，外部源码位置由文末提示词提供。以包内 `README.md`、`vad_service/app.py`、`vad_service/schemas.py`、`vad_service/config.py`、`vad_service/engine.py` 为接口依据。

- 首版用 `POST /vad/upload`，`multipart/form-data` 的 `file` 上传 `audio.raw` 的原始 PCM 字节，文件名用通用 `.pcm` 名称。传 `id=<父源稳定 ID>`、`audio_type=pcm`、`mode=stream`、`sample_rate=<有效源采样率>`，面谈 8000、外呼 16000。按入库场景逐条发送对应 `vad_config` 的 JSON 对象字符串，默认关闭 `return_probs` 和 `return_frames`。服务端先按原始采样率解码，再在 `engine.detect` 调用 `resample_to_16k`。FireRedVAD 模型采样率固定 16000。
- 在 `configs/vad/` 提供两个显式 profile。业务配置名中的 `vad_`（面谈）和 `wh_vad_`（外呼）仅为原配置前缀，映射成下列接口字段。服务拒绝未识别的键。

```yaml
shenpi_interview:
  source_sample_rate: 8000
  vad_config:
    smooth_window_size: 0
    speech_threshold: 0.64
    pad_start_frame: 25
    min_speech_frame: 15
    max_speech_frame: 3000
    chunk_max_frame: 30000
    min_silence_frame: 80
    merge_silence_frame: 0
    extend_speech_frame: 0
shenpi_outbound:
  source_sample_rate: 16000
  vad_config:
    smooth_window_size: 0
    speech_threshold: 0.62
    pad_start_frame: 25
    min_speech_frame: 13
    max_speech_frame: 3000
    chunk_max_frame: 30000
    min_silence_frame: 80
    merge_silence_frame: 0
    extend_speech_frame: 0
```

- 用户给出的六项参数按上述值透传。`min_silence_frame=80`、`merge_silence_frame=0`、`extend_speech_frame=0` 作为两类共用配置，暂按交付服务源码默认值显式固定，联调时确认实际审批部署口径。概率以外的计数单位为帧，模型帧移 10 ms。HTTP 的 `vad_config` 对象只传内部 VAD 参数，`source_sample_rate` 通过表单的 `sample_rate` 传递。
- 从 HTTP 成功响应校验 `status == "ok"`、`id`、`mode`、`timestamps` 为整数毫秒区间列表、`num_segments` 与长度一致、`sample_rate` 等于实际上传采样率、`model_sample_rate=16000`，并核对回显的有效 VAD 配置。`timestamps_sec`、`segments`、逐帧信息及耗时可供诊断，`GET /health` / `GET /vad/config/defaults` 可用于联调核对。
- 业务地址、连接及读超时、总并发和重试上限由调用端配置。核算分片和分片内并发的乘积，使用有界并发。网络异常、超时、限流及可恢复的服务端错误有限次重试，参数错误、音频超限和格式错误直接记录。默认传完整父音频，超服务限制时保留失败记录，便于按现场限制处理。
- 记录父源 ID、分类后的错误、有限诊断信息和耗时。音频字节、完整请求体和访问凭据不得写入日志、运行快照或 Parquet。复用现有 HTTP 实现或声明调用端直接依赖，确保按照文档安装即可运行审批链。

## 5. 裁切和 Parquet 契约

**本需求采用的边界口径**：审批直接使用服务返回的最终 `timestamps` 作为裁切基准，数字分身继续使用 FSMN v9 的流式分支。FireRedVAD 已应用审批配置下的起点回补和端点后处理。接口不提供 FSMN 的 pending、split/complete 和当时缓冲末尾等信息，因此审批无需再补 ±100 ms、套用 FSMN 短段阈值或补 EOF 尾段。两路复用的是父音频读取、裁切与落盘约定、身份/血缘和 Parquet 生成。模型产生的区间、片段数和片段 ID 可以不同。

1. 共享本地父音频读取逻辑，按 16 kHz PCM16LE、ch0 解码 `audio.resampled_16k`。HTTP 上传原始 `audio.raw`，本地裁切以规范化父音频为准。两者按同一父音频的零点和时长对齐。8 kHz 面谈在服务端与本地的重采样算法可能不同，字节比较只用于检验 HTTP 上传是否忠实于原始文件，以及子 WAV 是否忠实于本地规范化父音频对应切片。
2. 两类均将合法整数毫秒区间换算为 `start_ms * 16`、`end_ms * 16` 的本地采样点，采用 `[start_sample, end_sample)`。例如 `[[250, 750]]` 对两类都裁切本地 `[4000, 12000)`，得到 8000 个样本、0.5 秒。以本地实际 PCM 样本数约束尾部，记录是否裁到文件边界。负值、倒序、起点超界、裁后空区间或畸形条目视为协议异常；先校验整条父响应再发布子样本。按起止排序，完全相同的有效裁切区间去重；服务返回的不同重叠区间各自保留。持续时长以实际写出的样本数计算。
3. 把共享的片段 WAV 写盘及 `Sample` 构造从 `src/audio_engine/operators/audio/fsmn_vad_segment.py` 提取为可复用逻辑，沿用 `core/fsmn_vad/identity.py` 的 ID、内容哈希和标签构造。审批传入不同的处理版本（建议 `shenpi_vad_segment/v1`）和独立参数指纹。场景标记是来源元数据，数字分身同输入同配置时保持现有切段和片段 ID。缓存和恢复需感知新增元数据，避免读回缺标记的旧子样本。
4. 子 WAV 为 16 kHz 单声道 PCM16。`audio.resampled_16k` 指向子 WAV，`audio.normalized_16k`、`audio.parent_resampled_16k`、`audio.parent_raw` 沿用分身父路径语义。`id == labels.segment_id`，`sha256 == labels.original_audio_sha256 == labels.segment_pcm_sha256`。`labels.source_audio_id`、父 PCM 哈希、父规范化文件哈希、边界、父内序号、`segment_wav_sha256`、处理版本和配置指纹按现有结构填写。两路都从父样本显式继承 `labels.audio_scene` 和现有采样来源字段，建立子片段自己的 lineage/status，子片段的 ASR、标签判定和审核状态从空开始。历史无标签的分身兼容入口可根据原始父 ID 的 `fenshen` 前缀补场景。
5. 复用 `Manifest.save` / `Sample.to_flat_dict`。保证 `Sample` 字段、`audio/labels/quality` 的键与类型以及对应 `label_*` / `quality_*` 展平列兼容。新产出的分身和审批片段都包含字符串列 `label_audio_scene`，并在 `labels` JSON 中保留同一值。`clip_*` 表示实际切片采样边界，`vad_reported_*` 保留 HTTP 原始毫秒端点，`vad_end_ms` 与 `vad_inferred_end_ms` 使用返回的检测端点。`vad_start_ms` 用实际切片起点。`vad_clip_branch` 等诊断字段保留现有数据类型，可用明确的 HTTP 区间值。`buffer_end_sample` 可填用于裁切的规范化父 PCM 样本数并记录其含义。配置和原始上传统计附在父源运行记录或既有指纹的 JSON 内部。
6. 当前 Manifest writer 按数据推断 Arrow schema。使用同一批有代表性的父样本对比非空的两路 Parquet 实际列名、Arrow 类型、JSON 字段结构、加载后语义，并覆盖已有可选元数据齐全及缺失时的行为。若发现可空列因全空推断成不同类型，在两路 VAD 产出范围内解决；保留旧 Parquet 的读取兼容，避免引入全仓 Manifest 迁移。全批无语音须有可读取的零行结果，遵循既有空 Manifest 约定。

### 场景标记随片段流转

从入库父 Manifest 到片段 Manifest，以及可选话术过滤、Qwen/GLM ASR、DNSMOS、证据合并、自由打标后的各份样本级 Manifest，均以入库赋值为准。证据合并从片段底表保留 `labels.audio_scene`，分支证据和打标规则不得覆盖它。保存、重新加载及续跑后，按 `source_audio_id` 核对每个子片段的场景与父源一致。场景字段用于追踪和筛选，已有转写、评分和打标规则无需据此改变行为。本期验证到工序一打标产物，标注平台和训练数据导出属于独立接口。

## 6. 失败、身份与恢复

- `status=ok` 且 `timestamps=[]` 为成功处理但无语音，产出零行，写父源 `no_speech` 记录。网络、HTTP、解码、坏响应和写盘失败各自可定位。出现未解决的父源失败时，流水线需明确报出失败并阻止本次结果作为完整快照交给下游。保留已完成的分片/文件，续跑可以重试失败部分。原子写盘和重试不能导致重复行或坏文件被复用。
- 复用 `runs/`、进度查询和 `--resume`。运行参数快照包含规则及规范化版本、原始/目标采样率语义、两套有效配置、请求模式和部署标识。父源记录包含各自实际使用的场景、上传采样率和 profile。部署身份用可配置的版本/发布标识，服务 API 版本只记录为 API 版本。读取外部配置时把内容纳入指纹，配置、端点、部署身份或输入快照变化时，应使相应 checkpoint 失效或拒绝按旧运行续跑。片段身份结合父源及规范化内容、实际采样边界、处理版本和该父源有效 profile/部署指纹。
- 每批和每个场景至少能核对输入父源数、成功数、无语音数、失败数和片段数。失败父源及有效参数应能从该次 `runs/` 找到。正样本行数与可播放子 WAV 一一对应。

## 7. 验收和交付

- **混合入库**。用用户的三个审批示例和至少一个 `fenshen` 文件各写一秒的合成 PCM，同目录执行现有入库入口。四条的规范化产物均为 16000 帧、1 秒、16 kHz mono，原始源率和三类 `label_audio_scene` 正确。断言面谈波形与浮点参考重采样在合理量化误差内，核对外呼和分身逐样本等于原 PCM。覆盖 `fenshen` 开头但没有紧跟连字符、审批后缀内嵌 `V/P`、仅前缀/中段出现 `V/P`、小写后缀、畸形 ID、未知前缀、奇数字节、空文件、RIFF 冒充裸 PCM 和场景统计。
- **缓存与迁移**。给两个不同场景 ID 写相同裸 PCM 字节，预先用旧固定 16 kHz 跑过同批源文件，再跑新入库。两条正确处理为不同原始时长，分别引用自己的规范化音频；旧产物可读且内容不被覆盖。覆盖修改场景规则/profile 版本导致的缓存/续跑失效，以及从历史错误入库产物发起审批 VAD 时的明确报错。
- **入口**。用同一份混合入库产物分别运行两条 VAD，FSMN 只见分身，HTTP 只收到两种审批样本，未知样本两路均跳过。确认默认输出路径独立，审批调用端无需 FSMN 环境。核对单场景批次的空分支、显式路径、无场景字段的历史分身兼容及未标记审批拦截。
- **HTTP 与裁切**。替身 HTTP 同批接收面谈 8 kHz 原始字节和外呼 16 kHz 原始字节，核对表单及逐条 profile，分别回显源率 8000/16000、模型率 16000。对两类都返回 `[[250, 750]]`，逐样本核对本地子 WAV 等于各自 16 kHz 父音频 `[4000:12000]`。再覆盖多段、无语音、坏区间、尾边界、错误 ID、错误采样率或配置回显、原始文件变更、非成功响应、超时和重试后成功。至少验证一次有父源失败时无完整产物，修复后续跑成功且无重复。
- **兼容**。用当前分身测试及相同实际裁切边界的固定样例核对共享写盘，并核对两路生成的 Parquet 列/类型、反序列化、父子血缘、可选话术过滤匹配和 WAV 内容。用替身跑通两路片段的 ASR、DNSMOS、证据合并、自由打标，逐阶段核对三类场景值，覆盖分片合并、缓存命中和续跑后的字段保留，以及片段 ID 和内容哈希对齐。
- **服务器验收**。服务可访问后，选少量真实 8 kHz 面谈和 16 kHz 外呼原录音走混合批次，分别核对原始时长、服务回显配置、返回区间、实际切片和听感，再用两类片段跑通同一真实下游链。记录版本、运行目录、各场景父源数、片段数、失败数、服务用时和下游结果。未执行的真实服务验证标为待验。
- **交付**。修改现有入库和分身 VAD YAML，新增审批 VAD YAML、两套 profile、客户端和算子注册，完善 source naming、统一场景与采样率解析、标记传递、缓存与共享片段产出代码，提交以上回归用例。更新 `手册/dev/01-工序一-数据清洗落库打标.txt`、`单条流水线执行命令.txt` 和 045 操作手册，记录统一入库、选路、场景字段、下游别名、老产物重建与续跑命令。实施后记录验证结果、部署依赖和待验事项至 `docs/PROJECT_CONTEXT.md` 或其链接的专项文档。

## 给 Cursor 的提示词

```text
在 D:\Work\audio-data-engine 按需求文档实现工序一的统一场景识别、混合采样率入库和审批 HTTP VAD 调用方。
需求文档：docs/04-改进需求/进行中/050-工序一审批场景HTTP-VAD接入.md

先读 AGENTS.md 和 docs/PROJECT_CONTEXT.md，检查当前分支、工作区和实际代码，保留已有改动。再读需求文档、手册/dev/01-工序一-数据清洗落库打标.txt、045 操作手册和 docs/07-操作手册/流水线构建-AI执行手册.md。

改造现有 pipelines/audio_ingest.yaml，在解码和缓存查询前按原始文件名判定场景。用户确认分身均以 fenshen 开头。已知审批前缀为 shenpi_ai- 和 shepi_ai-，其中第二个连字符后的整个后缀含大写 V/P 为面谈 8000 Hz，否则为外呼 16000 Hz。三类标记分别用 labels.audio_scene=fenshen/shenpi_interview/shenpi_outbound，展开为 label_audio_scene。未知前缀按正文处理，保留父源 ID。修复 int16 重采样写 PCM16 尺度错误，保留逐条原始采样率和正确的 16 kHz mono 父音频。重点验相同 SHA 不同 ID 的缓存、规范化派生输入及旧路径冲突。历史审批从原始 PCM 重新入库。

同事交付的 HTTP 服务源码在 D:\Work\TMP\daily\1116\vad_http_service\vad_http_service。核对 README.md、vad_service/{app,schemas,config,engine,audio_io}.py，只实现本仓调用方。新增 pipelines/vad_segment_shenpi_batch.yaml，从审批入库 Manifest 逐条取原始 PCM 和源采样率，用 POST /vad/upload 发送。面谈用阈值 0.64 / 最短 15 帧，外呼用 0.62 / 最短 13 帧，其余配置按正文完整传递。读取整数毫秒 timestamps，两类均以本地 16 kHz 父音频计算裁切采样点。复用分身 WAV 写盘、Sample 构造、血缘和 Parquet writer。分身原入库命令与 FSMN v9 行为保持兼容。

两条 VAD 从同一输入 Manifest 按场景选行，分身只处理 fenshen，审批只处理面谈/外呼，并按场景读取 HTTP profile。审批 VAD 默认输出 segments_<batch>-shenpi。两个算子都要显式从父源继承 labels.audio_scene。用三类混在同目录的样例与替身 HTTP 做 CLI 端到端测试，核对两路分片和单批行为、三种标记一直保留到 ASR/DNSMOS、证据合并与自由打标，比较实际 Parquet 字段/类型及子音频字节，覆盖未知前缀、历史兼容和失败续跑。

完成实现、回归测试和手册更新后，汇报改动文件、从原始 PCM 开始的命令、实际执行的测试与结果、真实服务待验项。没有部署地址也可以先完成替身测试。部署地址按环境配置，禁止提交音频、服务代码、权重、凭据或临时产物。
```
