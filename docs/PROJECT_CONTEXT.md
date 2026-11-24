# Audio Data Engine 项目上下文

核对日期：2026-11-24。基于分支 `v4.1`、现有文档和部分相关历史对话整理；不是完整聊天导出。

## 评测入口主题化与 shenpi gold_dev_v001（2026-11-24）

评测脚本改名并引入同级主题参数 `fenshen|shenpi`：

- `scripts/run_asr_eval.sh <theme> <mode> <v0xx> ...`
- `scripts/eval_asr_xlsx.py <theme> <mode> <v0xx> ...`

金标/评测注册名改为 `gold_${theme}_{split}_${ver}.parquet`、`eval_${theme}_{split}_${ver}`；字准归档为 `datasets/stage3/asr/eval_${theme}_${ver}/`。旧名 `run_fenshen_1015_eval.sh` / `eval_fenshen_1015_xlsx.py` 已删除，不留兼容包装。历史磁盘产物仍可能为 `gold_fenshen_1015_*`。服务器批量改名：`python scripts/rename_fenshen_1015_theme_artifacts.py`（预览），确认后加 `--apply`；可选 `--include-runs`。改名前 `run_asr_eval.sh fenshen ...` 会找不到旧金标。

审批 exploratory 金标：`scripts/build_shenpi_gold_v001_dev.py` 从 `audio-qwen3-asr-train` 的 `exploratory/shenpi/train_v001` sample_map 写出 `datasets/stage1/derived/gold_shenpi_dev_v001.parquet`（2090：auto_gold 2033 / non_speech 57），正式评测门禁通过。明细 xlsx：`tmp/1124/gold_shenpi_dev_v001.xlsx`。评测示例：`bash scripts/run_asr_eval.sh shenpi full v001`。未跑真实转写。

## 工序一入库批次进度（2026-11-17）

入库分片按当前步骤上报已处理父音频数，父 run 汇总到 `progress.json` 与 `PROGRESS.log`。原进度从各步骤 checkpoint 取最大值，转码完成即可提前显示 100%，且清单行数不能代表处理过的父音频数。新汇总给出各步骤的批次分母、成功分片处理的父音频数、质检保留和剔除数；扫描和场景剔除仍可查 `prepare/`。CLI 单独入库、一步到位入口及 Web 节点日志的使用方法见 [045 父 run 查询](07-操作手册/045-入库VAD多路ASR独立DNSMOS自由标签.md#父-run-查询--续跑--失败重试)。

本机用合成 8 kHz/16 kHz PCM 验证两分片入库、场景剔除、最终清单与续跑，扫描 3 条、场景剔除 1 条、最终保留 2 条。进度、分片、045、050 和 Web API 相关测试共 39 项通过。Web 集成测试使用符合 050 场景规则的分身样例，并核对入库节点日志中的全批次总数。真实服务器批次待验证。

## fenshen_1015 gold_dev_v031（2026-11-13）

从 `datasets/stage1/derived/gold_fenshen_1015_dev_v030.parquet`（601）去掉 `transcripts.qwen_1.text == 【听不清】` 的 7 条，写出 `gold_fenshen_1015_dev_v031.parquet`（594），兼写 jsonl / meta。过滤脚本：`tmp/1113/filter_v030_to_v031.py`。正式评测门禁通过；不做 type split。其中 2 条被删样本金标非空（`你想干啥？` / `哪里？`），其余 5 条为空金标 non_speech。注意：仓库里另有 `scripts/build_fenshen_1015_gold_v031_from_export.py`（1102 导出另建 v031），与本次「v030 去听不清」产物同名路径，勿混跑覆盖。

## fenshen_1015 空金标计分（2026-11-11）

`scripts/eval_asr_xlsx.py`：噪声 / 空转写（计分参考为空）不再参与总字准率。正确率上，识别经 `plain_transcript_text` 后只要还有文字即字准=0，**不去语气词**（避免「什么」等被词表抹成空后误算正确）。空金标只累计编辑距离（插入字数），在 `non_speech` scope 的「总字准」列仍改写为该编辑距离整数。金标非空样本仍走 plain → 删语气词 → 重复折叠 → 字准。相关单测：`tests/test_eval_asr_xlsx.py`。

## tmp/1110 证据聚合（2026-11-10）

对 `20260813-n-1` / `20260814-n-1` 在 `tmp/1110` 跑 `pipelines/evidence_join.yaml`：底表用同目录 `qwen1_asr_*`，`--join-manifest` 接入 qwen1/glm1/sv1，`AUDIO_ENGINE_DNSMOS_MANIFEST` 指向同目录 `dnsmos_*`，输出 `tmp/1110/prepared_evidence_<batch>.parquet`（兼写 jsonl）。行数 53907 / 62641；alignment 全 aligned，三路与 DNSMOS 无 missing/extra/hash/snapshot 缺口；`label_evidence_asr_routes=["glm1","qwen1","sv1"]`，DNSMOS 全 success。三路文本与 ovrl/status 相对源 parquet 零差异（qwen 源侧本就有少量空文本：13 批 1 条、14 批 3 条）。未写入 `datasets/stage1/derived/`。

复听候选合并包 `tmp/1110/qwen_unclear_or_nonspeech_candidates_20260813_20260814.parquet`（12103）已将业务 `batch` 统一为 `qwen_unclear_or_nonspeech_20260813_20260814`，原批次保留在 `upstream_batch`，便于 audio-label-master 单文件单池导入。

## fenshen_1015 gold_dev_v028（2026-11-10）

`scripts/build_fenshen_1015_gold_v028_dev.py` 从 `audio-qwen3-asr-train` 的 `train_v028` sample_map 写出 `datasets/stage1/derived/gold_fenshen_1015_dev_v028.parquet`：2815 条，`label_eval_type_split=true`（auto_gold 1732 / hardcase 583 / non_speech 500）。non_speech 含 v027 原空金标 22 与 SIG<=2.0 新增候选 478；计分参考均为空串。明细 xlsx：`tmp/1110/gold_fenshen_1015_dev_v028.xlsx`。未跑评测转写。

## fenshen_1015 gold_dev_v027（2026-11-09）

`scripts/build_fenshen_1015_gold_v027_dev.py` 从 `audio-qwen3-asr-train` 的 `train_v027` sample_map 写出 `datasets/stage1/derived/gold_fenshen_1015_dev_v027.parquet`：2337 条，`label_eval_type_split=true`（auto_gold 1732 / hardcase 583 / non_speech 22）。对应训练集为 v024 全量 + 1104-qwen短语筛选（只留 speech/accept 与 background_suspected）。明细 xlsx：`tmp/1109/gold_fenshen_1015_dev_v027.xlsx`。未跑评测转写。

## 工序一 Web 接入（2026-11-09）

Web 主链复用 045 编排和现有算子，支持可选 XLSX 剔除，按路次绑定服务器 Qwen/GLM 连接，并在成功后登记独立任务目录中的阶段清单与汇总表。模型页面只发起服务器 TCP 端口探测。FSMN 与 DNSMOS 使用服务器上的运行目录和 ONNX 文件。入口、配置要求及续跑方式见 [045 服务器 Web 操作台](07-操作手册/045-入库VAD多路ASR独立DNSMOS自由标签.md#服务器-web-操作台)。

替身验收见 `tests/web/test_api.py`，覆盖原始音频入库、VAD、话术剔除、并行分支、DNSMOS 失败后复用 ASR 续跑、分步证据合并和自由打标、节点日志及产物查询。Web、插件、045、话术过滤和转写清洗相关测试共 65 项通过，前端生产构建通过。真实 vLLM、FSMN、DNSMOS 和正式批次仍待服务器联调。

## TTS 文件夹金标评测（2026-11-06）

入口 `scripts/eval_tts_folder_qwen_base.py`：扫描 `/data2/data-cp/xijujun/Project/TTS/后处理音频`（可改 `--audio-root`），一级子文件夹名即真实转写；默认请求本机 `5555` qwen-base vLLM。默认沿用 `configs/asr/qwen_asr.yaml` 的 vocabulary context，可用 `--no-context` 关闭。正确判定：`plain_transcript_text` 后再删 `configs/normalization/fenshen_1015_filler_tokens.yaml`；删完编辑距离为 0 不算 mismatch。`qwen_base_mismatch.parquet` 可直接「转换 parquet」导入 audio-label-master：`audio.resampled_16k` 播放；预标注来源选 `tts_gold`（子文件夹名）；原始机转参考选 `qwen_base`（另有顶列 `candidate_text`）；`is_vad_segment=false`，数据单位用 legacy/整段。产物默认 `tmp/tts_folder_qwen_base/`。

## Qwen 落盘保留标点（2026-11-06）

2026-11-17，服务器 Qwen ASR 时间戳日志出现两类请求期异常。中间件收到 HTTP 200 的 JSON `null` 后访问 `body["choices"]`，抛出 `TypeError`，尚未进入强制对齐推理。vLLM 客户端断开后的取消路径可能产生该响应，现场需用客户端超时日志确认。另有单条请求在 EngineCore 的多模态接收缓存找不到 `mm_hash` 而返回 500，可用 `--mm-processor-cache-gb 0` 做服务器复测。证据为现场 11-17 10:42–10:52 的 `err.txt`，其中首次 `TypeError` 在第 30 行，缓存断言及 HTTP 500 在第 104–106 行。当前仅完成日志与源码核对。

同日修复 `scripts/serve_qwen_base_with_timestamps.py` 的时间戳响应处理。中间件先验证上游 JSON 的 `choices[].message.content`，对整个 `null`、不可解析 JSON 或无效结构记录 vLLM 响应异常并返回 502。有效的空转写仍返回空时间戳列表，原生非 200 响应透传。用替身 ASGI 复现旧版异常并验证修复，连同 Qwen 和 vLLM 相关测试共 33 项通过。上线 `5555` 和按请求核对 `null` 的实际触发原因待服务器执行。

2026-11-16，新增 `scripts/serve_qwen_base_with_timestamps.py`。在 qwen-base 服务器环境执行该入口，会以 GPU 4、`127.0.0.1:5555` 和原模型、模板、显存与并行参数运行 `vllm serve`，通过 `--middleware` 在 ASGI 启动阶段加载本地 ForcedAligner。单音频 base64 非流式 Chat 请求在 `choices[].message.timestamps` 返回相对输入音频的毫秒时间戳，`return_time_stamps=false` 可关闭。现有客户端可继续读取 `message.content`，持久化服务端时间戳需读取新增字段。服务器须使用支持 `--middleware` 的 vLLM 环境，并安装 `qwen_asr`。本机替身验证了启动参数、预加载和启动失败、时间戳响应、流式透传和原生 HTTP 错误，现有 Qwen 测试 10 项通过。真实 5555 和 GPU 联调待服务器执行。

2026-11-16，新增 `scripts/run_qwen_asr_batch_with_timestamps.py`。该独立入口复用 `scripts/run_qwen_asr_batch.py` 的 vLLM 请求和转写格式，默认调用 qwen-base 5555，单个进程加载本地 Qwen3-ForcedAligner 权重。单条用 `--audio`，直接输出转写和时间戳 JSON；批量用 `input.parquet -o output.parquet`，默认写 `transcripts.qwen-base-timestamps.extra.timestamps`（`text`、`start_ms`、`end_ms`，以输入音频开头为零点）。空转写、纯标点和 `【听不清】` 跳过对齐；失败记录到样本状态，下次可续跑。原批脚本及评测流水线保持各自入口。替身 HTTP 和对齐器验证了单条请求、Parquet 落盘、跳过、限量、失败重试与续跑；真实 5555 和 GPU 待服务器验证。

当 5555 运行官方 `qwen-asr-demo` 时，客户端加 `--backend gradio --api-base http://127.0.0.1:5555`，并在调用端安装 `gradio-client`。该模式通过 Demo 的 `/run` 上传音频并请求服务器生成时间戳，读取秒值后转换成原有毫秒字段，无需本地对齐权重。Demo 接口接受语言选择，现有 vLLM 热词模板不参与这条调用。已用本机 Gradio 6 和替身推理验证音频上传、单条 JSON、并发批量落盘和续跑；远端实际推理待验证。

`pipelines/qwen_asr_batch.yaml` 写盘 profile 为 `configs/normalization/zh_asr_keep_punct_v1.yaml`：`transcripts.*.text` 保留标点（仍去空白/控制标签）；`blank_exact_qwen_v1` 仍在无标点 plain 上匹配/删跨度，命中后残余仍为 plain。`extra.raw_text` 仍是模型原串。GLM/SenseVoice 仍走 `zh_asr_v1` 去标点。下游分拣/字准比对继续用 `plain_transcript_text`。旧批次无标点 parquet 不自动回填，需重跑 Qwen ASR。

## 工序一审批 HTTP VAD（2026-11-16，已实现调用方）

[050 统一场景识别、混合采样率入库与审批 HTTP VAD](04-改进需求/进行中/050-工序一审批场景HTTP-VAD接入.md)已在 `v4.1` 落地调用方（不含服务部署）。`pipelines/audio_ingest.yaml` 在 `pcm_to_wav` 前跑 `ingest.resolve_audio_scene`：`fenshen*` → `fenshen`；`shenpi_ai-`/`shepi_ai-` 按第二个 `-` 后后缀是否含大写 `V/P` → `shenpi_interview`(8 kHz) / `shenpi_outbound`(16 kHz)；未知 → `unknown`。裸 int16 重采样先转浮点幅度再写 PCM16；转换缓存与派生路径纳入 `sample.id`/场景/有效源率并隔离 `pcm_to_wav_v2`。分身 `vad_segment_batch` 只选分身（历史缺标记分身仍兼容；未标记审批前缀拦截）；审批 `pipelines/vad_segment_shenpi_batch.yaml` + `audio.shenpi_vad_segment` 读 `configs/vad/shenpi_vad_profiles.yaml`，`POST /vad/upload` 上传原始 PCM，毫秒区间按本地 16 kHz 父音频裁切，默认输出 `segments_<batch>-shenpi`。共享发布见 `core/fsmn_vad/publish.py`，两路显式继承 `labels.audio_scene`。本机替身：`tests/test_audio_scene.py`、`tests/test_shenpi_050_ingest_vad.py`（含 stub HTTP）与 `tests/test_pcm_to_wav.py` 通过。真实 `SHENPI_VAD_API_BASE`、真实 8k/16k 听感与正式下游链仍待服务器联调。历史错误审批须换批次从 `audio.raw` 重入库。

## VAD 裁切（2026-11-05）

FSMN 切段版本 `fsmn_vad_segment/v9`：应用层裁切与 `ph_ai_model_asr_rt_cpu` 的 `vad_asr_core_new` 同步——split-end 为 `pending-100`～`end+100`，完整端点/EOF 起点 `−100` 且完整端点仍裁到当时缓冲末尾。旧 v8 片段须从父音频重跑。

## 话术标签剔除（2026-11-05）

分步工序在 VAD 与 ASR/DNSMOS 之间可选执行 `scripts/filter_segments_drop_identity_verify.py`：用业务 xlsx 的 `file_id` + `AI话术分类标签`，去掉「接洽（身份核实）」父源的全部切段。匹配键是父源 `source_audio_id`，不是切段 `seg_...` id。默认备份全量 `segments_<batch>_vad_all.parquet` 后写回 `segments_<batch>.parquet`。一步到位总入口不含此步。

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

## ASR 评测知识（原 fenshen_1015 评测）

入口 `scripts/run_asr_eval.sh`，主题与版本同级（`fenshen` / `shenpi`），调用转写和 XLSX 评测链，不是重新给原始音频做生产分拣。示例（先准备金标、模型服务与依赖）：

```bash
bash scripts/run_asr_eval.sh fenshen full v024
bash scripts/run_asr_eval.sh fenshen lora-full v030
bash scripts/run_asr_eval.sh fenshen omni v031
bash scripts/run_asr_eval.sh fenshen sophon v032
```

Shell 入口默认 `dev best`，仍可显式指定切分与权重。端口为 base 5555、Full best/over 5560/5561、LoRA best/over 5565/5566、RL 5580、Omni 6191、Sophon 6179。RL、Omni 和 Sophon 只提供单端口，`all` 会补跑基线。输入金标位于 `datasets/stage1/derived/`，按 train/dev 与版本命名。

2026-11-17，Omni 评测改为独立 `pipelines/qwen_omni_asr_batch.yaml` + `chat_format=omni`：端口 6191，模型 ID 仍从 `/v1/models` 读取并写入 `QWEN_ASR_MODEL`。请求对齐 `audio_model_v3.py`——单条 user，`audio_url` 后接固定提示词「这段音频是电话贷款营销业务场景中客户的回答，大部分客户会表示不用了、不需要等否定意图，输出ASR转译文本。」，`temperature=0.1`，`max_tokens=20480`，`timeout=590`，不发 system vocabulary / `chat_template_kwargs`。响应按纯文本落盘后再走既有保标点清洗与字准 xlsx。本机 `tests/test_vllm_asr.py` omni 请求体单测已加；真实 `bash scripts/run_asr_eval.sh fenshen omni v031` 待服务器执行。

2026-11-13，Omni Thinking 在单卡 A800 上读完 16 个分片并启动。当时 `omni v030` 复用已落盘的 5555 基线，实际向 6190 发起最多 32 路并发请求，且仍共用 Qwen3-ASR chat 请求（无显式转写提示 / `max_new_tokens`）。该形态已被 2026-11-17 的 omni 专用 pipeline 取代。证据位置为服务器 `runs/fenshen_1015_eval_omni_v030_dev_best_20261113_103305/best.log` 和同次 6190 服务日志。

`lora-full` 并行转写 LoRA 和 Full，两路共用同一份金标和 qwen-base 基线。默认比较两路 best，写出 `datasets/stage3/asr/eval_v030/<月日_小时分钟>/v030_dev_lora-full.xlsx`。选 `both` 时每个切分各出一份联合报表。本机替身端口已验证默认和 `both all` 的路由、基线去重、联合出表及一路失败时不出表，真实模型联合评测待服务器执行。

Sophon 使用内网 Sophon-Realtime-32B-Tag。评测从 `/v1/models` 取实际 ID，提交语音和逐字转写提示并只请求文本，沿用同一套转写落盘、清洗和字准评测。已有 qwen-base 转写时直接复用，默认模式下缺失才调用 5555。替身服务已验证模型 ID、基线复用与补跑、Parquet 和 XLSX 落盘，真实金标评测待服务器执行。

2026-11-10，v030 在 6179 收到 HTTP 500，日志定位到 vLLM-Omni 渲染音频占位符时的 `Failed to apply prompt replacement for mm_items['audio'][0]`。用 Tag 权重的本地模板验证了结构化请求中普通音频部件不会生成占位符，Sophon 请求显式携带的 `<|audio|>` 会保留。先核对服务器评测环境中 `audio-data` 实际加载的 `operators/asr/qwen.py` 和 `operators/asr/vllm.py`，再用官方客户端验证真实语音请求。现场请求体和服务端模板仍待核对。

### Sophon HTTP 纯转写配置（2026-11-10）

当前评测调用 HTTP `/v1/chat/completions`，只请求文本。WebSocket 的 `session.turn_detection` 和 Sophon 的 `SOPHON_TURN_*` 作用于实时会话。生产 FSMN 保持 `fsmn_vad_segment/v9` 与 fenshen 参数；已切片如需增加声学证据，独立对比 `configs/quality/silero_vad_v1.yaml` 的 `0.50 / 120 ms` 和试验组 `0.60 / 160 ms`（试验组同时调整检测与判定的最短语音时长）。低 SIG、VAD `uncertain` 与缺失证据保留给复核；短词、弱语音纳入回归。

评测入口默认 `SOPHON_PROFILE=v1-hotwords` → `pipelines/sophon_asr_v1-hotwords.yaml`：`language: zh`，`context` 对齐 fenshen 生产 `vllm_hotwords.default`（`Vocabulary：不方便，平安融易，测额，额度，还贷，征信，没有，暂时不要，不可以，不用，不要谢谢，不需要。`）；评测无场景路由，不用「接洽-身份核实」词表。无热词对照用 `SOPHON_PROFILE=v1`（`context: ""`）。配置独立于 Qwen YAML，保留 `temperature=0`、`max_tokens=256`、`modalities=["text"]`，用 system 指令要求只转写可辨认词句和真实语气词，纯噪声与纯非语言声返回空串。ASR 按 [官方 Tag 标签语法](https://github.com/adventists-ai/sophon-realtime/blob/3585943/tag/plugin/sophon_realtime/proc.py#L21-L24) 去除标签、纯笑声片段及常见非语言标注，词句继续经过现有保标点规范化。`transcripts.<alias>.extra.raw_text` 保存原串，HTTP 调用失败缺少转写字段时终止新版 Sophon 报表，避免把失败当成无语音。

继续调热词或提示词时复制现有 profile 为新版本，只改试验组的 `params.context` / `system_prompt`。非空词表附在 system 提示，`SOPHON_PROFILE=legacy` 使用原 Qwen 配置和原 Sophon 提示。配置由评测客户端加载，更新评测代码后启动新任务即可。

```bash
bash scripts/run_asr_eval.sh fenshen sophon v032
SOPHON_PROFILE=v1 bash scripts/run_asr_eval.sh fenshen sophon v032
SOPHON_PROFILE=legacy bash scripts/run_asr_eval.sh fenshen sophon v032
```

每次运行在 `runs/fenshen_1015_eval_sophon_*` 归档原始 YAML、实际使用的 `sophon.pipeline.yaml`、引用的 YAML 和 `sophon.run.txt`（profile、run_tag、模型 ID、API 地址）。Sophon 转写别名为 `qwen-sophon-<split>-best-<run_tag>`，旧转写仍可读取。根据 `sophon.run.txt` 重出对应的报表：`python scripts/eval_asr_xlsx.py fenshen sophon v032 --split dev --sophon-run-tag <run_tag>`。

本机相关测试 56 项通过，Bash 语法检查通过。替身 HTTP 通过完整评测入口验证 `v1`、自定义词表和 `legacy` 均有各自的配置快照、Parquet 与 XLSX，并复用同一份 qwen-base。真实音频误转写率与服务端兼容性仍待验收。

本地 Parquet 核对 `gold_fenshen_1015_dev_v028` 有 500 条空参考（其中 478 条为 SIG<=2.0 的候选），可用于噪声压力测试；`v032` 有 3 条空参考。上线前在服务器用同一份语音比较成对结果，分别统计确认无语音样本的非空输出率、短词和弱语音的漏转率、发布文本标签残留率。候选需复听区分纯噪声与低质量人声；本机无法读取 v028 所引用的音频路径。

分类实现见 `scripts/eval_asr_xlsx.py`：显式 `label_eval_type` 可指定分类；否则 non_speech/empty 优先，再按候选与金标 plain 文本相等分 auto_gold 或 hardcase。所有行关闭 `label_eval_type_split` 时只出 overall。评测桶不代表当前待评模型识别是否正确，也不等同声学五类。字准 XLSX 每次运行归档到 `datasets/stage3/asr/eval_<主题>_<版本>/<月日_小时分钟>/`，文件名为 `<主题>_<版本>_<切分>_<模式>.xlsx`，例如 `datasets/stage3/asr/eval_fenshen_v030/1110_1430/fenshen_v030_dev_full.xlsx`。同次命令的 train 和 dev 共用归档目录，同一分钟内重复运行时给目录加序号。

`gold_fenshen_1015_dev_v024` 已于 2026-11-06 打开分类：`label_eval_type_split=true`，`label_eval_type` 取自 `train_v024` sample_map 的 `eval_type`（auto_gold 1732 / hardcase 433 / non_speech 2）。旧无分类 parquet 备份为 `gold_fenshen_1015_dev_v024.parquet.bak_no_type_split`。已有转写时只需重跑 `python scripts/eval_asr_xlsx.py fenshen <mode> v024 --split dev`。

`gold_fenshen_1015_dev_v027` 已于 2026-11-09 生成：同样打开分类，dev 2337（auto_gold 1732 / hardcase 583 / non_speech 22）。评测示例：`bash scripts/run_asr_eval.sh fenshen lora v027 dev best`。

`gold_fenshen_1015_dev_v028` 已于 2026-11-10 生成：同样打开分类，dev 2815（auto_gold 1732 / hardcase 583 / non_speech 500）。评测示例：`bash scripts/run_asr_eval.sh fenshen lora v028 dev best`。

历史排查发现：极少数超长复读输出可让句级正确率很高而汇总字准很低。比较结果必须确认删语气词、重复折叠、空金标等计分口径；保留原始输出，另存诊断修正版，不能把手工修正后的指标当原始模型成绩。空金标自 2026-11-11 起不进总字准、正确率不受语气词表影响，见上文「空金标计分」。

## 验证与后续接手

2026-11-16，新增单条强制对齐验证入口 `scripts/test_qwen3_forced_aligner.py`。它用本地 Qwen3-ForcedAligner 权重，把现有转写对齐到对应音频，输出相对音频起点的秒级时间戳。默认权重路径指向扁平化后的模型目录，也可传 `--model-dir` 覆盖。本机替身验证了缺配置报错、加载参数和 TSV 输出。真实权重、音频和 GPU 仍待服务器验证。

安装入口见根 README：`pip install -e .`；开发依赖见 `pyproject.toml`。代码修改后按模块选择 `python -m pytest tests/<相关路径>`，不要为阅读文档启动生产流水线。

本次只建立知识同步入口，未运行数据加工、模型推理或服务器验收。下一项业务任务由用户指定；执行前核对所需批次、数据版本和相关运行手册，不能从旧聊天推断当前服务器任务仍在运行。
