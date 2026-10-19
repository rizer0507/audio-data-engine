# 049：FSMN VAD 与分身实时服务完全对齐

日期：2026-10-16。

## 1. 目标与交付约束

修复 `audio-data-engine` 的 FSMN VAD，使同一份规范音频在相同依赖和 fenshen 模型配置下，生成与 `ph_ai_model_asr_rt_cpu` 分身场景一致的模型事件、片段数量、顺序、实际裁切起止采样点和 PCM 内容。

“完全一致”的比较对象是原服务实际送入 `_call_vllm_transcribe(segment, ...)` 的 `segment` 字节，不是 ASR 文本，也不是仅比较接口返回的 start/end。不能只删除 675 ms 补尾后就宣称完成：分包、结束调度、短段过滤、完整端点分支和 EOF 补尾也必须对齐。

本次由 Cursor 修改代码和必要配置、操作说明。**不新增、不执行测试，不运行 pytest、mock、smoke、推理脚本、对比脚本或其他验证命令，不安装或升级依赖。** 允许静态阅读源码和检查修改差异。用户自行在服务器运行现有对比脚本完成验收；交付时必须明确说明未运行测试。

本需求取代旧 041/045 文档中与本需求冲突的“保留 675 ms、按 1 秒过滤、丢弃 EOF 补尾/EOF 端点”等 VAD 语义要求。原有父子血缘、Manifest、原子提交及并发隔离契约继续保留。

## 2. 源码与范围

当前项目：`D:/Work/audio-data-engine`。

对照项目：`D:/Work/ph_ai_model_asr_rt_cpu`，只读，不修改。

实现前必须阅读以下原服务文件，以实际执行代码为基线，不依据注释猜测行为：

- `D:/Work/ph_ai_model_asr_rt_cpu/app/core/vad_asr_core_new.py`：`process_audio_chunk`、`process_vad_and_asr`、`process_str_message`、`_pcm_to_float32`。
- `D:/Work/ph_ai_model_asr_rt_cpu/app/core/asr_model.py`：模型初始化。
- `D:/Work/ph_ai_model_asr_rt_cpu/app/main.py`：确认分身走 FSMN 分支。
- `D:/Work/ph_ai_model_asr_rt_cpu/model_configs/fenshen_vad_config.yaml`。
- `D:/Work/ph_ai_model_asr_rt_cpu/app/config/stg_fenshen.yaml` 和 `prd_fenshen.yaml`：`firered_vad_switch=false`、`send_end_result=true`。
- `D:/Work/ph_ai_model_asr_rt_cpu/start.sh`：模型配置覆盖路径。

本次一致性契约限定为相同的 **16 kHz、单声道、PCM16LE** 输入。WAV 去掉文件头后比较 PCM；无头 PCM 必须按该格式解释。保留当前项目对其他输入的正常规范化能力，不复刻原服务 8 kHz 输入未正确送入重采样结果的缺陷，不增加降噪、响度归一化等处理。

文件输入采用确定性客户端回放：默认每包 100 ms，与现有对比脚本一致。若支持其他包长，必须显式配置并记录；同一参数应贯通生产核心和对比脚本。网络包长未知的历史请求不能仅凭同一文件保证严格复现。

## 3. 已有服务器证据

完整结果：`F:/1016/3/wav/vad_compare.json`。

实际音频：`F:/1016/3/wav/vad_compare_files_02f8_kmw/engine/0000.wav` 与同目录 `service/0000.wav`。

输入为 `str_con_AkfQNU3KCR_174540_176610.wav`，总时长 4070 ms。两边使用同一个 Python 环境，FunASR 1.3.1、Torch/Torchaudio 2.10.0。两边权重、CMVN、模型源码、完整 fenshen 配置相同，21 次模型调用的输入和原始端点相同。

- 原始模型事件均为 `[940, -1]`、`[-1, 2680]`。
- 原服务实际片段：`[940, 2680)` ms，1740 ms，采样范围 `[15040, 42880)`。
- 当前项目实际片段：`[940, 3355)` ms，2415 ms，采样范围 `[15040, 53680)`。
- 原服务音频是当前项目音频的完全相同前缀，当前项目额外包含 10800 个采样点，即 675 ms。
- 本例没有 EOF 截断，尾部差异确定来自当前项目补尾。

以上是用户已完成的服务器实测证据，不是要求 Cursor 重新执行的测试。

## 4. 必须使用完整 fenshen 配置

当前项目的配置入口为 `D:/Work/audio-data-engine/configs/vad/fenshen_vad_config.yaml`，应与对照项目的完整 YAML 一致。不能只覆盖两个阈值，其他模型、前端、encoder 参数也必须完整保留。

已确认的配置 SHA256：`9057c765320b48d5a1247528dd8c5ea9fa818d4d6e3d073db0036ec4b81b79e6`。

关键参数包括：`max_end_silence_time=1100`、`max_start_silence_time=4000`、`sil_to_speech_time_thres=200`、`speech_to_sil_time_thres=50`、`speech_noise_thres=0.85`、`max_single_segment_time=10000`、`lookback_time_start_point=200`、`lookahead_time_end_point=100`。

生产入口必须保证真正加载 fenshen 配置，不能仅记录声明配置路径：

1. 准备独立运行目录，复制模型资产，将完整 fenshen YAML 作为运行目录 `config.yaml`；不修改共享原始权重目录。
2. 已准备的运行目录必须校验有效 YAML；若仍为原始默认配置 `800 / 0.6 / 60000`，明确失败并给出重新准备目录的方法，不能静默运行。
3. 校验模型实际加载的 `vad_opts` 与 YAML 的 `model_conf` 一致，并记录配置来源、有效配置哈希和加载后的参数。
4. 对 `generate_options` 等运行时覆盖进行冲突检查，不能悄悄改回默认阈值。不要引入原服务没有传递的推理参数或自行更换 FunASR 版本。
5. 应用层 `vad_eos=1000` 与模型层 `max_end_silence_time=1100` 是不同字段。原 FSMN 调用未把请求 `eos` 传给模型，不得用 `vad_eos` 覆盖模型 YAML。

现有对比脚本会独立覆盖 fenshen 配置，它的成功不能替代生产运行目录的配置校验。必须同时修复生产入口。

## 5. 分包、缓存与结束调度

按照原服务调用顺序处理，而不是“文件最后一块一律 final”：

1. 默认按 100 ms，即 1600 个采样点/3200 字节，模拟客户端发送音频包；最后一包可以不足 100 ms。
2. 收到音频包先追加到当前已接收 PCM 缓冲区。每收到两个包，合并未推理的包，调用 `generate(input=..., cache=..., is_final=False, chunk_size=200)`。
3. float32 转换采用 `int16.astype(float32) / 32767`。每条父音频独立 cache，同一条音频持续复用；每个并发 worker 的模型与流状态不得串流。
4. 所有音频包处理完，再模拟客户端 `signal=end`。分身配置 `send_end_result=true`，应执行应用结束处理。
5. 若仍有未推理的非空缓冲，执行一次 `is_final=True` 的模型调用；若缓冲为空，跳过 `generate`，但仍执行应用层 EOF 逻辑。
6. 不追加静音，不对空输入强行调用模型 flush，不把已经以 `is_final=False` 推理过的末块再推理一次。
7. 区分模型调用日志和应用结束处理日志。不能把没有发生的模型调用伪记为 final generate。

例如总长为 1000 ms、100 ms 分包时，5 次模型调用均为 `is_final=False`；结束时只执行应用结束处理。总长为 900 ms 时，前四次为 200 ms 非 final，最后一个 100 ms 残留在 end 时 final。此处为行为说明，不要求执行验证。

## 6. 逐事件消费与实际裁切

保持原事件顺序，不能先按“统一区间”归并后丢失事件分支、pending 状态和当时缓冲末尾。所有时间戳为整条流的绝对位置；裁切上界使用该次事件发生时**已接收**的缓冲范围，不能偷用完整文件 EOF。

### 6.1 只有起点 `[start, -1]`

直接设置 `pending_start_ms=start`。原服务会覆盖已有 pending；不得因已有 pending 而新增失败条件。

### 6.2 只有终点 `[-1, end]`

1. 没有 pending 时忽略该事件，不让整个父音频失败。
2. 当 `end - pending_start_ms <= 150` ms 时丢弃并清空 pending；恰好 150 ms 也丢弃。
3. 其余片段实际使用：`start_byte=max(0, int(pending_start_ms * 32))`，`end_byte=min(current_received_bytes, int(end * 32))`，裁出 `[start_byte, end_byte)`。
4. **不加任何固定尾部，不向前扩展实际起点。** 删除此路径中的 675 ms 尾部保留及其他补尾逻辑。
5. 原服务用于返回/去重的 `seg_start_local_ms=pending_start_ms-100`，但实际音频从 pending 起点开始。两种时间戳必须分别记录，不能把返回值的 -100 ms 误用于 PCM 裁切。
6. 正常发出后清空 pending；重复事件的跳过顺序和 pending 状态遵循原方法。

### 6.3 完整端点 `[start, end]`

这是原服务与常规 VAD 区间裁切不同的分支，必须按本次“完全一致”目标保留其现有行为：

1. 仅在已有 pending 且 `end - pending_start_ms <= 200` ms 时丢弃并清空 pending；没有 pending 时不套用该 200 ms 过滤条件。
2. 实际起点为 `max(0, int(start * 32))`。
3. 原服务虽然计算 `end_pcm`，实际执行的是 `self.pcm_bytes[start_pcm:]`，即裁到**该次调用时已经收到的缓冲末尾**，不是模型 end，也不是整条文件末尾。
4. 模型 end、接口 end 与实际 clip end 可以不同，必须分别保留，不能为了“更合理”顺手将原服务行为改为 `[start, end)`。
5. 本分支按 `(start, end)` 去重；发出后清空 pending。

### 6.4 去重与忽略

对照原服务 `emitted` 的 key、更新时机、continue 与 pending 清理顺序实现。split-end 的 key 为 `(pending_start_ms-100, end)`，完整端点的 key 为 `(start, end)`。不要用扩展后片段范围取代原 key，不要擅自合并、排序、截短相邻片段。

原服务没有消费的 `[-1, -1]` 不产生片段。对真实异常可以记录清晰错误，但不能把原服务会忽略的事件升级成整条流失败。

## 7. 短段过滤和 EOF

取消当前 VAD 核心统一的“实际时长不足 1000 ms 丢弃”规则，以第 6 节的分支条件为准。不能把 1000 改成统一 150 或 200 后就认为对齐；完整端点条件依赖 pending 是否存在。

EOF 行为按分身的 `send_end_result=true` 实现：

- 结束处理仍有 pending 时，原服务裁出 pending 起点到当前源 PCM 末尾并送 ASR；当前项目应发布相同 PCM 片段，标记应用层 `eof_fallback`，不得凭空生成模型终点。
- 结束处理没有 pending 时不兜底发布整条音频。
- 删除“无模型 end 的 EOF 补尾一律丢弃”的硬规则；旧 `include_eof_fallback=false` 默认值/流水线配置不能继续屏蔽分身要求的行为。
- 删除“模型 end 到最后前端帧/EOF 就丢弃”的独立规则。若模型输出完整或 split-end，按对应事件分支处理，不因为贴近 EOF 再过滤。
- 不再使用 25 ms 窗、10 ms 步推算 EOF 边界来否决原服务接受的片段。
- EOF 补尾不额外套用 1 秒门限，保持原服务行为及输出顺序。

## 8. 当前项目的实施范围

至少检查以下文件及其实际调用方：

- `D:/Work/audio-data-engine/src/audio_engine/core/fsmn_vad/adapter.py`：输入分包、模型调用、final 调度、流状态与事件日志。
- `D:/Work/audio-data-engine/src/audio_engine/core/fsmn_vad/pairing.py`：分支裁切、去重、短段和 EOF 规则。
- `D:/Work/audio-data-engine/src/audio_engine/core/fsmn_vad/clip.py`、`types.py`：裁切契约、常量和结果类型。
- `D:/Work/audio-data-engine/src/audio_engine/core/fsmn_vad/runtime.py`：完整 fenshen 运行目录准备与有效配置检查。
- `D:/Work/audio-data-engine/src/audio_engine/core/fsmn_vad/identity.py`：片段身份、模型时间戳与实际裁切字段。
- `D:/Work/audio-data-engine/src/audio_engine/operators/audio/fsmn_vad_segment.py`：生产算子默认值、产物、指纹、模型加载与配置检查。
- `D:/Work/audio-data-engine/scripts/run_fsmn_vad.py`：单文件/目录入口必须复用同一生产核心。
- `D:/Work/audio-data-engine/scripts/compare_fsmn_vad_projects.py`：继续调用修复后的当前项目核心，保留原服务只读回放和强制 fenshen 配置。
- `D:/Work/audio-data-engine/pipelines/data_cleaning_source_A.yaml`、`vad_segment_batch.yaml`、`audio_ingest_vad_asr_dnsmos_tags.yaml` 及其他引用该算子的配置、相关操作手册。

不得只改对比脚本、CLI 或元数据来制造一致；实际生产流水线和落盘音频必须采用同一套语义。不得以运行时导入对照项目、AST 提取对照方法作为生产依赖；对照项目只作为实现基线。

对比脚本中已有的原服务 AST 回放是诊断专用，可以保留。确保它记录实际业务代码的结果，不增加隐藏的 end 修正或绕过过滤。

保留原有多 worker/分片机制，不做无关并发重构；不得共享跨父音频 pending、emitted 或 cache。

## 9. 元数据、版本和旧产物

1. 分别记录原始 `vad_start_ms/vad_end_ms`、原服务返回/去重时间戳、实际 `clip_start_sample/clip_end_sample/clip_duration_ms`。原服务 split-end 返回 start 提前 100 ms，不能污染实际裁切字段。
2. 正常 split-end 的 `extra_padding_ms=0`。完整端点裁到缓冲末尾产生的模型 end 与 clip end 差异，使用准确的来源/分支字段解释，不能标成固定尾部策略。
3. EOF 补尾的模型 end 应为空，实际末尾和应用推断末尾单独记录，不伪造成模型检测结果。
4. 指纹包括实际 fenshen 配置、依赖、分包、final 策略、分支过滤与 EOF 策略；旧的 675 ms/1 秒过滤/EOF 丢弃指纹不得继续沿用。
5. 升级 `VAD_PROCESSING_VERSION`（当前 v7）与算子版本，使新旧片段 ID、结果缓存和 checkpoint 能明确区分。检查父源已完成状态、步骤 checkpoint 与 `--resume`，避免更新代码后仍读旧产物。
6. 给用户明确的服务器重跑说明：从父音频重新 VAD，必要时使用 `--force`，输出到新目录/新 Manifest。不得自动删除旧片段、旧 Manifest 或用户数据，也不得声称旧 WAV 会自动更新。
7. 更新与本需求冲突的有效文档/配置注释。历史验证材料可以保留，但应注明旧规则已被本需求取代，不能伪造新的验证结果。

## 10. 用户在服务器验收

Cursor 本次不执行本节操作，只提供修改后可用的命令。

使用现有 `scripts/compare_fsmn_vad_projects.py`，保持两边使用完整 fenshen YAML、同一音频和同一 Python 环境：

```bash
/data2/data-cp/lizi/conda_envs/asr/bin/python \
  /data2/data-cp/lizi/audio-data-engine/scripts/compare_fsmn_vad_projects.py \
  --engine-project /data2/data-cp/lizi/audio-data-engine \
  --service-project /data2/data-cp/lizi/ph_ai_model_asr_rt_cpu \
  --engine-model-dir /data2/data-cp/xijujun/models/fsmn-vad \
  --service-model-dir /data2/data-cp/xijujun/models/fsmn-vad \
  --audio /data2/data-cp/lizi/ASR-data/mt3000/str_con_AkfQNU3KCR_174540_176610.wav \
  --output /data2/data-cp/lizi/tmp/1016/fenshen_aligned/vad_compare.json \
  --save-clips
```

已知样本预期：两边模型端点为 940–2680 ms，实际采样范围为 `[15040, 42880)`，实际时长 1740 ms；当前项目不能再生成 2415 ms 的音频。

严格验收要求：两边执行成功、完整 fenshen 配置一致、原始事件一致、片段数量和顺序一致、实际裁切范围与每段 PCM SHA256 一致。对于同起点的对应片段，`engine_minus_service_end_ms=0` 且 `pcm_equal=true`。同时检查未匹配片段，不能只挑匹配成功的一段报告通过。

用户还会自行使用不同长度、短语音、文件中途截断及不同端点事件形态的录音检查分支和 EOF。对单个样本的历史成功结果，不得推断其他分支已经实测通过。

## 11. 交付给 Cursor 的提示词

```text
请执行当前项目文档：
D:/Work/audio-data-engine/docs/04-改进需求/进行中/049-FSMN-VAD与分身实时服务完全对齐.md

目标：将 audio-data-engine 的生产 FSMN VAD 修复为与
D:/Work/ph_ai_model_asr_rt_cpu 的 fenshen 分身流程完全一致。
一致性以原服务实际传给 _call_vllm_transcribe 的 PCM segment 为准，包括
原始模型事件、片段数量、顺序、实际起止采样点和 PCM 内容。

先静态阅读对照项目 app/core/vad_asr_core_new.py 的实际代码，完整实现需求，
不要只把 END_SILENCE_KEEP_MS 从 675 改成 0 就结束。
必须同时对齐完整 fenshen 配置、100 ms 分包/每两包推理、客户端 end 与 final 调度、
split-end 和完整端点分支各自的过滤/裁切/去重，以及 send_end_result=true 的 EOF 补尾。
注意完整 [start,end] 分支在原服务实际裁到当时已收到的缓冲末尾；
split-end 返回的 start 提前 100 ms，但实际 PCM 起点不提前。
不要将这些现有行为擅自“修正”为另一种统一裁切规则。

修改必须覆盖生产核心、FSMN 算子、单文件脚本、相关 YAML、指纹/版本及操作说明。
完整 fenshen YAML 必须真正生效，并校验加载后的模型参数；不修改共享权重目录。
对照项目只读，不能作为生产运行时依赖；保持 ASR、DNSMOS、tag 和其他无关功能不变。
本需求明确取代旧 041/045 中关于 675 ms、1 秒过滤及 EOF 丢弃的冲突要求，
不要因这些旧要求保留差异，也不要为此再向用户请求确认。

本次不新增、不执行任何测试，不运行 pytest、mock、smoke、推理、对比脚本，
不安装或升级依赖。仅做静态阅读、编辑和 diff 检查；不要为了兼容旧测试保留旧行为。
用户会自行在服务器使用 scripts/compare_fsmn_vad_projects.py 验收。
完成后简洁说明改动、旧产物重跑方法、服务器命令以及“未运行测试”，不要编造结果。
```
