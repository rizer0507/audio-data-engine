# 009 · GLM vLLM 识别流水线

> 状态：**已落地（2026-09）**  
> 生产入口：`audio-data pipeline run pipelines/glm_asr_batch.yaml`  
> 现行说明：[GLM-ASR-vLLM识别流水线](../../07-操作手册/GLM-ASR-vLLM识别流水线.md)、[工序总览](../../01-项目架构/工序总览.md)

---

## 落地摘要（实现对照）

| 项 | 位置 |
| --- | --- |
| YAML | `pipelines/glm_asr_batch.yaml`：单步 `asr.glm_batch`；默认 1 shard；无 `gpus` |
| 配置 | `configs/asr/glm.yaml`：`model: glm-asr`、`concurrency: 8`、`api_base: null` |
| 算子 | `src/audio_engine/operators/asr/glm.py`：`asr.glm` + `asr.glm_batch`（vLLM-only） |
| 探针 | `scripts/probe_glm_vllm.py`：短 WAV / 目录并发 / `--repeat 2`；**没有 `--pad`** |
| 单测 | `tests/test_glm_asr.py` |
| 分拣 | `zh_asr_v1.yaml` / `zh_asr_v2.yaml` 追加独立 `glm` family |
| 手册 | `docs/07-操作手册/GLM-ASR-vLLM识别流水线.md` |

产物契约：`glm_asr_*` / `transcripts.glm`，join 名 `glm`。未设置 `GLM_ASR_API_BASE` 时拒绝本地加载。  
权重路径：`/data2/data-cp/models/GLM-ASR-Nano-2512`。空闲 GPU / 端口落地当天确认（占位 `:5570`）。

> **定位：GLM-ASR 专属 vLLM 流水线。** 参照 Qwen transcription HTTP + 并发，不参照 Kimi pad，不在引擎进程里加载权重。

---

## 用户原话（真相源）

1. 参照 qwen3-asr 的 vLLM 加载方式
2. 设计一条流水线，完美嵌入工序一以及工序三
3. 要能满足 batch 并发转写，像 qwen3-asr 一样

与原话冲突时：**停下来问用户，不要自行改需求含义。**

---

## 定位

做一条 **GLM-ASR 专属** 的 vLLM 识别流水线，行为对齐现网 `qwen_asr_batch`：

- 服务端：`vllm serve`，OpenAI 兼容 `/v1/audio/transcriptions`
- 客户端：不加载模型、不占 GPU；一条 HTTP 送一条 WAV；用分片 + 线程池把请求打进 vLLM 的 continuous batching
- 产物契约与其它 ASR 相同：同 `id`、`transcripts.<alias>`、`{alias}_asr_<source|eval>.parquet`

**参照 Qwen，不参照 Kimi pad。** Kimi 的时长桶 pad 是 Kimi-Audio encoder 混 `T` 的专属补丁，不要抄进 GLM。除非上线前探针证明 GLM 也会因长短混合打挂，再另开需求，不要在本需求里预埋 pad 步。

与 Qwen / SenseVoice / 豆包 / Kimi-vLLM / Kimi 本地 **解耦**：不改它们的 YAML、算子和 join 名。

三大工序对照：

```text
工序一 ①清洗  cleaned_<BATCH>（resampled_16k）
    │
    ├─ qwen_asr_batch / sensevoice / doubao / kimi_asr_batch / kimi_audio_asr_batch
    └─ 【本需求】glm_asr_batch（vLLM HTTP 并发转写）
            │
            ▼  产物契约与其它 ASR 相同
工序一 ②  multi_asr_aggregate  --join-manifest glm
工序一 ③  classify_*  →  gold / classified / release
            │
            ├─► 工序二 训练（Release 用 resampled_16k，不感知 GLM）
            └─► 工序三 评测
                    eval register
                    glm_asr_batch --eval-name … --asr-run glm
                    eval_aggregate --join-manifest glm
                    eval_metric_pipeline --eval-model glm
```

---

## 我想做什么

现网 Qwen3-ASR 已经是「先起 vLLM，再 `audio-data pipeline run pipelines/qwen_asr_batch.yaml`」：客户端调 transcription API，片内 `concurrency` 并发 HTTP，服务端自己做 batch。GLM 要走同一条路，成为工序一多模型识别里的又一个可选模型，并在工序三用同一条 YAML 评它自己。

这条流水线必须：

1. **专属**：只服务 GLM-ASR。生产入口一条命令：`audio-data pipeline run pipelines/glm_asr_batch.yaml`
2. **对齐 Qwen 加载方式**：`vllm serve` + `--served-model-name` + `GLM_ASR_API_BASE`；复用现有 `call_vllm_transcription`；batch 算子 **只走 vLLM**，禁止再做一套进程内本地加载当生产路径
3. **嵌入工序一 / 工序三**：`--source-name` / `--asr-run` / `--eval-name` / `--join-manifest glm` 与现网其它 ASR 同一套命名，聚合、分拣、Release、评测注册 **零改算子逻辑**
4. **batch 并发**：客户端并发 HTTP + 服务端 `--max-num-seqs`；不要把 YAML 里的 `batch_size` 当成 GPU 真 batch，也不要硬组变长音频 tensor

默认模型按官方 vLLM recipe 假设为 **GLM-ASR-Nano-2512**（`zai-org/GLM-ASR-Nano-2512`，约 1.5B）。落地时用服务器本地权重路径覆盖。不要和 GLM-4-Voice（语音对话）混用。

---

## 专属与解耦（必须遵守）


| 可以动                                                             | 不许动                                                                                                |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| 新建 `pipelines/glm_asr_batch.yaml`                               | `qwen_asr_batch` / `sensevoice_asr_batch` / `doubao_*` / `kimi_asr_batch` / `kimi_audio_asr_batch` |
| 新建 `configs/asr/glm.yaml`、`asr.glm` / `asr.glm_batch`           | `data_cleaning_source_A.yaml`；清洗产物仍只保证 `resampled_16k`                                             |
| 新建 `scripts/probe_glm_vllm.py`、GLM 操作手册                         | 把 GLM 接到现有 `asr.qwen_batch` 上「换个 model 名」凑合                                                        |
| 分拣配置里 **追加** `glm` 模型族                                          | 改 `multi_asr_aggregate` / `classify_`* / `eval_aggregate` 的算子语义                                    |
| 必要时给 `call_vllm_transcription` 加 **可选** 字段（默认关闭，Qwen/Kimi 行为不变） | 给 transcription 混入 Chat Completions 参数；抄 Qwen 的 `qwen3_asr_language.jinja`                         |


`glm_asr_batch` 这个 YAML 名已经匹配现有 `_ASR_PIPELINE_RE`（`glm` + `_asr_batch`），**默认不必改** `src/audio_engine/core/`。只有 CLI 报错文案里要提到新流水线时才允许改提示字符串。

---

## 数据从哪来、结果要什么

- **输入（工序一 · 建集）**：`datasets/stage1/cleaned/cleaned_<source>.parquet`，含 `resampled_16k`。
- **输入（工序三 · 评测跑批）**：已注册评测集（`--eval-name`），音频键仍是评测集里的 `resampled_16k`，不改评测集真值。
- **对外产物（与 Qwen 同一契约）**：


| 场景     | 命令要点                                       | 产物                                                                                |
| ------ | ------------------------------------------ | --------------------------------------------------------------------------------- |
| 建集默认   | `--source-name mt3000`                     | `datasets/stage1/asr/glm_asr_mt3000.parquet`，`transcripts.glm`                    |
| 建集别名   | `--source-name mt3000 --asr-run glm1`      | `glm1_asr_mt3000.parquet`，`transcripts.glm1`                                      |
| 评测默认   | `--eval-name eval_core_v001 --asr-run glm` | `datasets/stage3/asr/glm_asr_eval_core_v001.parquet`（或现网 `{alias}_asr_{eval}` 规则） |
| 评测 SFT | `--eval-name … --asr-run glm-sft-ep100`    | 对应 alias 产物；`GLM_ASR_API_BASE` / `GLM_ASR_MODEL` 指向该权重的 vLLM                      |


下游只认：**同 id、有** `transcripts.<alias>`**、路径符合** `{alias}_asr_`***。** 它们不需要知道 vLLM。

失败样本只标自身 `failed`，写入 `status` / `errors`，**不伪造转写**，不拖垮整批。

---

## 业务上有哪些规矩

### 1. vLLM 加载（对齐 Qwen，不要抄 Kimi pad / 不要抄 Qwen jinja）

官方入口（recipe）：`vllm serve zai-org/GLM-ASR-Nano-2512`，客户端走 `/v1/audio/transcriptions`。

本仓库约束：

- 每卡一份完整模型：`--tensor-parallel-size 1`。不要 TP=2 跑 ASR 批处理。
- `--served-model-name` 必须与 `GLM_ASR_MODEL`（建议默认 `glm-asr`）一致。
- 不要把 Qwen 的 `--chat-template …/qwen3_asr_language.jinja` 套到 GLM 上。
- 客户端复用 `src/audio_engine/operators/asr/vllm.py` 的 `call_vllm_transcription`。字段仍只发 transcription 认识的：`model` / `language` / `temperature` / 可选 `prompt` / `response_format`。
- 若 GLM 必须多一个字段（例如某些版本要 `max_tokens`）：做成 **opt-in**，默认不发送，回归测试保证 Qwen/Kimi 请求体不变。
- 不要用 `/v1/chat/completions` + `input_audio` 当生产路径（recipe 里有示例，但现网 Qwen/Kimi 都走 transcription）。

建议启动骨架（端口、GPU、权重路径落地前确认，避免和 Qwen `:5559` / Kimi `:5554` 撞车）：

```bash
# 示例：GPU 6 → :5570；路径换成服务器本地权重
CUDA_VISIBLE_DEVICES=6 \
vllm serve /data2/data-cp/models/GLM-ASR-Nano-2512 \
  --host 0.0.0.0 \
  --port 5570 \
  --served-model-name glm-asr \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"audio":1}'
```

环境变量（对齐 Qwen 的 `QWEN_ASR_*`）：


| 变量                  | 作用                                          |
| ------------------- | ------------------------------------------- |
| `GLM_ASR_API_BASE`  | 服务根或 `/v1`；客户端避免重复拼接 `/v1`                  |
| `GLM_ASR_MODEL`     | 覆盖 yaml 的 `model`，须等于 `--served-model-name` |
| `GLM_ASR_API_KEY`   | 可选，默认 `dummy`                               |
| `GLM_ASR_API_BASES` | 可选；多副本逗号分隔（见并发）                             |


依赖注意（写入手册，不算本仓库代码）：vLLM 需带 audio extras（官方 recipe：`vllm>=0.14.1` + `vllm[audio]`；transformers 版本按当时 recipe）。本流水线 **不** 在引擎进程里 `from_pretrained` GLM。

### 2. batch 并发（像 Qwen 那样）

「像 qwen3-asr 一样 batch」= **vLLM 服务端 continuous batching + 客户端多路 HTTP**，不是客户端把多条 WAV 叠成一个 GPU tensor。


| 层级         | 配置                                         | 作用                                        |
| ---------- | ------------------------------------------ | ----------------------------------------- |
| 数据分片       | YAML `sharding.shards` / `parallel_shards` | 多进程同时打 HTTP                               |
| 片内请求并发     | `configs/asr/glm.yaml` 的 `concurrency`     | 单 shard `ThreadPoolExecutor` 同时发出的 HTTP 数 |
| 服务端真 batch | vLLM `--max-num-seqs`                      | GPU 上同时处理的序列数                             |
| HTTP 切片    | yaml `batch_size`                          | 只是客户端切块大小，**不是** GPU batch                |


约束：

- `asr.glm_batch` 与 `asr.qwen_batch` 一样：**未设置** `api_base` **/** `GLM_ASR_API_BASE` **时直接报错**，禁止回退本地权重。
- 在途 HTTP ≈ `parallel_shards × concurrency`，应 ≤ 各副本 `--max-num-seqs` 之和。
- YAML **不要** 再写 `gpus` / `instances_per_gpu`。那是 Qwen 旧本地加载遗留；本流水线客户端不占卡，服务端 GPU 只由 `CUDA_VISIBLE_DEVICES` + `vllm serve` 决定。
- 单副本默认建议：`shards: 1`、`parallel_shards: 1`、`concurrency` 对齐该副本 `--max-num-seqs`（先 4～8，探针通过再加）。
- 两卡两副本（同权重）：可学 Kimi，一次 `pipeline run` 吃 `GLM_ASR_API_BASE=http://127.0.0.1:5570,http://127.0.0.1:5571`，按 sample id 粘滞分配；YAML 用 2 shard。**不同权重 / 不同 SFT** 仍按 Qwen 评测习惯：各起各的端口，改 env 再跑一条 `--asr-run`。
- 不要 8 shard × 高 concurrency 去打一台 `max-num-seqs=2` 的服务——多出来的是排队不是吞吐。

### 3. 分拣模型族

GLM 是新的独立 family，同族多跑不算独立票：

- `configs/selection/zh_asr_v1.yaml` 与 `zh_asr_v2.yaml` 的 `model_families` **追加**：

```yaml
  glm:
    - glm
    - glm1
    - glm2
```

- 不要把 `glm*` 写进 `qwen` 族。`primary_family` 仍是 `qwen`，除非用户另改挑选规则。
- 分拣阈值、桶名、金标 medoid 规则不变。GLM 只是多一票转写。

### 4. 禁止的做法

- 不要改清洗产物的 `resampled_16k` / `duration`
- 不要做 GLM 专属 pad 目录当训练集或评测集真值
- 不要和正在跑的 Qwen / Kimi vLLM 抢同一张卡、同一个端口
- 不要旁路独立 CLI 当生产入口（禁止 `manifest shard` → `run-shards` → `merge` 三步手工当主路径）
- 不要用未设置 `GLM_ASR_API_BASE` 的旧缓存或 Qwen 缓存冒充 GLM 结果；缓存键必须含 model / api_base / 输入音频 key

---

## 下游怎么才算「完美接入」

本流水线停在「识别结果 parquet」。后面全部复用现有命令。

**工序一 · ② 聚拢**

```bash
export GLM_ASR_API_BASE=http://127.0.0.1:5570
export GLM_ASR_MODEL=glm-asr

audio-data pipeline run pipelines/glm_asr_batch.yaml --source-name mt3000
# 可选：--asr-run glm1  → glm1_asr_mt3000.parquet

audio-data pipeline run pipelines/multi_asr_aggregate.yaml --source-name mt3000 \
  --aggregate-base qwen1 \
  --join-manifest sensevoice1 \
  --join-manifest glm
```

- 与 Qwen/SenseVoice **按 id 左连接**；本流水线多出或缺失 id 的规则与现网其它 join 一致，不要为 GLM 改聚合语义。

**工序一 · ③ 分拣 → Release**

- `classify_dataset` / `classify_dataset_v2` / `classify_external_gold` 只读聚合后的 `transcripts.*` 与原始 `duration` / `resampled_16k`
- `release build` 的训练音频仍是 `resampled_16k`

**工序三 · 评测**

```bash
audio-data eval register "datasets/stage1/derived/classified_mt3000.parquet" \
  --name eval_core_v001

export GLM_ASR_API_BASE=http://127.0.0.1:5570
audio-data pipeline run pipelines/glm_asr_batch.yaml \
  --eval-name eval_core_v001 --asr-run glm

# 若评 SFT：另起端口，换 GLM_ASR_API_BASE / GLM_ASR_MODEL 再跑
# export GLM_ASR_API_BASE=http://127.0.0.1:5572
# audio-data pipeline run pipelines/glm_asr_batch.yaml \
#   --eval-name eval_core_v001 --asr-run glm-sft-ep100

audio-data pipeline run pipelines/eval_aggregate.yaml \
  --eval-name eval_core_v001 --join-manifest glm

audio-data pipeline run pipelines/eval_metric_pipeline.yaml \
  --eval-name eval_core_v001 --eval-model glm
```

- 评测集缺 id 则失败（现网 `eval_aggregate` 语义），推理侧多出 id 允许
- 指标相对 `gold_text`；空标桶从主 CER 剔除的规则不变
- **严禁** 只设一个全局 `GLM_ASR_API_BASE` 同时评两个权重（会串端口）。对照 Qwen 评测手册：评前 `unset` 建集用的 base

---

## 建议实现清单（给后续开发）

短名：`glm_asr_batch`。对照实现时以 Qwen 为蓝本，而不是 Kimi。


| 项    | 建议                                                                                                                                                                                  |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| YAML | `pipelines/glm_asr_batch.yaml`：单步 `asr.glm_batch`；`input_audio_key: resampled_16k`；`config_path: configs/asr/glm.yaml`；`fail_fast: false`；默认 1 shard（单副本）                           |
| 配置   | `configs/asr/glm.yaml`：`model: glm-asr`、`api_base: null`、`concurrency: 8`、`batch_size: 32`、`timeout: 120`、`language: zh`、`temperature: 0`、`response_format: json`                   |
| 算子   | `src/audio_engine/operators/asr/glm.py`：`asr.glm`（单条）+ `asr.glm_batch`（生产）；env 解析抄 `qwen.py` 的 `QWEN_ASR_*` → `GLM_ASR_*`；`process_batch` 抄 Qwen 的 skip / cache / ThreadPool / 失败隔离 |
| 注册   | `operators/asr/__init__.py` 导出新算子                                                                                                                                                   |
| 探针   | `scripts/probe_glm_vllm.py`：短 WAV；目录 `--concurrency`；同一条 `--repeat 2` 文本须稳定。**不要** `--pad`                                                                                          |
| 单测   | `tests/test_glm_asr.py`：对齐 `tests/test_qwen_asr.py`（拒本地加载、读 env、缓存复用、坏样本隔离、`transcript_key`、mock 跑通 pipeline）                                                                       |
| 分拣   | `zh_asr_v1.yaml` / `zh_asr_v2.yaml` 增加 `glm` family                                                                                                                                 |
| 文档   | 新建 `docs/07-操作手册/GLM-ASR-vLLM识别流水线.md`；更新工序总览 ASR 表、`评测流水线.md` 示例、`单条流水线执行命令.txt`、`手册/dev                                                                                           |


算子伪骨架（实现时按 Qwen 展开，不要新造旁路）：

```text
asr.glm_batch.process_batch
  → 无 api_base 且非 mock → ValueError("only supports vLLM")
  → should_skip / cache
  → ThreadPool(concurrency) × call_vllm_transcription(resampled_16k)
  → transcripts[alias] = {text, model, version, extra.language}
  → 单条失败 mark_failed，其余继续
```

---

## 上线前探针（必须过，再跑万级）

```bash
export GLM_ASR_API_BASE=http://127.0.0.1:5570
export GLM_ASR_MODEL=glm-asr

# 1) served model 名称一致
curl -fsS "http://127.0.0.1:5570/v1/models" | python -m json.tool

# 2) 短 WAV，text 必须非空（可先 curl transcriptions，再换成探针脚本）
python scripts/probe_glm_vllm.py /path/to/short.wav

# 3) 目录并发（对齐目标 concurrency / max-num-seqs）
python scripts/probe_glm_vllm.py /path/to/mixed_wavs --concurrency 8

# 4) 同一条连跑两次，文本须一致
python scripts/probe_glm_vllm.py /path/to/short.wav --repeat 2
```

建议退出码与 Kimi 探针对齐：`0` 全部非空且（若 `--repeat>1`）稳定；`1` 连接/配置/输入错误；`2` 空转写；`3` 重复识别不一致。

若第 3 档长短混合在高 `max-num-seqs` 下打挂：先把 seqs / concurrency 降到稳定值交付本需求；**不要擅自加 pad**。把现象记入手册，另开需求。

---

## 我怎么才算满意

1. **一条命令**完成建集识别：`audio-data pipeline run pipelines/glm_asr_batch.yaml --source-name <BATCH>`；评测同样一条：加 `--eval-name` + `--asr-run`
2. 未 export `GLM_ASR_API_BASE` 时 batch 算子拒绝本地加载
3. 产物能直接：
  - `--join-manifest glm` 进 `multi_asr_aggregate`
  - 再进 `classify_*` → `release build`
  - `--eval-name` 进工序三 → `eval_aggregate` → `eval_metric_pipeline --eval-model glm`
4. 不跑本流水线时，Qwen / SenseVoice / Kimi / 分拣 / 评测行为与现在完全一样
5. 并发可用：单副本下在途请求能吃满该副本 `--max-num-seqs`；坏样本只标自身失败
6. 探针短 WAV / 目录并发 / `--repeat 2` 通过后再跑生产批次
7. 更新 `单条流水线执行命令.txt`、工序总览、评测文档、dev/local 手册；明确 join 名 `glm`、端口不要和 Qwen/Kimi 冲突
8. 单测覆盖 vLLM-only、env 覆盖、缓存、失败隔离、`transcript_key`

---

## 其他我想说的

- **参考代码**：`pipelines/qwen_asr_batch.yaml`、`configs/asr/qwen_asr.yaml`、`src/audio_engine/operators/asr/qwen.py`、`tests/test_qwen_asr.py`、`call_vllm_transcription`
- **不要参考**：Kimi pad 算子、Qwen 本地 `Qwen3ASRModel.from_pretrained`、Qwen YAML 里的 `gpus/instances_per_gpu`
- **机器**：与现网 ASR 相同的 A800 环境；具体空闲卡、本地权重路径、正式端口 **落地当天确认**。上文 `GPU 6` / `:5570` / `/data2/data-cp/zcl/models/GLM-ASR-Nano-2512` 只是占位，避免文档真空
- **规模**：与 Qwen 同一量级（万～十万 VAD 片段）
- **生产入口**必须是 `audio-data pipeline run pipelines/glm_asr_batch.yaml`
- 不擅自 commit

落地前仍可确认（不阻塞把文档写清；实现会话里一次问清即可）：

1. 服务器上 GLM-ASR-Nano-2512 的实际目录？
2. 占用哪张卡、哪个端口？（不要和当时正在跑的 Qwen/Kimi 冲突）
3. 建集默认 join 名是否就叫 `glm`？（文档按 `glm` 写）

---

## 交给 AI 时可以说

```text
请严格按 docs/07-操作手册/流水线构建-AI执行手册.md 执行。
需求文档：docs/04-改进需求/已完成/009-GLM-VLLM加载流水线开发.md
用自然语言需求即可；缺关键信息先问我，其余工程细节你定。
做完更新 单条流水线执行命令.txt，保证一条 pipeline run 能跑。
本流水线必须是 GLM-vLLM 专属，参照 qwen_asr_batch（transcription HTTP + 并发），不要做 Kimi pad，不要本地加载。
产物按 glm / --asr-run / --eval-name 契约接入工序一聚拢分拣、工序三评测。
```

权重路径：/data2/data-cp/models/GLM-ASR-Nano-2512，空闲GPU和端口待指定，join名字用glm