# 009 · Kimi vLLM：按时长 pad 提速，不硬组变长 batch

> 状态：**已落地（2026-09）**  
> 生产入口：`audio-data pipeline run pipelines/kimi_asr_batch.yaml`  
> 现行说明：[Kimi-Audio-vLLM识别流水线](../../07-操作手册/Kimi-Audio-vLLM识别流水线.md)、[工序总览](../../01-项目架构/工序总览.md)

---

## 落地摘要（实现对照）

| 项 | 位置 |
| --- | --- |
| pad 算子 | `audio.kimi_duration_pad` → `src/audio_engine/operators/audio/kimi_pad.py` |
| vLLM ASR | `asr.kimi_batch` 4.0.0：多 `API_BASE` 粘滞分配；缓存键含 pad 桶 |
| 流水线 | `pipelines/kimi_asr_batch.yaml`（内部 pad → 识别；默认 2 shard × concurrency 4） |
| 探针 | `scripts/probe_kimi_vllm.py`：`--pad` / `--repeat` / 多端口 |
| 单测 | `tests/test_kimi_pad.py`、`tests/test_kimi_asr.py` |

`resampled_16k` 与原始 `duration` 不被改写。产物仍是 `kimi_asr_*` / `transcripts.kimi`，join 名 `kimi`。pad 音频不进 Release / 评测集真值。

> **定位：Kimi-vLLM 专属流水线。** 与 Qwen / SenseVoice / 豆包 / Kimi 本地加载完全解耦。pad 只服务本流水线的推理输入，不进清洗、不进其它 ASR、不进训练集和评测集的音频真值。
>
> 背景：本地 `kimia_infer` 两卡 A800、每卡两实例，1.25 万条约 5h24min。vLLM 版 Kimi **不能把长短不一的 WAV 可靠打进同一个音频 tensor batch**。本需求用「先按桶 pad 成相同时长，再提高在途请求数」换吞吐，而不是去改 vLLM 内核、也不是假装能做变长 GPU batch。

三大工序对照（见 [工序总览](../../01-项目架构/工序总览.md)）：

```text
工序一 ①清洗
    │
    ├─ qwen_asr_batch / sensevoice / doubao / kimi_audio_asr_batch   ← 不许改、不共享 pad
    └─ 【本需求】kimi_asr_batch（vLLM 专属：内部 pad → 识别）
            │
            ▼  产物契约与其它 ASR 相同
工序一 ②  multi_asr_aggregate  --join-manifest kimi
工序一 ③  classify_*  →  gold / classified / release
            │
            ├─► 工序二 训练（Release 用原 resampled_16k，不用 pad 音频）
            └─► 工序三 评测（评测集音频用原 resampled_16k；
                            若评 Kimi-vLLM 本身，再用本流水线 --eval-name 跑批）
```

---

## 我想做什么

做一条 **只属于 Kimi-vLLM** 的识别流水线：在进 vLLM 之前，按时长桶把音频 **尾部静音 pad 到该桶上限**，让同一窗口里的特征 `T` 一致，从而把 `--max-num-seqs` 从 2～4 提到约 8。客户端一条 HTTP 仍只送一条 WAV，**不硬组变长 GPU batch**。

这条流水线必须：

1. **专属**：pad + vLLM 识别包在同一条 YAML 里一次 `pipeline run` 跑完。不要做成「全模型共用的 pad 预处理」。
2. **解耦**：不改、不依赖 `qwen_asr_batch` / `sensevoice_asr_batch` / `doubao_*` / `kimi_audio_asr_batch` / `data_cleaning_source_A`。`multi_asr_aggregate`、`classify_*`、`eval_*` 只按现有契约消费产物，不感知 pad。
3. **能接工序二、工序三**：样本 `id`、转写字段、manifest 命名、`--source-name` / `--asr-run` / `--eval-name` / `--join-manifest kimi` 与现网其它 ASR 一致，聚合、分拣、Release、评测注册和评测跑批都不用为 Kimi 开特例。

配套：两张 A800 各起一个完整 vLLM（`--tensor-parallel-size 1`），在途请求对齐两台 `max-num-seqs` 之和。现在单 `KIMI_ASR_API_BASE` + 8 shard × 4 并发打一台 `--max-num-seqs 2`，多出来的是排队不是吞吐。

本地 `kimi_audio_asr_batch`（`kimia_infer` 一次一条 `generate`）不在本需求改造范围内，join 名继续是 `kimi_audio`，禁止和 vLLM 的 `kimi` 混用。

---

## 专属与解耦（必须遵守）

| 可以动 | 不许动 |
|--------|--------|
| `pipelines/kimi_asr_batch.yaml`（在本 YAML **内部**加 pad 步，或等价专属 YAML 但对外仍产出 `kimi_asr_*`） | `qwen_asr_batch` / `sensevoice_asr_batch` / `doubao_*` / `kimi_audio_asr_batch` |
| `configs/asr/kimi.yaml`、Kimi vLLM 探针与操作手册 | `data_cleaning_source_A.yaml`（清洗产物仍只保证 `resampled_16k`） |
| 新建 **仅 Kimi-vLLM 使用** 的 pad 算子 | 把 pad 做成清洗步骤或其它模型的默认输入 |
| 本流水线的多 `API_BASE` | 改 `multi_asr_aggregate` / `classify_*` / `eval_aggregate` 的算子逻辑来迁就 pad |

pad 音频是 **推理侧派生文件**：每条需要补静音的样本写一份新 WAV，manifest 增加 key（建议 `kimi_padded_16k`）。**不是**按桶把整库 `resampled_16k` 复制多套。已对齐桶上限的样本可不写新文件，新 key 指向原路径。

---

## 数据从哪来、结果要什么

- **输入（工序一 · 建集）**：`cleaned_<source>`，含 `resampled_16k` 和 **原始** `duration`。
- **输入（工序三 · 评测跑批）**：已注册评测集（`--eval-name`），同样用评测集里的 `resampled_16k` 做 pad 源，不得改评测集真值音频。
- **中间产物**：`audio.kimi_padded_16k`。不覆盖 `resampled_16k`。pad 只加静音，不改采样率、声道、已有转写。
- **对外产物（与现网一致）**：
  - 建集：`datasets/stage1/asr/{alias}_asr_<source>.parquet`（默认 alias=`kimi` → `kimi_asr_<source>.parquet`）
  - 评测：`datasets/stage3/asr/{alias}_asr_eval_<eval>.parquet`
  - 转写：`transcripts.kimi.text`（若 `--asr-run` 另指定 alias，按现有 ASR 别名规则写 transcript key，与 Qwen/SenseVoice 同一套）
- **规模**：万级到十万级 VAD 片段；实测参考约 1.25 万条。

下游只认：**同 id、有 `transcripts.kimi`、路径符合 `{alias}_asr_*`。** 它们不需要知道 pad 的存在。

---

## 业务上有哪些规矩

### 1. 禁止的做法

- 不要把不同时长的原始 WAV 当成一个 GPU 音频 batch 去叠。
- 不要把 `configs/asr/kimi.yaml` 的 `batch_size` 调到 32 当「真 batch」——那只是 HTTP 切片。
- 不要用 `--tensor-parallel-size 2` 跑 Kimi ASR 批处理。
- 不要一律 pad 到 30 秒。
- 不要本地 `kimi_audio_asr_batch` 和本 vLLM 服务抢同一张卡。
- 不要开 prefix cache（服务端 `--no-enable-prefix-caching`）。
- 不要把 `kimi_padded_16k` 当作训练集或评测集的音频真值；工序二 Release、工序三 `eval register` 继续用 `resampled_16k`（或评测集已有音频键）。
- 不要用 pad 后的时长覆盖 manifest 的 `duration`（分拣、时长均衡、评测切片都依赖原始时长）。
- pad / 识别失败只标自身 `failed`，不伪造转写，不拖垮整批。

### 2. 时长桶（必须按这个切）

16 kHz 尾部补零，补到该桶目标秒数：

| 原时长 `d` | pad 到 |
|------------|--------|
| `d ≤ 3s` | 3s |
| `3s < d ≤ 6s` | 6s |
| `6s < d ≤ 10s` | 10s |
| `10s < d ≤ 15s` | 15s |
| `15s < d ≤ 30s` | 30s |
| `d > 30s` | pad 步不硬截断冒充完整句；按 vLLM transcription 服务端切窗或单独标记。落地时用一条 >30s 探针确认，并写入手册。 |

### 3. vLLM 部署（两卡副本，不要 TP）

每卡一个完整模型，例如 GPU 4 → `:5554`，GPU 5 → `:5555`：

- `--tensor-parallel-size 1`、`--max-model-len 4096`、`--limit-mm-per-prompt '{"audio":1}'`
- `--no-enable-prefix-caching`、`--gpu-memory-utilization` 约 0.90
- **未 pad 前** `--max-num-seqs` 2～4；**pad 且混合时长探针通过后** 每卡可试到 8

### 4. 客户端在途请求对齐服务能力

在途 HTTP ≈ 各副本 `max-num-seqs` 之和。两副本 × 4 seq → 大约 8 路。不要 8 shard × 4 去打一台 seq=2 的服务。

本流水线要能配多个 `KIMI_ASR_API_BASE`（轮询或按 shard 绑定）。落地后仍是 **一条** `pipeline run`，不要让用户手工切两半 parquet。

### 5. 弱替代（不能代替 pad）

按 `duration` 排序只降低混 `T` 的概率，验收仍以分桶 pad 为准。

---

## 下游怎么才算「完美接入」

本流水线停在「识别结果 parquet」。后面全部复用现有命令，**零改聚合/分拣/训练/评测算子**。

**工序一 · ② 聚拢**

```bash
audio-data pipeline run pipelines/kimi_asr_batch.yaml --source-name mt3000
# 可选：--asr-run kimi1  → kimi1_asr_mt3000.parquet

audio-data pipeline run pipelines/multi_asr_aggregate.yaml --source-name mt3000 \
  --aggregate-base qwen1 \
  --join-manifest sensevoice1 \
  --join-manifest kimi
```

- 与 Qwen/SenseVoice **按 id 左连接**，允许本流水线多出或缺失 id（缺失记 join 空，规则与现网其它 join 一致，不要为 Kimi 改聚合语义）。
- join 名 `kimi` 对应 vLLM 产物；本地加载仍是 `--join-manifest kimi_audio`。

**工序一 · ③ 分拣 → Release（给工序二）**

- `classify_dataset` / `classify_dataset_v2` / `classify_external_gold` 只读聚合后的 `transcripts.*` 与原始 `duration` / `resampled_16k`。
- Kimi 只是多一票转写（`model_family` 按现有规则），不因 pad 改变阈值或桶定义。
- `release build` / 训练集导出的音频必须是 **未 pad** 的 `resampled_16k`。Kimi pad 文件不得进入 Dataset Release 的训练音频字段。

**工序二 · 训练**

- 只消费工序一冻结的 Release。本流水线不出现在训练 YAML 里。
- 训练框架拿到的 wav = 原始 16k，gold_text = 分拣结果；与是否用 Kimi 产金标无关。

**工序三 · 评测**

```bash
# 评测集仍来自 classified_/gold_/reviewed_，音频键 resampled_16k
audio-data eval register "datasets/stage1/derived/classified_mt3000.parquet" --name eval_core_v001

# 若要评 Kimi-vLLM 本身：用同一条专属流水线，不要另做评测专用 pad 逻辑
audio-data pipeline run pipelines/kimi_asr_batch.yaml \
  --eval-name eval_core_v001 --asr-run kimi

audio-data pipeline run pipelines/eval_aggregate.yaml \
  --eval-name eval_core_v001 --join-manifest kimi

audio-data pipeline run pipelines/eval_metric_pipeline.yaml \
  --eval-name eval_core_v001 --eval-model kimi
```

- 评测集缺 id 则失败（现网 `eval_aggregate` 语义），推理侧多出 id 允许。
- 报告相对 `gold_text`，不把 pad 静音当成「多出来的语音内容」去改指标口径。

---

## 我怎么才算满意

1. **一条命令**（建集或评测各一条）跑完 pad + Kimi-vLLM 识别，无需先跑别的 ASR。
2. `resampled_16k` 不被改写；`duration` 仍是原始时长；pad 文件可复现。
3. 产物能直接：
   - `--join-manifest kimi` 进 `multi_asr_aggregate`
   - 再进 `classify_*` → `release build`（工序二可用）
   - `--eval-name` 进工序三跑批 → `eval_aggregate` → `eval_metric_pipeline`
4. 不跑本流水线时，Qwen/SenseVoice/分拣/评测行为与现在完全一样。
5. 探针三档通过：短 WAV；**长短混合**目录（pad 后、目标 concurrency）；同一条连跑两次文本稳定。
6. 未 pad 时不把 `max-num-seqs` 默认抬到 8+。
7. 坏样本只标自身失败；两卡副本时在途数不超过 seq 预算。
8. 更新 `单条流水线执行命令.txt`、`docs/07-操作手册/Kimi-Audio-vLLM识别流水线.md`、dev/local 手册（含双端口、`--no-enable-prefix-caching`、join 名 `kimi`、评测 `--eval-name` 示例）。明确写：pad 音频不进 Release / 评测集真值。

加速预期（对照用，非硬 KPI）：相对本地约 6.2 秒/条，两副本 + seqs=4 约 2～4×；分桶 pad + seqs=8 约 4～8×。达不到 Qwen `batch_size=32` 也符合本需求。

---

## 其他我想说的

- 机器：两卡 A800 80G；权重现网 `Kimi-Audio-7B-Instruct`；`--served-model-name kimi-audio` 与 `KIMI_ASR_MODEL` 一致。
- 参考：现有 `kimi_asr_batch.yaml`、`configs/asr/kimi.yaml`、`scripts/probe_kimi_vllm.py`、工序总览里 ASR 任选多次 + `--asr-run`。
- 千万别动：其它 ASR YAML；清洗的 `resampled_16k` 语义；聚合/分拣/评测算子；本地 `kimi_audio` join 名。
- 生产入口必须是 `audio-data pipeline run pipelines/kimi_asr_batch.yaml`（或对外契约相同的专属 YAML），按 [流水线构建-AI执行手册](../../07-操作手册/流水线构建-AI执行手册.md)。
- 缓存键必须包含 pad 桶/目标时长，禁止用未 pad 音频的旧 ASR 缓存冒充新结果。

---

## 交给 AI 时可以说

```text
请严格按 docs/07-操作手册/流水线构建-AI执行手册.md 执行。
需求文档：docs/04-改进需求/已完成/009-Kimi-vLLM按时长pad提速.md
用自然语言需求即可；缺关键信息先问我，其余工程细节你定。
做完更新 单条流水线执行命令.txt，保证一条 pipeline run 能跑。
本流水线必须是 Kimi-vLLM 专属，与其它 ASR 解耦；产物按现有 kimi / --asr-run / --eval-name 契约接入工序一聚拢分拣、工序二 Release、工序三评测。
```
