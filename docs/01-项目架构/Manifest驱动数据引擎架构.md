> **文档定位（治理说明）**：本文保留 Manifest / Operator / Pipeline 分层等架构蓝图。其中「物理批次入库到 `data/batches/`、`datasets/batches/`」等段落属于早期愿景，**与现行「工程不持有物理音频、仅登记 `resources/manifest.yaml`」不一致**。涉及数据源登记时，以 [数据源登记规范](../02-规范规则/数据源登记规范.md) 与 [数据源落入流水线](../03-流水线/数据源落入流水线.md) 为准。现行工序见 [工序总览](./工序总览.md)。

可以。对于你这种场景，我建议不要继续发展成：

```text
wav/
├── pcm_to_wav.py
├── denoise.py
├── sensevoice.py
├── qwen_asr.py
├── augment.py
└── final_v2_final_new.xlsx
```

这种体系短期快，半年以后会非常难维护。

更适合的是建立一个 **Manifest-driven Audio Data Pipeline（清单驱动的音频数据处理流水线）**。它的核心思想是：

> **WAV 文件只是物理资产，Manifest 才是数据集的“真相源”；所有处理都是 Operator，Pipeline 只是 Operator 的 DAG 组合。**

这也是 NeMo、Lhotse 等现代 Speech AI 工具链大量采用的思路。NeMo 的 ASR 数据集本身就是“音频文件 + JSONL manifest”，Lhotse 则进一步把 Recording、Supervision、Cut 抽象成统一的数据对象。([NVIDIA Docs][1])

---

# 一、我建议你的整体架构

你可以把整个工程拆成 **7 层**（在原有 6 层之上，增加「数据批次资产 → 数据源」登记层）：

```text
                    ┌──────────────────────┐
                    │      Pipeline YAML   │
                    │  定义“这批数据怎么跑” │
                    └──────────┬───────────┘
                               │
                               ▼
┌──────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│ Raw Audio    │────▶│   Batch Asset Layer  │────▶│   Pipeline Engine    │
│ WAV / PCM    │     │  ingest → 数据源注册  │     │ Prefect / 自研 Runner│
└──────────────┘     └──────────┬───────────┘     └──────────┬───────────┘
                                │                            │
                                │  数据源 A / 数据源 B         │
                                └────────────────────────────┘
                                             │
             ┌───────────────────────────────┼──────────────────┐
             ▼                               ▼                  ▼
      Audio Operators                   ASR Operators       QC Operators
      pcm_to_wav                        sensevoice          duration
      resample                          qwen_asr            snr
      denoise                           whisper             cer
      vad                               remote_api          filter
      augment                                                 validate
             │                               │                  │
             └───────────────────────────────┼──────────────────┘
                                             ▼
                    ┌──────────────────────┐
                    │ Manifest / Metadata  │
                    │ JSONL / Parquet      │
                    └──────────┬───────────┘
                               │
                               ▼
                 Dataset / Train / Evaluation
```

这里最重要的不是某个框架，而是**分层边界**。

**关键变化**：Pipeline 的输入不再直接绑定「某个 raw 文件夹」，而是绑定体系内已登记的 **数据源**（如 `数据源A`）。清洗、标注等流水线都从这里读 manifest，而不是每次重新扫描 wav。

---

# 二、数据批次资产：从物理文件到可引用数据源

当你同时持有 **两批（或多批）** 物理数据时，不要把它们混在同一个 `data/raw/` 里跑。应先分别登记为 **数据批次资产**，再在体系内以 **数据源 A / 数据源 B** 的形式被后续流水线引用。

## 2.1 核心概念

| 概念 | 含义 |
|------|------|
| **数据批次（Batch）** | 一次物理交付的原始音频集合，如供应商 8/15 交付的 1.5 万条 wav |
| **批次 Manifest** | 该批次下每条样本的 manifest（`id`、路径、`sha256`、时长等） |
| **批次元数据（batch_meta）** | 批次级真相：入库时间、来源、交付方、文件统计、状态 |
| **数据源（Data Source）** | 批次登记完成后在体系内的稳定引用名，如 `数据源A`，供 Pipeline 作为 `input.source` |

关系链：

```text
物理文件（批次 A）
    ↓ ingest
批次资产 A（batch_meta + manifest.parquet）
    ↓ register
数据源 A（source_A.yaml）
    ↓ 作为 input
数据清洗流水线 → cleaned_A_*.parquet
    ↓
标注流水线 → train.jsonl
```

两批数据完全独立登记，互不覆盖：

```text
批次 A 物理文件  →  数据源 A  →  cleaned_A  →  train_A
批次 B 物理文件  →  数据源 B  →  cleaned_B  →  train_B
                              ↘  也可 merge / compare
```

## 2.2 目录结构

在原有目录上，为每个批次增加 **批次资产目录** 和 **数据源注册目录**：

```text
audio-data-engine/
│
├── data/
│   ├── batches/                    # 各批次物理文件（raw 永不修改）
│   │   ├── batch_A/
│   │   │   ├── 000001.wav
│   │   │   └── ...
│   │   └── batch_B/
│   │       ├── 000001.wav          # 可与 A 同名，因 batch_id 隔离
│   │       └── ...
│   ├── derived/                    # 按 source_id 分子目录
│   │   ├── source_A/
│   │   └── source_B/
│   └── exports/
│
├── datasets/
│   ├── batches/                    # 批次资产（真相源）
│   │   ├── batch_A/
│   │   │   ├── batch_meta.yaml     # 入库时间、来源、统计
│   │   │   ├── manifest.jsonl
│   │   │   └── manifest.parquet
│   │   └── batch_B/
│   │       ├── batch_meta.yaml
│   │       ├── manifest.jsonl
│   │       └── manifest.parquet
│   │
│   └── sources/                    # 数据源注册（Pipeline 引用入口）
│       ├── source_A.yaml
│       └── source_B.yaml
│
└── pipelines/
    ├── ingest_batch.yaml           # 批次入库专用
    ├── clean_source_A.yaml         # 输入：数据源 A
    └── clean_source_B.yaml
```

**区分原则**：

- `data/batches/` — 物理文件，只读
- `datasets/batches/` — 批次 manifest + 元数据，ingest 产物
- `datasets/sources/` — 逻辑数据源指针，Pipeline 只认这里

## 2.3 批次元数据 Schema（batch_meta.yaml）

```yaml
batch_id: batch_A
source_id: source_A              # 登记后对外名称：数据源 A
source_name: 数据源A             # 可读别名

ingested_at: "2026-08-20T10:30:00+08:00"
ingested_by: "zhangsan"

origin:
  type: vendor_delivery          # vendor_delivery | internal_record | crawl | ...
  vendor: "某供应商"
  delivery_id: "20260815_v1"
  delivery_date: "2026-08-15"
  description: "第一批客服外呼录音，8k pcm"
  tags:
    - customer_service
    - outbound

storage:
  root: data/batches/batch_A
  manifest: datasets/batches/batch_A/manifest.parquet

stats:
  file_count: 15000
  duration_hours: 20.3
  formats:
    wav: 14628
    pcm: 372
  broken: 17

status: registered               # ingesting | registered | archived
```

## 2.4 数据源注册 Schema（source_A.yaml）

批次 ingest 完成后，生成数据源注册文件。Pipeline 通过 `source_id` 引用，而不是硬编码路径：

```yaml
source_id: source_A
source_name: 数据源A
batch_id: batch_A

manifest: datasets/batches/batch_A/manifest.parquet
batch_meta: datasets/batches/batch_A/batch_meta.yaml

created_at: "2026-08-20T10:35:00+08:00"

# 默认音频入口 key（清洗流水线可覆盖）
default_audio_key: raw

# 可选：与其他数据源的关系
relations:
  - type: successor
    target: source_B
    note: "B 为 A 的补充交付"
```

## 2.5 批次入库 Operator 与 CLI

批次入库是 **Pipeline 之前** 的独立步骤，只做扫描 + 登记，不做清洗：

```bash
# 批次 A 入库
audio-data batch ingest \
    --batch-id batch_A \
    --source-id source_A \
    --source-name 数据源A \
    --input data/batches/batch_A \
    --origin vendor=某供应商,delivery_id=20260815_v1

# 输出：
# datasets/batches/batch_A/batch_meta.yaml
# datasets/batches/batch_A/manifest.parquet
# datasets/sources/source_A.yaml

# 批次 B 同理
audio-data batch ingest \
    --batch-id batch_B \
    --source-id source_B \
    --source-name 数据源B \
    --input data/batches/batch_B \
    --origin vendor=另一来源,delivery_id=20260818_v1
```

查看批次 / 数据源状态：

```bash
audio-data batch stats batch_A
audio-data source info source_A

# Files        15000
# Duration     20.3 h
# Ingested     2026-08-20 10:30
# Origin       某供应商 / 20260815_v1
# Status       registered
```

## 2.6 与数据清洗流水线的衔接

[`数据清洗流水线`](../03-流水线/数据清洗流水线.md) 中，**原「ingest 扫描 raw wav」步骤前移到批次资产层**。清洗流水线的输入从「原始 wav 目录」改为「已登记的数据源」：

**之前**（直接扫 raw）：

```yaml
input:
  manifest: datasets/manifests/raw_YYYYMMDD.parquet   # ingest 在清洗流水线内
```

**之后**（引用数据源）：

```yaml
name: data_cleaning_source_A

input:
  source: source_A                                    # → datasets/sources/source_A.yaml
  # 等价于 manifest: datasets/batches/batch_A/manifest.parquet

output:
  manifest: datasets/manifests/cleaned_source_A_20260820.parquet

pipeline:
  - { operator: audio.pcm_to_wav, params: { sample_rate: 8000 } }
  - { operator: audio.resample, params: { sample_rate: 16000, input_audio_key: raw, output_audio_key: resampled_16k } }
  - { operator: quality.probe, params: { input_audio_key: resampled_16k } }
  - { operator: quality.filter, params: { expr: "label_broken != True and duration > 0", label_key: audio_pass } }
```

执行：

```bash
audio-data pipeline run pipelines/clean_source_A.yaml
audio-data pipeline run pipelines/clean_source_B.yaml
```

清洗产物的 manifest 中应 **继承批次血缘**，便于追溯：

```json
{
  "id": "000001",
  "batch_id": "batch_A",
  "source_id": "source_A",
  "ingested_at": "2026-08-20T10:30:00+08:00",
  "origin": { "vendor": "某供应商", "delivery_id": "20260815_v1" },
  "audio": { "raw": "data/batches/batch_A/000001.wav", "resampled_16k": "..." },
  "labels": { "audio_pass": true }
}
```

## 2.7 多批次常见操作

```bash
# 分别清洗
audio-data pipeline run pipelines/clean_source_A.yaml
audio-data pipeline run pipelines/clean_source_B.yaml

# 合并清洗结果做对比实验
audio-data source merge \
    --sources source_A,source_B \
    --output datasets/manifests/cleaned_AB_merged.parquet

# 跨源查重（sha256 碰撞）
audio-data source dedupe --sources source_A,source_B

# 跨源统计
audio-data source compare source_A source_B
```

## 2.8 与 DVC / 版本的关系

批次资产适合作为 **DVC 版本化的最小单元**：

```text
batch_A manifest  →  dvc add datasets/batches/batch_A/
cleaned_source_A  →  dvc add datasets/manifests/cleaned_source_A_*.parquet
```

这样「数据源 A 在第 3 版清洗配置下产出的 cleaned manifest」可以完整复现，而不需要重新扫描原始 wav。

---

# 三、最核心的东西：不要通过“文件夹”管理数据，而要通过 Manifest

例如原始数据：

```text
data/raw/
├── 000001.wav
├── 000002.wav
├── 000003.pcm
└── ...
```

不要认为：

```text
data/denoise/
data/qwen/
data/sensevoice/
data/augment/
```

这些文件夹本身就是你的数据集。

真正的数据集应该是：

```json
{
  "id": "000001",
  "source_path": "data/raw/000001.wav",
  "sha256": "89f2...",
  "sample_rate": 8000,
  "channels": 1,
  "duration": 4.72,

  "audio": {
    "raw": "data/raw/000001.wav",
    "resampled_16k": "data/derived/resample/000001.wav",
    "denoised": "data/derived/denoise/000001.wav"
  },

  "transcripts": {
    "sensevoice": {
      "text": "我不需要",
      "model": "sensevoice-small",
      "version": "20260820"
    },
    "qwen_asr": {
      "text": "我不需要",
      "model": "qwen3-asr",
      "version": "xxx"
    }
  },

  "quality": {
    "snr": 12.7,
    "speech_ratio": 0.83
  },

  "labels": {
    "gold_text": "我不需要",
    "badcase": "negative_intent"
  }
}
```

一条音频就是一个逻辑 Sample。

于是你以后问：

> “找所有 Qwen-ASR 转写成‘需要’，但是 SenseVoice 转写成‘不需要’的数据。”

就不需要在十几个文件夹之间找了。

直接：

```python
df[
    (df.qwen_text == "需要") &
    (df.sensevoice_text == "不需要")
]
```

NeMo Curator 现在的 audio manifest 设计基本就是这个思想，官方也允许在 `audio_filepath/text/duration` 之外增加 language、speaker、sample rate 和自定义 metadata。([NVIDIA Docs][2])

我建议你实际落地时：

**交换格式：JSONL**

**分析格式：Parquet**

也就是：

```text
manifest.jsonl
manifest.parquet
```

两者可以互转。

---

# 四、第二个核心：所有处理能力统一做成 Operator

不要：

```python
python denoise.py xxx
python qwen.py xxx
python sensevoice.py xxx
```

而是定义统一接口：

```python
class AudioOperator:

    name: str
    version: str

    def process(self, sample, config):
        ...
        return result
```

然后实现：

```text
operators/

audio/
    pcm_to_wav.py
    resample.py
    normalize.py
    trim_silence.py
    vad.py
    denoise.py

augment/
    add_noise.py
    speed_perturb.py
    volume_perturb.py
    codec_simulation.py

asr/
    sensevoice.py
    qwen_asr.py
    whisper.py
    remote_api.py

quality/
    audio_info.py
    snr.py
    cer.py
    transcript_diff.py
    filter.py
```

比如：

```python
class QwenASROperator(AudioOperator):

    name = "qwen_asr"

    def process(self, sample, config):
        audio = sample.audio_path

        text = self.client.transcribe(audio)

        return {
            "text": text,
            "model": config.model,
        }
```

SenseVoice：

```python
class SenseVoiceOperator(AudioOperator):

    name = "sensevoice"

    def process(self, sample, config):
        ...
```

这样上层 Pipeline 根本不关心 Qwen 和 SenseVoice 怎么调用。

它只知道：

```text
audio
  ↓
qwen_asr
  ↓
transcript
```

这叫 **Adapter / Plugin architecture**。

以后哪怕增加：

```text
FunASR
Paraformer
Whisper
腾讯ASR API
阿里ASR API
内部ASR服务
```

Pipeline 都不用改。

---

# 五、第三个核心：Pipeline 应该用配置描述，而不是写死

例如：

```yaml
name: qwen_denoise_experiment

input:
  source: source_A                    # 或 manifest: datasets/batches/batch_A/manifest.parquet

pipeline:

  - name: pcm_convert
    operator: audio.pcm_to_wav
    params:
      sample_rate: 8000

  - name: resample
    operator: audio.resample
    params:
      sample_rate: 16000

  - name: denoise
    operator: audio.denoise
    params:
      model: dns

  - name: qwen_asr
    operator: asr.qwen
    params:
      model: qwen3-asr

  - name: sensevoice
    operator: asr.sensevoice

  - name: compare
    operator: quality.transcript_diff
```

执行：

```bash
audio-pipeline run pipelines/qwen_denoise.yaml
```

这样你会逐渐拥有：

```text
pipelines/

01_raw_to_wav.yaml

02_sensevoice_baseline.yaml

03_qwen_baseline.yaml

04_denoise_qwen.yaml

05_denoise_sensevoice.yaml

06_noise_augmentation.yaml

07_training_dataset.yaml
```

这比维护十几个 shell/python 脚本强很多。

---

# 六、一定要做“断点续跑 + 幂等 + Cache”

这是大量 WAV 最容易踩坑的地方。

比如你有：

```text
100000 个 wav
```

跑 Qwen-ASR 到第：

```text
78324
```

服务器挂了。

第二天绝对不能重新跑：

```text
1 → 100000
```

而应该知道：

```text
000001 completed
000002 completed
...
078324 completed
078325 pending
```

Operator 的 cache key 可以设计成：

```text
hash(
    input_audio_sha256
    + operator_name
    + operator_version
    + params
    + model_version
)
```

例如：

```text
wav hash
    +
qwen_asr
    +
qwen3-asr-v1
    +
temperature=0
```

没变：

```text
CACHE HIT
```

直接跳过。

模型换了：

```text
qwen3-asr-v2
```

自动重新计算。

这也是为什么 Prefect 对你的体系比较合适：它的 Task 原生就是可 retry、cache、并发和状态追踪的工作单元。([Prefect][3])

---

# 七、目录我建议你直接这样定

这是我比较推荐你实际建仓库时使用的结构：

```text
audio-data-engine/
│
├── configs/
│   ├── asr/
│   │   ├── qwen_asr.yaml
│   │   └── sensevoice.yaml
│   ├── augmentation/
│   └── denoise/
│
├── pipelines/
│   ├── ingest_batch.yaml           # 批次入库
│   ├── clean_source_A.yaml         # 清洗：输入数据源 A
│   ├── clean_source_B.yaml
│   ├── baseline_qwen.yaml
│   ├── baseline_sensevoice.yaml
│   ├── denoise_qwen.yaml
│   └── build_training_set.yaml
│
├── src/audio_engine/
│
│   ├── core/
│   │   ├── sample.py
│   │   ├── manifest.py
│   │   ├── operator.py
│   │   ├── pipeline.py
│   │   ├── batch.py                # 批次资产 / 数据源
│   │   └── registry.py
│   │
│   ├── operators/
│   │   ├── batch/
│   │   │   └── ingest.py           # 扫描物理文件 → manifest
│   │   ├── audio/
│   │   │   ├── pcm.py
│   │   │   ├── resample.py
│   │   │   ├── vad.py
│   │   │   └── denoise.py
│   │   │
│   │   ├── augmentation/
│   │   │   ├── noise.py
│   │   │   ├── speed.py
│   │   │   └── volume.py
│   │   │
│   │   ├── asr/
│   │   │   ├── qwen.py
│   │   │   ├── sensevoice.py
│   │   │   └── base.py
│   │   │
│   │   └── quality/
│   │       ├── cer.py
│   │       ├── snr.py
│   │       └── filter.py
│   │
│   └── cli/
│       └── main.py
│
├── datasets/
│   ├── batches/                    # 批次资产
│   │   ├── batch_A/
│   │   │   ├── batch_meta.yaml
│   │   │   └── manifest.parquet
│   │   └── batch_B/
│   ├── sources/                    # 数据源注册
│   │   ├── source_A.yaml
│   │   └── source_B.yaml
│   ├── manifests/                  # 流水线产出 manifest
│   └── README.md
│
├── data/
│   ├── batches/                    # 各批次物理文件（raw，永不修改）
│   │   ├── batch_A/
│   │   └── batch_B/
│   ├── derived/                    # 按 source_id 分子目录
│   │   ├── source_A/
│   │   └── source_B/
│   ├── cache/
│   └── exports/
│
├── runs/
│   └── 20260820_113012/
│       ├── config.yaml
│       ├── manifest.parquet
│       ├── metrics.json
│       └── run.log
│
├── tests/
│
├── pyproject.toml
└── README.md
```

其中最关键的是区分：

```text
batches（物理 + 批次 manifest）
sources（逻辑数据源，Pipeline 入口）
derived（按 source_id 隔离的处理产物）
cache（可删）
exports（最终交付）
```

### batches / raw

永不修改。

```text
data/batches/batch_A/000001.wav
```

### batches manifest

批次登记产物，含入库时间与来源。

```text
datasets/batches/batch_A/manifest.parquet
datasets/batches/batch_A/batch_meta.yaml
```

### sources

数据源注册，Pipeline 通过 `input.source` 引用。

```text
datasets/sources/source_A.yaml
```

### derived

处理产生的正式中间资产，按数据源隔离。

```text
data/derived/source_A/
    resample_16k/
    denoise/
```

---

# 八、你以后实际上会出现“分叉 DAG”

你的任务天然不是线性 Pipeline：

```text
                           ┌─ SenseVoice ───────┐
                           │                    │
RAW → Decode → Resample ───┼─ Qwen-ASR ─────────┼→ Compare
                           │                    │
                           └─ Denoise → Qwen ───┘
```

再例如：

```text
                         ┌→ 原始 → Qwen
                         │
RAW → VAD → Resample ────┼→ 降噪 → Qwen
                         │
                         ├→ 加噪 → Qwen
                         │
                         └→ speed perturb → Qwen
```

这时候你就会真正看到工程体系的收益：

你不是复制四份 WAV。

而是维护：

```text
sample
+
artifact lineage
+
DAG
```

---

# 九、数据血缘一定要保存

假设最终有：

```text
000023_aug_noise.wav
```

半年之后你一定会问：

> 这个音频到底怎么来的？

系统应该直接告诉你：

```text
000023.wav
│
├─ resample
│    sample_rate = 16000
│    version = 1.2
│
├─ add_noise
│    noise = cafe_003.wav
│    snr = 5dB
│    seed = 381928
│
└─ output
     000023_aug_noise.wav
```

也就是说 augmentation 甚至应该记录：

```json
{
  "operator": "add_noise",
  "params": {
    "snr": 5,
    "noise_id": "cafe_003",
    "seed": 381928
  }
}
```

否则你的增强数据**不可复现**。

---

# 十、你可以参考的三个体系

### 1. NeMo Curator——最值得参考

这个跟你现在描述的需求非常接近。

最新 NeMo Curator Audio Curation 已经提供：

```text
Load
↓
ASR inference
↓
quality assessment
↓
VAD
↓
quality filtering
↓
speaker separation
↓
export
```

其中质量过滤 pipeline 已经包含 mono conversion、VAD、band filtering、UTMOS/SIGMOS、speaker separation 等独立 stage。([NVIDIA Docs][4])

官方甚至有完整 example：

```text
tutorials/audio/fleurs/

README.md
pipeline.py
pipeline.yaml
run.py
```

非常建议你直接研究这个目录。([NVIDIA Docs][5])

[NeMo Curator Audio Curation](https://docs.nvidia.com/nemo/curator/latest/curate-audio?utm_source=chatgpt.com)

你的系统不一定真的使用 NeMo Curator，但**架构可以大量模仿它**。

---

### 2. Lhotse——重点学习数据模型

Lhotse 最值得你学习的是：

```text
Recording
Supervision
Cut
CutSet
```

它不是简单认为：

```text
一个 wav = 一条数据
```

而是：

```text
Recording
   ↓
Cut
   ↓
Supervision
```

所以非常适合：

```text
长音频
VAD切段
说话人
时间戳
文本标注
数据增强
训练数据
```

而且 CutSet 本身支持 pad、mix、concatenate、augment 等操作。([Lhotse][6])

[Lhotse 文档](https://lhotse.readthedocs.io/en/latest/?utm_source=chatgpt.com)

如果以后你的数据从“几秒一个 wav”发展成：

```text
原始通话
↓
VAD
↓
客户片段
↓
ASR
↓
标注
```

Lhotse 这种模型尤其值得借鉴。

---

### 3. Prefect——学习任务编排

你的大量：

```text
wav × operator
```

本质是 Map：

```python
for wav in dataset:
    qwen_asr(wav)
```

Prefect 非常适合：

```text
并发
重试
超时
cache
状态管理
失败任务重跑
```

而且还是 Python-first，不需要一开始就搭 Airflow 那么重。([Prefect][3])

---

# 十一、DVC 应该放在哪里

DVC 可以用，但不要让 DVC 承担所有功能。

它适合：

```text
Git
 │
代码版本
 │
DVC
 │
数据版本
```

例如：

```text
dataset_v1
dataset_v2
dataset_v3
```

以及：

```text
RAW
 ↓
clean
 ↓
train
```

这种 reproducible pipeline。

DVC Pipeline 根据 stage 的 `deps/outs` 构建 DAG，并用 `dvc.lock` 保存实际依赖版本。([DVC][7])

但我不会建议你拿 DVC 去调度：

```text
10万 WAV × 3 ASR Service
```

这个应该交给 Prefect / Ray / 自研 Worker。

所以：

```text
DVC
    数据集版本

Prefect
    计算任务调度

Manifest
    数据状态

Operator
    数据处理能力
```

四者职责不同。

---

# 十二、我最推荐你的技术组合

如果让我现在直接替你定技术栈，我会用：

```text
Python 3.10+

音频
    soundfile
    torchaudio
    ffmpeg

数据描述
    Pydantic

Manifest
    JSONL
    Parquet
    PyArrow / Polars

配置
    YAML
    Hydra（可选）

Operator
    自己写

Pipeline
    Prefect

CLI
    Typer

日志
    loguru / logging

数据版本
    DVC（第二阶段再加）

训练数据抽象
    借鉴 / 部分使用 Lhotse
```

**不建议第一版直接上：**

```text
Airflow
Kafka
Spark
Kubernetes
MinIO
MLflow
Dagster
Ray
```

不是这些东西不好，而是你现在的核心矛盾还不是分布式计算，而是：

> **把“音频 → 多种加工 → 多模型转写 → 数据筛选 → 训练集”这条链路标准化。**

先把这个抽象正确。

---

# 十三、最终你希望达到的使用体验

例如你拿到两批 WAV，分别来自不同供应商。

**第一步：批次入库，形成数据源**

```bash
# 批次 A → 数据源 A
audio-data batch ingest \
    --batch-id batch_A \
    --source-id source_A \
    --source-name 数据源A \
    --input data/batches/batch_A \
    --origin vendor=供应商甲,delivery_id=20260815_v1

# 批次 B → 数据源 B
audio-data batch ingest \
    --batch-id batch_B \
    --source-id source_B \
    --source-name 数据源B \
    --input data/batches/batch_B \
    --origin vendor=供应商乙,delivery_id=20260818_v1
```

查看：

```bash
audio-data source info source_A
audio-data source info source_B
```

**第二步：对数据源跑清洗流水线**

```bash
audio-data pipeline run pipelines/clean_source_A.yaml
audio-data pipeline run pipelines/clean_source_B.yaml
```

**第三步：对清洗结果跑标注流水线**

```bash
audio-data pipeline run pipelines/label_source_A.yaml
```

查看统计：

```bash
audio-data stats cleaned_source_A
```

得到：

```text
Source       数据源A (batch_A)
Files        15000
Duration     20.3 h
Ingested     2026-08-20 10:30
Origin       供应商甲 / 20260815_v1
Cleaned      14203 pass / 797 filtered
```

跑 SenseVoice（基于数据源，而非 raw 目录）：

```bash
audio-data run sensevoice \
    --source source_A
```

跑 Qwen：

```bash
audio-data run qwen_asr \
    --source source_A
```

只跑噪音 badcase：

```bash
audio-data run qwen_asr \
    --source source_A \
    --filter "badcase == 'noise'"
```

跨源对比：

```bash
audio-data source compare source_A source_B
```

最后：

```bash
audio-data compare \
    qwen_asr \
    sensevoice \
    --source source_A
```

输出：

```text
runs/20260820_qwen_vs_sensevoice/

manifest.parquet
summary.json
badcases.xlsx
```

到了这个阶段，你拥有的就不再是一个“ASR 数据处理脚本库”，而是一个真正意义上的：

> **Audio Data Engine / Speech Data Engineering Platform**

---

对于你的实际场景，我尤其建议**先不要把 NeMo Curator 整套搬进来**。最合理的路线是借它的 **Manifest + Stage + Pipeline** 思想，借 Lhotse 的 **Recording/Cut/Supervision** 思想，然后自己做一层非常薄的 `audio_engine`。因为你还需要同时兼容 SenseVoice、Qwen-ASR、内部服务、PCM、各种业务标签，这些恰好是自定义 Operator 最有价值的地方。([NVIDIA Docs][8])

如果继续往下做，我建议下一步直接确定 **`audio-data-engine` V1 的工程目录、Manifest Schema、Operator 接口、Pipeline YAML Schema 和第一批 10 个 Operator**。这一步确定好，后面基本就是往框架里不断加积木。

[1]: https://docs.nvidia.com/nemo/speech/nightly/asr/datasets.html?utm_source=chatgpt.com "Datasets — NeMo-Speech"
[2]: https://docs.nvidia.com/nemo/curator/v26.04/curate-audio/load-data/custom-manifests?utm_source=chatgpt.com "Custom Manifests | NeMo Curator"
[3]: https://docs.prefect.io/v3/concepts/tasks?utm_source=chatgpt.com "Tasks"
[4]: https://docs.nvidia.com/nemo/curator/curate-audio/process-data/quality-filtering?utm_source=chatgpt.com "Overview | NeMo Curator"
[5]: https://docs.nvidia.com/nemo/curator/latest/curate-audio/tutorials/beginner?utm_source=chatgpt.com "Beginner Tutorial | NeMo Curator"
[6]: https://lhotse.readthedocs.io/en/latest/cuts.html?utm_source=chatgpt.com "Cuts — lhotse 2.0.0a3.dev documentation"
[7]: https://dvc.org/blog/end-to-end-computer-vision-api-part-1-data-versioning-and-ml-pipelines/?utm_source=chatgpt.com "End-to-End Computer Vision API, Part 1: Data Versioning and ML Pipelines – DVC"
[8]: https://docs.nvidia.com/nemo/curator/curate-audio/tutorials/beginner?utm_source=chatgpt.com "Beginner Tutorial | NeMo Curator"
