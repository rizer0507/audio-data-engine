服务器上模型权重路径：/data2/data-cp/zcl/models/Qwen3-ASR-1___7B
vllm服务无法启动，踩了一个坑：
很好，这个结果说明关键链路已经打通：

```text
qwen_asr
    ↓
Qwen3ASRModel.from_pretrained()
    ↓
Qwen3-ASR-1.7B 权重
    ↓
加载成功
```

因此可以确认：

1. **模型文件完整**

   * safetensors 权重正常
   * tokenizer / processor 配置正常

2. **qwen_asr 环境正常**

   * Qwen3-ASR 官方 Python 包可以识别 `qwen3_asr`

3. **之前的 vLLM 报错原因已经明确**

   * 不是模型损坏
   * 不是 transformers 版本低
   * 而是 vLLM 的加载路径走的是：

```text
vllm
 ↓
transformers.AutoConfig
 ↓
config.json
 ↓
model_type=qwen3_asr
 ↓
KeyError
```

它没有经过：

```text
qwen_asr
 ↓
Qwen3ASRModel
```

这条官方链路。

---

## 下一步建议不要继续硬怼 vLLM

对于 Qwen3-ASR-1.7B，推荐先验证官方推理。

例如：

```python
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "/data2/data-cp/zcl/models/Qwen3-ASR-1___7B"
)

result = model.transcribe(
    audio="/path/to/test.wav"
)

print(result)
```

具体 API 需要看你安装的 `qwen_asr` 版本，可以查看：

```bash
python - <<'PY'
from qwen_asr import Qwen3ASRModel
import inspect

print(inspect.signature(Qwen3ASRModel))
print(dir(Qwen3ASRModel))
PY
```

---

## 关于部署方式

你原来的目标：

> 启动一个服务，让其他程序调用 ASR

更推荐：

```
客户端
  |
  | wav
  ↓
FastAPI
  |
  ↓
qwen_asr.Qwen3ASRModel
  |
  ↓
文本
```

结构：

```
                 GPU
                  |
FastAPI Worker ---- Qwen3ASRModel
                  |
              Qwen3-ASR-1.7B
```

而不是：

```
vLLM OpenAI Server
        |
        ↓
Qwen3-ASR
```

原因：

vLLM 的优势主要是：

* continuous batching
* KV cache
* 多请求 LLM 推理

而 ASR 服务核心瓶颈通常是：

* 音频 encoder
* batch 音频处理
* GPU kernel

两者优化目标不同。

---

## 如果你后续想做生产级 ASR 服务

你的场景（审批电话、人声关键词提取）更适合：

```
                wav
                 |
                 ↓
              VAD
                 |
                 ↓
        Qwen3-ASR Batch Inference
                 |
                 ↓
             文本结果
                 |
                 ↓
          关键词/意愿判断
```

而不是单纯 chat server。

---

## 现在建议你做三个验证

### 1. 单条 wav 推理

确认模型真的能识别：

```text
wav → text
```

### 2. batch 推理速度

测试：

* batch=16
* batch=32
* batch=64

看四张 A800 的利用率、显存占用和整体 samples/s。

### 3. 再决定服务框架

如果：

* 并发低（几十路以内）
  → FastAPI + batch queue 足够

* 高并发电话流
  → Triton / 自建 batching engine 更合适

---

你现在已经绕过了最容易踩坑的地方。下一步重点不是 vLLM，而是确认 **Qwen3-ASR 在四张 A800 80G 环境下的最优 batch 和整体吞吐**。这和你之前做 SenseVoice/Qwen-ASR 替换的业务链路更相关。


现在我需要一个新的能够支持高并发处理（毕竟我有近十万多条wav数据要处理）的qwen-asr服务启动识别的流水线

---

## 工程落地（2026-08-21）

已按 `流水线构建统一流程.md` 接入统一执行链：

```text
PipelineRunner
  → 8 个独立进程（4 张 A800 × 每卡 2 个模型实例）
  → 每个进程运行 asr.qwen_batch（默认 batch=32）
  → 逐样本 transcript / lineage / status / cache
  → checkpoint / manifest / metrics
```

实现不依赖 vLLM。模型通过官方 `qwen_asr.Qwen3ASRModel.from_pretrained()` 加载，
默认使用服务器路径 `/data2/data-cp/zcl/models/Qwen3-ASR-1___7B`。也可通过环境变量
`QWEN_ASR_MODEL_PATH` 或流水线参数 `model_path` 覆盖。

### 安装

```bash
pip install -e ".[asr]"
```

### 四张 A800 80G 并发运行

先完成数据清洗，得到包含 `resampled_16k` 音频 key 的 manifest：

配置默认使用 `bfloat16` 和 `batch_size=32`。A800 80G 单卡可同时驻留多个 1.7B
实例。推荐 **4 卡 × 每卡 2 实例 = 8 进程**；分片数必须等于并发实例数，这样每个
进程只加载一次模型：

```bash
audio-data manifest shard datasets/manifests/cleaned_source_A.parquet \
  --shards 8 \
  --strategy duration-balanced \
  --output-dir datasets/shards/qwen_source_A

audio-data pipeline run-shards pipelines/qwen_asr_batch.yaml \
  --shard-dir datasets/shards/qwen_source_A \
  --parallel-shards 8 \
  --gpus 0,1,2,3 \
  --instances-per-gpu 2 \
  --run-root runs/qwen_source_A

audio-data manifest merge "runs/qwen_source_A/shard-*.parquet" \
  --output datasets/manifests/qwen_asr_source_A.parquet \
  --expected-shards 8
```

`run-shards` 按 `--instances-per-gpu` 把进程轮转到各 GPU，上限为
`GPU 数 × instances-per-gpu`。`nvidia-smi` 上每张卡应看到 2 个 python 进程。
若 GPU 利用率仍低，可试每卡 3 实例（12 分片）；若 OOM，把 `batch_size` 降到 16。