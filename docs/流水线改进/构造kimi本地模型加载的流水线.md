# Kimi-Audio 本地模型加载识别流水线

按 SenseVoice 本地加载模式实现，与 vLLM HTTP 版 `kimi_asr_batch.yaml` **独立**。
vLLM 链路（`asr.kimi_batch` / `scripts/probe_kimi_vllm.py` / `pipelines/kimi_asr_batch.yaml`）保持不变。

## 实现位置

| 组件 | 路径 |
|------|------|
| 探针 | `scripts/probe_kimi_audio.py`（单 WAV 或 WAV 文件夹） |
| 批处理 operator | `asr.kimi_audio_batch` → `src/audio_engine/operators/asr/kimi_audio.py` |
| 单条 operator | `asr.kimi_audio`（同样写入 `transcripts.kimi`） |
| 模型配置 | `configs/asr/kimi_audio.yaml` |
| 流水线 | `pipelines/kimi_audio_asr_batch.yaml` |

## 并发模型（两卡 A800）

官方 `KimiAudio.generate` **一次一条**。吞吐靠多进程，不靠进程内 batch / 多线程共享 CUDA 模型：

1. **进程级并行**：manifest 按时长均衡切成 `shards` 片；`parallel_shards = len(gpus) * instances_per_gpu`。
2. 每个 shard 进程通过 `CUDA_VISIBLE_DEVICES` 绑定一张卡，进程内只加载一次模型，然后逐条 `generate`。
3. 默认两卡：`gpus: [0, 1]`，`instances_per_gpu: 1`，`batch_size: 1`。共享集群请改成两张空闲卡，不要占用 Qwen / SenseVoice 正在用的 GPU。
4. **逐样本容错**：坏 WAV 只标记自身失败，不拖垮整片。

## 环境与配置

```bash
pip install -e '.[kimi-audio]'
export KIMI_AUDIO_MODEL_PATH=/data2/data-cp/models/kimi-audio
```

可调参数见 `configs/asr/kimi_audio.yaml`：

- `batch_size: 1`：进程内一次一条（不要为了吞吐去加大；加大也不会变成真正的 GPU batch）
- `inference_threads: 1`：共享 CUDA 模型时必须为 1
- `load_detokenizer: false`：ASR 仅需文本，关闭 detokenizer 节省显存
- `generation_kwargs`：传给 `model.generate` 的采样参数

## 探针（必须先过，再跑 Parquet）

```bash
# 单条短音频。stdout 每条一条 JSON，text 必须非空
python scripts/probe_kimi_audio.py /path/to/short.wav

# WAV 文件夹（当前目录；子目录加 --recursive）
python scripts/probe_kimi_audio.py /path/to/wav_folder --limit 8
python scripts/probe_kimi_audio.py /path/to/wav_folder --recursive --limit 8

# 覆盖模型路径 / 设备
python scripts/probe_kimi_audio.py /path/to/short.wav \
  --model-path /data2/data-cp/models/kimi-audio --device cuda
```

探针退出码：`0`=全部非空，`1`=输入/模型/推理错误，`2`=至少一条空转写。
单文件和文件夹探针都为 0 后，才跑整个 `cleaned_<source>.parquet`。

## 执行命令

```bash
audio-data pipeline run pipelines/kimi_audio_asr_batch.yaml --source-name mt3000
# → cleaned_mt3000 → kimi_audio_asr_mt3000.parquet
```

产物中转写位于 `transcripts.kimi.text`。聚合时 join 名是 `kimi_audio`（文件名），不要和 vLLM 的 `kimi` 混用：

```bash
audio-data pipeline run pipelines/multi_asr_aggregate.yaml \
  --source-name mt3000 \
  --join-manifest sensevoice \
  --join-manifest kimi_audio
```

## 与 vLLM 版对比

| | `kimi_asr_batch` (vLLM) | `kimi_audio_asr_batch` (本地) |
|--|-------------------------|-------------------------------|
| operator | `asr.kimi_batch` | `asr.kimi_audio_batch` |
| 依赖 | vLLM 服务 + HTTP | kimia_infer 本地加载 |
| 探针 | `scripts/probe_kimi_vllm.py` | `scripts/probe_kimi_audio.py` |
| 配置 | `configs/asr/kimi.yaml` | `configs/asr/kimi_audio.yaml` |
| 产物 | `kimi_asr_<name>.parquet` | `kimi_audio_asr_<name>.parquet` |
| 环境变量 | `KIMI_ASR_API_BASE` | `KIMI_AUDIO_MODEL_PATH` |
| 并发 | HTTP `concurrency` × shards | 两卡 A800 × 每卡 1 进程，一次一条 |
