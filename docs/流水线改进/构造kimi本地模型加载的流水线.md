# Kimi-Audio 本地模型加载识别流水线

按 SenseVoice 本地加载模式实现，与 vLLM HTTP 版 `kimi_asr_batch.yaml` 独立。

## 实现位置

| 组件 | 路径 |
|------|------|
| 批处理 operator | `asr.kimi_audio_batch` → `src/audio_engine/operators/asr/kimi_audio.py` |
| 单条 operator | `asr.kimi_audio` |
| 模型配置 | `configs/asr/kimi_audio.yaml` |
| 流水线 | `pipelines/kimi_audio_asr_batch.yaml` |

## 并发模型

1. **进程级并行**：manifest 按时长均衡切片，每个 shard 绑定一个 GPU slot，进程内只加载一次模型。
2. **模型级批处理**：按 `batch_size` 分组；若 wrapper 提供 `transcribe_batch` 则一次送入整批，否则回退为逐条 `generate`。
3. **逐样本容错**：整批失败时自动拆成单条重试，坏 WAV 只标记自身失败。

## 环境与配置

```bash
pip install 'audio-data-engine[kimi-audio]'
export KIMI_AUDIO_MODEL_PATH=/data/models/Kimi-Audio-7B-Instruct
```

可调参数见 `configs/asr/kimi_audio.yaml`：

- `batch_size`：每批文件数（官方 generate 接口下实际为顺序推理分组大小）
- `inference_threads`：generate-only 回退时的线程数，共享 CUDA 模型时保持 `1`
- `load_detokenizer: false`：ASR 仅需文本，关闭 detokenizer 节省显存
- `generation_kwargs`：传给 `model.generate` 的采样参数

## 执行命令

```bash
audio-data pipeline run pipelines/kimi_audio_asr_batch.yaml --source-name mt3000
# → cleaned_mt3000 → kimi_audio_asr_mt3000.parquet
```

产物中转写位于 `transcripts.kimi.text`。聚合时可：

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
| 配置 | `configs/asr/kimi.yaml` | `configs/asr/kimi_audio.yaml` |
| 产物 | `kimi_asr_<name>.parquet` | `kimi_audio_asr_<name>.parquet` |
| 环境变量 | `KIMI_ASR_API_BASE` | `KIMI_AUDIO_MODEL_PATH` |
