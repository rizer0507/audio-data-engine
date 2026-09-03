# Kimi-Audio 本地高并发识别流水线（兼容模式）

Kimi 的标准入口 `asr.kimi` / `asr.kimi_batch` 使用 vLLM HTTP；本地权重直载由
`asr.kimi_audio` / `asr.kimi_audio_batch` 作为**独立**入口提供。本页只说明本地模式；
vLLM 的探针与 Parquet 批量流程见 `docs/流水线改进/构造kimi-audio识别流水线.md`。

## 安装与配置

```bash
pip install -e '.[kimi-audio]'
export KIMI_AUDIO_MODEL_PATH=/data2/data-cp/models/kimi-audio
```

模型路径也可写在 `configs/asr/kimi_audio.yaml`。环境变量优先于 operator 参数和 YAML。
`load_detokenizer: false` 可降低只做 ASR 时的显存占用。`batch_size` 默认 `1`：官方
`generate()` 一次一条。

## 探针

```bash
python scripts/probe_kimi_audio.py /path/to/short.wav
python scripts/probe_kimi_audio.py /path/to/wav_folder --limit 8
python scripts/probe_kimi_audio.py /path/to/wav_folder --recursive --limit 8
```

stdout 每条音频一条 JSON，`text` 必须非空。退出码：`0` 全部非空，`1` 配置/模型/推理错误，
`2` 至少一条空转写。两个探针都通过后再跑 Parquet。

也可用已有 Manifest：

```bash
audio-data run kimi_audio --dataset <manifest> --config configs/asr/kimi_audio.yaml
```

## 高并发设计（两卡 A800，一次一条，多进程）

```bash
audio-data pipeline run pipelines/kimi_audio_asr_batch.yaml --source-name mt3000
```

流水线按音频时长均衡切成 2 片，两个 shard 进程分别绑定 `CUDA_VISIBLE_DEVICES=0` 和 `1`。
每个进程只加载一次模型，然后逐条 `generate`。并发发生在 GPU 进程之间，不要用
`execution.workers` 在同一进程里抢同一张卡。

部署时必须按机器修改 `sharding.gpus`：

- `parallel_shards = len(gpus) * instances_per_gpu`（默认 2 × 1 = 2）；
- 7B 在 A800 上先保持 `instances_per_gpu: 1`；
- `batch_size` / `inference_threads` 保持 `1`；
- 共享集群不要占用 Qwen / SenseVoice 正在使用的卡。

坏音频只会标记自身失败。缓存键包含模型路径、版本、设备、提示词和 `generation_kwargs`。
输出为 `datasets/manifests/kimi_audio_asr_<source>.parquet`，文本位于 `transcripts.kimi.text`。
聚合 join 名是 `--join-manifest kimi_audio`。
