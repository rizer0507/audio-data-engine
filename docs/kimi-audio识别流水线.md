# Kimi-Audio 高并发识别流水线

## 现状与实现位置

此前仓库只在输出命名和多模型聚合中预留了 `kimi` 名称，没有 Kimi-Audio
推理 operator 或可执行 pipeline。本次补齐：

- `asr.kimi_batch`：批推理、逐样本缓存、坏音频隔离、断点友好的批处理 operator；
- `configs/asr/kimi_audio.yaml`：模型、提示词与生成参数；
- `pipelines/kimi_asr_batch.yaml`：时长均衡切片、多 GPU 多进程执行与最终合并；
- `KIMI_AUDIO_MODEL_PATH`：服务器本地权重路径覆盖。

## 并发模型

高吞吐由两层组成：

1. **进程级并行**：manifest 先按时长均衡切片，每个并行 shard 绑定一个 GPU
   slot，并独立加载一次模型。这避免多线程共享 CUDA 模型以及短 shard 等待长音频 shard。
2. **模型级批处理**：每个进程将待识别数据按 `batch_size` 分组。若部署 wrapper
   提供 `transcribe_batch` 或支持列表输入的 `transcribe`，一次送入整批；官方
   `build_prompt`/`generate` 接口则兼容回退为逐条生成。

`checkpoint_every: 500` 会定期落盘。进程中断后重跑同一命令时，operator 的逐样本
cache 会跳过已成功的数据；指定原 run 目录还可以用 pipeline 的 resume 能力恢复 checkpoint。
一个批次失败时会自动拆成单条重试，因此损坏 WAV 只标记自身失败，不拖垮整个 shard。

## 环境和配置

按 Kimi-Audio 官方仓库安装 `kimia_infer` 及其匹配的 PyTorch/CUDA 依赖，然后设置：

```bash
export KIMI_AUDIO_MODEL_PATH=/data/models/Kimi-Audio-7B-Instruct
```

先根据单实例显存测量调整 `configs/asr/kimi_audio.yaml`：

- `batch_size`：原生 batch wrapper 每次接收的文件数，建议从 1/2/4/8 梯度压测；
- `inference_threads`：仅 generate-only wrapper 使用。共享单个 CUDA 模型时保持 `1`；
- `generation_kwargs`：确定性识别默认 temperature 为 0；
- `prompt`：要求仅输出转写文本，避免解释性回答污染训练语料。

再调整 pipeline 的 `gpus`、`parallel_shards`、`instances_per_gpu`。必须满足：

```text
parallel_shards <= len(gpus) * instances_per_gpu
```

Kimi-Audio 7B 通常先从每卡一个实例开始，确认显存余量后才增加每卡实例数。

## 十万条数据执行

pipeline YAML 已包含 sharding，常规执行一条命令即可完成 split、并发 run 和 merge：

```bash
audio-data pipeline run pipelines/kimi_asr_batch.yaml --source-name mt3000
```

产物为 `datasets/manifests/kimi_asr_mt3000.parquet`，转写位于
`transcripts.kimi.text`。建议先对小 manifest 做真实模型冒烟测试，查看 run 目录中的
`run.log`、`metrics.json` 和失败样本，然后再放大到全量。

需要手动控制时可沿用 Qwen 的三段式操作：manifest split、pipeline run-shards、
manifest merge。不要在 pipeline 内把 `execution.workers` 调大来共享同一 GPU 模型；GPU
并发应由 shard 进程和 GPU slot 显式控制。
