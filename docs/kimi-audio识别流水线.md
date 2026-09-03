# Kimi-Audio 本地高并发识别流水线

Kimi 的标准入口已改为**直接加载本地权重**，不再依赖 vLLM、HTTP 服务或端口。
`asr.kimi` 和 `asr.kimi_batch` 均复用进程内模型缓存；旧的 `asr.kimi_audio*`
名称仅作为兼容别名保留。

## 安装与配置

```bash
pip install -e '.[kimi-audio]'
export KIMI_AUDIO_MODEL_PATH=/data/models/Kimi-Audio-7B-Instruct
```

模型路径也可写在 `configs/asr/kimi.yaml`。环境变量优先于 operator 参数和 YAML。
`load_detokenizer: false` 可降低只做 ASR 时的显存占用。

单文件探针：

```bash
PYTHONPATH=src python scripts/test_kimi_single.py --audio /path/to/sample.wav
```

## 高并发设计

```bash
audio-data pipeline run pipelines/kimi_asr_batch.yaml --source-name mt3000
```

流水线按音频时长均衡切片，每个 shard 是独立进程并通过
`CUDA_VISIBLE_DEVICES` 绑定一张 GPU。每个进程只加载一次模型，并优先调用后端的
`transcribe_batch`，由 `batch_size` 控制单次送入模型的文件数。这样并发发生在隔离的
GPU 进程之间，不会让多个线程不安全地共享同一个模型实例。

部署时必须按机器修改 `sharding.gpus`。通常设置：

- `parallel_shards = len(gpus) * instances_per_gpu`；
- 显存紧张时保持 `instances_per_gpu: 1`；
- 逐步提高 `batch_size`，直到吞吐不再上升或接近显存上限；
- 仅当所安装后端明确保证 `generate()` 线程安全时，才提高 `inference_threads`。

批次失败会自动降级为逐文件重试，因此坏音频只会标记自身失败。缓存键包含最终解析的
模型路径、版本、设备和提示词，重复运行可直接复用结果。输出仍为
`datasets/manifests/kimi_asr_<source>.parquet`，文本位于 `transcripts.kimi.text`。
