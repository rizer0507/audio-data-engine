# Kimi-Audio 识别流水线

## vLLM 服务启动

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve /data2/data-cp/zcl/models/Kimi-Audio-7B-Instruct \
  --host 0.0.0.0 \
  --port 5554 \
  --served-model-name kimi-audio \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 2 \
  --limit-mm-per-prompt '{"audio":1}'
```

## 流水线执行

前置：完成数据清洗，manifest 含 `resampled_16k`。

```bash
# 可选：覆盖 vLLM 地址 / 模型名
export KIMI_ASR_API_BASE=http://127.0.0.1:5554
export KIMI_ASR_MODEL=kimi-audio

audio-data pipeline run pipelines/kimi_asr_batch.yaml --source-name mt3000
# → cleaned_mt3000 → kimi_asr_mt3000.parquet
```

## 并行说明

本流水线通过 **vLLM HTTP API** 识别，客户端不加载模型、不占 GPU：

| 层级 | 配置 | 作用 |
|------|------|------|
| 数据分片 | `sharding.shards` / `parallel_shards` | 多子进程并行处理不同数据片 |
| 片内并发 | `execution.workers` + `executor: thread` | 每片多线程并发调用 `/v1/audio/transcriptions` |
| API 并发上限 | `configs/asr/kimi.yaml` 的 `concurrency` | 单 batch 内同时发出的 HTTP 请求数 |

默认 8 片 × 4 线程 = 最多 32 路并发请求。请根据 vLLM 的 `--max-num-seqs` 和服务器负载调整 `sharding` 与 `concurrency`。

## 聚合到其他模型结果

```bash
audio-data pipeline run pipelines/multi_asr_aggregate.yaml --source-name mt3000 \
  --join-manifest sensevoice --join-manifest kimi
```
