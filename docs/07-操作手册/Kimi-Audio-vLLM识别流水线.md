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

# 第一级探针：served model 必须与 KIMI_ASR_MODEL 一致
curl -fsS "${KIMI_ASR_API_BASE%/}/v1/models" | python -m json.tool

# 第二级探针：先测一条真实短音频，输出 JSONL，text 必须非空
python scripts/probe_kimi_vllm.py /path/to/short.wav

# 第三级探针：再测整个 WAV 文件夹；需要子目录时加 --recursive
python scripts/probe_kimi_vllm.py /path/to/wav_folder --concurrency 4
python scripts/probe_kimi_vllm.py /path/to/wav_folder --recursive --concurrency 4

# 探针全部通过后，才运行整个 cleaned_<source>.parquet
audio-data pipeline run pipelines/kimi_asr_batch.yaml --source-name mt3000
# → cleaned_mt3000 → kimi_asr_mt3000.parquet
```

探针每个音频向标准输出写一条 JSON，汇总写到标准错误；任一转写为空时退出码为
`2`，连接、配置或输入错误时退出码为 `1`，因此可直接用于上线前脚本的失败拦截。

## 并行说明

本流水线通过 **vLLM HTTP API** 识别，客户端不加载模型、不占 GPU：

| 层级 | 配置 | 作用 |
|------|------|------|
| 数据分片 | `sharding.shards` / `parallel_shards` | 多子进程并行处理不同数据片 |
| 片内请求并发 | `configs/asr/kimi.yaml` 的 `concurrency` | operator 单批同时发出的 HTTP 请求数 |
| shard 调度 | `execution.workers` + `executor: thread` | 调度 shard 内样本；实际 HTTP 扇出仍受 `concurrency` 限制 |

默认 8 片 × 4 线程 = 最多 32 路并发请求。请根据 vLLM 的 `--max-num-seqs` 和服务器负载调整 `sharding` 与 `concurrency`。

`KIMI_ASR_API_BASE` 可填写服务根地址（如 `http://127.0.0.1:5554`）或
OpenAI 基地址（如 `http://127.0.0.1:5554/v1`），客户端会避免重复拼接 `/v1`。
请求仅发送 vLLM transcription 接口支持的字段，不会把
`max_completion_tokens` 等 Chat Completions 参数混入 multipart 请求。

批量 operator 保留输入顺序、逐条写入 `transcripts.kimi.text` 并使用缓存；一批请求失败后
会逐文件重试，将坏音频标记为 `failed`，不会让单条坏数据污染整个 Parquet 的其他结果。

## 聚合到其他模型结果

```bash
audio-data pipeline run pipelines/multi_asr_aggregate.yaml --source-name mt3000 \
  --join-manifest sensevoice --join-manifest kimi
```
