# GLM-ASR 识别流水线（vLLM）

GLM-ASR **专属**流水线：`pipelines/glm_asr_batch.yaml` 客户端调 vLLM OpenAI 兼容 `/v1/audio/transcriptions`，片内 `concurrency` 并发 HTTP，服务端 continuous batching。  
与 Qwen / SenseVoice / 豆包 / Kimi-vLLM / Kimi 本地 **解耦**。不要抄 Kimi 的时长桶 pad，也不要在引擎进程里 `from_pretrained` GLM。

默认权重：**GLM-ASR-Nano-2512**（`zai-org/GLM-ASR-Nano-2512`）。不要和 GLM-4-Voice（语音对话）混用。

生产入口（建集或评测各一条命令）：

```bash
audio-data pipeline run pipelines/glm_asr_batch.yaml --source-name mt3000
audio-data pipeline run pipelines/glm_asr_batch.yaml --eval-name eval_core_v001 --asr-run glm
```

## vLLM 服务启动（每卡一份，不要 TP）

每卡一个完整模型：`--tensor-parallel-size 1`。不要 `--tensor-parallel-size 2`，也不要和正在跑的 Qwen / Kimi 抢同一张卡、同一个端口。

空闲 GPU 与正式端口 **落地当天确认**。下文 `GPU 6` / `:5570` 只是占位，避免和 Qwen `:5559` / Kimi `:5554` 撞车。

```bash
# 示例：GPU 6 → :5570；路径换成服务器本地权重
CUDA_VISIBLE_DEVICES=6 \
vllm serve /data2/data-cp/models/GLM-ASR-Nano-2512 \
  --host 0.0.0.0 \
  --port 5570 \
  --served-model-name glm-asr \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"audio":1}'
```

`--served-model-name` 必须与 `GLM_ASR_MODEL`（默认 `glm-asr`）一致。  
不要把 Qwen 的 `--chat-template …/qwen3_asr_language.jinja` 套到 GLM 上。

依赖（写在值班笔记，不算本仓库代码）：vLLM 需带 audio extras（官方 recipe：`vllm>=0.14.1` + `vllm[audio]`；transformers 版本按当时 recipe）。

## 流水线执行

前置：完成数据清洗，manifest 含 `resampled_16k`。评测跑批用 `--eval-name`，音频键仍是评测集里的 `resampled_16k`，不改评测集真值。

```bash
export GLM_ASR_API_BASE=http://127.0.0.1:5570
export GLM_ASR_MODEL=glm-asr

# 第一级：served model 必须与 GLM_ASR_MODEL 一致
curl -fsS "http://127.0.0.1:5570/v1/models" | python -m json.tool

# 第二级：短 WAV，text 必须非空
python scripts/probe_glm_vllm.py /path/to/short.wav

# 第三级：目录并发（对齐目标 concurrency / max-num-seqs）
python scripts/probe_glm_vllm.py /path/to/mixed_wavs --concurrency 8

# 稳定性：同一条连跑两次，文本须一致（不一致退出码 3）
python scripts/probe_glm_vllm.py /path/to/short.wav --repeat 2

# 探针全部通过后，一条命令跑识别
audio-data pipeline run pipelines/glm_asr_batch.yaml --source-name mt3000
# → cleaned_mt3000 → glm_asr_mt3000.parquet
# 可选：--asr-run glm1  → glm1_asr_mt3000.parquet / transcripts.glm1
```

探针每个音频向标准输出写一条 JSON，汇总写到标准错误。  
退出码：`0` 全部非空且（若 `--repeat>1`）文本稳定；`1` 连接/配置/输入错误；`2` 至少一条空转写；`3` 重复识别文本不一致。

**没有 `--pad`。** 若长短混合在高 `max-num-seqs` 下打挂：先把 seqs / concurrency 降到稳定值；不要擅自加 pad，把现象记入值班笔记并另开需求。

## 并行说明

本流水线通过 **vLLM HTTP API** 识别，客户端不加载模型、不占 GPU：

| 层级 | 配置 | 作用 |
|------|------|------|
| 数据分片 | YAML `sharding.shards` / `parallel_shards` | 默认 1 片（单副本） |
| 片内请求并发 | `configs/asr/glm.yaml` 的 `concurrency` | 单 shard 同时发出的 HTTP 数 |
| 服务端真 batch | vLLM `--max-num-seqs` | GPU 上同时处理的序列数 |
| HTTP 切片 | yaml `batch_size` | 只是客户端切块大小，**不是** GPU batch |
| 多副本 | `GLM_ASR_API_BASE` 多值 | 按 sample id 粘滞分配，一条 `pipeline run` 打两台 |

在途 HTTP ≈ `parallel_shards × concurrency`，应 ≤ 各副本 `--max-num-seqs` 之和。

YAML **不要** 写 `gpus` / `instances_per_gpu`。服务端 GPU 只由 `CUDA_VISIBLE_DEVICES` + `vllm serve` 决定。

| 部署 | 建议 |
|------|------|
| 单副本 × seqs=8 | YAML 默认：1 shard × concurrency 8 |
| 两副本 × seqs=8（同权重） | YAML 改成 2 shard；`GLM_ASR_API_BASE=http://127.0.0.1:5570,http://127.0.0.1:5571` |
| 不同权重 / 不同 SFT | 各起各的端口，改 env 再跑一条 `--asr-run` |

`GLM_ASR_API_BASE` 可填服务根地址（如 `http://127.0.0.1:5570`）或 OpenAI 基地址（如 `http://127.0.0.1:5570/v1`），客户端会避免重复拼接 `/v1`。  
请求只发送 transcription 接口支持的字段（`model` / `language` / `temperature` / 可选 `prompt` / `response_format`），不会把 Chat Completions 参数混入 multipart。

未 export `GLM_ASR_API_BASE` 时 batch 算子拒绝本地加载。缓存键含 model / api_base / 输入音频 key，Qwen 缓存不会冒充 GLM 结果。  
失败样本只标自身 `failed`，不伪造转写，不拖垮整批。

## 聚合、分拣、评测（零改下游算子）

join 名 **`glm`**：

```bash
audio-data pipeline run pipelines/multi_asr_aggregate.yaml --source-name mt3000 \
  --aggregate-base qwen1 \
  --join-manifest sensevoice1 \
  --join-manifest glm
```

分拣 / `release build` 继续读聚合后的 `transcripts.*` 与原始 `duration` / `resampled_16k`。  
`zh_asr_v1` / `zh_asr_v2` 已追加独立 `glm` family（`glm` / `glm1` / `glm2`），不要把 `glm*` 写进 `qwen` 族。

评测 GLM 本身时用同一条 YAML：

```bash
# 评前 unset 建集用的 GLM_ASR_API_BASE，避免串端口
export GLM_ASR_API_BASE=http://127.0.0.1:5570
audio-data pipeline run pipelines/glm_asr_batch.yaml \
  --eval-name eval_core_v001 --asr-run glm

# 若评 SFT：另起端口，换 GLM_ASR_API_BASE / GLM_ASR_MODEL 再跑
# export GLM_ASR_API_BASE=http://127.0.0.1:5572
# audio-data pipeline run pipelines/glm_asr_batch.yaml \
#   --eval-name eval_core_v001 --asr-run glm-sft-ep100

audio-data pipeline run pipelines/eval_aggregate.yaml \
  --eval-name eval_core_v001 --join-manifest glm

audio-data pipeline run pipelines/eval_metric_pipeline.yaml \
  --eval-name eval_core_v001 --eval-model glm
```

指标相对 `gold_text`。严禁只设一个全局 `GLM_ASR_API_BASE` 同时评两个权重。
