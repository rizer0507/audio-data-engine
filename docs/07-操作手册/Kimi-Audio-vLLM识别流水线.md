# Kimi-Audio 识别流水线（vLLM）

Kimi-vLLM **专属**流水线：`pipelines/kimi_asr_batch.yaml` 内部先按时长桶尾部静音 pad，再调 transcription API。  
与 Qwen / SenseVoice / 豆包 / 本地 `kimi_audio_asr_batch` **解耦**。pad 音频只作本流水线推理输入，**不进**清洗产物、其它 ASR、Dataset Release、评测集真值。

生产入口（建集或评测各一条命令，pad + 识别一次跑完）：

```bash
audio-data pipeline run pipelines/kimi_asr_batch.yaml --source-name mt3000
audio-data pipeline run pipelines/kimi_asr_batch.yaml --eval-name eval_core_v001 --asr-run kimi
```

## vLLM 服务启动（两卡副本，不要 TP）

每卡一个完整模型：`--tensor-parallel-size 1`。不要 `--tensor-parallel-size 2`，也不要和本地 `kimi_audio` 抢同一张卡。

**未做分桶 pad / 混合时长探针未通过前**，`--max-num-seqs` 保持 2～4，不要默认抬到 8+。  
**pad 且长短混合探针通过后**，每卡可试到 8。

```bash
# GPU 4 → :5554
CUDA_VISIBLE_DEVICES=4 \
vllm serve /data2/data-cp/zcl/models/Kimi-Audio-7B-Instruct \
  --host 0.0.0.0 \
  --port 5554 \
  --served-model-name kimi-audio \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"audio":1}'

# GPU 5 → :5555（第二副本；单卡可省略）
CUDA_VISIBLE_DEVICES=5 \
vllm serve /data2/data-cp/zcl/models/Kimi-Audio-7B-Instruct \
  --host 0.0.0.0 \
  --port 5555 \
  --served-model-name kimi-audio \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"audio":1}'
```

`--served-model-name` 必须与 `KIMI_ASR_MODEL`（默认 `kimi-audio`）一致。

## 流水线执行

前置：完成数据清洗，manifest 含 `resampled_16k` 和原始 `duration`。评测跑批用 `--eval-name`，同样以评测集里的 `resampled_16k` 为 pad 源，不改评测集真值音频。

```bash
# 两副本时写多个 API_BASE（逗号分隔或 KIMI_ASR_API_BASES）
export KIMI_ASR_API_BASE=http://127.0.0.1:5554,http://127.0.0.1:5555
# export KIMI_ASR_API_BASES=http://127.0.0.1:5554,http://127.0.0.1:5555
export KIMI_ASR_MODEL=kimi-audio

# 第一级：served model 必须与 KIMI_ASR_MODEL 一致
curl -fsS "http://127.0.0.1:5554/v1/models" | python -m json.tool
curl -fsS "http://127.0.0.1:5555/v1/models" | python -m json.tool

# 第二级：短 WAV，text 必须非空
python scripts/probe_kimi_vllm.py /path/to/short.wav

# 第三级：长短混合目录（必须 --pad）。
# 无权限改他人 conda 时：不要打补丁。重启 vLLM 并加 --max-num-seqs 1，探针用 --concurrency 1。
python scripts/probe_kimi_vllm.py /path/to/mixed_wavs --pad --concurrency 1

# 稳定性：同一条连跑两次，文本须一致（不一致退出码 3）
python scripts/probe_kimi_vllm.py /path/to/short.wav --repeat 2

# >30s：pad 步不截断，原文件直送；用一条长音频确认服务端行为后写进值班笔记
python scripts/probe_kimi_vllm.py /path/to/over30s.wav --pad

# 探针全部通过后，一条命令跑完 pad + 识别
audio-data pipeline run pipelines/kimi_asr_batch.yaml --source-name mt3000
# → cleaned_mt3000 → kimi_asr_mt3000.parquet
```

探针每个音频向标准输出写一条 JSON，汇总写到标准错误。  
退出码：`0` 全部非空且（若 `--repeat>1`）文本稳定；`1` 连接/配置/输入错误；`2` 至少一条空转写；`3` 重复识别文本不一致。

## pad 规则（只服务本流水线）

16 kHz 尾部补零到该桶上限。已对齐上限的样本不写新文件，`audio.kimi_padded_16k` 指向原 `resampled_16k`。

| 原时长 `d` | pad 到 |
|------------|--------|
| `d ≤ 3s` | 3s |
| `3s < d ≤ 6s` | 6s |
| `6s < d ≤ 10s` | 10s |
| `10s < d ≤ 15s` | 15s |
| `15s < d ≤ 30s` | 30s |
| `d > 30s` | **不截断**；`kimi_padded_16k` 指向原文件，标记 `over_30s`，交给 vLLM transcription 服务端处理 |

- **不覆盖** `resampled_16k`，**不改** manifest `duration`（分拣 / 时长均衡 / 评测切片用原始时长）。
- pad 文件不得进入 `release build` 训练音频字段，也不得当作评测集真值。
- 客户端一条 HTTP 仍只送一条 WAV，不硬组变长 GPU batch。
- 不要把 `configs/asr/kimi.yaml` 的 `batch_size` 当成 GPU 真 batch。

## 并行说明

本流水线通过 **vLLM HTTP API** 识别，客户端不加载模型、不占 GPU：

| 层级 | 配置 | 作用 |
|------|------|------|
| 时长 pad | `audio.kimi_duration_pad` | 推理侧派生 `kimi_padded_16k` |
| 数据分片 | `sharding.shards` / `parallel_shards` | 默认 2 片并行（对齐两副本） |
| 片内请求并发 | `configs/asr/kimi.yaml` 的 `concurrency` | 单 shard 同时发出的 HTTP 数 |
| 多副本 | `KIMI_ASR_API_BASE` 多值 | 按 sample id 粘滞分配，一条 `pipeline run` 打两台 |

在途 HTTP ≈ `parallel_shards × concurrency`，应 ≤ 各副本 `--max-num-seqs` 之和。
客户端会 **按 pad 桶分组再并发**：同一 HTTP 窗口只送相同目标时长；`>30s` 强制一路。探针 `--pad` 同样按桶拆窗口。分片目录名 `shard-000` / `shard-001` 会绑到对应 `API_BASE`，避免两 shard 把不同桶打进同一台 GPU。

**单卡只起一份 vLLM 时**，把 YAML 的 `shards` / `parallel_shards` 改成 `1`。两个 shard 打同一端口仍会在 GPU 上混 T。

**他人 conda 无写权限时不要给 site-packages 打补丁。** 改启动参数和客户端并发即可：`--max-num-seqs 1`，`concurrency: 1` / `--concurrency 1`。同一时刻只有一条音频进 encoder，就不会触发 `list.dim()`。吞吐会降，但服务能稳定跑。

| 部署 | 建议 |

| 部署 | 建议 |
|------|------|
| 两副本 × seqs=4 | YAML 默认：2 shard × concurrency 4 = 8 路 |
| 两副本 × seqs=8（混合探针通过后） | 把 `concurrency`（及可选 `batch_size`）调到 8 |
| 单副本 × seqs=2 | `parallel_shards: 1` 且 `concurrency: 2`，不要 8 shard × 4 |

`KIMI_ASR_API_BASE` 可填服务根地址（如 `http://127.0.0.1:5554`）或 OpenAI 基地址（如 `http://127.0.0.1:5554/v1`），客户端会避免重复拼接 `/v1`。  
请求只发送 transcription 接口支持的字段，不会把 `max_completion_tokens` 等 Chat Completions 参数混入 multipart。

ASR 缓存键包含 pad 桶 / 目标时长与输入音频 key，未 pad 的旧缓存不会冒充新结果。  
pad / 识别失败只标自身 `failed`，不伪造转写，不拖垮整批。

## 聚合、分拣、评测（零改下游算子）

join 名 **`kimi`**（本地加载是 `kimi_audio`，禁止混用）：

```bash
audio-data pipeline run pipelines/multi_asr_aggregate.yaml --source-name mt3000 \
  --aggregate-base qwen1 \
  --join-manifest sensevoice1 \
  --join-manifest kimi
```

分拣 / `release build` 继续读聚合后的 `transcripts.*` 与原始 `duration` / `resampled_16k`。

评测 Kimi-vLLM 本身时用同一条 YAML，不要另做评测专用 pad：

```bash
audio-data pipeline run pipelines/kimi_asr_batch.yaml \
  --eval-name eval_core_v001 --asr-run kimi

audio-data pipeline run pipelines/eval_aggregate.yaml \
  --eval-name eval_core_v001 --join-manifest kimi

audio-data pipeline run pipelines/eval_metric_pipeline.yaml \
  --eval-name eval_core_v001 --eval-model kimi
```

指标相对 `gold_text`，不把 pad 静音当成多出来的语音内容。
