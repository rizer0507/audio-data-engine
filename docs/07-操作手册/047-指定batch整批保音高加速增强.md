# 047｜指定 batch 整批保音高加速增强

生产入口：`pipelines/augment_batch_speed.yaml`（算子 `augmentation.tempo`）。

## 1. 做什么

指定一个 batch，对该 batch **单一输入快照**中的全部样本做**加快语速、保持音高**的加速增强，生成独立派生样本与配套 Parquet。默认倍率 `1.2`。原始音频与原 batch 不覆盖。

算法后端：FFmpeg `atempo`（可串联处理 >2 的倍率）。**不是**旧算子 `augmentation.speed_perturb`（`signal.resample`，会同时提高音高）。

## 2. 输入与产物

| 项 | 默认 |
|----|------|
| `--source-name` | 批次标识（与其它流水线同名参数） |
| 默认输入 | `datasets/stage1/segments/segments_<batch>.parquet` |
| 显式输入 | `--input-manifest`（同 batch 的清洗/仓库等快照；不扫描混阶段） |
| 默认 audio key | `resampled_16k` |
| 产物根 | `datasets/stage1/augmented/<batch>/tempo_<factor_tag>_<recipe_digest>/` |

产物文件：

- `manifest.parquet`：仅当全批成功时发布（输入顺序）
- `manifest.partial.parquet`：有失败时的成功子集（非正式完成）
- `results.parquet`：覆盖每条输入（成功/失败/缓存命中）
- `recipe.json` / `summary.json`
- `audio/<prefix>/*.wav`：PCM_16，采样率与声道与输入一致

## 3. 命令

```bash
# 默认 1.2 倍
audio-data pipeline run pipelines/augment_batch_speed.yaml --source-name "$BATCH"

# CLI 指定倍率（覆盖 YAML）
audio-data pipeline run pipelines/augment_batch_speed.yaml \
  --source-name "$BATCH" --speed-factor 1.3

# 显式仓库 Manifest
audio-data pipeline run pipelines/augment_batch_speed.yaml \
  --source-name "$BATCH" \
  --input-manifest datasets/stage1/warehouses/$BATCH/manifest.parquet \
  --speed-factor 1.2

# 断点续跑（使用原 run 冻结的 batch / 倍率 / 输入）
audio-data pipeline run pipelines/augment_batch_speed.yaml --resume runs/<原run目录>
```

缺省不传 `--speed-factor` 时**不会**用 CLI 默认值覆盖 YAML；YAML 缺省再用 `1.2`。

## 4. 参数优先级与分片

- CLI 显式 `--speed-factor` > YAML `params.speed_factor` > 1.2
- 分片时父进程写出 `runs/.../pipeline.resolved.yaml`，子进程携带同一倍率；merge 后 finalize 正式 Manifest
- 并发预算：`parallel_shards × workers × backend_threads`；FFmpeg 子进程受 `max_in_flight` 门闩约束

## 5. 身份与证据

- 派生 `id` 稳定可复现；当前 `sha256` / `source_path` 指向增强音频
- 原 ASR / DNSMOS 等归档在 `labels.augmentation.source`，**不**保留对增强音频的 `completed` 伪装
- 金标文本、训练禁用、防泄漏分组等保留并标记继承

## 6. 依赖

系统需安装 FFmpeg，且 `ffmpeg -filters` 含 `atempo`。可通过 `params.ffmpeg_bin` 指定绝对路径。

## 7. 验收状态（工程）

- 单元/集成：`pytest tests/test_augment_tempo.py`（含真实 FFmpeg 单频保音高、整批守恒、失败不发布正式清单、缓存幂等）
- **待验**：真实语音听感、目标服务器大批量吞吐
