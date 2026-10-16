# 045｜入库 → VAD → 多路 ASR∥独立 DNSMOS → 证据合并 → 自由标签

新生产主链。DNSMOS 是与 ASR 同级的正式独立流水线，不是质量侧车。

## 一条命令

```bash
# 服务器原始权重（生产固定）
export FSMN_VAD_SOURCE_MODEL_DIR=/data2/data-cp/xijujun/models/fsmn-vad
# 使用已准备好的本地 runtime；首次准备见下文，不能填写占位路径。
export FSMN_VAD_RUNTIME_DIR="$PWD/var/models/fsmn-vad-fenshen-v1"
# configs/quality/dnsmos_p835.yaml 中 model_path 须可读；calibrated 保持 false
# Qwen/GLM 等按各 ASR YAML / 环境变量配置，勿照抄历史卡号端口

audio-data pipeline run pipelines/audio_ingest_vad_asr_dnsmos_tags.yaml \
  --source-name "$BATCH" \
  --source-dir "$SOURCE_DIR"
```

`$FSMN_VAD_RUNTIME_DIR` 必须包含 `config.yaml`（已覆盖 fenshen 参数）、`configuration.json`、`model.pt`、`am.mvn`。
首次部署：

```bash
python scripts/prepare_fsmn_vad_runtime.py \
  --source-model-dir /data2/data-cp/xijujun/models/fsmn-vad \
  --fenshen-config configs/vad/fenshen_vad_config.yaml \
  --runtime-dir "$FSMN_VAD_RUNTIME_DIR"
```

原始权重目录为服务器只读资产，运行时使用工程内 prepare 后的 runtime。分片父进程会在切分输入和启动子进程之前校验路径；独立算子和单文件脚本也会在调用 FunASR 之前拦截缺失的本地模型，避免进入 ModelScope 下载锁等待。

VAD 的实际并发是 `parallel_shards × source_workers`，当前默认 `8 × 2 = 16` 个父源任务；`execution.workers` 不控制这个 Manifest 算子。每个分片先串行初始化各 worker 模型，全部成功后才提交音频，初始化异常直接退出。`max_in_flight_sources` 同时限制执行与排队任务，模型按线程长期复用。日志分别显示初始化、模型就绪、父源完成数；汇总进度按源结果 JSON 统计，而非切出的片段数。

若日志出现 `Still waiting to acquire lock ... ___path___to___fsmn-vad-runtime`，先中止对应的旧运行，修正模型目录并更新代码后再续跑。不要删除共享 ModelScope 锁文件；单独增加并发数不能解决错误路径导致的初始化等待。`--resume` 复用完整步骤 checkpoint；未完成的 VAD Manifest 步骤仍会重跑父源，已有同 ID 的片段 WAV 会复用。

路次在 `pipelines/audio_ingest_vad_asr_dnsmos_tags.yaml` 的 `production.asr_routes` 配置，可增删，不写死三家族六路。

## 产物契约

| 阶段 | 权威路径 |
|------|----------|
| 入库 | `datasets/stage1/ingested/ingested_<batch>.parquet` |
| VAD 片段 | `datasets/stage1/segments/segments_<batch>.parquet` |
| ASR | `datasets/stage1/asr/<route_id>_asr_<batch>.parquet` |
| DNSMOS | `datasets/stage1/quality/dnsmos_<batch>.parquet` |
| 合并证据 | `datasets/stage1/derived/prepared_evidence_<batch>.parquet` |
| 自由 tag | `datasets/stage1/derived/tagged_free_tags_v2_<batch>.parquet` |

历史 `quality_sidecar_*` / `cleaned_*`（041 合并入口）仍可读，但新链默认不发现、不依赖。

## 父 run 查询 / 续跑 / 失败重试

```bash
# 进度：可见 ASR 与 DNSMOS 同时 running，以及谁在阻塞 join
audio-data pipeline progress "runs/<parent-run>" --follow

# 续跑：已成功节点跳过（产物仍在则复用）
audio-data pipeline run pipelines/audio_ingest_vad_asr_dnsmos_tags.yaml \
  --source-name "$BATCH" --source-dir "$SOURCE_DIR" \
  --resume "runs/<parent-run>"
```

独立失败恢复示例：

```bash
# 只重跑 DNSMOS（不重跑已成功 ASR）
audio-data pipeline run pipelines/dnsmos_score_batch.yaml --source-name "$BATCH"
# 再合并 + 打标
audio-data pipeline run pipelines/evidence_join.yaml --source-name "$BATCH" \
  --join-manifest qwen1 --join-manifest glm1
audio-data pipeline run pipelines/tag_dataset_free_tags_v2.yaml \
  --source-name "$BATCH" --config configs/tagging/zh_asr_free_tags_v2.yaml
```

父 run 目录要点：`state.json`、`progress_view.json`、`overlap_evidence.json`、`events.jsonl`、`nodes/<node>/work/`（子流水线 runs）。

## 独立节点

```bash
audio-data pipeline run pipelines/audio_ingest.yaml --source-name "$BATCH" --source-dir "$SOURCE_DIR"
audio-data pipeline run pipelines/vad_segment_batch.yaml --source-name "$BATCH"
audio-data pipeline run pipelines/dnsmos_score_batch.yaml --source-name "$BATCH"
```

自由 tag 使用 `evidence_mode: existing`，消费已提交的 DNSMOS，不会隐式重跑评分。FSMN 切分不会冒充 Silero / energy 事实；缺失时规则保持三态 unknown。

## 与旧入口关系

| 旧 | 新 |
|----|----|
| `data_cleaning_source_A.yaml`（入库+VAD） | `audio_ingest.yaml` + `vad_segment_batch.yaml` |
| `audio_quality_sidecar.yaml` → `quality_sidecar_*` | `dnsmos_score_batch.yaml` → `dnsmos_*` |
| `stage1 run` 固定三家族 / 旧 classify 门控 | 配置驱动 N 路 + 自由 tag（不经 selection/classify） |
