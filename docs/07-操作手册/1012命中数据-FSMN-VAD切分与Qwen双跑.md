# 1012 命中数据：服务器 FSMN VAD 切分与 Qwen 双跑

适用日期：2026-10-14。输入是本次筛出的 **10,449 条父音频**，使用项目现有 `pipelines/data_cleaning_source_A.yaml` 完成规范化和 **FSMN VAD 实际切段**，输出片段 Parquet，随后交给 Qwen 双跑。

## 1. 服务器准备

以下命令均在服务器项目根目录、已安装本项目的 Python 环境中执行。服务器代码须包含当前 `audio.fsmn_vad_segment` 实现、`src/audio_engine/core/fsmn_vad/`、现行清洗 YAML 和模型准备脚本。沿用已跑通的 FunASR / Torch / Torchaudio 环境。

将输入文件复制到服务器项目内的同一相对位置：

```text
tmp/1012/free_tags_run_20261014/results/matched_samples.parquet
```

Parquet 存储音频路径引用，不包含音频字节。两批原始音频路径分别为：

```text
/data2/data-cp/datasets/asr_datasets/fenshen/20260813/1/   4,702 条 PCM
/data2/data-cp/datasets/asr_datasets/fenshen/20260814/1/   5,747 条 PCM
```

**一个 Parquet 可以引用多个音频目录，无需合并目录，也无需拆成两次 VAD 运行。** 每条样本的 `audio.raw` 都记录自己的完整路径，流水线按行读取；上面的两个目录均可访问时，直接使用本文的同一条命令。

现有 `audio.pcm_wav` / `audio.resampled_16k` 的 WAV 引用位于 `/data2/data-cp/lizi/audio-data-engine/data/derived/pcm_to_wav/` 下的不同子目录。本文命令带 `--force`，会从每条 `audio.raw` 重新规范化，再读取本次生成的 WAV 切段。

若服务器移动了这两个源目录，分别把两组路径前缀映射到两个实际目录，更新 `audio.raw`，并同步更新 `source_path` 后另存一份输入 Parquet；VAD 命令只需把 `--input-manifest` 指向该文件。无需指定 `--source-dir`。字段中的路径打不开时，程序不会自动搜索其它目录。

如果服务器只保留了现成 WAV，可将每行 `audio.raw` 指向它对应的可读 WAV，再走同一规范化和切分流程；项目会按 WAV 文件头读取采样率与声道。不要仅修改 `audio.resampled_16k` 而留下不可读的 `audio.raw`。

首次使用该 FSMN 模型时，准备一次独立运行目录。服务器原始权重固定为：

```text
/data2/data-cp/xijujun/models/fsmn-vad
```

```bash
export FSMN_VAD_SOURCE_MODEL_DIR=/data2/data-cp/xijujun/models/fsmn-vad
python scripts/prepare_fsmn_vad_runtime.py \
  --source-model-dir "$FSMN_VAD_SOURCE_MODEL_DIR" \
  --fenshen-config configs/vad/fenshen_vad_config.yaml \
  --runtime-dir "$PWD/var/models/fsmn-vad-fenshen-v1"
```

脚本复制模型资产，并用项目的完整 fenshen 配置覆盖运行目录的 `config.yaml`，默认校验模型与配置指纹。已有按此方式准备好的运行目录可直接复用，在下一条命令中填入它的绝对路径。

## 2. 一条命令完成 VAD 切分

准备完成后，直接执行下面**一条命令**。换行仅用于阅读，整段是同一次 `pipeline run`：

```bash
FSMN_VAD_RUNTIME_DIR="$PWD/var/models/fsmn-vad-fenshen-v1" \
AUDIO_ENGINE_VAD_MOCK=0 \
audio-data pipeline run pipelines/data_cleaning_source_A.yaml \
  --input-manifest tmp/1012/free_tags_run_20261014/results/matched_samples.parquet \
  --output-manifest datasets/stage1/cleaned/cleaned_1012_tag_vad.parquet \
  --runs-dir runs/1012_tag_vad \
  --force
```

执行链路为：**读取指定 Manifest → PCM 转 WAV → 16 kHz / 单声道规范化 → 音频探测与有效性筛选 → FSMN VAD → 裁切 WAV → 合并片段 Parquet**。`--force` 让带有旧清洗完成状态的父样本重新经过当前规范化流程。

命令保留现有 YAML 的分片设置：8 个 hash 分片并行执行，完成后自动合并。不需要手动拆分或合并。

这里使用的是 `audio.fsmn_vad_segment` 切段算子。规则原样来自当前项目：

- FSMN 在 CPU 上推理，每份音频独立状态，按 200 ms 分块；多声道取第 0 声道，统一 16 kHz PCM16。
- 按模型返回的 `[start, end)` 裁切，不再额外加尾；不设最小时长；不补造静音，不拼接短段。
- 有有效起点但直到文件结束仍没有模型终点的 EOF 补尾，无论时长一律丢弃。
- 模型原始 end 到达文件截止或最后 VAD 帧的片段也丢弃，记录原因 `endpoint_at_eof`。按当前 25 ms 窗、10 ms 步长判定，覆盖 EOF 强制 end 比文件时长早约 15–25 ms 的情况（例如 1000 ms 文件的 980 ms end）。
- 无有效起点、无语音、只有无终点的 EOF 补尾或处理失败的父音频不生成下游片段。
- 沿用项目完整 fenshen 配置（模型侧 `max_end_silence_time=1100` 只影响端点检测，不用于裁切加尾）；最长单段模型参数 10000 ms 不是额外固定切片长度。
- **049 起**：切分语义与分身实时服务完全对齐（100 ms 分包 / 每两包推理、`send_end_result` EOF 补尾、无 675 ms 固定尾、无统一 1 秒门限）。旧 041/045 中冲突规则已废止；版本为 `fsmn_vad_segment/v8`，旧片段须从父音频 `--force` 重跑。

**保持上述命令参数即可。** 对这个原清洗 YAML，不要额外添加 `--source-name`、`--source-dir`、`--skip-step ingest` 或 `--no-sharding`：显式 Manifest 与当前分片入口会自动跳过目录扫描；这些额外参数可能触发源目录校验或破坏子进程的 ingest 跳过逻辑。

## 3. 生成什么，后续使用哪个文件

后续 Qwen 优先读取合并结果：

```text
datasets/stage1/cleaned/cleaned_1012_tag_vad.parquet
```

同时会生成同名 `.jsonl`。输出行数是**保留片段数**，不要求等于输入的 10,449：一条父音频可以产生零个、一个或多个片段。

运行日志会打印实际运行目录，例如：

```text
runs/1012_tag_vad/<时间>_data_cleaning_source_A_shards/
```

该目录根下 `shard-000.parquet` 至 `shard-007.parquet` 是各分片的**切分结果**，也可以分别供后续 ASR 使用。其 `shards/shard-*.parquet` 是**切分前的输入分片**，不要把它们当作 VAD 输出。

片段 WAV 位于项目的 `data/derived/vad_segments/`，具体绝对路径写入每条片段的 `audio.resampled_16k`。每个分片的 `shard-000/vad_source_results/` 等目录保存父音频的区间、丢弃原因与 `_batch_summary.json`。

后续读取片段时使用 `audio.resampled_16k`。`source_path` 和 `audio.parent_resampled_16k` 保留父音频路径，不能拿它们代替片段输入。将 Parquet 搬到另一台服务器时，需要同时保留片段 WAV 的可访问路径。

每行是新的 `seg_…` 样本，使用片段 PCM 内容哈希，旧父音频的六路转写、标签和 ASR 完成状态不继承。可通过 `labels.source_audio_id`、`parent_pcm_sha256`、`clip_start_sample`、`clip_end_sample`、`vad_end_reason` 等字段回溯父音频。本次原始命中文件继续作为父样本与标签来源对照。

中断后保留运行目录，在原 VAD 命令末尾追加 `--resume runs/1012_tag_vad/<实际运行目录>`，使用相同输入、配置及模型继续执行。

EOF 终点丢弃规则自 `fsmn_vad_segment/v4`（算子 `1.4.0`）生效；自 `v5`（算子 `1.5.0`）起裁切不再额外加 1100 ms 尾。已有旧版片段需从父音频按上面的 `--force` 命令重跑 VAD，后续 Qwen 读取新 Manifest；旧版产物不会自动删除。规则版本进入片段 ID，重跑后的片段 ID 会更新。

## 4. 后续 Qwen 双跑

以下命令调用现有 `pipelines/qwen_asr_batch.yaml`，直接读取上面生成的片段 Parquet。两张卡分别启动的 Qwen vLLM 服务监听 `5555`、`5556` 时，可以让两轮全量识别同时运行：**第一轮使用 5555，第二轮使用 5556；每轮都读取全部片段。** 两个服务使用相同模型、推理配置及 served-model-name。

打开两个终端，在项目根目录、同一项目环境中分别设置公共参数；`qwen3-asr` 按实际 served-model-name 替换：

```bash
export QWEN_ASR_API_KEY=dummy
export QWEN_ASR_MODEL=qwen3-asr
```

`asr.qwen_batch` 需要已启动的 HTTP 服务；普通 `pipeline run` 不会自动启动模型服务。下面地址假设客户端与服务在同一台机器；跨机器时将 `127.0.0.1` 换成服务器地址。

终端 1：第一轮使用 5555。

```bash
QWEN_ASR_API_BASE=http://127.0.0.1:5555 \
audio-data pipeline run pipelines/qwen_asr_batch.yaml \
  --no-sharding \
  --input-manifest datasets/stage1/cleaned/cleaned_1012_tag_vad.parquet \
  --output-manifest datasets/stage1/asr/qwen_1_asr_1012_tag_vad.parquet \
  --transcript-key qwen_1 \
  --runs-dir runs/1012_tag_vad_qwen_1 \
  --force
```

终端 2：同时启动第二轮，使用 5556，仍从**同一份全量 VAD 结果**开始。无需等待终端 1 完成。

```bash
QWEN_ASR_API_BASE=http://127.0.0.1:5556 \
audio-data pipeline run pipelines/qwen_asr_batch.yaml \
  --no-sharding \
  --input-manifest datasets/stage1/cleaned/cleaned_1012_tag_vad.parquet \
  --output-manifest datasets/stage1/asr/qwen_2_asr_1012_tag_vad.parquet \
  --transcript-key qwen_2 \
  --runs-dir runs/1012_tag_vad_qwen_2 \
  --force
```

两次结果分别含 `transcripts.qwen_1`、`transcripts.qwen_2`；使用相同片段 ID / SHA256，便于后续聚合。两个输出路径及运行目录独立，`--force` 确保实际重新推理。命令前的环境变量只对各自进程生效，因此两个终端分别绑定不同服务。

这里的 `--no-sharding` 仅用于 Qwen 客户端，避免套用原 Qwen YAML 固定的 GPU `[4,5]`；两张 GPU 已由各自 vLLM 服务使用。当前每个客户端默认 HTTP 并发为 4，`batch_size=32`，两进程各有自己的并发。`--workers` 不控制 Qwen batch 算子的请求并发；需要调优时使用 `configs/asr/qwen_asr.yaml` 的 `concurrency`，先按当前配置运行即可。

当前 Qwen 客户端的 `QWEN_ASR_API_BASE` 只接受一个 URL，不能把 `5555,5556` 拼在同一个值中。采用上述两进程并行即可使用两个服务，无需改动项目代码。两轮各处理全量，才保证每个片段有两次识别；不要把输入分成两半各识别一次。

两轮同时占用两张卡，双跑总时间可缩短；实际提速取决于单卡利用率、服务配置和音频读写。当前 `temperature=0`，两次结果相同也可能正常。

如需更换这批数据的名字，统一修改命令中的 `1012_tag_vad`；保留两次运行各自的 `qwen_1`、`qwen_2` 别名和不同输出路径。项目 CLI 会写入指定输出，重新实验时使用新的批次名或结果路径。

## 实现依据与范围

本说明按当前工作区实际代码与配置编写：`pipelines/data_cleaning_source_A.yaml`、`audio.fsmn_vad_segment`、`core/fsmn_vad/`、`scripts/prepare_fsmn_vad_runtime.py`、`pipelines/qwen_asr_batch.yaml` 和 `configs/asr/qwen_asr.yaml`。本次仅编写执行说明，没有在服务器运行 VAD 或 Qwen。
