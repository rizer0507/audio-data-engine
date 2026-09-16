# 服务器端五类完整自动分类与 XLSX 导出

本文说明在服务器已经完成三模型家族、每家族两次 ASR 推理后，如何在服务器继续完成数据准备、声学评分、五类自动分类和 XLSX 导出。

本文采用以下业务前提：

- 输入音频已经过上游 VAD；本项目不重复执行 VAD。
- 三个模型家族分别为 Qwen、GLM 和 SenseVoice，每家族有两次独立推理结果。
- 服务器能够读取 Manifest 中的 `resampled_16k` 音频路径。
- Parquet Manifest 是权威结果，XLSX 只作为查看和交付格式。

## 1. 重要结论

当前仓库不能仅凭现有 `quality_sidecar` 自动产出 `environment_noise`。

现有侧车只输出 DNSMOS 的 `sig`、`bak`、`ovrl`。默认配置为 `calibrated: false`，因此 `noise_band=unknown`、`noise_risk=null`。五类分类器只在存在可信的 `no_target_speech`、串音或背景噪声确认字段时输出 `environment_noise`，不会把低 DNSMOS 或全空 ASR 直接当作环境噪声。

因此，执行本文命令前必须先落地第 4 节定义的自动环境噪声规则。未完成该改造时，即使侧车评分成功并重新运行分类，`environment_noise` 仍可能为 0。

## 2. 目标产物

以批次名 `${BATCH}` 为例，完整流程应产出：

```text
datasets/stage1/asr/prepared_asr_v3_${BATCH}.parquet
datasets/stage1/derived/quality_sidecar_${BATCH}.parquet
datasets/stage1/derived/classified_five_class_v1_${BATCH}.parquet
data/exports/summary_five_class_v1_${BATCH}-part-001.xlsx
data/exports/summary_five_class_v1_${BATCH}-part-002.xlsx  # 超过 20,000 行时
```

分类结果使用以下五个业务类别：

```text
gold_candidate
voicemail
semantic_risk
hardcase
environment_noise
```

无法满足自动归类证据的样本允许保留 `manual_annotation`，不得通过伪造声学状态强行塞入某个类别。

## 3. 服务器运行前检查

### 3.1 环境

```bash
cd /data2/data-cp/lizi/audio-data-engine
source .venv/bin/activate

audio-data --help
python -c "import onnxruntime, soundfile; print('quality runtime ok')"
```

### 3.2 批次变量

```bash
export BATCH=0914-mixed-30000
export DATASET_CFG=configs/datasets/zh_asr_v3_0914_mixed_30000.yaml
export SELECTION_CFG=configs/selection/zh_asr_five_class_v1_0914_mixed_30000.yaml
export QUALITY_CFG=configs/quality/dnsmos_p835_server_auto.yaml
```

`dnsmos_p835_server_auto.yaml` 是第 4 节要求新增的服务器生产配置。不要直接把现有 `dnsmos_p835.yaml` 的 `calibrated` 从 `false` 改成 `true`。

### 3.3 六路 ASR 文件

确认以下六路结果均存在，实际文件名以 `${DATASET_CFG}` 中登记的 run 为准：

```bash
ls -lh \
  datasets/stage1/asr/glm-base-1_asr_${BATCH}.parquet \
  datasets/stage1/asr/glm-base-2_asr_${BATCH}.parquet \
  datasets/stage1/asr/qwen-basr-1_asr_${BATCH}.parquet \
  datasets/stage1/asr/qwen-basr-2_asr_${BATCH}.parquet \
  datasets/stage1/asr/sv-base-1_asr_${BATCH}.parquet \
  datasets/stage1/asr/sv-base-2_asr_${BATCH}.parquet
```

六路 Manifest 必须使用同一批样本，并能通过 `sample_id + original_audio_sha256` 对齐。禁止按行号合并。

### 3.4 音频可读性

DNSMOS 必须在服务器执行。运行分类前，抽查 `resampled_16k` 路径确实存在；如果 Manifest 中仍是另一台机器的绝对路径，先修正路径映射，不要把失败结果传到 Windows 后补救。

## 4. 自动 `environment_noise` 的生产规则

### 4.1 必须新增的显式配置

建议在 selection 配置中新增独立配置块，不要复用模糊的 `calibrated: true`：

```yaml
environment_noise:
  auto_classify: true
  policy_version: vad_asr_dnsmos_v1
  upstream_vad_applied: true
  require_all_families_stable_empty: true
  require_dnsmos_success: true
  require_approved_thresholds: true
  reject_if_any_valid_target_text: true
  reject_if_voicemail: true
  reject_if_semantic_risk: true
```

质量配置必须记录经业务批准的阈值和版本：

```yaml
quality_policy_version: quality_policy_server_auto_v1
calibrated: true
calibration_set: <人工听音校准集名称或版本>
calibration_report: <校准报告路径>
thresholds:
  clean_bak: <已批准值>
  clean_ovrl: <已批准值>
  moderate_bak: <已批准值>
  moderate_ovrl: <已批准值>
```

阈值必须从人工听音校准集获得。现有示例阈值不能仅通过改布尔值冒充已经校准。

### 4.2 保守自动判定条件

在“音频已经过上游 VAD”的前提下，`vad_asr_dnsmos_v1` 仅在以下条件全部满足时自动输出 `environment_noise`：

1. 样本不是无效音频、语音信箱或严格语义风险。
2. 三个模型家族均有两路可用推理结果。
3. 每个家族的两路结果在分类文本清洗后均稳定为空。
4. 没有任何未排除路包含有效目标文本。
5. DNSMOS 执行成功。
6. 使用已批准阈值得到 `noise_band=noisy`。
7. 记录自动证据来源、阈值版本和规则版本。

满足条件后，应写入类似以下声学证据：

```json
{
  "state": "environment_confirmed",
  "background_only": true,
  "no_target_speech": true,
  "no_target_speech_trusted": true,
  "source": "vad_asr_dnsmos_v1",
  "quality_policy_version": "quality_policy_server_auto_v1"
}
```

只要任一路存在有效目标文本，就不能仅因 DNSMOS 较低归入 `environment_noise`。这类样本应继续进入 `gold_candidate`、`hardcase`、`semantic_risk` 或人工任务。

### 4.3 串音边界

DNSMOS 不能判断说话人是否属于目标用户。因此，仅靠六路 ASR 和 DNSMOS 不能可靠自动识别“内容清楚但属于旁人”的串音。

若业务要求自动覆盖串音，服务器侧车还必须增加经过校准的串音或目标说话人检测器，并输出：

```text
overlap_detected=true
overlap_detected_trusted=true
```

或输出版本化的目标说话人缺失证据。没有此类检测器时，疑似串音应归入 `hardcase` 或人工任务，而不是伪装成已确认环境噪声。

### 4.4 当前代码需要补齐的内容

在运行第 5 节之前，至少需要完成：

1. 新增服务器质量配置 `configs/quality/dnsmos_p835_server_auto.yaml`。
2. 在五类 selection 配置中加入 `environment_noise` 自动策略配置。
3. 在五类分类器中实现 `vad_asr_dnsmos_v1`：读取家族稳定空结果和已校准 DNSMOS，生成版本化的可信声学证据。
4. 新增测试，覆盖自动环境噪声、存在有效文本不得判噪声、未校准不得判噪声、评分失败不得判噪声。
5. 建议新增一条服务器专用流水线，将全量 DNSMOS 和五类分类放在同一次服务器运行中，避免 Windows 中转。

当前 `quality.asr_anomaly_noise` 明确拒绝 `calibrated=true`，因此不能通过修改现有 YAML 绕过上述开发工作。

## 5. 服务器完整执行顺序

以下命令假定第 4 节改造已经落地。

### 5.1 准备并挂载六路 ASR

如果已经存在 `prepared_asr_v3_${BATCH}.parquet`，并确认它对应当前六路推理结果，可以跳过本小节。

```bash
audio-data pipeline run pipelines/prepare_dataset_v3.yaml \
  --source-name "$BATCH" \
  --config "$DATASET_CFG" \
  --force

audio-data pipeline run pipelines/attach_asr_v3.yaml \
  --source-name "$BATCH" \
  --config "$DATASET_CFG" \
  --force
```

产物：

```text
datasets/stage1/asr/prepared_asr_v3_${BATCH}.parquet
```

### 5.2 在服务器运行全量质量侧车

```bash
audio-data pipeline run pipelines/audio_quality_sidecar.yaml \
  --source-name "$BATCH" \
  --config "$QUALITY_CFG" \
  --force
```

产物：

```text
datasets/stage1/derived/quality_sidecar_${BATCH}.parquet
```

要求：

- 行数与批次样本数一致。
- `dnsmos_status=success`；失败样本必须保留失败原因。
- 成功样本的 `noise_band` 不能全部为 `unknown`。
- 配置和结果中必须包含质量策略版本及校准来源。

### 5.3 运行五类分类

`--source-name` 会自动发现同批次的 `quality_sidecar_${BATCH}.parquet`，并按 `sample_id + original_audio_sha256` 合并。

```bash
audio-data pipeline run pipelines/classify_dataset_five_class_v1.yaml \
  --source-name "$BATCH" \
  --config "$SELECTION_CFG" \
  --force
```

产物：

```text
datasets/stage1/derived/classified_five_class_v1_${BATCH}.parquet
```

生产上更推荐新增服务器一体化流水线，例如：

```text
pipelines/classify_dataset_five_class_v1_server_auto.yaml
```

该流水线应按顺序执行：

```text
prepared_asr_v3
  → 全量 DNSMOS
  → vad_asr_dnsmos_v1 声学证据
  → selection_five_class_v1 分类
  → classified_five_class_v1
```

这样无需把 ASR 或质量侧车传到 Windows 后重新分类。

### 5.4 导出 XLSX

```bash
audio-data review export-summary \
  "datasets/stage1/derived/classified_five_class_v1_${BATCH}.parquet" \
  --output "data/exports/summary_five_class_v1_${BATCH}.xlsx" \
  --max-rows 20000
```

超过 20,000 行时会自动输出：

```text
summary_five_class_v1_${BATCH}-part-001.xlsx
summary_five_class_v1_${BATCH}-part-002.xlsx
```

## 6. 结果验收

### 6.1 分类数量

```bash
python - <<'PY'
import pandas as pd

path = "datasets/stage1/derived/classified_five_class_v1_0914-mixed-30000.parquet"
df = pd.read_parquet(path, columns=[
    "label_category",
    "label_outcome",
    "quality_dnsmos_status",
    "quality_noise_band",
])

print("rows:", len(df))
print("\ncategory:")
print(df["label_category"].fillna("<NA>").value_counts())
print("\noutcome:")
print(df["label_outcome"].fillna("<NA>").value_counts())
print("\nDNSMOS status:")
print(df["quality_dnsmos_status"].fillna("<NA>").value_counts())
print("\nnoise band:")
print(df["quality_noise_band"].fillna("<NA>").value_counts())
PY
```

不要把“`environment_noise` 必须大于 0”写成无条件门禁。如果某批数据确实没有满足强条件的样本，0 可以是真实结果。验收重点是候选集是否按同一规则重算、证据是否完整，以及抽样听音的准确率。

### 6.2 `environment_noise` 证据审计

每条自动环境噪声至少应保留：

```text
category=environment_noise
classification_reason=environment_noise_confirmed
acoustic_evidence.state=environment_confirmed
acoustic_evidence.source=vad_asr_dnsmos_v1
quality.dnsmos_status=success
quality.noise_band=noisy
quality.quality_policy_version=<已批准版本>
selection_policy_version=<当前五类规则版本>
```

不得出现：

- `dnsmos_status=failed` 却自动判为环境噪声。
- `noise_band=unknown` 却自动判为环境噪声。
- 存在有效目标文本，仅因 DNSMOS 低分自动判为环境噪声。
- 未记录阈值或规则版本。

### 6.3 人工抽样验收

首次启用或阈值变更后，对自动 `environment_noise` 至少进行分层抽样听音：

- 按 BAK/OVRL 分数区间抽样。
- 按数据来源和日期抽样。
- 单独抽查极短音频、单字短应答和 SenseVoice 控制标签样本。
- 记录误判为环境噪声的真实目标语音比例。

只有当误判率达到业务接受标准后，才能把对应策略版本作为生产默认。

## 7. 故障排查

### `environment_noise` 仍为 0

依次检查：

1. 分类运行日志是否显示已合并 `quality_sidecar_${BATCH}.parquet`。
2. `noise_band` 是否仍全部为 `unknown`。
3. 是否真的启用了 `vad_asr_dnsmos_v1`，而不是旧的 `selection_five_class_v1` 默认路径。
4. 三个家族是否都能形成稳定空家族；ASR 失败不能冒充稳定空。
5. `acoustic_evidence.state` 是否仍全部为 `unknown`。
6. 质量策略版本与阈值版本是否写入最终 Manifest。

### DNSMOS 全部 `failed`

检查：

- `resampled_16k` 是否为服务器可读路径。
- ONNX 模型路径是否存在。
- `onnxruntime` 和 `soundfile` 是否安装。
- 音频哈希是否与 Manifest 一致。

### Excel 没有 `category` 或 `outcome`

确认导出输入是：

```text
classified_five_class_v1_${BATCH}.parquet
```

不要误用 ASR 聚合结果、prepared Manifest 或旧版 `classified_v3_*`。

## 8. 迁移原则

完成服务器一体化流水线后，推荐停止以下往返流程：

```text
服务器 ASR → Windows 分类 → 服务器质量侧车 → Windows 再分类
```

统一改为：

```text
服务器 ASR
  → 服务器 prepare/attach
  → 服务器声学评分与可信证据
  → 服务器五类分类
  → 服务器导出 XLSX
  → 只传最终 Parquet/XLSX
```

这可以消除 Linux/Windows 音频路径差异，并保证 ASR、声学证据、规则版本和最终分类来自同一次可审计运行。
