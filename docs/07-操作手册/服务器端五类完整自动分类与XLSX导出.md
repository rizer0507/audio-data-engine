# 服务器端五类完整自动分类与 XLSX 导出

本文说明在服务器已经完成多模型家族 ASR 推理后，如何完成五类分类与 XLSX 导出。

## 0. 最短操作（030 stage1 单命令，推荐）

前置：补齐 `configs/stage1/server.yaml` 必填项（`authorized_gpus`、`engine_python`、`families.qwen.vllm_bin`、`dnsmos.onnx_path`），确认授权仅两张卡。

```bash
cd /data2/data-cp/lizi/audio-data-engine
source /data2/data-cp/zk/env_hub/miniconda312_qw3tts/bin/activate   # 或你们的引擎解释器
export AUTHORIZED_GPUS="<授权卡号,逗号分隔>"   # 禁止照抄历史 4；以部署配置为准

# 1) 首次配置预检
audio-data stage1 check-config --runtime-config configs/stage1/server.yaml

# 2) 正常运行（后台；SSH 断开可继续）
audio-data stage1 run \
  --batch "$BATCH" \
  --source "$SOURCE_DIR" \
  --model qwen=/data2/data-cp/zcl/models/Qwen3-ASR-1___7B \
  --model glm=/data2/data-cp/models/GLM-ASR-Nano-2512 \
  --model sensevoice=/data2/data-cp/xijujun/models/SenseVoiceSmall \
  --runtime-config configs/stage1/server.yaml \
  --gpus "$AUTHORIZED_GPUS"

# 3) 查进度（对账通过前不会显示 100%/完成）
audio-data stage1 status <job_id> --watch
audio-data stage1 status <job_id> --json

# 4) 等待正式成功（仅 reconcile.ok 时 exit 0）
audio-data stage1 wait <job_id>

# 5) 补跑 / 续跑（第三步）
audio-data stage1 resume <job_id>                                    # 中断后续跑；跳过已成功产物
audio-data stage1 retry <job_id> --family glm --run 2 --failed-only  # 只补指定路失败/缺失
audio-data stage1 retry <job_id> --export-only                       # 导出失败只恢复导出，不重跑 ASR
```

**产物位置（权威）：**

| 产物 | 路径 |
|------|------|
| job 状态/快照 | `runs/stage1/jobs/<job_id>/`（`state.json`、`reconcile.json`、`config_snapshot/`、`checkpoints/`、`worker.lock`） |
| cleaned | `datasets/stage1/cleaned/cleaned_${BATCH}.parquet` |
| 六路 ASR | `datasets/stage1/asr/{qwen,glm,sensevoice}_{1,2}_asr_${BATCH}.parquet` |
| prepared / attached | `datasets/stage1/derived/prepared_v3_${BATCH}.parquet`、`prepared_asr_v3_${BATCH}.parquet` |
| 分类（v2.2） | `datasets/stage1/derived/classified_five_class_v2_2_auto_noise_${BATCH}.parquet` |
| XLSX | `data/exports/summary_five_class_v2_2_auto_noise_${BATCH}.xlsx`（>20000 自动 `-part-001…`） |

说明：提交成功 ≠ 批次成功。`status`/`wait` 在对账通过前不得视为完成；`needs_attention` 时 `wait` exit=3。双卡调度默认开启（`scheduler.dual_gpu`）；吞吐对比须服务器实测。下文保留「手工分步」完整步骤，供对照与回退。

### 0.1 六路 ASR 已齐、只做后置分类（031）

不启 ASR、不占 GPU。别名：显式 `--family` > dataset YAML > stage1 固定 `qwen_1/2` 等。详见 [已有六路 ASR 一键分类导出](已有六路ASR一键分类导出.md)。

```bash
python scripts/classify_asr_to_xlsx.py --batch "$BATCH"
# 或：audio-data stage1 classify --batch "$BATCH"
```

默认写出 `summary_five_class_v2_2_auto_noise_${BATCH}.xlsx` 与 `classified_five_class_v2_2_auto_noise_${BATCH}.parquet`。

---

**优先入口（029 / v2.2 DNSMOS 联合）：**

```text
pipelines/classify_dataset_five_class_v2_2_auto_noise.yaml
```

现行联合规则：[数据挑选规则 v2.2](../02-规范规则/数据挑选规则_v2_2_auto_noise.md)。需求依据：[029](../04-改进需求/已完成/029-五分类v2.2-DNSMOS联合判定修复.md)。

**v2 基线入口（028，无 DNSMOS 联合语义）：**

```text
pipelines/classify_dataset_five_class_v2_auto_noise.yaml
```

v2 规则：[数据挑选规则 v2（自动噪声）](../02-规范规则/数据挑选规则_v2_auto_noise.md)。需求依据：[028](../04-改进需求/已完成/028-五分类自动噪声归类与互斥准入修复.md)。

v1 仅作对照/回退。

## 1. 前提与边界

- 输入音频已经过上游 VAD；本项目不重新执行 VAD。
- 默认使用 Qwen、GLM、SenseVoice 三个模型家族，每家族两次独立推理。
- 六路 ASR 结果已注册到批次 dataset 配置，且能按 `sample_id + original_audio_sha256` 对齐。
- 分类在能够读取 `resampled_16k` 音频的 Linux 服务器执行。
- Parquet Manifest 是权威结果；XLSX 是查看、审计和交付格式。
- v2.2 产物与 v1/v2 完全隔离，不覆盖历史 v1、v2、v3、审核包或 release。

成功进入业务分类的样本只会得到以下五类之一：

```text
voicemail
semantic_risk
environment_noise
gold_candidate
hardcase
```

物理无效音频或全部路次被前置规则排除时，仍按输入契约输出 `outcome=excluded, category=null`；它们不是第六个业务类别。其他成功分类样本不得出现空类别或 `manual_annotation` 主出口。

## 2. v2.2 自动噪声规则

分类顺序固定为：

```text
voicemail → semantic_risk → environment_noise → gold_candidate → hardcase
```

同一家族的两次结果先归并为：

- `stable_text`：两路成功有字且内容等价。
- `stable_empty`：两路成功且清洗后均为空。
- `unstable`：一空一有字、双跑内容实质冲突或仅一路成功。
- `unavailable`：两路均失败、缺失或无法形成证据。

失败、缺失、外语排除和回声排除不能冒充 `stable_empty`。

自动环境噪声子类：

- `background`：全部家族 `stable_empty`，且 `energy_state=audible`。
- `silence`：全部家族 `stable_empty`，且 `energy_state=inaudible`。
- `audio_too_short`：`energy_state=too_short`，且未先命中语音信箱或语义风险。
- `human_noise`：至少两个 `stable_empty`、恰好一个 `stable_text`、无不稳定或不可用家族、音频可闻，且唯一文本不是关键短应答。
- `crosstalk`：先满足 `human_noise`，并已有可信 `human_crosstalk_confirmed` 字段。

`failed` 能量、关键短应答、家族内部不稳定或证据不足统一进入 `hardcase`。`borderline` 只有在 DNSMOS 判为 `noisy` 时才可按规则收敛为环境噪声，否则进入 `hardcase`。

DNSMOS 在 v2.2 中是联合证据而不是单独真值：它可以增强 `background/human_noise` 置信度、解决 noisy borderline，或把“全空 + audible + clean + strong speech”的矛盾样本送入 `hardcase`。`dnsmos_p835.yaml` 的 `calibrated` 只影响 legacy `noise_band`；v2.2 使用独立版本化的 `dnsmos_decision_v2_2.yaml`，不要通过手工把 `calibrated` 改成 `true` 来影响分类。

## 3. 代码和依赖准备

进入服务器工程并激活环境：

```bash
cd /data2/data-cp/lizi/audio-data-engine
source .venv/bin/activate

audio-data --help
python -c "import numpy, soundfile; print('audio energy runtime ok')"
```

如果服务器尚未安装当前工作区代码，按项目现有部署方式更新后执行：

```bash
pip install -e .
```

轻量能量计算只依赖 NumPy 和 SoundFile。v2.2 默认启用 DNSMOS 候选评分，还必须保证 ONNX 模型和 `onnxruntime` 在服务器可用：

```bash
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
grep -n "model_path" configs/quality/dnsmos_p835.yaml
test -f /服务器实际路径/sig_bak_ovr.onnx
```

仓库配置中的 `model_path` 可能是开发机路径；部署到服务器后必须改成服务器可读的绝对路径。不要只复制 ASR Parquet 而遗漏模型文件或音频路径。若完全复用已有同哈希 DNSMOS 分数，候选算子不会重复评分，但未命中复用条件的候选仍需要该模型。

## 4. 设置批次变量

0914 三万条示例：

```bash
export BATCH=0914-mixed-30000
export DATASET_CFG=configs/datasets/zh_asr_v3_0914_mixed_30000.yaml
export SELECTION_CFG=configs/selection/zh_asr_five_class_v2_2_auto_noise_0914_mixed_30000.yaml
export ENERGY_CFG=configs/quality/audio_energy_v1.yaml
export DNSMOS_DECISION_CFG=configs/quality/dnsmos_decision_v2_2.yaml
# v2 对照时改用：configs/selection/zh_asr_five_class_v2_auto_noise_0914_mixed_30000.yaml
```

其他批次需要复制并修改批次 dataset 配置和 selection 配置，确保：

- `model_families` 与实际家族一致。
- `run_aliases` 与六路 ASR 的 transcript key 一致。
- `runs` 中登记真实 artifact、execution、模型和输入音频身份。
- selection 配置中的家族和 run ID 与 dataset 配置一致。

不要对其他批次直接复用 0914 的 run ID。

## 5. 检查 ASR 和音频输入

### 5.1 六路 ASR

0914 示例应包含：

```text
glm-base-1
glm-base-2
qwen-basr-1
qwen-basr-2
sv-base-1
sv-base-2
```

检查 dataset 配置中的登记信息：

```bash
grep -nE "run_id:|artifact_id:|execution_id:" "$DATASET_CFG"
```

如果 ASR 结果尚未注册到 catalog/run identity，先按现有 ASR 操作手册完成登记。`attach_asr_v3` 读取登记结果，不按文件行号直接拼接。

### 5.2 音频路径

`quality.audio_energy` 默认读取 `resampled_16k`。抽查 prepared/cleaned Manifest 中对应路径在当前服务器存在且可读。

Windows 路径或另一台服务器的绝对路径必须先完成路径映射。能量读取失败不会被伪造成噪声，而会形成 `energy_state=failed`，最终进入 `hardcase`。

### 5.3 能量阈值

当前 [audio_energy_v1.yaml](../../configs/quality/audio_energy_v1.yaml) 默认值：

```yaml
min_duration_ms: 300
min_rms_dbfs: -50.0
min_peak_dbfs: -40.0
min_non_silent_ratio: 0.02
borderline_margin_db: 3.0
silence_frame_dbfs: -45.0
frame_ms: 20
```

阈值变更必须同时提升 `policy_version`，使缓存和 lineage 可区分。不要只改数值而继续沿用 `audio_energy_v1`。

## 6. 从六路 ASR 生成 prepared Manifest

如果已经存在并确认以下文件对应当前六路 ASR，可以跳到第 7 节：

```text
datasets/stage1/derived/prepared_asr_v3_${BATCH}.parquet
```

否则运行：

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

第一步从同批 cleaned 音频底表建立契约、分组和 reservation；第二步按登记身份挂载各家族 ASR。不要用某一路 ASR 文件代替完整音频底表。

## 7. 一条命令完成能量、DNSMOS 候选和五分类（v2.2）

```bash
audio-data pipeline run pipelines/classify_dataset_five_class_v2_2_auto_noise.yaml \
  --source-name "$BATCH" \
  --config "$SELECTION_CFG" \
  --force
```

流水线实际执行：

1. `quality.audio_energy`：计算时长、RMS、峰值、非静音占比及 `energy_state`。
2. `quality.dnsmos_v2_2_candidates`：对全空有声/临界、两空一有字等候选复用或计算 DNSMOS；按 `sample_id + original_audio_sha256` 合并侧车。
3. `quality.classify`：执行 `selection_five_class_v2_2_auto_noise`。

目标产物：

```text
datasets/stage1/derived/classified_five_class_v2_2_auto_noise_${BATCH}.parquet
```

若已有全量 `quality_sidecar_*`，source-name 会将其挂到 classify；候选算子也可通过参数复用同哈希分数。缺模型时候选失败记为 DNSMOS unavailable，分类审计回退 v2，不伪造 noisy。

### 7.0 v2 基线（无 DNSMOS 联合）

```bash
audio-data pipeline run pipelines/classify_dataset_five_class_v2_auto_noise.yaml \
  --source-name "$BATCH" \
  --config configs/selection/zh_asr_five_class_v2_auto_noise_0914_mixed_30000.yaml \
  --force
```

产物：`classified_five_class_v2_auto_noise_${BATCH}.parquet`。

### 7.1 不强制重算 DNSMOS

若仅复用已有侧车分数、或本轮允许 unavailable 回退 v2：确保侧车按哈希可合并即可。显式关闭联合决策时，在 selection 配置中设 `dnsmos_decision.enabled: false`（会写入 `decision_trace.dnsmos_decision=disabled`）。

### 7.2 续跑

需要从已有 run 续跑时，使用原命令并追加：

```bash
--resume runs/<run目录>
```

不要同时对同一目标产物执行多个写入任务。

## 8. 导出最终 XLSX

```bash
audio-data review export-summary \
  "datasets/stage1/derived/classified_five_class_v2_2_auto_noise_${BATCH}.parquet" \
  --output "data/exports/summary_five_class_v2_2_auto_noise_${BATCH}.xlsx" \
  --max-rows 20000
```

超过 20,000 行时自动写为：

```text
data/exports/summary_five_class_v2_2_auto_noise_${BATCH}-part-001.xlsx
data/exports/summary_five_class_v2_2_auto_noise_${BATCH}-part-002.xlsx
```

导出命令会在终端打印 `type`、`category`、`noise_kind`、`dnsmos_noise_state`、`dnsmos_status`、`v2_fallback` 与 `borderline_resolved_by_dnsmos` 计数。

XLSX 至少包含：

- 五类及子类：`category`、`semantic_subtype`、`noise_kind`。
- 决策信息：`classification_reason`、`classification_source`、`classification_confidence`、`rule_version`、`evidence_sources`、`decision_trace`。
- 家族信息：四种家族计数、`family_state_by_name`、选中文本的 family/run。
- 能量信息：`duration_ms`、`rms_dbfs`、`peak_dbfs`、`non_silent_ratio`、`energy_state`、`energy_policy_version`。
- DNSMOS：`dnsmos_sig/bak/ovrl/status`、`dnsmos_noise_state`、`dnsmos_speech_state`、`dnsmos_decision_policy_version`。
- 质量标签：`quality_tag`、`background_quality_risk`、`v2_fallback`、`borderline_resolved_by_dnsmos`。
- 复核信息：`needs_review`、`review_reason`、`hardcase_reason`。
- 各 ASR run 的原始文本列。

如需同时输出带标准化 `type` 的 Manifest：

```bash
audio-data review export-summary \
  "datasets/stage1/derived/classified_five_class_v2_2_auto_noise_${BATCH}.parquet" \
  --output "data/exports/summary_five_class_v2_2_auto_noise_${BATCH}.xlsx" \
  --output-manifest "datasets/stage1/derived/summary_five_class_v2_2_auto_noise_${BATCH}.parquet" \
  --max-rows 20000
```

## 9. 分类结果验收

### 9.1 自动测试

```bash
pytest tests/test_selection_five_class_v2_2_auto_noise.py \
       tests/test_selection_five_class_v2_auto_noise.py \
       tests/test_selection_five_class_v1.py -q
```

当前仓库对应回归结果应为全部通过（v1+v2+v2.2）。若代码继续演进，以实际收集到的测试数为准，不要把固定数量当作永久门禁。

### 9.2 Parquet 计数核对

```bash
python - <<'PY'
import os
import pandas as pd

batch = os.environ["BATCH"]
path = f"datasets/stage1/derived/classified_five_class_v2_2_auto_noise_{batch}.parquet"
cols = [
    "label_outcome",
    "label_category",
    "label_noise_kind",
    "label_energy_state",
    "label_hardcase_reason",
    "label_dnsmos_noise_state",
    "label_v2_fallback",
    "label_borderline_resolved_by_dnsmos",
]
df = pd.read_parquet(path, columns=cols)

print("rows:", len(df))
print("\noutcome:")
print(df["label_outcome"].fillna("<NA>").value_counts())
print("\ncategory:")
print(df["label_category"].fillna("<NA>").value_counts())
print("\nnoise kind:")
print(df["label_noise_kind"].fillna("<NA>").value_counts())
print("\ndnsmos_noise_state:")
print(df["label_dnsmos_noise_state"].fillna("<NA>").value_counts())
print("\nv2_fallback:", int(df["label_v2_fallback"].fillna(False).astype(bool).sum()))
print("borderline_resolved:", int(df["label_borderline_resolved_by_dnsmos"].fillna(False).astype(bool).sum()))

classified = df[df["label_outcome"].eq("classified")]
allowed = {
    "voicemail",
    "semantic_risk",
    "environment_noise",
    "gold_candidate",
    "hardcase",
}
actual = set(classified["label_category"].dropna().astype(str))
assert classified["label_category"].notna().all(), "classified 中存在空 category"
assert actual <= allowed, f"出现五类之外的 category: {sorted(actual - allowed)}"
assert not df["label_outcome"].eq("manual_annotation").any(), "出现 manual_annotation 主出口"
print("\nvalidation: OK")
PY
```

验收原则：

- `classified` 行的五类计数之和等于 `classified` 样本数。
- `excluded` 可以有 `category=null`，但必须有明确排除原因。
- `environment_noise` 应能按 `background/human_noise/silence/audio_too_short/crosstalk` 拆分。
- 不要求任何真实批次的噪声数量必须大于 0；应检查符合规则的候选是否确实被归入对应子类。
- `energy_state=failed`、关键短应答和家族不稳定应进入 `hardcase`；`borderline` 仅可在 DNSMOS=`noisy` 且满足对应家族规则时收敛为噪声。

## 10. 常见故障

### 10.1 `energy_state=failed` 很多

检查：

- `resampled_16k` 是否为当前服务器可读路径。
- 音频文件是否存在、权限是否正确。
- `soundfile` 是否能读取实际编码。
- Manifest 的音频哈希和路径是否来自同一批次。

不要把 `failed` 当静音或背景噪声；代码会将其送入 `hardcase`。

### 10.2 `environment_noise` 仍为 0

依次检查：

1. 实际运行的是 v2.2 pipeline，而不是 `classify_dataset_five_class_v1.yaml`。
2. 输出文件名是否包含 `classified_five_class_v2_2_auto_noise_`（或对照用的 `v2_auto_noise_` / `v1_`）。
3. selection 配置的 run ID 是否与 Manifest transcript key 完全一致。
4. 家族是否形成 `stable_empty`，而不是 `unstable/unavailable`。
5. `energy_state` 是否正常产生，而不是全部 `failed/borderline`。
6. 全空样本是否满足全部家族双跑成功空。
7. 人声噪声候选是否因关键短应答或家族缺失正确进入了 `hardcase`。

### 10.3 DNSMOS 失败或大量 `unavailable`

检查 `configs/quality/dnsmos_p835.yaml` 的 ONNX 模型路径、`onnxruntime`、音频路径和文件哈希。候选评分失败时 v2.2 会写 `dnsmos_status`/`dnsmos_noise_state=unavailable`，并按配置审计回退到 v2 强规则；不会伪造 `noisy`。修复依赖后用同一 v2.2 命令加 `--force` 重跑，不要回到 Windows 重新分类，也不要跳过一个不存在于 v2.2 流水线中的旧 `diagnose_asr_anomaly` 步骤。

### 10.4 导出时报缺少 `classification_bucket/type`

确认导出输入是 v2 classified Manifest，而不是 prepared ASR、能量中间产物或单路 ASR 文件。分类器会为五类和排除结果写入对应分桶字段。

### 10.5 结果大部分进入 `hardcase`

按 `hardcase_reason` 排查：

- `energy_failed`：音频路径或读取问题。
- `energy_borderline`：能量位于阈值边界。
- `family_unstable`：同一家族双跑不稳定。
- `critical_short_response_unsupported`：唯一文本是关键短应答。
- `insufficient_family_evidence`：家族失败、缺失或混合状态。
- `substantive_divergence_non_strict_risk`：多家族内容实质分歧但不构成严格语义风险。

不要通过把失败路次改为空票来减少 hardcase。

## 11. v1 对照运行

如需影子对照，可以另跑 v1：

```bash
audio-data pipeline run pipelines/classify_dataset_five_class_v1.yaml \
  --source-name "$BATCH" \
  --config configs/selection/zh_asr_five_class_v1_0914_mixed_30000.yaml
```

v1 产物为 `classified_five_class_v1_*`；v2 产物为 `classified_five_class_v2_auto_noise_*`；v2.2 产物为 `classified_five_class_v2_2_auto_noise_*`。三者不能互相覆盖。

## 12. 推荐的服务器交付流程

```text
服务器完成六路 ASR
  → 登记 ASR run identity/artifact
  → prepare_dataset_v3
  → attach_asr_v3
  → classify_dataset_five_class_v2_2_auto_noise
     ├─ audio_energy
     ├─ dnsmos_v2_2_candidates（复用侧车或按需评分）
     └─ five_class_v2_2
  → export-summary
  → 仅传输最终 Parquet 和 XLSX
```

根目录命令速查见 `单条流水线执行命令.txt` §2f-v3-029 / §2f-v3-028。
