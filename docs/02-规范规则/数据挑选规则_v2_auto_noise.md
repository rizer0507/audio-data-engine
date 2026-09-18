# 数据挑选规则 v2（自动噪声）

| 项 | 值 |
| --- | --- |
| 规范版本 | `selection_five_class_v2_auto_noise`（简称 **v2**） |
| 修订日期 | 2026-09-16 |
| 正式入口 | `pipelines/classify_dataset_five_class_v2_auto_noise.yaml` |
| 需求依据 | [028-五分类自动噪声归类与互斥准入修复](../04-改进需求/已完成/028-五分类自动噪声归类与互斥准入修复.md) |
| 并行保留 | [数据挑选规则_v1](./数据挑选规则_v1.md)（`selection_five_class_v1`）不得覆盖 |
| 增量修订 | [数据挑选规则_v2_2_auto_noise](./数据挑选规则_v2_2_auto_noise.md)（029 DNSMOS 联合；独立产物） |

与 v1 冲突时以 **028 / 本文** 为准。v2 产物、缓存、lineage 与 v1 隔离。DNSMOS 在 v2 中仅作补充侧车；真正参与联合判定见 v2.2。

---

## 1. 目标

在上游已做 VAD、多家族双跑 ASR 已就绪的前提下，用家族级稳定空/有字证据 + 版本化音频能量，**自动**完成五分类。人工听音不再是 `environment_noise` 的前置条件；只有模糊或技术证据不足的样本进入 `hardcase`（可附加复核）。

本项目不重新执行 VAD。分类与读音频须在服务器完成；Windows 只接收最终 Parquet/XLSX。

---

## 2. 样本出口

```text
成功进入分类决策的样本 → 必为五类之一（outcome=classified）
全路前置排除 / 物理无效 → outcome=excluded（category=null，非业务第六类）
```

禁止：`category=null` 的业务分类、`manual_annotation` 作主出口、`pending_evidence`。

---

## 3. 五个业务类别（互斥）

| 类别 | 一句话 |
| --- | --- |
| `voicemail` | 任一有效路次命中信箱正则 |
| `semantic_risk` | ≥2 个 `stable_text` 家族同命题明确冲突 |
| `environment_noise` | 家族状态 + 能量自动判噪（见 §6） |
| `gold_candidate` | ≥2 个稳定有字家族支持同一文本 |
| `hardcase` | 前四类未命中的唯一兜底 |

---

## 4. 决策顺序（命中即结束）

```text
voicemail → semantic_risk → environment_noise → gold_candidate → hardcase
```

---

## 5. 家族四态

对每个配置家族的双跑归并：

| 状态 | 条件 |
| --- | --- |
| `stable_text` | 两路成功且 `classify_text` 非空，相等或仅无害差异 |
| `stable_empty` | 两路成功且均为空（真实空/仅标签标点） |
| `unstable` | 一空一有字、实质冲突、或仅一路成功 |
| `unavailable` | 两路均失败/缺失/无法形成证据 |

**推理失败、缺失、外语/回声排除不得计为 `stable_empty`。**

---

## 6. `environment_noise`（全自动）

| noise_kind | 条件摘要 |
| --- | --- |
| `background` | 全部家族 `stable_empty` 且 `energy_state=audible` |
| `silence` | 全部家族 `stable_empty` 且 `energy_state=inaudible` |
| `audio_too_short` | `energy_state=too_short`（未先命中信箱/语义风险） |
| `human_noise` | ≥2 稳定空 + 恰好 1 稳定有字 + audible + 非关键短应答 |

禁止自动入噪：多家族同文支持、严格语义冲突、族内不稳定、失败冒充空、`borderline/failed`、关键短应答。v2 中 DNSMOS 只作补充；`calibrated=false` 不得把噪声候选整批退回人工。DNSMOS 联合置信度 / borderline 收敛见 [v2.2 规范](./数据挑选规则_v2_2_auto_noise.md)。

能量阈值见 `configs/quality/audio_energy_v1.yaml`（改阈值须升 `policy_version`）。

---

## 7. 金标与 hardcase

- 金标：≥2 稳定有字家族两两等价；第三家族稳定空可审计不否决；任一 `unstable` 否决。
- hardcase：唯一兜底；须写具体 `hardcase_reason`；`needs_review` / `review_reason` 仅为附属动作。

---

## 8. 产物与命令

```bash
audio-data pipeline run pipelines/classify_dataset_five_class_v2_auto_noise.yaml \
  --source-name "$BATCH" \
  --config configs/selection/zh_asr_five_class_v2_auto_noise.yaml \
  --force

audio-data review export-summary \
  "datasets/stage1/derived/classified_five_class_v2_auto_noise_${BATCH}.parquet" \
  --output "data/exports/summary_five_class_v2_auto_noise_${BATCH}.xlsx" \
  --max-rows 20000
```

产物：`classified_five_class_v2_auto_noise_*`（禁止覆盖 `classified_five_class_v1_*`）。
