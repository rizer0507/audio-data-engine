# 数据挑选规则 v2.2（DNSMOS 联合自动噪声）



| 项 | 值 |

| --- | --- |

| 规范版本 | `selection_five_class_v2_2_auto_noise`（简称 **v2.2**） |

| 修订日期 | 2026-09-16 |

| 正式入口 | `pipelines/classify_dataset_five_class_v2_2_auto_noise.yaml` |

| 需求依据 | [029-五分类 v2.2 DNSMOS 联合判定修复](../04-改进需求/已完成/029-五分类v2.2-DNSMOS联合判定修复.md) |

| 基线 | [数据挑选规则_v2_auto_noise](./数据挑选规则_v2_auto_noise.md)（028）；本规范仅增量修订 DNSMOS 角色 |

| 完整总览 | [五类分拣规则与决策流程](./五类分拣规则与决策流程.md)（类别 + 全流程；优先对照） |

| 并行保留 | v1 / v2 产物与缓存不得覆盖 |



与 v2 冲突时：**家族四态、能量规则、五类顺序以 028/v2 为准**；DNSMOS 联合分支以 **029 / 本文** 为准。完整决策图与五类说明见总览文档。



---



## 1. 目标



在 v2 的家族状态 + 音频能量之上，引入版本化 DNSMOS 决策状态（`dnsmos_noise_state` / `dnsmos_speech_state`），用于：



- 增强 `background` / `human_noise` 置信度；

- 在 `energy_state=borderline` 且 DNSMOS=`noisy` 时自动收敛到环境噪声；

- 识别「全空 ASR + 有声 + clean+strong speech」矛盾 → `hardcase`；

- 金标候选可附 `quality_tag=background_noisy`，**不得**因 DNSMOS 改主类。



DNSMOS 不可用时审计回退到 v2 强规则，不整批进 hardcase，不伪造 noisy。



---



## 2. 不变部分（继承 v2）



- 五类：`voicemail / semantic_risk / environment_noise / gold_candidate / hardcase`

- 决策顺序：`voicemail → semantic_risk → environment_noise → gold_candidate → hardcase`

- 家族四态：`stable_text / stable_empty / unstable / unavailable`

- `audio_too_short` / 明确 `silence` 不依赖 DNSMOS

- `hardcase` 唯一业务兜底



---



## 3. DNSMOS 决策配置



见 `configs/quality/dnsmos_decision_v2_2.yaml`（`policy_version: dnsmos_decision_v2_2`）。



| noise_state | 条件 |

| --- | --- |

| `unavailable` | status≠success 或分数缺失 |

| `noisy` | BAK &lt; noisy_bak **或** OVRL &lt; noisy_ovrl |

| `clean` | BAK ≥ clean_bak **且** OVRL ≥ clean_ovrl |

| `moderate` | 其余有效分数 |



SIG 仅形成 `speech_state=strong|weak|unknown`，不单独证明目标说话人。



---



## 4. 联合判定要点



| 场景 | 结果 |

| --- | --- |

| 全空 + audible + DNSMOS noisy | `environment_noise/background`，confidence=high |

| 全空 + audible + DNSMOS unavailable | `background`，medium（v2 回退） |

| 全空 + audible + clean + strong | `hardcase/empty_asr_but_clean_strong_speech` |

| 全空 + borderline + noisy | `background`，medium |

| 两空一有字 + audible + noisy | `human_noise`，high |

| 两空一有字 + borderline + noisy | `human_noise`，medium |

| ≥2 稳定同文 + DNSMOS noisy | 仍 `gold_candidate` + `quality_tag=background_noisy` |



---



## 5. 产物与命令



```bash

audio-data pipeline run pipelines/classify_dataset_five_class_v2_2_auto_noise.yaml \

  --source-name "$BATCH" \

  --config configs/selection/zh_asr_five_class_v2_2_auto_noise.yaml \

  --force



audio-data review export-summary \

  "datasets/stage1/derived/classified_five_class_v2_2_auto_noise_${BATCH}.parquet" \

  --output "data/exports/summary_five_class_v2_2_auto_noise_${BATCH}.xlsx" \

  --max-rows 20000

```



产物：`classified_five_class_v2_2_auto_noise_*`（禁止覆盖 v1 / v2_auto_noise）。


