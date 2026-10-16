# Tag 规则库总览

> 本文档为**派生产物**，由 `audio-data tags docs --generate` 从 `configs/tagging/library/` 生成。
> 修改定义后请重新生成；可用 `audio-data tags docs --check` 检查是否过期。
> 生成与查看过程**不调用 ASR / 声学推理**。

生成时间：2026-10-15 05:56:38 UTC
定义库目录：`D:/Work/audio-data-engine/configs/tagging/library`
source_content_hash: `d88e52babf1f182b6ef1240549e5218e8adc364775c7178006e7da6ed427ce6e`

<!-- tag-library-doc:begin -->

## 目录

| tag_id | 名称 | 版本 | 状态 | 规则数 | 定义文件 |
|--------|------|------|------|--------|----------|
| `critical_short` | 关键短应答 | `1` | enabled | 1 | [`critical_short.yaml`](../../configs/tagging/library/critical_short.yaml) |
| `human_noise_sig_low` | 人声且SIG低档 | `1` | enabled | 1 | [`human_noise_sig_low.yaml`](../../configs/tagging/library/human_noise_sig_low.yaml) |
| `polarity_conflict` | 极性冲突 | `1` | enabled | 1 | [`polarity_conflict.yaml`](../../configs/tagging/library/polarity_conflict.yaml) |
| `qwen_stable_empty_diag` | Qwen稳定空诊断 | `1` | enabled | 1 | [`qwen_stable_empty_diag.yaml`](../../configs/tagging/library/qwen_stable_empty_diag.yaml) |
| `qwen_unclear_speech_candidate` | 疑似含糊人声（Qwen有字、跨模型分歧且SIG低） | `2` | enabled | 1 | [`qwen_unclear_speech_candidate.yaml`](../../configs/tagging/library/qwen_unclear_speech_candidate.yaml) |
| `semantic_risk_text_disagreement` | 语义风险词分歧或Qwen短字分歧 | `5` | enabled | 2 | [`semantic_risk_text_disagreement.yaml`](../../configs/tagging/library/semantic_risk_text_disagreement.yaml) |
| `silence_strict_zero` | 严格全零静默 | `1` | enabled | 1 | [`silence_strict_zero.yaml`](../../configs/tagging/library/silence_strict_zero.yaml) |
| `tag_1` | SenseVoice/GLM单字且与Qwen分歧 | `1` | enabled | 1 | [`tag_1.yaml`](../../configs/tagging/library/tag_1.yaml) |
| `voicemail_raw` | 语音信箱原文命中 | `1` | enabled | 1 | [`voicemail_raw.yaml`](../../configs/tagging/library/voicemail_raw.yaml) |

---

## 公共条件片段

来源：`D:/Work/audio-data-engine/configs/tagging/fragments_v1.yaml`

### `all_families_dual_success` (v1)

当前配置的全部家族双跑成功

```yaml
capability: family_all_dual_success
```

### `sensevoice_glm_equivalent` (v1)

SenseVoice 与 GLM 代表文本无害等价

```yaml
capability: texts_equivalent
left:
  source: family_representative
  family: sensevoice
right:
  source: family_representative
  family: glm
```

### `three_families_stable_text` (v1)

SenseVoice / GLM / Qwen 均稳定有字

```yaml
capability: family_state
families:
- sensevoice
- glm
- qwen
state: stable_text
quantifier: all
```

## `critical_short` — 关键短应答

- **版本**：`1`
- **状态**：enabled
- **定义文件**：`D:/Work/audio-data-engine/configs/tagging/library/critical_short.yaml`

### 业务定义

SenseVoice 家族状态为 stable_text，且其代表文本属于现行 selection 配置中的关键短应答目录
（capability: is_critical_short_response，阈值与目录出处见 selection_config）。

### 摘要

SenseVoice 稳定代表文本属于关键短应答目录（可与 tag_1 同时命中）

### 依赖

- 公共片段：（无显式声明）
- 词表：（无显式声明）
- 需要声学证据：否

### 例子

- **命中** (`hit`):
  - SenseVoice 稳定文本为「好」且属于关键短应答目录。
    - 说明：规则示例，非真实模型验证记录。
- **不命中** (`miss`):
  - SenseVoice 稳定文本为较长句子，不在关键短应答目录。
- **证据不足 / 无法判断** (`unknown`):
  - SenseVoice 无稳定代表文本（非 stable_text）。

> 例子为规则示意，不能宣称为未做过的真实模型验证。

### 规则

#### `rule_critical_short_v1` (v1)

- enabled: `True`
- 说明：与 tag_1 可重叠的纯文本规则：SenseVoice 关键短应答
- 完整条件（展开公共片段）：
```yaml
all:
- capability: family_state
  family: sensevoice
  state: stable_text
- capability: is_critical_short_response
  text:
    source: family_representative
    family: sensevoice
```

---

## `human_noise_sig_low` — 人声且SIG低档

- **版本**：`1`
- **状态**：enabled
- **定义文件**：`D:/Work/audio-data-engine/configs/tagging/library/human_noise_sig_low.yaml`

### 业务定义

Silero VAD 判定 speech present，且 DNSMOS SIG 低于 selection/acoustic 配置中的低档阈值
（capabilities: vad_speech_present + dnsmos_sig_low）。默认 evidence_mode=existing 时，
缺声学证据为 unknown，不伪造、不自动重跑推理。

### 摘要

VAD present 且 DNSMOS SIG 低档（依赖声学；默认 existing 模式缺证据为 unknown）

### 依赖

- 公共片段：（无显式声明）
- 词表：（无显式声明）
- 需要声学证据：是

### 例子

- **命中** (`hit`):
  - 已有 VAD=present 且 DNSMOS SIG 低于阈值。
    - 说明：规则示例，非真实模型验证记录。
- **不命中** (`miss`):
  - VAD=absent 或 SIG 不在低档。
- **证据不足 / 无法判断** (`unknown`):
  - existing 模式下样本缺少 VAD/DNSMOS 证据。

> 例子为规则示意，不能宣称为未做过的真实模型验证。

### 规则

#### `rule_human_noise_sig_low_v1` (v1)

- enabled: `True`
- 完整条件（展开公共片段）：
```yaml
all:
- capability: vad_speech_present
- capability: dnsmos_sig_low
```

---

## `polarity_conflict` — 极性冲突

- **版本**：`1`
- **状态**：enabled
- **定义文件**：`D:/Work/audio-data-engine/configs/tagging/library/polarity_conflict.yaml`

### 业务定义

沿用 selection_v3 / 五类分拣中的极性冲突判定（capability: polarity_conflict），
阈值与极性词表出处见运行配置引用的 selection_config。

### 摘要

现行极性冲突语义（模板）

### 依赖

- 公共片段：（无显式声明）
- 词表：（无显式声明）
- 需要声学证据：否

### 例子

- **命中** (`hit`):
  - 文本满足现行 polarity_conflict 能力条件。
    - 说明：规则示例，非真实模型验证记录。
- **不命中** (`miss`):
  - 无极性冲突。
- **证据不足 / 无法判断** (`unknown`):
  - 证据不足以完成极性判定。

> 例子为规则示意，不能宣称为未做过的真实模型验证。

### 规则

#### `rule_polarity_conflict_v1` (v1)

- enabled: `True`
- 完整条件（展开公共片段）：
```yaml
capability: polarity_conflict
```

---

## `qwen_stable_empty_diag` — Qwen稳定空诊断

- **版本**：`1`
- **状态**：enabled
- **定义文件**：`D:/Work/audio-data-engine/configs/tagging/library/qwen_stable_empty_diag.yaml`

### 业务定义

Qwen 家族状态明确为 stable_empty（capability: family_state）。
不得用 not(texts_equivalent) 等间接方式推断「空」而误命中其它分歧类 tag。

### 摘要

明文要求 Qwen=stable_empty；不经 not(一致) 间接推断

### 依赖

- 公共片段：（无显式声明）
- 词表：（无显式声明）
- 需要声学证据：否

### 例子

- **命中** (`hit`):
  - Qwen 两路成功且文本均为空 → stable_empty。
    - 说明：规则示例，非真实模型验证记录。
- **不命中** (`miss`):
  - Qwen 为 stable_text。
- **证据不足 / 无法判断** (`unknown`):
  - Qwen 路次失败导致家族状态非稳定态。

> 例子为规则示意，不能宣称为未做过的真实模型验证。

### 规则

#### `rule_qwen_stable_empty_v1` (v1)

- enabled: `True`
- 完整条件（展开公共片段）：
```yaml
capability: family_state
family: qwen
state: stable_empty
```

---

## `qwen_unclear_speech_candidate` — 疑似含糊人声（Qwen有字、跨模型分歧且SIG低）

- **版本**：`2`
- **状态**：enabled
- **定义文件**：`D:/Work/audio-data-engine/configs/tagging/library/qwen_unclear_speech_candidate.yaml`

### 业务定义

用于寻找说话含糊、可能难以听清的候选，供人工听音确认，不代表已确认发音含糊。
必须同时满足：
1. DNSMOS SIG 低档（evidence 配置阈值 dnsmos_weak_sig=2.8，capability dnsmos_sig_low，比较为 SIG<=阈值）。
2. Qwen 至少一条成功路次的清洗前原始转写去除首尾空白后非空；即使清洗后为空，仍算发生过转写。
3. SenseVoice、GLM、Qwen 各至少有一条可用于比较的成功路次；比较全部跨家族成功路次的
   清洗后 comparison_text，任意一对按现行无害等价判定不一致即构成分歧。
不要求任何家族双跑稳定，也不要求全部六路成功。成功空输出参与比较，有字与空构成分歧；
失败、缺失和排除路次不参与比较，不能伪装为空输出。某家族没有可比较成功结果时，分歧为 unknown。
不依赖 Silero VAD；VAD 缺失、absent 或失败均不参与本 tag 判定，也不因本 tag 补算 VAD。
默认 evidence_mode=existing；缺少 DNSMOS 证据时为 unknown。compute_acoustic 模式下文本条件可成立时
仅补算 DNSMOS，不触发 VAD/energy。

### 摘要

DNSMOS SIG 低档；Qwen 至少一次成功运行产生原始非空转写；SenseVoice/GLM/Qwen 的成功结果存在跨模型分歧。
允许双跑不稳定以及成功空输出；不依赖 Silero VAD；缺 DNSMOS 时 compute_acoustic 仅补算 DNSMOS。

### 依赖

- 公共片段：（无显式声明）
- 词表：（无显式声明）
- 需要声学证据：是

### 例子

- **命中** (`hit`):
  - SIG=1.5；SV/GLM 成功文本均为「需要」，Qwen 成功文本为「不需要」；无 VAD 仍命中。
    - 说明：规则示例，非真实模型验证记录，也非已确认含糊人声。
  - SIG<=2.8；SV/GLM 成功输出为空，Qwen 至少一次成功原始有字；有字与空构成跨模型分歧。
  - SIG<=2.8；Qwen 一路失败、另一路成功为「不需要」，SV/GLM 各有成功结果「需要」；仍命中。
- **不命中** (`miss`):
  - 三家成功结果仅有标点等无害差异，全部等价；即使 SIG 很低也不命中。
  - Qwen 有成功路次，但所有成功原始输出均为空；失败路次残留有字也不能满足 Qwen 有字。
  - SIG>2.8；即使 Qwen 有字且模型分歧，也不命中。
- **证据不足 / 无法判断** (`unknown`):
  - Qwen 有字且三家分歧，但 existing 模式下缺少 DNSMOS 证据。
  - 某家族全部失败或缺失，没有可比较成功结果；不能把它当作空输出制造分歧。

> 例子为规则示意，不能宣称为未做过的真实模型验证。

### 规则

#### `rule_qwen_unclear_speech_candidate_v1` (v2)

- enabled: `True`
- 说明：SIG低档 + Qwen至少一次成功原始有字 + 三家成功结果任意跨家族分歧；不依赖VAD。
- 完整条件（展开公共片段）：
```yaml
all:
- capability: dnsmos_sig_low
- capability: family_has_successful_raw_text
  family: qwen
- capability: cross_family_text_disagreement
  families:
  - sensevoice
  - glm
  - qwen
  include_empty: true
```

---

## `semantic_risk_text_disagreement` — 语义风险词分歧或Qwen短字分歧

- **版本**：`5`
- **状态**：enabled
- **定义文件**：`D:/Work/audio-data-engine/configs/tagging/library/semantic_risk_text_disagreement.yaml`

### 业务定义

在全部配置家族双跑成功、SenseVoice/GLM/Qwen均为stable_text，且Qwen稳定代表文本未命中semantic_risk_qwen_exclusions的前提下：
A：任一六路原文命中semantic_risk_keywords中的多字需求、意愿、接受、明确拒绝或指定准入表达，且SV与GLM无害等价、Qwen与SV不等价；
或B：Qwen的稳定代表文本恰好为1个、2个或3个汉字（整段文本长度等于汉字数），且三族代表文本两两不等价；SV和GLM不限文本长度。
准入词表不含单字；指定准入表达包括“对呀、谢谢、我有、有啊、可以、确实、对啊、是啊、行啊”等。
B要求SV≉GLM、SV≉Qwen、GLM≉Qwen三对均成立；不要求准入词表命中；一致性与字数判断沿用现行引擎。
排除词表对两个分支均生效：Qwen清洗后的稳定代表文本包含“不需要、不用、不要、不可用、没有”任一项即不检出本tag；“沒有”同样排除。
排除只检查Qwen，不因SV/GLM中出现排除词而剔除Qwen肯定的样本。B不依赖准入词表，但仍执行排除词表。

### 摘要

全族双跑成功且三族稳定有字、Qwen未命中排除词；语义准入词命中并SG同文Qwen异文，或Qwen稳定代表文本为1~3个汉字且三族两两不等价。

### 依赖

- 公共片段：`all_families_dual_success`, `three_families_stable_text`, `sensevoice_glm_equivalent`
- 词表：`semantic_risk_keywords`, `semantic_risk_qwen_exclusions`
- 需要声学证据：否

### 例子

- **命中** (`hit`):
  - SV/GLM各双跑稳定为“需要贷款”，Qwen双跑稳定为“天气晴朗”，原文关键词命中，分支A成立。
  - SV/GLM/Qwen各双跑稳定为“甲/乙/丙”；Qwen为单字且三族两两不等价，分支B成立。
  - SV/GLM/Qwen稳定为“甲乙/丙丁/戊己”；Qwen为双字且三族两两不等价，分支B成立。
  - SV/GLM稳定为“甲乙丙/丁戊己”，Qwen稳定为“庚辛壬”；Qwen为三字且三族两两不等价，分支B成立。
- **不命中** (`miss`):
  - 三族均稳定为“需要贷款”；仅有关键词命中，没有分歧，且Qwen不是1~3字。
  - SV/GLM/Qwen稳定为“甲/甲/甲”；Qwen虽为单字，但三族无分歧。
  - SV/GLM稳定为“甲乙”，Qwen稳定为“甲丙”；无关键词且虽有分歧，但Qwen不是恰好1~3个汉字整段。
  - SV/GLM稳定为“甲/乙”，Qwen稳定为“甲”；Qwen与SV等价，不满足三族两两不等价。
  - SV/GLM稳定为“需要”，Qwen稳定为“不需要了谢谢”；虽满足原分支A，仍因Qwen命中排除词“不需要”而不检出。
- **证据不足 / 无法判断** (`unknown`):
  - 比较所需的家族代表文本无法取得时，该比较为unknown；整条规则仍按前置条件及三态AND合成。

> 例子为规则示意，不能宣称为未做过的真实模型验证。

### 规则

#### `rule_semantic_risk_keyword_sg_qwen_v1` (v2)

- enabled: `True`
- 说明：分支A：Qwen未命中排除词，多字强倾向语义词命中且SV/GLM同文、Qwen异文。
- 完整条件（展开公共片段）：
```yaml
all:
- fragment: all_families_dual_success
  version: '1'
  expanded:
    capability: family_all_dual_success
- fragment: three_families_stable_text
  version: '1'
  expanded:
    capability: family_state
    families:
    - sensevoice
    - glm
    - qwen
    state: stable_text
    quantifier: all
- not:
    capability: lexicon_match
    lexicon: semantic_risk_qwen_exclusions
    text:
      source: family_representative
      family: qwen
- fragment: sensevoice_glm_equivalent
  version: '1'
  expanded:
    capability: texts_equivalent
    left:
      source: family_representative
      family: sensevoice
    right:
      source: family_representative
      family: glm
- capability: texts_not_equivalent
  left:
    source: family_representative
    family: qwen
  right:
    source: family_representative
    family: sensevoice
- any:
  - capability: lexicon_match
    lexicon: semantic_risk_keywords
    text:
      source: route_raw
      run_id: sensevoice_1
  - capability: lexicon_match
    lexicon: semantic_risk_keywords
    text:
      source: route_raw
      run_id: sensevoice_2
  - capability: lexicon_match
    lexicon: semantic_risk_keywords
    text:
      source: route_raw
      run_id: glm_1
  - capability: lexicon_match
    lexicon: semantic_risk_keywords
    text:
      source: route_raw
      run_id: glm_2
  - capability: lexicon_match
    lexicon: semantic_risk_keywords
    text:
      source: route_raw
      run_id: qwen_1
  - capability: lexicon_match
    lexicon: semantic_risk_keywords
    text:
      source: route_raw
      run_id: qwen_2
```

#### `rule_semantic_risk_single_char_pairwise_v1` (v5)

- enabled: `True`
- 说明：分支B：Qwen未命中排除词、稳定代表文本为1~3个汉字，且三族两两不等价；SV/GLM不限长度，无需准入关键词。
- 完整条件（展开公共片段）：
```yaml
all:
- fragment: all_families_dual_success
  version: '1'
  expanded:
    capability: family_all_dual_success
- fragment: three_families_stable_text
  version: '1'
  expanded:
    capability: family_state
    families:
    - sensevoice
    - glm
    - qwen
    state: stable_text
    quantifier: all
- not:
    capability: lexicon_match
    lexicon: semantic_risk_qwen_exclusions
    text:
      source: family_representative
      family: qwen
- any:
  - capability: single_han_char
    text:
      source: family_representative
      family: qwen
  - all:
    - capability: han_char_count
      text:
        source: family_representative
        family: qwen
      op: eq
      n: 2
    - capability: text_length
      text:
        source: family_representative
        family: qwen
      op: eq
      n: 2
  - all:
    - capability: han_char_count
      text:
        source: family_representative
        family: qwen
      op: eq
      n: 3
    - capability: text_length
      text:
        source: family_representative
        family: qwen
      op: eq
      n: 3
- capability: texts_not_equivalent
  left:
    source: family_representative
    family: sensevoice
  right:
    source: family_representative
    family: glm
- capability: texts_not_equivalent
  left:
    source: family_representative
    family: sensevoice
  right:
    source: family_representative
    family: qwen
- capability: texts_not_equivalent
  left:
    source: family_representative
    family: glm
  right:
    source: family_representative
    family: qwen
```

---

## `silence_strict_zero` — 严格全零静默

- **版本**：`1`
- **状态**：enabled
- **定义文件**：`D:/Work/audio-data-engine/configs/tagging/library/silence_strict_zero.yaml`

### 业务定义

严格全通道采样值为零（capability: strict_all_channels_zero），与「低能量」不等同。
依赖能量/静默类声学证据；existing 模式缺证据为 unknown。

### 摘要

严格全通道采样为零（与低能量不混淆）

### 依赖

- 公共片段：（无显式声明）
- 词表：（无显式声明）
- 需要声学证据：是

### 例子

- **命中** (`hit`):
  - 能量分析确认全通道采样严格为零。
    - 说明：规则示例，非真实模型验证记录。
- **不命中** (`miss`):
  - 存在非零采样（即使能量很低）。
- **证据不足 / 无法判断** (`unknown`):
  - 缺少能量/静默证据。

> 例子为规则示意，不能宣称为未做过的真实模型验证。

### 规则

#### `rule_silence_strict_zero_v1` (v1)

- enabled: `True`
- 完整条件（展开公共片段）：
```yaml
capability: strict_all_channels_zero
```

---

## `tag_1` — SenseVoice/GLM单字且与Qwen分歧

- **版本**：`1`
- **状态**：enabled
- **定义文件**：`D:/Work/audio-data-engine/configs/tagging/library/tag_1.yaml`

### 业务定义

在全部 ASR 家族双跑成功、三族均有稳定代表文本的前提下：SenseVoice 与 GLM 代表文本无害等价，
且该共同代表文本命中 tag_1 词表并恰好为一个汉字；同时 Qwen 代表文本与 SenseVoice 代表文本
按无害等价判定为不一致。可与 critical_short 等其它 tag 同时命中。

### 摘要

全族双跑成功；三族稳定有字；SenseVoice/GLM 无害等价且共同代表文本命中 tag_1 词表并恰为一汉字；
Qwen 与 SenseVoice 按无害等价不一致。

### 依赖

- 公共片段：`all_families_dual_success`, `three_families_stable_text`, `sensevoice_glm_equivalent`
- 词表：`tag_1`
- 需要声学证据：否

### 例子

- **命中** (`hit`):
  - SenseVoice/GLM 均为「好」，Qwen 为「要」；词表命中且单汉字。
    - 说明：规则示例，非真实模型验证记录。
- **不命中** (`miss`):
  - SenseVoice/GLM/Qwen 均为「好」——无分歧，不命中。
  - SenseVoice/GLM 为「好啊」——非整段单汉字，不命中。
- **证据不足 / 无法判断** (`unknown`):
  - Qwen 某一路失败导致家族非 dual_success——前置片段不满足或整规则 unknown。

> 例子为规则示意，不能宣称为未做过的真实模型验证。

### 规则

#### `rule_tag_1_v1` (v1)

- enabled: `True`
- 说明：文档 §4.2 tag_1 完整条件
- text_binding:
```yaml
sg_text:
  source: family_representative
  family: sensevoice
```
- 完整条件（展开公共片段）：
```yaml
all:
- fragment: all_families_dual_success
  version: '1'
  expanded:
    capability: family_all_dual_success
- fragment: three_families_stable_text
  version: '1'
  expanded:
    capability: family_state
    families:
    - sensevoice
    - glm
    - qwen
    state: stable_text
    quantifier: all
- fragment: sensevoice_glm_equivalent
  version: '1'
  expanded:
    capability: texts_equivalent
    left:
      source: family_representative
      family: sensevoice
    right:
      source: family_representative
      family: glm
- capability: lexicon_match
  lexicon: tag_1
  text:
    binding: sg_text
- capability: single_han_char
  text:
    binding: sg_text
- not:
    capability: texts_equivalent
    left:
      source: family_representative
      family: qwen
    right:
      binding: sg_text
```

---

## `voicemail_raw` — 语音信箱原文命中

- **版本**：`1`
- **状态**：enabled
- **定义文件**：`D:/Work/audio-data-engine/configs/tagging/library/voicemail_raw.yaml`

### 业务定义

任一路成功 ASR 原始文本命中语音信箱词表（capability: voicemail_raw_hit）。
词表路径由运行配置 acoustic.voicemail_patterns_path 或 selection 配置提供；不阻断其他规则。

### 摘要

任一路成功原始文本命中语音信箱词表（不阻断其他规则）

### 依赖

- 公共片段：（无显式声明）
- 词表：`voicemail`
- 需要声学证据：否

### 例子

- **命中** (`hit`):
  - 某路成功 raw_text 命中语音信箱正则。
    - 说明：规则示例，非真实模型验证记录。
- **不命中** (`miss`):
  - 全部成功路次均未命中语音信箱词表。
- **证据不足 / 无法判断** (`unknown`):
  - 无可用成功文本时由引擎按既有语义判定。

> 例子为规则示意，不能宣称为未做过的真实模型验证。

### 规则

#### `rule_voicemail_raw_v1` (v1)

- enabled: `True`
- 完整条件（展开公共片段）：
```yaml
capability: voicemail_raw_hit
```

---

<!-- tag-library-doc:end -->
