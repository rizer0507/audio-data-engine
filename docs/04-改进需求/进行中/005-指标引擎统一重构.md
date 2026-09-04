# ASR 字准率 / 文本指标统一流水线改造说明

> 用途：本文档用于指导 Cursor 对当前 `audio-data-engine` 进行工程改造。  
> 核心目标：将“生产金标阶段的字准率计算”和“模型评测阶段的字准率计算”统一到底层同一套 Metric Engine 中，同时在上层保留不同的业务配置和指标语义。

---

## 1. 当前数据处理工序

当前工程的数据处理链路如下：

```text
① 数据清洗、原始数据落库
   data_cleaning_source_A.yaml

        ↓

② 生产金标 / 训练集 / 评测集
   multi_asr_aggregate.yaml

   内部主要流程：
   - 全量 Qwen ASR
   - 全量 SenseVoice
   - 强模型 Model C（如有）
   - 文本清洗
   - 多模型结果比较 / 打分

        ↓

③ 分拣、挑选、收集、筛选
   当前主要由人工完成
   后续计划流水线化

        ↓

④ 训练流水线
   当前位于另一套训练框架
   尚未集成至本工程

        ↓

⑤ 模型评测流水线
   使用旧模型 + 新模型
   在③生成的评测集上重新识别
   与最终 Gold 进行比较
   输出模型评测结果
```

当前存在两个需要计算 CER（Character Error Rate，字错误率）的场景：

1. **生产 Gold / 数据筛选阶段**
2. **最终模型 Evaluation 阶段**

两个场景都会计算 CER，但 `reference/base` 字段不同，且指标语义不同。

本次改造禁止复制两套 CER 实现。

---

# 2. 核心设计原则

## 2.1 不创建两套 CER Pipeline

错误方案：

```text
gold_cer_pipeline.py
eval_cer_pipeline.py
```

或：

```python
calculate_gold_cer(...)
calculate_eval_cer(...)
```

这种方案会导致：

- CER 逻辑重复；
- normalization 规则容易不一致；
- bug 修复需要修改两份代码；
- 后续 WER、关键词召回率等指标继续复制；
- 不利于统一版本管理和指标追溯。

正确方案：

```text
一个统一 Metric Engine
+
不同业务场景的 Metric Profile / YAML Config
```

即：

```text
                        Metric Engine
                             │
                 ┌───────────┴───────────┐
                 │                       │
                 ▼                       ▼
        Gold Production             Model Evaluation
                 │                       │
      reference = Model C          reference = Gold
      hypothesis = Model A/B       hypothesis = New/Old Model
```

---

# 3. CER 底层统一抽象

CER Engine 不应该知道：

- Qwen；
- SenseVoice；
- Model C；
- 人工标注；
- 生产金标；
- 模型评测；
- train / eval。

底层只接受两个标准角色：

```text
reference
hypothesis
```

统一接口示例：

```python
calculate_cer(
    reference: str,
    hypothesis: str,
    normalization_config: dict | None = None,
)
```

禁止使用：

```python
calculate_cer(qwen_text, sensevoice_text)
```

禁止使用：

```python
calculate_eval_cer(gold_text, model_text)
```

---

# 4. 两种 CER 的业务语义必须区分

## 4.1 Gold Production 阶段

示例：

```text
Model A v1:
我现在不需要贷款

Model B v1:
我现在不需要贷款

Model C:
我现在不需要代款
```

如果使用：

```text
reference = Model C
hypothesis = Model A
```

计算：

```text
CER(Model A, Model C)
```

该指标**不能直接解释为 Model A 的真实识别准确率**。

因为 Model C 本身也可能错误。

因此生产 Gold 阶段的 CER 建议定义为：

```text
agreement_cer
pairwise_cer
pseudo_ref_cer
```

推荐统一使用：

```text
agreement_cer
```

例如：

```text
qwen_v1_vs_model_c_agreement_cer
sensevoice_v1_vs_model_c_agreement_cer
```

含义：

> 某候选 ASR 与指定 pseudo-reference / 强模型之间的文本差异程度。

不要将该指标称为：

```text
accuracy
true_cer
model_accuracy
```

---

## 4.2 Model Evaluation 阶段

当③经过筛选和人工确认后得到：

```text
gold_text = "我现在不需要贷款"
```

模型输出：

```text
old_model_text = "我现在不需要贷款"
new_model_text = "我现在需要贷款"
```

此时：

```text
reference = gold_text
hypothesis = model prediction
```

计算的 CER 才是真正意义上的：

```text
evaluation_cer
```

例如：

```text
old_model_cer
new_model_cer
```

---

# 5. 工程目录改造目标

建议形成如下结构：

```text
audio-data-engine/
│
├── pipelines/
│   │
│   ├── ingestion/
│   │   └── data_cleaning_source_A.yaml
│   │
│   ├── annotation/
│   │   └── multi_asr_aggregate.yaml
│   │
│   ├── selection/
│   │   └── build_dataset.yaml
│   │
│   ├── training/
│   │   └── ...
│   │
│   └── evaluation/
│       ├── asr_eval.yaml
│       └── regression_eval.yaml
│
├── components/
│   │
│   ├── cleaning/
│   │
│   ├── asr/
│   │
│   └── metrics/
│       ├── __init__.py
│       ├── cer.py
│       ├── wer.py
│       ├── align.py
│       ├── normalization.py
│       └── runner.py
│
├── configs/
│   │
│   ├── metrics/
│   │   ├── gold_agreement.yaml
│   │   └── model_eval.yaml
│   │
│   └── normalization/
│       └── zh_asr_v1.yaml
│
└── datasets/
```

如果当前项目已有固定目录结构，不强制机械迁移目录，但必须保证逻辑上存在以下三层：

```text
Pipeline
Component
Config
```

职责分别为：

```text
Pipeline:
负责 orchestration / 流程编排

Component:
负责原子能力，例如 CER、normalize、ASR inference

Config:
负责说明 reference / hypothesis / normalization / output 等业务语义
```

---

# 6. Metric Engine 设计

建议新增：

```text
components/metrics/
```

其中至少包含：

```text
normalization.py
cer.py
align.py
runner.py
```

---

## 6.1 normalization.py

负责统一文本归一化。

推荐接口：

```python
def normalize_text(
    text: str,
    config: dict,
) -> str:
    ...
```

所有 CER 计算必须经过统一 normalization 层。

禁止在各 Pipeline 内自己临时写：

```python
text.replace("，", "")
text.replace(" ", "")
```

---

## 6.2 cer.py

负责纯 CER 计算。

推荐返回：

```python
{
    "cer": float,
    "substitutions": int,
    "deletions": int,
    "insertions": int,
    "reference_length": int,
}
```

CER 定义：

```text
CER = (S + D + I) / N
```

其中：

```text
S = substitution
D = deletion
I = insertion
N = reference 字符数
```

需要明确处理空 reference：

```text
reference == ""
```

不能直接出现除零。

建议行为：

- reference 和 hypothesis 均为空：CER = 0；
- reference 为空而 hypothesis 非空：由代码明确约定；
- 不允许静默产生 NaN / inf；
- 对特殊情况写单元测试。

---

## 6.3 align.py

用于字符级 alignment。

后续应支持输出：

```text
reference char
hypothesis char
operation
```

例如：

```text
不  -> NULL   deletion
贷  -> 代     substitution
NULL -> 啊    insertion
```

此模块未来可以服务于：

- CER error analysis；
- 否定词错误统计；
- 关键词错误统计；
- badcase 可视化。

---

## 6.4 runner.py

这是上层统一调用入口。

建议抽象：

```python
run_metric(
    dataframe,
    metric_config,
)
```

或：

```python
run_text_comparison(
    dataframe,
    reference_field,
    hypothesis_field,
    output_prefix,
    normalization_config,
    metrics,
)
```

不要在 API 中绑定具体模型名字。

推荐输入：

```python
reference_field="gold_text"
hypothesis_field="qwen3_asr_v2_text"
```

推荐输出：

```text
qwen3_asr_v2_cer
qwen3_asr_v2_substitutions
qwen3_asr_v2_deletions
qwen3_asr_v2_insertions
```

---

# 7. Metric Config 统一 Schema

建议所有文本指标统一使用如下配置结构：

```yaml
metric:
  name: qwen_vs_gold
  type: cer

  purpose: model_evaluation

  reference:
    field: gold_text

  hypothesis:
    field: qwen3_asr_v2_text

  normalization:
    profile: zh_asr_v1

  output:
    prefix: qwen3_asr_v2
```

如果未来支持多个指标：

```yaml
comparison:
  name: qwen_vs_gold

  purpose: model_evaluation

  reference:
    field: gold_text

  hypothesis:
    field: qwen3_asr_v2_text

  normalization:
    profile: zh_asr_v1

  metrics:
    - cer
    - wer
    - keyword_recall
    - negation_accuracy

  output:
    prefix: qwen3_asr_v2
```

优先考虑第二种结构，因为后续扩展性更好。

---

# 8. Gold Production 配置

新增或整理：

```text
configs/metrics/gold_agreement.yaml
```

示例：

```yaml
comparisons:

  - name: qwen_v1_vs_model_c

    purpose: gold_generation_agreement

    reference:
      field: model_c_text

    hypothesis:
      field: qwen_v1_text

    normalization:
      profile: zh_asr_v1

    metrics:
      - cer

    output:
      prefix: qwen_v1_vs_model_c


  - name: sensevoice_v1_vs_model_c

    purpose: gold_generation_agreement

    reference:
      field: model_c_text

    hypothesis:
      field: sensevoice_v1_text

    normalization:
      profile: zh_asr_v1

    metrics:
      - cer

    output:
      prefix: sensevoice_v1_vs_model_c
```

最终字段建议：

```text
qwen_v1_vs_model_c_cer
sensevoice_v1_vs_model_c_cer
```

如果要强调业务语义，也可以：

```text
qwen_v1_vs_model_c_agreement_cer
sensevoice_v1_vs_model_c_agreement_cer
```

工程内保持一种命名方式即可。

推荐后者。

---

# 9. Model Evaluation 配置

新增：

```text
configs/metrics/model_eval.yaml
```

示例：

```yaml
comparisons:

  - name: old_model_vs_gold

    purpose: model_evaluation

    reference:
      field: gold_text

    hypothesis:
      field: old_model_text

    normalization:
      profile: zh_asr_v1

    metrics:
      - cer

    output:
      prefix: old_model


  - name: new_model_vs_gold

    purpose: model_evaluation

    reference:
      field: gold_text

    hypothesis:
      field: new_model_text

    normalization:
      profile: zh_asr_v1

    metrics:
      - cer

    output:
      prefix: new_model
```

输出：

```text
old_model_cer
new_model_cer
```

注意：

生产 Gold 阶段和模型 Evaluation 阶段必须调用同一个 `MetricRunner`。

---

# 10. normalization 必须统一版本化

真正容易造成指标污染的并不是 CER 算法，而是文本预处理规则不一致。

必须建立独立配置：

```text
configs/normalization/zh_asr_v1.yaml
```

建议：

```yaml
name: zh_asr_v1
version: 1

unicode:
  normalize: true
  form: NFKC

punctuation:
  remove: true

whitespace:
  remove: true

english:
  lowercase: true

number:
  normalize: false

filler:
  remove: false
```

---

## 10.1 当前业务特别要求

当前 ASR 的下游业务会使用：

```text
嗯
啊
需要
不需要
```

等表达进行客户意愿判断。

因此当前 normalization 不允许默认删除：

```text
嗯
啊
呃
```

也就是说：

```yaml
filler:
  remove: false
```

除非后续明确建立另一套 profile。

禁止在 CER 代码中硬编码 filler 删除规则。

---

# 11. Gold 数据必须保留 provenance

最终形成的 `gold_text` 不能只有文本本身。

至少建议保留：

```text
gold_text
gold_source
gold_status
gold_version
```

示例一：

```yaml
gold_text: 不需要
gold_source: human
gold_status: verified
gold_version: gold_v1
```

示例二：

```yaml
gold_text: 不需要
gold_source: model_c
gold_status: pseudo_gold
gold_version: gold_v1
```

示例三：

```yaml
gold_text: 不需要
gold_source: model_consensus+human_review
gold_status: adjudicated
gold_version: gold_v2
```

如果数据结构允许，推荐进一步记录：

```yaml
annotation:
  text: 不需要

  source: human_review

  source_models:
    - qwen_v1
    - sensevoice_v1
    - model_c

  status: verified

  version: gold_v2
```

目的：

后续 Evaluation 可以区分：

```text
CER@human_gold
CER@verified_gold
CER@pseudo_gold
```

而不是把所有 Gold 混在一起。

---

# 12. 推荐的指标三层抽象

整体指标体系建议拆成三层。

---

## Layer 1：Primitive Metric

纯算法，无业务语义：

```text
CER
WER
Edit Distance
Substitution
Deletion
Insertion
```

只接受：

```text
reference
hypothesis
```

---

## Layer 2：Comparison

定义谁和谁比较：

```text
qwen_v1_vs_model_c
sensevoice_v1_vs_model_c

old_model_vs_gold
new_model_vs_gold
```

配置：

```yaml
reference:
  field: xxx

hypothesis:
  field: xxx
```

---

## Layer 3：Purpose

说明为什么比较：

```text
gold_generation_agreement
model_evaluation
data_quality
regression_test
```

推荐结果元数据：

```json
{
  "metric": "cer",
  "metric_version": "cer_v1",
  "normalizer": "zh_asr_v1",

  "purpose": "model_evaluation",

  "reference": {
    "field": "gold_text",
    "source": "human_verified"
  },

  "hypothesis": {
    "field": "qwen3_asr_v2_text",
    "model": "qwen3_asr_v2"
  },

  "value": 0.034
}
```

不要求第一版就把所有 metadata 以 JSON 方式落库，但代码设计必须允许后续增加这些字段。

---

# 13. Pipeline 的最终调用关系

最终目标：

```text
multi_asr_aggregate.yaml
        │
        │
        ▼
  MetricRunner
        │
        ├── reference = model_c_text
        │
        ├── hypothesis = qwen_v1_text
        │
        └── purpose = gold_generation_agreement
```

以及：

```text
asr_eval.yaml
        │
        │
        ▼
  MetricRunner
        │
        ├── reference = gold_text
        │
        ├── hypothesis = new_model_text / old_model_text
        │
        └── purpose = model_evaluation
```

重点：

```text
两个 Pipeline
调用
同一个 MetricRunner
```

---

# 14. 不要把 Metric Engine 只设计成 CER Engine

虽然当前主要需求是 CER，但模块名称不要锁死。

不推荐：

```text
CERPipeline
CERManager
CERService
```

推荐：

```text
TextMetricRunner
TextMetricEngine
ASRMetricRunner
```

未来需要自然扩展：

```yaml
metrics:
  - cer
  - wer
  - keyword_recall
  - keyword_precision
  - negation_accuracy
```

---

# 15. 未来必须支持业务指标

CER 并不能完全表示当前 ASR 业务风险。

示例：

Gold：

```text
不需要
```

Prediction：

```text
需要
```

这里只缺失一个：

```text
不
```

CER 从字符层面看未必极高，但业务结果发生完全反转。

因此 Metric Engine 后续必须能够扩展：

```text
keyword_recall
keyword_precision
negation_accuracy
business_critical_error
```

其中重点关注：

```text
不需要 -> 需要
需要 -> 不需要
```

这种业务极性翻转。

当前改造不一定立即实现这些指标，但 API 和配置必须为其预留扩展空间。

---

# 16. Cursor 本次具体修改任务

Cursor 应按以下顺序改造。

## Task 1：定位现有 CER 实现

搜索整个项目：

```text
cer
character_error
edit_distance
levenshtein
字准率
准确率
```

确认：

- 当前 CER 在哪里计算；
- 是否存在重复实现；
- normalization 是否分散；
- `multi_asr_aggregate.yaml` 如何调用 CER；
- CER 的 reference 当前是否写死。

不要立即删除旧代码。

---

## Task 2：抽离统一 normalization

将所有 CER 相关文本归一化逻辑集中至：

```text
components/metrics/normalization.py
```

如果已有通用 text normalization 模块，则复用，不要重复造轮子。

要求：

- normalization 可配置；
- 不硬编码 Model 名；
- 不硬编码业务 Pipeline；
- 支持版本化 profile。

---

## Task 3：抽离统一 CER Primitive

形成：

```text
components/metrics/cer.py
```

输入：

```text
reference
hypothesis
```

输出至少包含：

```text
cer
substitutions
deletions
insertions
reference_length
```

---

## Task 4：建立 MetricRunner

新增统一入口：

```text
components/metrics/runner.py
```

支持：

```text
reference_field
hypothesis_field
metrics
normalization_profile
output_prefix
purpose
```

要求可以直接对 dataframe / record list 执行。

---

## Task 5：改造 Gold Production 流水线

修改：

```text
multi_asr_aggregate.yaml
```

不要再直接耦合某个 CER 函数。

改为调用统一：

```text
MetricRunner
```

reference 可通过 YAML 指定。

生产 Gold 阶段推荐：

```text
reference = model_c_text
hypothesis = qwen_v1_text / sensevoice_v1_text
```

输出指标语义使用：

```text
agreement_cer
```

不要宣称为模型真实准确率。

---

## Task 6：为 Evaluation 预留统一调用接口

即使⑤当前尚未实现，也需要增加一个最小示例：

```text
pipelines/evaluation/asr_eval.yaml
```

展示未来如何：

```text
reference = gold_text
hypothesis = old_model_text
hypothesis = new_model_text
```

调用同一个 MetricRunner。

不要求实现完整模型 inference pipeline，至少确保 metric 部分可直接工作。

---

## Task 7：加入 Gold provenance 字段支持

确认数据 Schema 能够支持：

```text
gold_text
gold_source
gold_status
gold_version
```

若当前已有类似字段，优先兼容现有 Schema，而不是重复创建语义相同的字段。

---

## Task 8：补充测试

至少增加如下测试：

### CER 基本测试

```text
reference == hypothesis
CER = 0
```

### substitution

```text
不需要
需要
```

应正确识别 deletion / substitution 的实际 alignment 结果。

### insertion

```text
需要
啊需要
```

### empty reference

```text
reference = ""
hypothesis = ""
```

以及：

```text
reference = ""
hypothesis = "需要"
```

必须有明确行为。

### normalization 一致性

```text
我不需要，贷款。
我不需要贷款
```

在 `zh_asr_v1` 下结果符合预期。

### Pipeline 配置测试

确保：

```text
Gold Production
```

和：

```text
Evaluation
```

使用的是同一 MetricRunner，而不是两套实现。

---

# 17. 向后兼容要求

本次修改尽量避免一次性破坏现有生产流水线。

如果旧代码已有：

```python
calculate_cer(...)
```

可以暂时保留 wrapper：

```python
def calculate_cer(...):
    return new_metric_engine(...)
```

并标记：

```text
deprecated
```

后续再删除。

不能因为工程重构导致当前：

```text
data_cleaning_source_A.yaml
multi_asr_aggregate.yaml
```

无法执行。

---

# 18. 日志要求

MetricRunner 执行时至少记录：

```text
metric
reference_field
hypothesis_field
normalization_profile
output_field
record_count
```

例如：

```text
[MetricRunner]
metric=cer
purpose=gold_generation_agreement
reference=model_c_text
hypothesis=qwen_v1_text
normalizer=zh_asr_v1
output=qwen_v1_vs_model_c_agreement_cer
records=30000
```

禁止仅打印：

```text
calculating cer...
```

否则未来很难排查指标来源。

---

# 19. 配置校验要求

启动 Pipeline 时，对 metric config 做 fail-fast 校验。

如果：

```yaml
reference:
  field: xxx
```

字段不存在，应立即抛错：

```text
MetricConfigError:
reference field `xxx` not found in dataset
```

同理检查：

```text
hypothesis field
normalization profile
metric name
output field collision
```

不要静默跳过。

---

# 20. 输出字段冲突处理

若配置生成：

```text
new_model_cer
```

但 dataframe 中已存在同名字段：

默认建议报错。

例如：

```text
OutputFieldAlreadyExistsError
```

除非 YAML 明确：

```yaml
overwrite: true
```

避免重复执行 Pipeline 时无意覆盖历史结果。

---

# 21. 推荐最终 Schema 示例

一条样本在 Gold Production 后可能类似：

```json
{
  "id": "sample_000001",

  "audio_path": "...",

  "qwen_v1_text": "不需要",
  "sensevoice_v1_text": "需要",
  "model_c_text": "不需要",

  "qwen_v1_vs_model_c_agreement_cer": 0.0,
  "sensevoice_v1_vs_model_c_agreement_cer": 0.3333,

  "gold_text": "不需要",
  "gold_source": "human_review",
  "gold_status": "verified",
  "gold_version": "gold_v1"
}
```

Evaluation 后：

```json
{
  "id": "sample_000001",

  "gold_text": "不需要",

  "old_model_text": "需要",
  "new_model_text": "不需要",

  "old_model_cer": 0.3333,
  "new_model_cer": 0.0
}
```

如果数据表实际为 CSV / XLSX / Parquet，字段思想保持一致。

---

# 22. 最终验收标准

完成改造后必须满足以下条件。

## 架构验收

- [ ] 工程中只有一套 CER 核心算法实现。
- [ ] Gold Production 和 Evaluation 共用同一个 MetricRunner。
- [ ] Metric Engine 不依赖具体 ASR 模型名称。
- [ ] reference / hypothesis 可以通过配置指定。
- [ ] normalization 已抽离并可版本化。
- [ ] Pipeline 与 Metric Component 解耦。

## 数据语义验收

- [ ] Gold Production 的指标明确标记为 agreement / pseudo-reference comparison。
- [ ] Evaluation 使用最终 `gold_text` 作为 reference。
- [ ] Gold 支持 provenance 字段。
- [ ] 不再使用含义模糊的 `base` 字段表达 comparison 角色。

## 工程质量验收

- [ ] CER 有单元测试。
- [ ] normalization 有单元测试。
- [ ] 空文本有明确行为。
- [ ] 字段不存在时 fail-fast。
- [ ] 输出字段冲突有保护。
- [ ] 旧 Pipeline 不因重构直接失效。
- [ ] 日志能够追踪 reference、hypothesis、normalizer 和 metric purpose。

---

# 23. Cursor 修改时的约束

请 Cursor 遵循以下约束：

1. **先阅读现有工程，不要直接按照本文档机械创建重复模块。**
2. 如果现有项目已经存在 Metric / Normalizer 抽象，应在其基础上改造。
3. 尽量小步修改，不要无关重构。
4. 不修改 ASR 推理结果本身。
5. 不修改现有数据内容。
6. 不改变 CER 数学定义，除非现有实现存在明确 bug。
7. normalization 行为如果发生变化，必须明确说明。
8. 所有新增配置应提供默认值或兼容旧流程。
9. 每个关键修改点补测试。
10. 修改完成后输出一份 change summary，说明：
    - 修改了哪些文件；
    - 新增了哪些文件；
    - 旧 CER 逻辑如何迁移；
    - Gold Production 如何调用；
    - Evaluation 如何调用；
    - 是否存在兼容性风险。

---

# 24. 最终设计结论

本次改造的核心不是：

```text
生产金标 CER Pipeline
+
模型评测 CER Pipeline
```

而是：

```text
                  Unified Text Metric Engine
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
  gold_agreement.yaml                model_eval.yaml
            │                               │
reference = pseudo reference         reference = gold
hypothesis = candidate ASR           hypothesis = evaluated model
purpose = gold_generation            purpose = evaluation
```

最终应形成以下工程原则：

> **Pipeline 负责流程编排，Component 负责指标能力，Config 负责比较语义。**

CER 只是一个 Primitive Metric。

Gold Production 和 Model Evaluation 是两个不同的业务场景，但它们不应该拥有两套不同的 CER 实现。

真正需要变化的只有：

```text
reference
hypothesis
purpose
output
```

而不是 CER 算法本身。
