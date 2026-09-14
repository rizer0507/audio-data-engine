# 021 · SenseVoice 控制标签不得进入可比文本与候选金标

> 状态：**进行中**（022 已让分拣候选正文去掉控制标签；ASR 落盘 `text` 白名单仍保留未知标签，本项不因此标完成）  
> 日期：2026-09-11  
> 触发：工序一产物里 SenseVoice 转写仍然带着 `<|zh|>`、`<|NEUTRAL|>`、`<|Speech|>`、`<|withitn|>` 这类控制字段。下游表、人审候选、训练目标看上去像「没洗干净」。  
> 前置 / 关联：标签解析见 `src/audio_engine/operators/asr/sensevoice.py`（`parse_sensevoice_text`）与 [SenseVoice识别流水线](../../07-操作手册/SenseVoice识别流水线.md) §7.1；全标签剥离函数在 `src/audio_engine/core/transcript_reconcile.py`（`clean_control_tags`）；v3 候选文本取自 `selection_v3/consensus.py` 的 `medoid.raw_text`。清洗边界与 [018](./018-转写语速不可能防护栏.md) 同类：**不要把本项塞进①音频清洗 DAG**。  
> 范围边界：**对外可见的转写 / 候选 / 训练目标不得残留 FunASR 控制标签**；模型原始串只留在审计字段。**不改**分桶阈值、相似度公式、家族投票规则。① `data_cleaning_source_A` 不改。

---

## 1. 当前事实（结论）

### 1.1 标签还在，但不是「清洗流水线忘了跑」

SenseVoice / FunASR 的原始输出形态固定为控制标签前缀 + 口语正文，例如：

```text
<|zh|><|NEUTRAL|><|Speech|><|withitn|>客户表示暂时不需要
```

现行配置 `configs/asr/sensevoice.yaml` 默认 `use_itn: true`，模型几乎总会再吐一个 ITN 标记 `<|withitn|>` 或 `<|woitn|>`。这两枚 **不在解析白名单里**。

用户看到的 `<| |>` 就是这类控制字段，不是音频清洗漏掉的尖括号正文。

### 1.2 两条清洗口径互相打架

| 位置 | 实际行为 | 结果 |
| --- | --- | --- |
| ASR 落盘 `parse_sensevoice_text` | **只删白名单**：语言 `zh/en/yue/ja/ko/nospeech`，情感 `HAPPY/SAD/ANGRY/NEUTRAL`，事件 `Speech/BGM/Applause/Laughter/Cry/Sneeze/Breath/Cough` | 未知标签**留在** `transcripts.*.text`。单测写死：`"<|future|>你好"` 必须原样保留 |
| 比对用 `clean_control_tags` / `comparison_text` | 删除**全部** `<\|…\|>`，外加残缺情感形如 `<EMO_UNKNOW>\|` | 投票、相似度通常看不到标签 |
| v3 分拣写候选 | `candidate_text = medoid.raw_text or medoid.comparison_text` | **优先用带标签的原始串** |
| `quality.normalize_transcripts` | 会把正文洗成纯字 | **只挂在评测/字准流水线**，不在 `sensevoice_asr_batch`、聚拢、`classify_dataset_v3` |

因此：

- 比较往往是干净的，**看起来洗过了**；
- 落盘 `text`、人审 `candidate_text` / `label`、以及从候选拷走的 `train_target_text` **仍然带着 `<| |>`**。

SenseVoice 被选成 medoid（或该路没有 `comparison_text` 可退）时，候选金标就是：

```text
<|zh|><|NEUTRAL|><|Speech|><|withitn|>客户表示暂时不需要
```

这就是「清洗完为什么还带 `<| |>`」。

### 1.3 白名单漏掉的常见标签（生产里会进 `text`）

| 标签 | 来源 | 现在落在 `text` 里？ |
| --- | --- | --- |
| `<|withitn|>` / `<|woitn|>` | ITN 开关，默认开启 | **是** |
| `<|EMO_UNKNOWN|>` / `<|EMO_UNKNOW|>` | 情感未知（仓库测试和旁路脚本已出现后一种拼写） | **是** |
| 新语言 / 新事件 / 模型升级多出来的 `<|…|>` | FunASR 版本漂移 | **是**（故意保留，见手册 §7.1） |
| `<|zh|>` `<|NEUTRAL|>` `<|Speech|>` | 白名单内 | 从 `text` 去掉，但仍完整留在 `extra.raw_text`，并被候选金标重新捞回 |

手册 §7.1 的原意是：未知标签记到 `extra.unknown_tags`，**不要静默丢审计信息**；并避免把用户真说出的尖括号内容删掉。  
落地时把「审计保留」做成了「可比正文也保留」，和「`text` 必须是去掉控制标签后的可比对纯文本」互相矛盾。真实口语几乎不会说出 `<|withitn|>` 这种 token。

### 1.4 为什么不能改① `data_cleaning_source_A`

①清洗发生在多 ASR **之前**，当时没有 SenseVoice 输出，删不掉标签。DAG 仍然是：

```text
ingest → pcm_to_wav → resample_16k → probe → filter(duration>0) → select(audio_pass)
```

文件头写死「只做音频质量，不做标注决策」。与 [018](./018-转写语速不可能防护栏.md) 相同：

- **不要**把文本清洗塞进 pcm / resample / probe。
- **要**在 SenseVoice **写出可比正文时**剥掉全部控制标签，并在分拣 **写出给人看 / 给训练用的文本时** 禁止回写原始串。

已跑完的 ASR parquet 里已经有 `extra.raw_text`，修复**不必重跑 GPU**。

---

## 2. 我想做什么

SenseVoice 的控制字段只允许出现在审计副本。任何会被人读、被当金标候选、被当训练目标的字段，都不得再带 `<| |>`。

处理原则：

1. **模型原串不丢。** `extra.raw_text` 继续保存 FunASR 原文，供回溯、语言/情感/事件解析、语音信箱正则扫原始串。
2. **可比正文必须干净。** `transcripts.<sensevoice别名>.text` 去掉全部控制标签后再落盘。未知标签记入 `extra.unknown_tags`，**不要留在正文**。
3. **候选与标签跟干净正文走。** v3 `candidate_text`、`label`、以及从它们派生的训练/评测目标，使用已去标签的文本（与 `comparison_text` 同一口径的正文部分），禁止 `medoid.raw_text` 直接外露。
4. **已有批次可离线修补。** 用已存的 `raw_text` / 带标签 `text` 重写可比字段，不重新推理。

---

## 3. 数据从哪来、结果要什么

- **输入：** 已有 SenseVoice 结果（`sensevoice_asr_*.parquet`、聚拢后的 `transcripts.sensevoice*`），以及由它们产生的 `classified_v3_*`。新跑的 `pipelines/sensevoice_asr_batch.yaml` 同样适用。
- **跑完希望得到：**
  - `transcripts.*.text`：口语正文，无 `<|…|>`，无残缺情感标签。
  - `extra.raw_text`：仍是模型原文（可以带标签）。
  - `extra.language` / `emotion` / `events`：仍从原文解析；白名单外的标签进 `unknown_tags`，不丢。
  - `labels.candidate_text` / `labels.label`：与给人看的正文一致，无控制标签。
- **数据量：** 存量批次（含 `0908-30000` 一类已落盘结果）+ 之后所有新跑。不清数具体条数，按 manifest 全量处理。

①清洗产物 `cleaned_*` **不含转写**，本需求不改它，也不要它去读 ASR。

---

## 4. 业务上有哪些规矩

### 4.1 什么必须从可比正文删掉

一律视为 SenseVoice / FunASR 控制字段，从 `text` 与对外文本删除：

- 标准形：`<|…|>`（标签内允许空白，例如 `<| zh |>`、`<|withitn|>`）。
- 仓库里已经出现的残缺形：`<EMO_UNKNOW>|`、`</EMO_xxx>`（现有 `_LOOSE_EMO_TAG_RE` 覆盖的那类）。
- 已知业务标签即使不在旧白名单，也必须删：`withitn`、`woitn`、`within`、`EMO_UNKNOWN`、`EMO_UNKNOW`，以及任意新的 `<|标签|>`。

删除后 `strip`。若全文只剩标签（例如只有 `<|nospeech|><|NEUTRAL|><|Speech|><|withitn|>`），`text` 必须是空串，不能把 `<|withitn|>` 当成「有转写」。

### 4.2 什么不能删

- `extra.raw_text` 一字不改。
- 用户真实说出、但**不像控制标签**的尖括号正文（例如 `小于a大于`、普通 `()`、`【】`）。判定标准：只删 `<|…|>` 与已证实的残缺 `EMO_*` 形，不另做「凡是尖括号都删」。
- 标点、语气词、数字、否定词。本需求**只去控制标签**，不把 `plain_transcript_text` 的去标点规则提前到 ASR 落盘。（比对仍按现有 `comparison_text` 去标点；那是另一层。）

### 4.3 解析与审计

- 语言 / 情感 / 事件继续从 **原文** 解析，规则可沿用现有白名单。
- 原文里出现、但未归入 language/emotion/events 的标签，写入 `extra.unknown_tags`（去重、保持出现顺序）。删正文 ≠ 丢标签。
- 语音信箱等规则若需要扫原始串，继续读 `raw_text`；不要改成只扫已清洗正文，以免漏掉只写在标签旁的提示音文案。投票相似度继续走 `comparison_text`，本需求不改它的阈值。

### 4.4 候选金标口径

`candidate_text` 表示「建议给人看、给人改、给训练用的那句话」，不是模型 dump。

- 取值改为：该 medoid 路次的**已去控制标签正文**（实现上可与 `comparison_text` 共用去标签，但候选可以保留标点，只要标签没了；若现网候选本就不保留标点，保持现网，不要借机改标点策略）。
- **禁止** `candidate_text = medoid.raw_text` 直接外露。
- 选定 SenseVoice 路次时，正文不得带回 `<|zh|>` 前缀。选定 Qwen / GLM / Kimi 时行为不变（它们本来就没有这套标签）；若其 `raw_text` 偶然含 `<| |>`，同样剥掉再写入候选，避免只修 SenseVoice 别名。

分桶、`decision`、`min_similarity`、支持家族数 **不得** 因「标签从候选上消失」而改变。理由：投票早就用 `comparison_text`，标签不参与相似度。本需求只改外露字符串，不改裁决。

### 4.5 存量与缓存

- 新推理：`asr.sensevoice` / `asr.sensevoice_batch` 写出的 `text` 即已清洗。`transcript_key` 别名（`sensevoice-asr-1` 等）同样适用。
- 已落盘 parquet：提供可重复执行的修补（脚本或算子皆可），只改 `text` / `candidate_text` / `label` 等可比字段，**不重跑模型、不删 `raw_text`**。
- ASR cache：若 cache 里存的是带标签的 `text`，修补逻辑或 cache key 必须让 `--force` 之前的旧 cache 不会把脏 `text` 再写回去。优先：读 cache 时再剥一次标签（便宜、不强迫全量重推）。
- 分拣 cache：投票输入不变则桶不变；重跑分拣或离线改写候选均可。不要为了去标签而要求重推三万条音频。

### 4.6 明确不做

- 不改 `pipelines/data_cleaning_source_A.yaml`。
- 不重写 FunASR 解码、不关 `use_itn`（ITN 正文要留，只丢掉 `<|withitn|>` 这个标记）。
- 不把情感/事件从结构化字段删掉；它们仍供分析，只是不进转写句子。
- 不调整 v3 阈值、家族契约、语速栏、DNSMOS。

---

## 5. 我怎么才算满意

1. 给定 `"<|zh|><|HAPPY|><|Laughter|><|withitn|>你好"`，落盘 `text == "你好"`；`extra.raw_text` 仍是全串；`unknown_tags` 含 `withitn`；`language/emotion/events` 仍正确。
2. 给定 `"<|zh|><|EMO_UNKNOW|><|within|>你 好"`，`text` 为 `"你 好"`（或仅 strip 空白后的同等正文），不再含 `<|`。
3. 全文只有控制标签时，`text == ""`，分拣视为空转写，不得把标签本身当成一句话。
4. `candidate_text` 与 `label` 抽检（含 SenseVoice 为 medoid 的样本）不含正则 `<\|.*?\|>`，也不含 `<EMO_…>|`。
5. 同一批样本在修复前后：`type` / `decision` / `min_similarity` / 支持家族集合不变（允许 `candidate_text` 字符串变化）。若有变化，视为本需求做坏了，先停。
6. 单测覆盖：白名单标签、ITN 标签、未知标签、残缺情感标签、纯标签空文本、`candidate_text` 不再等于 `raw_text`。现有 `test_parse_sensevoice_tags_preserves_unknown_tags` 改为「未知标签不进 `text`，只进 `unknown_tags`」。
7. 手册 §7.1 与本需求对齐：`text` 去掉**全部**控制标签；未知标签只留在 `extra`，不再出现在正文。

---

## 6. 其他我想说的

- 根因一句话：**审计串和可比正文被写成了同一个字段。** `raw_text` 该脏，`text` 和 `candidate_text` 不该脏。
- 优先改解析与候选写出；存量用离线重写补。不要用「再跑一遍 SenseVoice」当修复方案。
- 旧手册「未知标签留在正文，以免误删口语」作废。误删防护改为：只匹配控制标签语法，不删除其它尖括号。
- 参考实现已在仓库：`clean_control_tags` 比 `parse_sensevoice_text` 更接近正确口径。新逻辑应共用这一处，避免 ASR 与分拣各写一套正则。
