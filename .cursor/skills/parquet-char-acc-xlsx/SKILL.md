---
name: parquet-char-acc-xlsx
description: >-
  Build a character-accuracy evaluation xlsx from a gold parquet and one or
  more ASR parquets. Use when the user gives parquet files and wants 字准, 评测,
  正确率, or a final evaluation xlsx. Match the workbook layout of
  tmp/1021/evaluation_final_train_v002_best_vs_over.xlsx: sheets 正确率, 透视对比,
  明细, 说明. qwen-base is the baseline when that model is present.
---

# 评测 xlsx

只交付一个 xlsx。版式对齐 `tmp/1021/evaluation_final_train_v002_best_vs_over.xlsx`，不要改成「只留字准不为 1」的表。

在 `audio-data-engine` 下调用 `audio_engine.core.transcript_reconcile.levenshtein_ops` 和 `plain_transcript_text`。不要另写编辑距离。

## 输入

- 金标 parquet：参考文本是 `label_gold_text`。预标注是 `qwen_1_text`。
- 每个识别 parquet：按 `id` 对齐金标。转写列是该模型的 `*_text`（如 `qwen-base_text`、`qwen-lora_text`）。
- 有 `qwen-base` 时它是 baseline。相对列只相对它。

## type

- `non_speech`：`label_source_gold_kind=non_speech` 或 `label_decision=empty`。计分金标当空串。
- `auto_gold`：speech 且 `plain(qwen_1_text)==plain(label_gold_text)`。
- `hardcase`：speech 且两者 plain 后不一致。
- 计分参考：non_speech 用空串，其余用 `label_gold_text` 原文。

`字准率==1` 才算正确。标点差异在 plain 之后不算错。拉丁字母保留。

## 工作表

`正确率`：每个 scope（`overall`、`auto_gold`、`hardcase`、`non_speech`；后三个条数为 0 则省略）下每个模型一行。

列：`scope` `模型` `样本数` `正确数(字准=1)` `正确率` `相对 qwen-base` `总编辑距离` `总基准字数` `总字准` `相对 qwen-base总字准`

- 正确率写成 `79.09%` 这种百分数字符串。baseline 的相对列写 `baseline`。其他模型写带符号的百分点，如 `-0.81%`、`+1.70%`。
- 总字准 = `1 - 总编辑距离/总基准字数`，同样写成百分数。
- `non_speech` 总基准字数为 0 时，总字准改写总编辑距离的整数；相对列改写与 qwen-base 的编辑距离差（`+45` 这种整数），不写百分数。

`透视对比`：每个 scope 一行。列先是 `scope` `标注总条数` `样本数(有识别)`，然后每个模型一组：`{模型}_正确率%` `{模型}_正确数` `{模型}_样本数` `{模型}_总字准%` `{模型}_总编辑距离` `{模型}_总基准字数`。正确率和总字准这里是数值（79.09），不是带 `%` 的字符串。无基准字时总字准% 仍填编辑距离整数。

`明细`：全部对齐行，不只放错例。列顺序：

`id` `type` `label` `candidate_text` `gold_text_ann` `gold_vs_candidate_mismatch` `category` `decision` `gold_kind` `noise_kind` `exclude_from_training` `uncertain` `label_source` `wav_path`

然后每个模型：`{模型}_text` `{模型}_字准率` `{模型}_exact_correct`

最后 `in_model_eval` `eval_scored`。

`label` / `gold_text_ann`：speech 填金标原文，non_speech 留空。`candidate_text` 填 `qwen_1_text`。`gold_vs_candidate_mismatch` 为 plain 后是否不一致（1/0）。`{模型}_text` 用识别列原文。`exact_correct` 为 0/1。有识别则 `in_model_eval=1` 且 `eval_scored=True`。

`说明`：两列 `字段/口径`、`说明`。写清金标路径、各模型文件与文本列、type 三条规则、正确率与总字准口径、non_speech 无基准字时的改写、条数。

文件名：`evaluation_final_{切分}_{模型简称}.xlsx`，放在识别 parquet 同目录。
