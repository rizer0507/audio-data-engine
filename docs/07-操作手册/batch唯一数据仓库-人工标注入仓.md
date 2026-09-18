# batch 唯一数据仓库（033）：人工标注包导出 → 回填 → 冻结入仓

日期：2026-09-17。对应需求：`docs/04-改进需求/已完成/033-人工标注到batch唯一数据仓库.md`。

## 范围（本期已实现）

- 分类 Manifest → 整批人工标注包（覆盖全部分类样本，含 `environment_noise`）
- 复用 annotation_v3 听音/回填/复审；可选 `reviewed_category`
- 回填校验后冻结 `datasets/stage1/warehouses/<batch>/`；catalog 一 batch 一正式仓库
- **两段生产命令**，中间保留人工操作；不宣称人审自动完成

本期**不实现**：跨仓抽取、label 生成、train/val 拆分、训练启动、训练集 Release 门禁。

## 前置

1. 已完成五类分拣，存在  
   `datasets/stage1/derived/classified_five_class_v2_2_auto_noise_<batch>.parquet`
2. 本机可读音频路径（冻结时校验内容摘要；明确拒绝/无效可无音频但须保留记录）

## 段一：导出标注包（自动）

```bash
# PowerShell
$BATCH = "0914-mixed-30000"
audio-data pipeline run pipelines/warehouse_export_annotation.yaml `
  --source-name $BATCH `
  --config configs/warehouse/export_r1.yaml
```

产物：

- `datasets/stage1/review/warehouse/<batch>/r1/pack.xlsx` + `.jsonl` + `.meta.json`
- 旁路 Manifest：`datasets/stage1/derived/warehouse_export_<batch>.parquet`

`meta.json` 含 `warehouse_binding`（batch、classified_digest）。重复导出同身份包**不会覆盖**已填写内容。

## 人工环节（非自动）

1. 听原音频，填写 `gold_text` / `decision` 等（沿用 v3 契约）。
2. 改类写入 `reviewed_category`；空白=保持自动 `category`。确认无语音用 `__EMPTY__`，空白=未完成。
3. 可分包多次标注；整批未完成前不要跑段二。

回填（可多次；按 sample_id + 音频身份，禁止按行号）：

```bash
# 首次：从分类 Manifest 导入
audio-data review import `
  datasets/stage1/derived/classified_five_class_v2_2_auto_noise_$BATCH.parquet `
  --input datasets/stage1/review/warehouse/$BATCH/r1/pack.jsonl `
  --output datasets/stage1/derived/reviewed_warehouse_$BATCH.parquet `
  --revision r1 --protocol v3 --pass first --actor-id ann1 `
  --batch $BATCH

# 若需双审：对 reviewed_warehouse 再 import --pass second --actor-id rev1
```

分类汇总 XLSX **不能**直接当 v3 包导入。

## 段二：冻结唯一仓库（自动）

确认 `reviewed_warehouse_<batch>.parquet` 已覆盖全部分类样本且无 pending/冲突后：

```bash
audio-data pipeline run pipelines/warehouse_freeze.yaml `
  --source-name $BATCH `
  --config configs/warehouse/freeze_default.yaml
```

正式仓库：

- `datasets/stage1/warehouses/<batch>/manifest.parquet`
- `datasets/stage1/warehouses/<batch>/warehouse.json`
- catalog：`data/catalog/warehouses/<batch>.json` → `warehouse_id=wh_<batch>`

同输入重跑返回原仓库；内容变化或并发第二个正式仓会报错。失败可安全重试（不留可消费半成品）。

## 类别扩展

改 `configs/warehouse/categories_five_class_v2_2.yaml` 的 `allowed_categories` 即可；不必改入仓代码分支。

## 测试

```bash
pytest tests/test_warehouse_033.py -q
```

覆盖：整批导出、幂等防覆盖、reviewed_category、完成门禁、幂等冻结、内容变更拒绝、并发唯一性。  
**未做**：真实服务器大 batch 端到端人审验证（勿写成已通过）。
