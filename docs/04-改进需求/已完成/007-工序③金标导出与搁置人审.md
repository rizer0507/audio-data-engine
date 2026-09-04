# 人工分拣 / 人工审核 / 冻结 Dataset 工序改进（已落地）

## 需求
1. classify 之后先导出全部多模型一致结果（auto_gold），xlsx 新增 `label` 列并填入金标。
2. 金标可聚拢成 Manifest，直接进入评测；不一致样本（人审）可先搁置。

## 落地
- `quality.classify`：auto_gold 同时写 `labels.gold_text` 与 `labels.label`
- `audio-data review export-gold`：导出含 `label` 的 xlsx，并可写 `gold_${BATCH}.parquet`
- `audio-data release build --allow-unresolved-review`：允许跳过未审完的 review_queue/hardcase
- 手册工序③：classify → export-gold →（可搁置）人审 export/import → release
- 评测推荐：`audio-data eval register` summary/gold → `--eval-name eval_$BATCH` 解耦评测

## 操作入口
见 `全自动训练评测闭环执行手册-local.txt` / `-dev.txt` 第二节 step 6–9；评测见第四节。
