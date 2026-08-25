当前在执行流水线的时候，所有传入的参数都是靠修改配置文件来执行的。我希望在命令中输入这些参数，从而对所有生成的文件有一个统一的命名逻辑。
比如我在执行data_cleaning_source_A.yaml，我不需要去在manifest.yaml先去配置id和name，之后我还得配置out的文件的名字。我希望我在给出我当前source源头的名字，比如mt3000，自动在datasets的manifests文件夹中生成一个叫cleaned_mt3000.jsonl。
之后我在执行multi_asr_aggregate.yaml流水线的时候我只需要传入mt3000，则会自动找到cleaned_mt3000.jsonl/cleaned_mt3000.parquet文件，并生成一个qwen_asr_mt3000.parquet，以及multi_asr_aggregate_mt3000.parquet文件

---

已落地（CLI，工序②解耦后）：

```bash
audio-data pipeline run pipelines/data_cleaning_source_A.yaml \
  --source-name mt3000 --source-dir /path/to/wav

audio-data pipeline run pipelines/qwen_asr_batch.yaml --source-name mt3000
# → qwen_asr_mt3000.parquet

audio-data pipeline run pipelines/sensevoice_asr_batch.yaml --source-name mt3000
# → sensevoice_asr_mt3000.parquet

audio-data pipeline run pipelines/multi_asr_aggregate.yaml --source-name mt3000
# → join sensevoice_asr_mt3000 → multi_asr_aggregate_mt3000.parquet

audio-data pipeline run pipelines/asr_metric_pipeline.yaml --source-name mt3000
# → multi_asr_metrics_mt3000.parquet
```

命名：`cleaned_<name>` → `{model}_asr_<name>` → `multi_asr_aggregate_<name>` → `multi_asr_metrics_<name>`。
详见 `单条流水线执行命令.txt`。
