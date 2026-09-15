# 0914-mixed-30000 · ASR run identity（方案 B）

| family | transcript_key | parquet |
| --- | --- | --- |
| glm | `glm-base-1` / `glm-base-2` | `datasets/stage1/asr/glm-base-{1,2}_asr_0914-mixed-30000.parquet` |
| qwen | `qwen-basr-1` / `qwen-basr-2` | `datasets/stage1/asr/qwen-basr-{1,2}_asr_0914-mixed-30000.parquet` |
| sensevoice | `sv-base-1` / `sv-base-2` | `datasets/stage1/asr/sv-base-{1,2}_asr_0914-mixed-30000.parquet` |

来源：`tmp/0915/`。清洗底表由 `scripts/rebuild_cleaned_from_asr.py` 从 qwen-basr-1 重建。
