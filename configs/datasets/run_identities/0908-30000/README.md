# 0908-30000 · ASR run identity 模板（方案 B）

对应 ASR：

| family | transcript_key | parquet |
| --- | --- | --- |
| glm | `glm-asr-1` / `glm-asr-2` | `datasets/stage1/asr/glm-asr-{1,2}_asr_0908-30000.parquet` |
| qwen | `qwen3-asr-1` / `qwen3-asr-2` | `datasets/stage1/asr/qwen3-asr-{1,2}_asr_0908-30000.parquet` |
| sensevoice | `sensevoice-asr-1` / `sensevoice-asr-2` | `datasets/stage1/asr/sensevoice-asr-{1,2}_asr_0908-30000.parquet` |

## 用法

1. 复制 `*_identity.template.yaml` → 去掉 `.template`，填入真实 `execution_id` / digests / `created_at`。
2. 确保已有 `datasets/stage1/cleaned/cleaned_0908-30000.parquet`（或从 ASR 重建底表）。
3. 对每路执行：

```bash
audio-data artifact register-asr-run \
  "datasets/stage1/asr/<alias>_asr_0908-30000.parquet" \
  --identity "configs/datasets/run_identities/0908-30000/<alias>_identity.yaml" \
  --audio-base "datasets/stage1/cleaned/cleaned_0908-30000.parquet" \
  --output-identity "configs/datasets/run_identities/0908-30000/<alias>_registered.yaml"
```

4. 将 6 份 `*_registered.yaml` 合并进 `configs/datasets/zh_asr_v3_0908_30000.yaml` 的 `runs:`。
5. 再跑 prepare → attach → sidecar → classify（见 `单条流水线执行命令.txt` §2f-v3）。

**禁止**用空 digest 或复制同一 artifact 冒充双跑。
