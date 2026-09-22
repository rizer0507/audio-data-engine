# semantic_risk 听写标注（VS Code Remote / Mac Mini）

适用于五类分拣完成后，只对 `category=semantic_risk` 做**人工听写转写**，再回填到分类 Manifest。  
目标场景：代码与音频都在**公司服务器**上；本机（如 Mac Mini）通过 **VS Code / Cursor Remote** 连接，仅作播放器与键盘输入。

脚本：[`scripts/annotate_semantic_risk_transcripts.py`](../../scripts/annotate_semantic_risk_transcripts.py)  
命令台账：[手册/dev/05-一条命令多步骤.txt](../../手册/dev/05-一条命令多步骤.txt) §3  
上游分类入口：[已有六路ASR一键分类导出.md](./已有六路ASR一键分类导出.md)（`classify_asr_to_xlsx.py` / `stage1 classify`）

> 与正式 annotation_v3「整包导出 → XLSX/JSONL → review import」不同：本入口是**轻量 CLI 听写**。  
> 合并键仍为 `sample_id` + `original_audio_sha256`，写入 `labels.gold_text` 等字段；正式双审/入仓仍走 [batch唯一数据仓库-人工标注入仓.md](./batch唯一数据仓库-人工标注入仓.md)。

---

## 1. 流程总览

```text
五分类 parquet
    │
    ▼  prepare
队列 queue.parquet + playlist.md +（可选）wav/ 序号软链
    │
    ▼  label（边听边写）
annotations.parquet（可断点续标）
    │
    ▼  merge
classified_..._${BATCH}_with_gold.parquet
（未标注行原样保留；已标行写入 gold_text）
```

三步对应三个子命令，**不要跳过 prepare**（`label` 依赖队列）。

---

## 2. 环境与前置

在**服务器**项目根执行（与分类同一 conda）：

```bash
cd /data2/data-cp/lizi/audio-data-engine
source /data2/data-cp/zk/env_hub/miniconda312_qw3tts/bin/activate
export BATCH=<批次名>
```

前置检查：

| 项 | 要求 |
|----|------|
| 分类产物 | 默认 `datasets/stage1/derived/classified_five_class_v2_3_min_acoustic_${BATCH}.parquet` |
| 音频可读 | 样本 `audio.resampled_16k`（或 `pcm_wav` / `pcm_to_wav`）在服务器上存在 |
| Remote | VS Code / Cursor 已 Remote-SSH 到该服务器；资源管理器能打开并播放 `.wav` |
| 批次数 | `--batch` 与分类时一致 |

若尚无分类产物，先跑手册 §2 / [已有六路ASR一键分类导出](./已有六路ASR一键分类导出.md)。

---

## 3. 最短命令

```bash
# ① 筛 semantic_risk，生成队列 / 播放清单 / 序号软链
python scripts/annotate_semantic_risk_transcripts.py prepare --batch "$BATCH"

# ② 交互听写（自动尝试用 cursor/code 打开下一条 wav）
python scripts/annotate_semantic_risk_transcripts.py label \
  --batch "$BATCH" \
  --annotator-id <你的id>

# ③ 回填五分类 parquet（默认不覆盖已有输出，需 --overwrite）
python scripts/annotate_semantic_risk_transcripts.py merge \
  --batch "$BATCH" \
  --overwrite
```

---

## 4. 逐步说明

### 4.1 prepare — 建队列

做什么：

1. 读取五分类 parquet，筛 `labels.category == semantic_risk`（兼容 `type`）。
2. 按 `sample_id` 排序，写入 `queue_index`（从 1 起）。
3. 解析听音路径（优先 `resampled_16k` → `pcm_wav` → `pcm_to_wav` → 容器格式 `source_path`）。
4. 默认在工作目录建 `wav/00001_<sample_id>.wav` **软链**（失败则复制）；并写 `playlist.md`。

默认工作目录：

```text
datasets/stage1/review/semantic_risk_transcripts/${BATCH}/
  queue.parquet
  playlist.md
  wav/                  # 序号软链，便于资源管理器按序点击
```

常用选项：

| 选项 | 含义 |
|------|------|
| `--classified PATH` | 指定分类 parquet（非默认路径时用） |
| `--rule-stem STEM` | 默认 `classified_five_class_v2_3_min_acoustic`；legacy 可改 v2.2 stem |
| `--work-dir DIR` | 自定义工作目录 |
| `--no-link-wav` | 不建软链（仍写 queue + playlist；听音用绝对路径） |
| `--overwrite-links` | 重建 `wav/` 目录 |

示例（legacy v2.2 分类产物）：

```bash
python scripts/annotate_semantic_risk_transcripts.py prepare \
  --batch "$BATCH" \
  --rule-stem classified_five_class_v2_2_auto_noise
```

### 4.2 label — 边听边写

做什么：

1. 读 `queue.parquet`，跳过 `annotations` 里已是 `annotated` / `empty` 的样本（断点续标）。
2. 每条打印 `sample_id`、sha256、WAV 绝对路径、自动选文 `candidate_text`（**仅参考，勿照抄**）。
3. 尝试 `cursor` / `code` / `code-insiders` 打开当前 wav（与资源管理器点击等价，在 Mac Mini 客户端播放）。
4. 你输入转写后回车 → 写入 `annotations.parquet` → 自动下一条。

```bash
python scripts/annotate_semantic_risk_transcripts.py label \
  --batch "$BATCH" \
  --annotator-id ann1

# 从某一序号起（含）
python scripts/annotate_semantic_risk_transcripts.py label \
  --batch "$BATCH" \
  --annotator-id ann1 \
  --start-index 120
```

**交互命令（单独一行）：**

| 输入 | 行为 |
|------|------|
| 普通文本 | 记为 `gold_kind=speech`，`status=annotated`，进入下一条 |
| `/empty` | 确认无语音：`gold_kind=non_speech`，`gold_text=""`，`status=empty` |
| `/skip` | 跳过（`status=skipped`，下次 `label` 仍会再遇到，可重做） |
| `/replay` | 再次打开当前 wav |
| `/show` | 重新打印本条信息并再次尝试打开 |
| `/quit` | 保存进度退出 |
| `/help` | 显示帮助 |

**播放打不开时：**

1. 在 VS Code 资源管理器打开  
   `datasets/stage1/review/semantic_risk_transcripts/${BATCH}/wav/`  
   按 `00001_...` 序号点击播放。
2. 或打开终端打印的 **WAV 绝对路径**，在资源管理器定位后点击。
3. 对照 `playlist.md` 表格中的路径列。

脚本**不会**在服务器上调用 `afplay`/`ffplay`；Remote 场景下以客户端播放器为准。

### 4.3 merge — 回填分类表

做什么：

1. 读取 `annotations.parquet` 中 `status ∈ {annotated, empty}` 的行。
2. 按 **`sample_id` + `original_audio_sha256`** 匹配五分类 Manifest（禁止只按行号/文件名合并）。
3. 命中则写回 labels；**未标注样本整行不变**；**不改**自动 `category`。
4. 写出新 parquet + `.merge_report.json`（默认路径带 `_with_gold` 后缀，不覆盖原分类文件）。

```bash
python scripts/annotate_semantic_risk_transcripts.py merge \
  --batch "$BATCH" \
  --overwrite

# 严格模式：sha256 不匹配或标注孤儿 → 非零退出
python scripts/annotate_semantic_risk_transcripts.py merge \
  --batch "$BATCH" \
  --overwrite \
  --strict
```

常用选项：

| 选项 | 含义 |
|------|------|
| `--classified PATH` | 输入五分类 parquet |
| `--annotations PATH` | 标注旁路表（默认 work-dir 下） |
| `--output PATH` | 合并输出路径 |
| `--overwrite` | 允许覆盖已有输出 |
| `--strict` | 校验失败则失败退出 |
| `--rule-stem` | 与 prepare 一致，用于拼默认路径 |

默认输出：

```text
datasets/stage1/derived/classified_five_class_v2_3_min_acoustic_${BATCH}_with_gold.parquet
datasets/stage1/derived/classified_five_class_v2_3_min_acoustic_${BATCH}_with_gold.parquet.merge_report.json
```

---

## 5. 产物与字段

### 5.1 目录一览

| 路径 | 说明 |
|------|------|
| `.../semantic_risk_transcripts/${BATCH}/queue.parquet` | 待标队列 |
| `.../semantic_risk_transcripts/${BATCH}/annotations.parquet` | 听写结果（可反复 upsert） |
| `.../semantic_risk_transcripts/${BATCH}/wav/` | 序号软链，便于点播 |
| `.../semantic_risk_transcripts/${BATCH}/playlist.md` | 人类可读清单 |
| `.../derived/classified_*_${BATCH}_with_gold.parquet` | 合并后的 Manifest |

### 5.2 annotations.parquet 列

| 列 | 说明 |
|----|------|
| `sample_id` | 合并主键之一（= Manifest `id`） |
| `original_audio_sha256` | 合并主键之二（与 classification 身份哈希一致） |
| `batch` / `category` / `queue_index` | 批次、应为 `semantic_risk`、队列序号 |
| `wav_path` / `candidate_text` | 听音路径、自动候选文（参考） |
| `gold_text` / `gold_kind` | 人工转写；`speech` 或 `non_speech` |
| `decision` / `status` | `accept`/`empty`/`skip`；`annotated`/`empty`/`skipped` |
| `annotator_id` / `annotated_at` | 标注人与 UTC 时间 |

### 5.3 merge 写入 Manifest 的 labels

仅对命中且 `status` 为 `annotated`/`empty` 的样本更新：

| 字段 | 值 |
|------|----|
| `gold_text` | 人工转写（无语音为空串） |
| `gold_kind` | `speech` / `non_speech` |
| `annotation_state` | `annotated` |
| `annotator_id` / `annotated_at` | 来自 annotations |
| `label_source` | `human_transcript_cli` |
| `transcript_annotation_protocol` | `semantic_risk_cli_v1` |
| `is_human_verified` | `True`（表示本 CLI 听写已确认；**正式双审金标**仍走 annotation_v3） |
| `label` | 有语音时同步为 `gold_text` |
| `category` | **不修改** |

未出现在 annotations、或仅为 `skipped` 的样本：**完全不变**。

---

## 6. Mac Mini / Remote 操作建议

1. 用 Cursor / VS Code **Remote-SSH** 打开服务器上的工程目录。  
2. 终端跑 `prepare` → `label`；编辑器侧栏打开 `wav/` 目录备用。  
3. 若 `cursor`/`code` CLI 未进 PATH，自动打开会失败，但不影响标注：改用侧栏点击对应序号 wav。  
4. 网络抖动导致会话断开后：重新连上再跑同一条 `label` 命令即可续标（已 `annotated`/`empty` 会跳过）。  
5. 不要把 wav 拷到 Mac 本地再标——合并依赖服务器上的 `sample_id` / sha256 与路径一致性。

---

## 7. 验收清单

- [ ] `prepare` 后 `queue.parquet` 条数 ≈ XLSX/分类表中 `semantic_risk` 计数  
- [ ] `playlist.md` / `wav/` 中路径可在 Remote 资源管理器播放  
- [ ] `label` 中途 `/quit` 再启动，从下一条未完成继续  
- [ ] `merge` 报告 `applied` = 有效标注条数；`sha256_mismatches=0`  
- [ ] 抽查：有标注样本 `labels.gold_text` 正确；无标注样本与原 classified 一致；`category` 未变  

快速抽查示例：

```bash
python - <<'PY'
from audio_engine.core.manifest import Manifest
m = Manifest.load(
  f"datasets/stage1/derived/classified_five_class_v2_3_min_acoustic_{BATCH}_with_gold.parquet"
)
n_gold = sum(1 for s in m.samples if str(s.labels.get("gold_text") or "").strip()
             or s.labels.get("gold_kind") == "non_speech"
             and s.labels.get("label_source") == "human_transcript_cli")
n_risk = sum(1 for s in m.samples if s.labels.get("category") == "semantic_risk")
print("semantic_risk=", n_risk, "cli_gold_rows≈", n_gold, "total=", len(m.samples))
PY
```

（把 `BATCH` 换成实际批次名，或在 shell 里 `export` 后嵌入。）

---

## 8. 常见问题

| 现象 | 处理 |
|------|------|
| `分类 parquet 不存在` | 先跑 §2 分类；或 `--classified` 指到实际路径 |
| `未找到 category=semantic_risk` | 确认规则版本与 XLSX 统计；必要时换 `--rule-stem` |
| 缺 wav / playlist 显示 missing | 检查 cleaned/derived 上 `resampled_16k` 是否可读；或手工指 `--classified` 含路径的表 |
| `cursor`/`code` 打不开音频 | 用资源管理器点 `wav/` 软链；功能不依赖自动打开 |
| merge 提示输出已存在 | 加 `--overwrite`，或换 `--output` |
| sha256 不匹配 | 分类表与标注表不是同一批身份；勿混用不同 classified 再标 |
| 想改类别 | 本脚本**不改** `category`；改类用正式 warehouse / review 包的 `reviewed_category` |

---

## 9. 与正式人审入口的关系

| | 本 CLI 听写 | annotation_v3 / 仓库入仓 |
|--|------------|-------------------------|
| 范围 | 仅 `semantic_risk` | 可整批 / 按队列 |
| 交互 | 终端 + Remote 播 wav | XLSX/JSONL 包 |
| 合并键 | `sample_id` + `original_audio_sha256` | 相同 + queue 元数据 |
| 输出 | `*_with_gold.parquet` 旁路 | `review import` → reviewed / warehouse |
| 双审门禁 | 无 | 有 |

本入口适合快速听写校准、补 `gold_text`；若要进入正式训练金标与 batch 仓库冻结，请转 [batch唯一数据仓库-人工标注入仓.md](./batch唯一数据仓库-人工标注入仓.md)。
