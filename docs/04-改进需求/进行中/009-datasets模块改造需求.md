# 009 · datasets 模块按工序分层改造需求

> 状态：**进行中（Phase A + B 已落地；Phase C 迁移/清理工具未做）**。  
> 目标：厘清 `datasets/` 与 `runs/` 的职责边界，按三大工序与「昂贵/可重算」价值分级存放 durable 产物；评测报告离开 `runs/`。  
> 关联现行说明：[工序总览](../../01-项目架构/工序总览.md)、[评测流水线](../../03-流水线/评测流水线.md)、[datasets/README](../../../datasets/README.md)。  
> 命名约定代码权威：`src/audio_engine/core/source_naming.py`。

---

## 1. 问题陈述

当前几乎所有工序的 durable Parquet 都扁平落在 `datasets/manifests/`，靠文件名前缀区分；每次 pipeline 运行又在 `runs/<ts>_<name>/` 再存一份 `manifest.parquet` + checkpoint；评测最终报告硬编码写在 `runs/.../reports/`。

体感上：

1. **三大工序产出堆在一起**：清洗、ASR、聚拢、分拣、评测集、评测推理、评测指标全部平铺同一目录。
2. **`datasets` 与 `runs` 没有本质区别**：二者都在堆完整 manifest；缺少「可复用业务产物」vs「单次执行痕迹」的清晰分层。
3. **昂贵产物与廉价产物同级**：模型识别 Parquet（跑批极久）与组合筛选/聚合（几十万条也很快）混放，备份与清理策略无法区分。
4. **评测交付物落在 runs**：`evaluation.json` / `evaluation.xlsx` 随 run 时间戳漂移，无稳定「最新报告」路径，且 `runs/` 被 gitignore，易丢、难分享。

### 现状对照

| 产物类型 | 现行落点 | 问题 |
| --- | --- | --- |
| 清洗 `cleaned_*` | `datasets/manifests/` | 与 ASR/评测混放 |
| ASR `{alias}_asr_*` | `datasets/manifests/` | 昂贵，却无独立保护区 |
| 聚拢 / 字准 / 分拣 | `datasets/manifests/` | 可快速重算，却与 ASR 同级 |
| 评测集 / 评测推理 / 指标 | `datasets/manifests/` | 工序三与工序一混目录 |
| 评测报告 | `runs/.../reports/` | 业务交付物误入执行痕迹 |
| 金标 / 汇总 xlsx | `data/exports/` | 相对合理，可保留 |
| 运行日志 / checkpoint | `runs/` | 本职正确 |

---

## 2. 设计原则

1. **`datasets/` = 可复用业务真相源**（按工序 + 价值分级）。  
2. **`runs/` = 单次执行痕迹**（config / log / metrics / checkpoint / 可选临时副本）。  
3. **文件名约定不变**：继续用 `--source-name` / `--asr-run` / `--eval-name` 生成 stem（如 `qwen_asr_mt3000`）；只改**目录根**，不改 stem 语义。  
4. **昂贵优先保护、廉价可重算**：ASR 识别结果 P0；aggregate / metrics / classified 可丢后重跑。  
5. **兼容旧路径**：解析时先新后旧；迁移期双读，写只写新路径。  
6. **不推翻 Operator 内部逻辑**：优先改路径契约层（`source_naming`、CLI、报告算子、YAML/手册）。

---

## 3. 目标布局

```text
datasets/
├── README.md
├── stage1/                         # 工序一：数据清洗落库打标
│   ├── cleaned/                    # ① 清洗落库
│   │   └── cleaned_{BATCH}.parquet
│   ├── asr/                        # ② ASR 识别（★ 昂贵，重点保护）
│   │   └── {alias}_asr_{BATCH}.parquet
│   ├── derived/                    # ②③ 可快速重算的派生表
│   │   ├── multi_asr_aggregate_{BATCH}.parquet
│   │   ├── multi_asr_metrics_{BATCH}.parquet
│   │   └── classified_{BATCH}.parquet
│   └── exports/                    # 可选：金标/汇总旁路副本（主路径仍可在 data/exports/）
│
├── stage3/                         # 工序三：评测
│   ├── eval_sets/                  # 注册评测集
│   │   └── eval_{BATCH}.parquet
│   ├── asr/                        # 评测集上的独立推理（★ 昂贵）
│   │   └── {alias}_asr_eval_{BATCH}.parquet
│   ├── derived/                    # 聚拢 / 指标（可重算）
│   │   ├── eval_aggregate_eval_{BATCH}.parquet
│   │   └── eval_metrics_eval_{BATCH}.parquet
│   └── reports/                    # ★ 评测交付报告（从 runs 迁出）
│       └── {eval_name}/
│           ├── evaluation.json
│           ├── evaluation.xlsx
│           └── latest -> 指向最近一次成功写出的报告（或固定文件名覆盖）
│
├── manifests/                      # 【兼容层】迁移完成前保留；新写入禁止默认落此
└── shards/                         # 分片临时目录（可继续用，或迁到 runs 下）

runs/
└── <YYYYMMDD_HHMMSS>_<pipeline>[_<source|eval>]/
    ├── config.yaml
    ├── run.log
    ├── metrics.json
    ├── checkpoints/
    ├── artifact.json               # 指向 datasets 正式产物 URI
    └── manifest.parquet            # 可选：运行快照；不再当作业务交付物
    # 不再作为 evaluation.xlsx / 业务报告的权威落点
```

### 价值分级

| 级别 | 目录 | 典型产物 | 运维策略 |
| --- | --- | --- | --- |
| **P0 昂贵** | `stage1/asr/`、`stage3/asr/` | `{alias}_asr_*.parquet` | 备份优先；清理脚本默认跳过 |
| **P1 派生** | `*/derived/`、`cleaned/`、`eval_sets/` | aggregate / metrics / classified / cleaned / eval | 可删后按命令重跑 |
| **P2 交付** | `stage3/reports/`、`data/exports/` | evaluation 报告、金标/汇总 xlsx | 稳定路径；可对外分享 |
| **痕迹** | `runs/` | log / checkpoint / 运行快照 | 可按保留策略定期清理 |

工序二（训练）产物仍以外部框架 + `data/catalog` Release / Model Registry 为主，本改造不强制往 `datasets/` 写权重。

---

## 4. datasets vs runs：职责重申

| | `datasets/` | `runs/` |
| --- | --- | --- |
| 问题 | 「这批数据现在是什么？」 | 「这次跑发生了什么？」 |
| 生命周期 | 跨多次运行复用 | 单次运行 |
| 内容 | 正式 Parquet / 评测报告 | config、log、metrics、checkpoint |
| 是否业务交付 | 是 | 否 |
| 清理 | 按价值分级 | 可按时间/磁盘策略清理 |

双写关系保持：PipelineRunner 仍可写 `runs/.../manifest.parquet` 作快照；CLI 配置了 `output.manifest` 时，**正式真相源**写入对应 `datasets/stage*/...`。

---

## 5. 路径契约（命名不变，根目录变）

文件名 stem 规则**冻结**（与现行一致）：

| 场景 | stem |
| --- | --- |
| 清洗 | `cleaned_{source}` |
| ASR | `{alias}_asr_{source}` |
| 聚拢 / 字准 | `multi_asr_aggregate_{source}` / `multi_asr_metrics_{source}` |
| 分拣 | `classified_{source}` |
| 评测集 | `eval_{batch}` |
| 评测推理 | `{alias}_asr_eval_{batch}` |
| 评测聚拢 / 指标 | `eval_aggregate_eval_{batch}` / `eval_metrics_eval_{batch}` |

目标路径映射：

| stem 模式 | 新根目录 |
| --- | --- |
| `cleaned_*` | `datasets/stage1/cleaned/` |
| `*_asr_*` 且不含 `_asr_eval_` | `datasets/stage1/asr/` |
| `multi_asr_*` / `classified_*` | `datasets/stage1/derived/` |
| `eval_*`（注册集，非 aggregate/metrics） | `datasets/stage3/eval_sets/` |
| `*_asr_eval_*` | `datasets/stage3/asr/` |
| `eval_aggregate_*` / `eval_metrics_*` | `datasets/stage3/derived/` |
| 评测报告 | `datasets/stage3/reports/{eval_name}/` |

实现建议：在 `source_naming.py` 增加 `manifest_kind_root(kind) -> Path`，`manifest_path()` 按 kind 选子目录；`resolve_existing_manifest()` **先搜新根，再回退 `datasets/manifests/`**。

---

## 6. 评测报告迁出 runs

### 现行问题

`src/audio_engine/operators/quality/evaluation_report.py` 硬编码：

```text
{run_dir}/reports/evaluation.json
{run_dir}/reports/evaluation.xlsx
```

### 目标行为

1. 权威写出：`datasets/stage3/reports/{eval_name}/evaluation.{json,xlsx}`  
2. `runs/.../reports/` 可写一份副本或仅留指针（`artifact.json` / 相对路径），**不再是唯一落点**。  
3. 同一 `eval_name` 重复成功跑批时：覆盖同名文件，或写带时间戳子目录 + 更新 `latest` 指针；默认推荐**覆盖同名 + 在 json 内记录 run_id / 生成时间**（路径稳定，便于手册写死）。  
4. 同类「业务报告」若仍写在 runs（如 alignment / inject 报告）：本次优先迁评测报告；其它可在后续小需求跟进。

---

## 7. 实施范围（建议分阶段）

### Phase A — 路径契约（核心，优先）

- [x] `source_naming.py`：按 kind 映射 stage 子目录；解析双读
- [x] `cli/main.py` / `apply_source_name_*` / `apply_eval_name_*`：写出新路径
- [x] `evaluation_report.py`：报告写到 `stage3/reports/`（`--eval-name` 注入；`runs/` 仍留副本）
- [x] `resolve_existing_manifest` / join 路径：旧 `datasets/manifests/` 仍能找到
- [x] 单测：命名映射、双读、报告路径（`tests/test_source_naming.py` 等）

### Phase B — 配置与手册同步

- [x] `pipelines/*.yaml` 中写死的 `datasets/manifests/...` 改为 staged 路径
- [x] `docs/03-流水线/*`、`docs/07-操作手册/*`、`手册/local|dev/*`、`单条流水线执行命令.txt`
- [x] 根 `README.md` 目录结构、`datasets/README.md`
- [x] 旁路脚本 `run_eval_from_summary.py` / `run_eval_from_label_xlsx.py` 等示例与写出路径

### Phase C — 迁移与清理工具（可选）

- [ ] 一次性迁移脚本：把现有 `datasets/manifests/*` 按 stem 规则搬到 stage 子目录（保留原文件或 soft-link 过渡期）  
- [ ] 清理脚本：默认保护 `*/asr/`，允许清理过期 `runs/` 与可选 `derived/`

**明确不做（本需求）**：

- 不改分拣规则 / 指标语义 / Operator 算法  
- 不强制物理音频入库（仍以 `resources/manifest.yaml` 登记为准）  
- 不引入新的「与 runs 平行的第二套执行系统」

---

## 8. 关键代码与文档入口

| 职责 | 路径 |
| --- | --- |
| 命名 / 路径推导 | `src/audio_engine/core/source_naming.py` |
| Manifest I/O 默认根 | `src/audio_engine/core/manifest.py` |
| CLI 写出正式产物 | `src/audio_engine/cli/main.py` |
| 评测报告落点 | `src/audio_engine/operators/quality/evaluation_report.py` |
| Catalog 注册 | `src/audio_engine/core/catalog.py`（跟随真实 path） |
| 工序权威说明 | `docs/01-项目架构/工序总览.md` |
| 评测产物说明 | `docs/03-流水线/评测流水线.md` |
| 本需求 | `docs/04-改进需求/进行中/009-datasets模块改造需求.md` |

---

## 9. 验收标准

1. 新跑的工序一 ASR 产物落在 `datasets/stage1/asr/`，不进扁平 `manifests/`。 ✅ Phase A  
2. 新跑的工序三评测报告落在 `datasets/stage3/reports/{eval_name}/`，不依赖翻找 `runs/<ts>_.../reports/`。 ✅ Phase A  
3. 旧路径下的 parquet 在未迁移时仍可被 `--join-manifest` / `resolve_existing_manifest` 解析。 ✅ Phase A  
4. `runs/` 清理不影响已落盘的 ASR parquet 与评测报告。 ✅ Phase A  
5. 手册中至少有一处「工序三完整四步」命令簇使用新报告路径。 ✅ Phase B（`手册/*/03-工序三-评测.txt`）  
6. `datasets/README.md` 与 `工序总览` 中的产物落点描述与实现一致。 ✅ Phase B  

---

## 10. 同步文档清单

| 文档 | 同步内容 |
| --- | --- |
| [datasets/README.md](../../../datasets/README.md) | 现行 stage 布局（Phase A+B） |
| [工序总览](../../01-项目架构/工序总览.md) | 「产物落点」节为现行 |
| [评测流水线](../../03-流水线/评测流水线.md) | 命令与产物路径为 stage3 |
| [设计决策 README](../../05-设计决策/README.md) | Phase A+B 已落地 |
| [流水线构建-AI执行手册](../../07-操作手册/流水线构建-AI执行手册.md) | 默认输出改为 staged |
| 根 [README.md](../../../README.md) | 目录结构含 stage1/stage3 |
| `pipelines/*.yaml` / `手册/*` / `单条流水线执行命令.txt` | Phase B 路径同步 |

Phase C（存量 `manifests/` → stage 迁移脚本）仍可选。全部完成后可将本文移入 `docs/04-改进需求/已完成/`。

---

## 11. 交给实现 AI 时可以说

```text
请按 docs/04-改进需求/进行中/009-datasets模块改造需求.md 落地 Phase A+B。
原则：stem 命名不变，只改目录根；resolve 双读旧 manifests；评测报告迁到 datasets/stage3/reports/。
先改 source_naming + evaluation_report + 单测，再扫 pipelines/docs/手册路径。
不要改分拣规则与指标语义。做完更新 datasets/README 与工序总览中的「现行」表述。
```
