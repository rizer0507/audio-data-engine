# 流水线构建执行手册（AI Coding）

> **给你（AI）用。** 用户新开会话构建流水线时，先读本文，再落地用户提供的  
> `docs/需求/<名字>_需求.md`（通常由 `docs/流水线需求文档模板_用户填写.md` 用**自然语言**写成）。  
> **用户原话是需求真相源**；本文负责工程约束与把自然语言翻译成可执行流水线。  
> 与用户原文冲突时：**停下来问用户，不要自行改需求含义。**

---

## 0. 会话启动必读（防遗忘）

每次接到「按需求文档建流水线」时，按顺序做：

1. 打开并通读用户指定的 `docs/需求/..._需求.md`（未指定则让用户给出路径或直接粘贴需求）。
2. **把自然语言整理成内部实现清单**（不必改用户原文）：短名、输入、输出、步骤、是否 sharding/GPU、验收。缺**会阻塞实现**的信息再问用户（一次问清）；其余按本文默认。
3. 通读本文剩余章节。
4. 对照 `单条流水线执行命令.txt` 与现有 `pipelines/*.yaml`，确认命名与数据链不冲突。
5. 盘点 `audio-data operators` / `scripts/`，能复用则复用。
6. 列出将改/将建文件 → 实现 → 按「默认验收」+ 用户「我怎么才算满意」自检。

**禁止**：旁路独立 CLI 当生产入口；强迫用户手敲 `manifest shard` → `run-shards` → `merge`。

### 0.1 从自然语言抽取时的默认

| 用户没写清时 | 你的默认 |
|--------------|----------|
| 短名 | 根据中文目标生成 snake_case，开工前用一句话告知用户 |
| 输入 | 优先衔接现有数据链（如 cleaned → qwen）；路径不明则问 |
| 输出路径 | `datasets/manifests/<短名>.parquet` |
| 并发 | 数据量大或涉及 ASR/GPU → YAML `sharding`；小数据可不分片 |
| 失败样本 | 记入 status/errors，不伪造结果；除非用户要求丢弃 |
| 日志 | 新脚本必须 `context.log` |
| 生产入口 | 必须是一条 `audio-data pipeline run pipelines/<短名>.yaml` |

只问阻塞问题，例如：输入文件到底在哪、用哪张 GPU、热词原文是什么。不要让用户填技术表格。

---

## 1. 工程不变式（必须遵守）

```text
自然语言需求
  → pipelines/<短名>.yaml（唯一生产编排）
  → 可选 scripts/<业务>.py（script.python）或沉淀算子
  → audio-data pipeline run pipelines/<短名>.yaml
  → runs/<ts>_<name>[_shards]/ + output.manifest
```

| 原则 | 要求 |
|------|------|
| 单条命令 | 生产入口只有 `audio-data pipeline run <yaml>`；环境变量可写在执行命令文档里 |
| Manifest 真相源 | 输入输出都是 Sample/Manifest；音频只引用 `sample.audio[key]` |
| 自由脚本优先 | 临时/业务逻辑 → `scripts/*.py` + `operator: script.python`；`process(sample, params, context)` |
| 正式算子 | 仅当能力稳定、多条流水线复用、或必须 Batch/Manifest 集合语义时，才 `@register_operator` |
| 日志 | 脚本业务日志用 `context.log` → `runs/.../script_logs/<step>.jsonl`；禁止只靠 print |
| 分片并行 | 大数据量在 YAML 顶层写 `sharding:`，由引擎 split→并行→merge；不要恢复三步手工命令 |
| 不改 core | 除非需求明确要求；默认不改 `src/audio_engine/core/` |

---

## 2. 从需求文档到实现的映射

用户文档通常只有这几块自然语言；你负责映射：

| 用户写的 | 你要产出什么 |
|----------|--------------|
| 我想做什么 | 流水线目标、步骤拆解、短名 |
| 数据从哪来、结果要什么 | `input.*`、`output.manifest`、字段/音频 key |
| 业务上有哪些规矩 | filter/热词/阈值/脚本逻辑 |
| 我怎么才算满意 | 验收用例 + 更新执行命令台账 |
| 其他 | 环境变量、模型路径、禁改范围 |

---

## 3. 实现决策树

```text
缺口能力能否用已有 operator 组合？
  ├─ 能 → 只写 / 改 YAML
  └─ 否 → 逻辑是否「单样本进、单样本出」？
        ├─ 是，且偏业务/易变 → scripts/xxx.py + script.python
        ├─ 是，且稳定通用 → operators/<cat>/<name>.py + @register_operator
        └─ 否（删并样本/全局统计）→ ManifestOperator
```

批量 GPU ASR 等已有 `BatchOperator`（如 `asr.qwen_batch`）优先扩展配置（如 `context` 热词），不要平行再写一套旁路脚本。

---

## 4. 标准交付物清单

每个新流水线交付时必须具备：

1. **`pipelines/<短名>.yaml`**（需要并行则含 `sharding`，且必有 `output.manifest`）
2. **缺口代码**（脚本和/或算子 + 注册）
3. **配置**（如有）
4. **更新 `单条流水线执行命令.txt`**（按现有范式：注释 + export + 一条 `pipeline run`）
5. 向用户用简体中文说明：怎么跑、怎么续跑；对照其「我怎么才算满意」逐条回应

---

## 5. YAML 骨架（内部用，不要让用户填）

```yaml
name: <短名>
input:
  manifest: datasets/manifests/<上游>.parquet
output:
  manifest: datasets/manifests/<短名产物>.parquet
execution:
  executor: thread
  workers: 4
  fail_fast: false
  checkpoint_every: 1000
sharding:
  shards: 8
  strategy: hash   # ASR 常用 duration-balanced
  parallel_shards: 8
pipeline:
  - name: <step>
    operator: <算子或 script.python>
    params: {}
```

自由脚本最小形：

```python
def process(sample, params, context):
    context.log("started")
    context.log("finished", key=value)
    return {"labels": {...}}
```

---

## 6. 执行清单

### A. 开工前

- [ ] 已读懂用户自然语言；阻塞项已问清
- [ ] 内部实现清单已对齐；短名已告知用户
- [ ] 已盘点复用；输出路径不会误覆盖（或已确认）

### B. 编码

- [ ] YAML + 脚本/算子；key 链正确
- [ ] 有 sharding 时不手写三步 CLI
- [ ] `context.log` / lineage / 失败不进伪造结果
- [ ] 缓存可失效（脚本 hash 或 version/params）

### C. 收尾

- [ ] 更新 `单条流水线执行命令.txt`
- [ ] 对照用户「我怎么才算满意」验收
- [ ] 不擅自 commit；汇报改动文件与运行方式

---

## 7. 开场提示词（用户可复制）

```text
请严格按 docs/流水线构建执行手册_AI.md 执行。
需求文档：docs/需求/<名字>_需求.md
用自然语言需求即可；缺关键信息先问我，其余工程细节你定。
做完更新 单条流水线执行命令.txt，保证一条 pipeline run 能跑。
```

---

## 8. 与旧文档的关系

| 文档 | 角色 |
|------|------|
| `docs/流水线需求文档模板_用户填写.md` | 用户用自然语言写需求 |
| **本文** | AI 翻译需求 + 工程落地 |
| `单条流水线执行命令.txt` | 生产命令台账（必须更新） |
| `docs/流水线构建统一流程.md` | 历史文档；与本文冲突时以**本文 + 用户原话**为准 |
