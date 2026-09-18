# 019 · Levenshtein / 文本相似度原生加速



> 状态：**已完成**  

> 日期：2026-09-10  

> 落地：`selection_v3.text._levenshtein` 改用 `rapidfuzz.distance.Levenshtein`；`rapidfuzz>=3.0` 写入 `pyproject.toml` 正式依赖；缺库导入即 `ImportError`（无静默纯 Python 回退）；单测 `tests/test_selection_v3_text_levenshtein.py` 与历史纯 Python DP 对照。  

> 触发：`selection_v3.text._levenshtein` 为纯 Python DP；`0908-30000` 分拣单核打满、墙钟可达数小时级。即便 [018](./018-转写语速不可能防护栏.md) 拦掉超长幻觉，正常长度文本上的多次两两比较仍偏慢。  

> 前置 / 关联：热路径位于 `src/audio_engine/core/selection_v3/text.py`（`text_similarity` / `_levenshtein`）；消费方为 `classify_dataset_v3`。总加速蓝图见 [017](../进行中/017-v3分拣加速.md)（本需求从中拆出 **编辑距离 P0**，单独可交付）。[018](./018-转写语速不可能防护栏.md) 与本需求正交、可叠加。  

> 范围边界：只替换编辑距离 / 相似度实现为**现成原生加速库**；**不改** `comparison_text` 归一化、分桶阈值、决策链、侧车、清洗 DAG。并发与 Manifest 改造仍归 017，本需求可不依赖它们单独落地。



---



## 1. 当前事实（结论）



| 项 | 现状（落地后） |

| --- | --- |

| 实现 | `rapidfuzz.distance.Levenshtein.distance`（C++）；边界短路仍在 Python |

| 公式 | `text_similarity = 1 - dist / max(len(a), len(b))`；双空→1.0；单空→0.0；结果 `round(..., 6)`（未改） |

| 调用面 | 族内双跑稳定性、medoid、教师共识、`pairwise_min_similarity`、Qwen vs teacher 等（每条样本多次） |

| 依赖 | `pyproject.toml`：`rapidfuzz>=3.0`（正式依赖，非 optional） |



实测（本机 2026-09-10）：80 对长度 180 的近重复串，热路径相对纯 Python DP ≈ **847×**（0.48s → 0.0006s）。v2 / CER 路径仍走 `metrics.cer`（需对齐回溯），未与本函数合并。



---



## 2. 我想做什么



把 Levenshtein（及由此算出的 `text_similarity`）换成**现成的 C/C++/Rust 扩展库**，保持与现纯 Python 实现**距离整数一致**（或可证明的等价），从而显著缩短 `classify_dataset_v3` 墙钟。



### 推荐库（优先序）



| 优先级 | 库 | 说明 |

| --- | --- | --- |

| **首选（已采用）** | [`rapidfuzz`](https://github.com/rapidfuzz/rapidfuzz) | C++ 实现；`rapidfuzz.distance.Levenshtein.distance`；维护活跃、轮子全、Windows/Linux 友好 |

| 备选 | `python-Levenshtein` / `Levenshtein` | 经典 C 扩展；API 简单 |

| 备选 | `edlib`（若已有生态偏好） | C++；偏生物序列，文本也可用 |



### 实现要点（已落实）



1. 仅替换 `_levenshtein`；`comparison_text` / NFKC / 标点剥离仍在 Python。

2. 边界：`a == b` → 0；空串 vs 非空 → 非空长度；Unicode 按字符计。

3. 缺依赖：模块导入即失败并提示 `pip install -e .` / `pip install 'rapidfuzz>=3.0'`；**无**静默回退。

4. v2 走 `character_similarity` / CER，非同构调用面，未强行合并。

5. 未改阈值与 `round(..., 6)`。



---



## 3. 数据从哪来、结果要什么



- **输入**：无新流水线；仍是现有 `classify_dataset_v3` 输入。

- **输出**：分拣结果与加速前一致（桶 / decision / reason）；墙钟明显下降。

- **规模**：以 `0908-30000` 或数千～一万子集做前后对比即可。



---



## 4. 业务上有哪些规矩



1. **语义不变**：同一对 `comparison_text`，新旧 `distance` 必须相等（整数）；`text_similarity` 在现有 `round` 规则下一致。

2. **分桶不变**：固定回归集（含短句、中长句、近重复、空串）加速前后 `type`/`decision` 一致。

3. **依赖可安装**：Windows（本机）+ Linux（若有 CI/服务器）均可 `pip install`；写入依赖与简短安装说明。

4. **不替代 018**：语速不可能仍先短路；本库加速服务「未超限」样本上的合法比较。

5. **不替代 017 的并发**：本需求单独合入就应有可见加速；并发是加分项，不是本需求门票。



---



## 5. 我怎么才算满意



- [x] `_levenshtein` / `text_similarity` 走原生库（`rapidfuzz`），依赖已进 `pyproject.toml`。

- [x] 单元测试：空串、相等、单侧空、中文短句、较长随机串 —— 与**保留的纯 Python 参考实现**（仅测用）逐条 `distance` 一致。

- [x] 分拣回归：`tests/test_selection_v3_classify.py` / `test_selection_v3_contract.py` 全绿（67 passed, 1 skipped）。

- [x] 同机热路径对照：长度 180×80 对 ≈ **847×**（目标 ≥10× 已满足）；整批 classify 墙钟另随 017 并发叠加。

- [x] 缺库行为：导入即 `ImportError` + 安装说明；生产路径不静默算错。

- [x] 本需求移入 `已完成/`；017 中「P0 编辑距离」勾成由本需求交付。



---



## 6. 其他我想说的



- 改动面应很小：主要是 `text.py` + 依赖 + 测试；适合作为 017 的**第一个可单独合入 PR**。

- 不要自己写 Cython/手写 C；优先现成库。

- 与 018 叠加后：超长幻觉被 exclude，剩余文本更短，原生库收益仍然很大（调用次数 × 中等长度）。



---



## 安装说明



```text

pip install -e .

# 或单独：

pip install "rapidfuzz>=3.0"

```



缺库时导入 `audio_engine.core.selection_v3.text` 会直接失败，不会退回慢实现。


