"""061 · 人工音频分类 / 训练排除 —— 类别别名读取与硬排除的单一事实来源。

契约（audio-label-master ``semantic_risk_web_v2``）：

- 前三类人工分类（silence / background_suspected / human_noise）
  → category=environment_noise, gold_text="", gold_kind=non_speech
- 多人说话（multi_speaker）
  → category=multi_speaker, gold_text=None（必须保留 null）,
    gold_kind=ambiguous_target, exclude_from_training=True
- 旧协议（semantic_risk_web_v1 / semantic_risk_cli_v1）仍可读取；
  无 annotation_type 的历史 empty 不得猜测为 silence。

所有产出训练样本的出口都必须通过 :func:`training_exclusion_reasons`，
且必须在文本归一化 / 文本回退 / 类别选择 / 配额 / 写文件之前执行。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

CATEGORY_MULTI_SPEAKER = "multi_speaker"
CATEGORY_ENVIRONMENT_NOISE = "environment_noise"

ANNOTATION_TYPE_SILENCE = "silence"
ANNOTATION_TYPE_BACKGROUND_SUSPECTED = "background_suspected"
ANNOTATION_TYPE_HUMAN_NOISE = "human_noise"
ANNOTATION_TYPE_MULTI_SPEAKER = "multi_speaker"

NOISE_ANNOTATION_TYPES = frozenset(
    {
        ANNOTATION_TYPE_SILENCE,
        ANNOTATION_TYPE_BACKGROUND_SUSPECTED,
        ANNOTATION_TYPE_HUMAN_NOISE,
    }
)
ANNOTATION_TYPES = NOISE_ANNOTATION_TYPES | {ANNOTATION_TYPE_MULTI_SPEAKER}

DECISION_MULTI_SPEAKER = "multi_speaker"

PROTOCOL_WEB_V1 = "semantic_risk_web_v1"
PROTOCOL_WEB_V2 = "semantic_risk_web_v2"
PROTOCOL_CLI_V1 = "semantic_risk_cli_v1"

EXCLUSION_REASON_MULTI_SPEAKER = "multi_speaker"

# 有效类别读取优先级（人工结果在前）。source_category 仅溯源，永不参与。
CATEGORY_ALIAS_PRIORITY: tuple[str, ...] = (
    "reviewed_category",
    "category",
    "type",
    "training_category",
)

# 排除检查时额外扫描的别名（含 classification_bucket 以防遗留写法）。
_CONFLICT_SCAN_KEYS: tuple[str, ...] = CATEGORY_ALIAS_PRIORITY

# 排除原因代码（用于报告统计）
REASON_CATEGORY = "category_multi_speaker"
REASON_ANNOTATION_TYPE = "annotation_type_multi_speaker"
REASON_FLAG = "exclude_from_training_flag"
REASON_ALIAS_CONFLICT = "category_alias_conflict_multi_speaker"


def is_blank(value: Any) -> bool:
    """None / NaN / 空白串视为空。"""
    if value is None:
        return True
    try:
        # pandas / numpy NaN
        if isinstance(value, float) and value != value:
            return True
    except Exception:  # noqa: BLE001
        pass
    if value.__class__.__name__ in {"NAType", "NaTType"}:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def clean_str(value: Any) -> str:
    return "" if is_blank(value) else str(value).strip()


def parse_bool(value: Any) -> bool:
    """宽松解析 exclude_from_training（bool / numpy bool / 字符串 / 数字）。"""
    if is_blank(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def effective_category(labels: Mapping[str, Any] | None) -> str:
    """有效主类别：reviewed_category > category > type > training_category。

    不读取 source_category，避免过期自动类别压过人工结果。
    """
    labels = labels or {}
    for key in CATEGORY_ALIAS_PRIORITY:
        val = clean_str(labels.get(key))
        if val:
            return val
    return ""


def category_alias_values(labels: Mapping[str, Any] | None) -> dict[str, str]:
    labels = labels or {}
    return {
        key: clean_str(labels.get(key))
        for key in _CONFLICT_SCAN_KEYS
        if clean_str(labels.get(key))
    }


def training_exclusion_reasons(labels: Mapping[str, Any] | None) -> list[str]:
    """返回硬排除原因代码；空列表表示不触发 061 排除（不豁免其他门禁）。

    规则（任一成立即排除）：
    1. 有效人工主类别 == multi_speaker
    2. annotation_type == multi_speaker
    3. exclude_from_training 为真
    4. 类别别名互相矛盾且任一别名为 multi_speaker（不得默默纳入）
    """
    labels = labels or {}
    reasons: list[str] = []
    eff = effective_category(labels)
    if eff == CATEGORY_MULTI_SPEAKER:
        reasons.append(REASON_CATEGORY)
    if clean_str(labels.get("annotation_type")) == ANNOTATION_TYPE_MULTI_SPEAKER:
        reasons.append(REASON_ANNOTATION_TYPE)
    if parse_bool(labels.get("exclude_from_training")):
        reasons.append(REASON_FLAG)
    aliases = category_alias_values(labels)
    if (
        CATEGORY_MULTI_SPEAKER in aliases.values()
        and eff != CATEGORY_MULTI_SPEAKER
    ):
        reasons.append(REASON_ALIAS_CONFLICT)
    return reasons


def is_multi_speaker(labels: Mapping[str, Any] | None) -> bool:
    """是否属于多人说话集合（用于训练集合交集校验）。"""
    reasons = training_exclusion_reasons(labels)
    return any(
        r in {REASON_CATEGORY, REASON_ANNOTATION_TYPE, REASON_ALIAS_CONFLICT}
        for r in reasons
    )


@dataclass(frozen=True)
class SampleIdentity:
    """样本身份（跨 batch 同名 sample_id 需带 batch 区分）。"""

    batch: str
    sample_id: str


def assert_no_multi_speaker_in_training(
    train_items: Iterable[tuple[SampleIdentity, Mapping[str, Any]]],
    multi_speaker_ids: Iterable[SampleIdentity],
) -> None:
    """训练 ID 集合 ∩ multi_speaker ID 集合必须为空；同时复核每条训练样本的有效标签。"""
    train_list = list(train_items)
    train_ids = {ident for ident, _ in train_list}
    overlap = train_ids & set(multi_speaker_ids)
    if overlap:
        sample = sorted((i.batch, i.sample_id) for i in overlap)[:5]
        raise AssertionError(
            f"训练集含 multi_speaker 样本 {len(overlap)} 条（禁止）: {sample}"
        )
    for ident, labels in train_list:
        if training_exclusion_reasons(labels):
            raise AssertionError(
                f"训练样本仍命中 061 排除规则: {ident.batch}/{ident.sample_id}"
            )