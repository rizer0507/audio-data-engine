"""Independent dual-run vs same-run resume cache boundaries."""

from __future__ import annotations

from dataclasses import dataclass

# Fixed route aliases for three-family dual runs (dataset_policy_v3 three_family).
FAMILY_RUN_ALIASES: dict[str, tuple[str, str]] = {
    "qwen": ("qwen_1", "qwen_2"),
    "glm": ("glm_1", "glm_2"),
    "sensevoice": ("sensevoice_1", "sensevoice_2"),
}

REQUIRED_FAMILIES = ("qwen", "glm", "sensevoice")


@dataclass(frozen=True)
class CacheBoundaryNote:
    """Documented contract for orchestrator and operators."""

    independent_dual_run: str = (
        "同一家族两路必须使用不同 --asr-run 别名（如 qwen_1 / qwen_2）。"
        "BaseOperator 缓存键包含完整 params（含 transcript_key），"
        "产物路径为 {alias}_asr_{batch}.parquet，因此跨路不会命中另一路缓存。"
    )
    same_run_resume: str = (
        "同路恢复：仅当该 stage 已 succeeded 且 ASR parquet + registered identity "
        "仍存在且 execution_id 未变时跳过；允许同 alias 继续，不重新占卡。"
    )
    forbidden: str = (
        "禁止复制第一路 parquet 冒充第二路；禁止复用 execution_id / artifact_id；"
        "禁止删掉失败家族后仍宣称三族完成。"
    )


CACHE_BOUNDARY = CacheBoundaryNote()


def all_run_aliases() -> list[str]:
    aliases: list[str] = []
    for family in REQUIRED_FAMILIES:
        aliases.extend(FAMILY_RUN_ALIASES[family])
    return aliases


def family_for_alias(alias: str) -> str:
    for family, pair in FAMILY_RUN_ALIASES.items():
        if alias in pair:
            return family
    raise KeyError(f"未知 ASR 路次别名: {alias}")


def assert_unique_execution_ids(execution_ids: list[str]) -> None:
    cleaned = [str(item).strip() for item in execution_ids if str(item).strip()]
    if len(cleaned) != len(set(cleaned)):
        raise ValueError("execution_id 必须六路互异；禁止复制同一执行身份冒充双跑")
