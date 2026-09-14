from __future__ import annotations

from typing import Any


def _looks_mixed(text: str) -> bool:
    value = text or ""
    cjk = sum(1 for ch in value if "\u4e00" <= ch <= "\u9fff")
    latin = sum(1 for ch in value if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    letters = cjk + latin
    if letters == 0 or not cjk or not latin:
        return False
    return cjk / letters >= 0.2 and latin / letters >= 0.2


def mixed_language_character_metrics(reference: str, hypothesis: str) -> dict[str, Any]:
    """Refuse whole-utterance Chinese CER after deleting Latin and concatenating CJK.

    Mixed text is not character-comparable as one Chinese sentence. Coverage is
    the share of CJK characters, not a score. No local CER is invented when the
    same-language spans are not a reliable alignment.
    """
    ref = reference or ""
    hyp = hypothesis or ""
    ref_cjk = sum(1 for ch in ref if "\u4e00" <= ch <= "\u9fff")
    hyp_cjk = sum(1 for ch in hyp if "\u4e00" <= ch <= "\u9fff")
    ref_latin = sum(1 for ch in ref if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    hyp_latin = sum(1 for ch in hyp if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    ref_letters = ref_cjk + ref_latin
    hyp_letters = hyp_cjk + hyp_latin
    return {
        "cer": None,
        "char_comparable": False,
        "reason": "mixed_language",
        "insertions": None,
        "reference_length": len(ref),
        "coverage": {
            "reference_cjk_share": (ref_cjk / ref_letters) if ref_letters else None,
            "hypothesis_cjk_share": (hyp_cjk / hyp_letters) if hyp_letters else None,
        },
        "local_cer": None,
        "note": "latin_was_not_stripped_and_cjk_was_not_concatenated",
    }


def reference_character_metrics(
    reference: str,
    hypothesis: str,
    *,
    char_comparable: bool = True,
) -> dict[str, Any]:
    """Reference CER. Consensus distance must not be reported as this metric.

    Empty reference has no CER; insertions are reported separately.
    Language mismatch is ``cer=None``, never 0 or 1.
    """
    if _looks_mixed(reference) or _looks_mixed(hypothesis):
        return mixed_language_character_metrics(reference, hypothesis)
    if not char_comparable:
        return {
            "cer": None,
            "char_comparable": False,
            "reason": "language_mismatch",
            "insertions": None,
            "reference_length": len(reference or ""),
        }
    if not reference:
        return {
            "cer": None,
            "char_comparable": True,
            "reason": "empty_reference",
            "insertions": len(hypothesis or ""),
            "reference_length": 0,
        }
    scored = calculate_cer(reference, hypothesis)
    edits = scored["substitutions"] + scored["deletions"] + scored["insertions"]
    scored["cer"] = edits / len(reference)
    scored["char_comparable"] = True
    scored["reason"] = "reference_length"
    return scored


def calculate_cer(reference: str, hypothesis: str) -> dict[str, Any]:
    """Calculate character edit operations. Empty reference uses insertion count."""
    rows, cols = len(reference), len(hypothesis)
    distance = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        distance[i][0] = i
    for j in range(cols + 1):
        distance[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            distance[i][j] = min(
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                distance[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
            )
    i, j = rows, cols
    substitutions = deletions = insertions = 0
    while i or j:
        if (
            i
            and j
            and reference[i - 1] == hypothesis[j - 1]
            and distance[i][j] == distance[i - 1][j - 1]
        ):
            i -= 1
            j -= 1
        elif i and j and distance[i][j] == distance[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif j and distance[i][j] == distance[i][j - 1] + 1:
            insertions += 1
            j -= 1
        else:
            deletions += 1
            i -= 1
    edits = substitutions + deletions + insertions
    cer = 0.0 if not reference and not hypothesis else edits / max(len(reference), 1)
    return {
        "cer": round(cer, 6),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "reference_length": len(reference),
    }
