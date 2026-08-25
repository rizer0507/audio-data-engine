"""Character-level alignment for CER analysis (reference vs hypothesis)."""

from __future__ import annotations

from typing import Any, Literal

Op = Literal["equal", "substitution", "deletion", "insertion"]


def align_characters(reference: str, hypothesis: str) -> list[dict[str, Any]]:
    """Return per-character alignment ops derived from Levenshtein backtrace.

    Each item:
      ``reference`` / ``hypothesis``: single char or None
      ``operation``: equal | substitution | deletion | insertion
    """
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

    ops: list[dict[str, Any]] = []
    i, j = rows, cols
    while i or j:
        if (
            i
            and j
            and reference[i - 1] == hypothesis[j - 1]
            and distance[i][j] == distance[i - 1][j - 1]
        ):
            ops.append(
                {
                    "reference": reference[i - 1],
                    "hypothesis": hypothesis[j - 1],
                    "operation": "equal",
                }
            )
            i -= 1
            j -= 1
        elif i and j and distance[i][j] == distance[i - 1][j - 1] + 1:
            ops.append(
                {
                    "reference": reference[i - 1],
                    "hypothesis": hypothesis[j - 1],
                    "operation": "substitution",
                }
            )
            i -= 1
            j -= 1
        elif j and distance[i][j] == distance[i][j - 1] + 1:
            ops.append(
                {
                    "reference": None,
                    "hypothesis": hypothesis[j - 1],
                    "operation": "insertion",
                }
            )
            j -= 1
        else:
            ops.append(
                {
                    "reference": reference[i - 1],
                    "hypothesis": None,
                    "operation": "deletion",
                }
            )
            i -= 1
    ops.reverse()
    return ops
