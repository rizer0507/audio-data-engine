from __future__ import annotations

from typing import Any


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
