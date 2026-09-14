"""019: rapidfuzz Levenshtein must match the historical pure-Python DP."""

from __future__ import annotations

import random
import time

import pytest

from audio_engine.core.selection_v3.text import _levenshtein, text_similarity


def _levenshtein_python_ref(left: str, right: str) -> int:
    """Historical row-rolling DP (tests only; not used in production)."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, ch_l in enumerate(left, start=1):
        curr = [i]
        for j, ch_r in enumerate(right, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ch_l == ch_r else 1)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


def _similarity_from_dist(a: str, b: str, dist: int) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return round(1.0 - dist / max(len(a), len(b)), 6)


CASES = [
    ("", ""),
    ("", "你好"),
    ("你好", ""),
    ("你好", "你好"),
    ("你好世界", "你好世界"),
    ("你好世界", "你好世间"),
    ("今天天气不错", "今天的天气不错"),
    ("abc", "abd"),
    ("kitten", "sitting"),
    ("全角ＡＢＣ", "半角ABC"),  # different code points until NFKC; raw distance still defined
    ("重复重复重复", "重复重复"),
]


@pytest.mark.parametrize("left,right", CASES)
def test_distance_matches_python_ref(left: str, right: str) -> None:
    assert _levenshtein(left, right) == _levenshtein_python_ref(left, right)


@pytest.mark.parametrize("left,right", CASES)
def test_text_similarity_matches_python_ref(left: str, right: str) -> None:
    ref_dist = _levenshtein_python_ref(left, right)
    assert text_similarity(left, right) == _similarity_from_dist(left, right, ref_dist)


def test_longer_random_strings_match_python_ref() -> None:
    rng = random.Random(19)
    alphabet = "你好世界天气不错今天的语音识别模型加速测试ABCDEFG0123456789"
    for length in (20, 50, 120, 200):
        for _ in range(8):
            left = "".join(rng.choice(alphabet) for _ in range(length))
            # Near-duplicate + noise
            right_chars = list(left)
            for _ in range(max(1, length // 20)):
                idx = rng.randrange(len(right_chars))
                op = rng.choice(("sub", "ins", "del"))
                if op == "sub":
                    right_chars[idx] = rng.choice(alphabet)
                elif op == "ins":
                    right_chars.insert(idx, rng.choice(alphabet))
                elif len(right_chars) > 1:
                    del right_chars[idx]
            right = "".join(right_chars)
            assert _levenshtein(left, right) == _levenshtein_python_ref(left, right)
            assert text_similarity(left, right) == _similarity_from_dist(
                left, right, _levenshtein_python_ref(left, right)
            )


def test_rapidfuzz_hot_path_faster_than_python_ref() -> None:
    """Sanity wall-clock: native path should dominate pure Python on mid-length pairs."""
    rng = random.Random(42)
    alphabet = "中文语音识别编辑距离加速验证abcdefghijklmnopqrstuvwxyz"
    pairs = []
    for _ in range(40):
        n = 180
        left = "".join(rng.choice(alphabet) for _ in range(n))
        right = list(left)
        for i in range(0, n, 17):
            right[i] = rng.choice(alphabet)
        pairs.append((left, "".join(right)))

    t0 = time.perf_counter()
    for a, b in pairs:
        _levenshtein_python_ref(a, b)
    py_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    for a, b in pairs:
        _levenshtein(a, b)
    native_s = time.perf_counter() - t1

    # Target in 019: hot path ≥10×; allow generous CI jitter floor.
    assert native_s > 0
    speedup = py_s / native_s
    assert speedup >= 5.0, f"expected significant speedup, got {speedup:.1f}x (py={py_s:.4f}s native={native_s:.4f}s)"
