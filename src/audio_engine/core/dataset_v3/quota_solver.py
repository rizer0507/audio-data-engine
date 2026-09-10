"""Solve joint train quotas when independent pool draws violate cross constraints."""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


def solve_train_quotas(pools, targets, config, *, used_ids, used_dups, call_counts):
    from audio_engine.core.dataset_v3.sampling import _call_key, _exact_dup_key
    candidates = sorted([(s, name) for name, samples in pools.items() for s in samples
                         if s.id not in used_ids and _exact_dup_key(s, config.exact_duplicate_fields) not in used_dups],
                        key=lambda pair: pair[0].id)
    if not candidates:
        return None
    rows, lower, upper = [], [], []

    def constraint(indices, lo, hi):
        rows.append(indices)
        lower.append(lo)
        upper.append(hi)

    for name, count in targets.items():
        constraint([i for i, (_, pool) in enumerate(candidates) if pool == name], count, count)
    cross = config.train_cross
    n = config.train_size
    for field, value, ratio in [("human_semantic", "positive", cross.min_positive_ratio),
                                ("human_semantic", "negative", cross.min_negative_ratio)]:
        constraint([i for i, (s, _) in enumerate(candidates) if s.labels.get(field) == value], math.ceil(n * ratio), np.inf)
    noisy = [i for i, (s, _) in enumerate(candidates) if s.labels.get("gold_kind") == "speech" and
             (s.labels.get("human_noise") == "noisy" or str(s.labels.get("human_crosstalk")).lower() == "true")]
    constraint(noisy, math.ceil(n * cross.min_noisy_crosstalk_ratio), np.inf)
    constraint([i for i, (s, _) in enumerate(candidates) if s.labels.get("gold_kind") == "non_speech"],
               0, math.floor(n * cross.max_non_speech_ratio))
    constraint([i for i, (_, p) in enumerate(candidates) if p == "pseudo_high_audited"],
               0, math.floor(n * cross.max_pseudo_high_ratio))
    calls, duplicates = defaultdict(list), defaultdict(list)
    for i, (s, _) in enumerate(candidates):
        call = _call_key(s)
        if call:
            calls[call].append(i)
        dup = _exact_dup_key(s, config.exact_duplicate_fields)
        if dup:
            duplicates[dup].append(i)
    for call, indices in calls.items():
        constraint(indices, 0, max(0, config.max_per_call - call_counts.get(call, 0)))
    for indices in duplicates.values():
        constraint(indices, 0, 1)
    matrix = lil_matrix((len(rows), len(candidates)), dtype=float)
    for i, indices in enumerate(rows):
        matrix[i, indices] = 1
    costs = [int(hashlib.sha256(f"{config.sampling_seed}|joint|{s.id}".encode()).hexdigest()[:12], 16) / 16**12
             for s, _ in candidates]
    result = milp(np.array(costs), integrality=np.ones(len(candidates)), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix.tocsc(), lower, upper),
                  options={"time_limit": 60.0, "mip_rel_gap": 0.0})
    if not result.success or result.x is None:
        return None  # Caller keeps explicit shortfall; never relax constraints.
    selected = [pair for pair, value in zip(candidates, result.x) if value > .5]
    return selected
