"""Replay evidence with production code without mutating source artifacts."""
import collections
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.classifier import classify_sample
from audio_engine.core.selection_v3.family_evidence import collect_route_views
from audio_engine.core.annotation_v3.config import AnnotationConfig
from audio_engine.core.annotation_v3.queue import select_review_batch
import pyarrow.parquet as pq

cfg = SelectionV3Config.from_yaml(ROOT / "configs/selection/zh_asr_v3_0908_30000.yaml")
counts = collections.defaultdict(collections.Counter)
samples = []
json_labels = {}
for line in (ROOT / "datasets/stage1/derived/classified_v3_0908-30000.jsonl").open(
    encoding="utf-8"
):
    s = Sample.model_validate(json.loads(line))
    samples.append(s)
    typ = s.labels["type"]
    json_labels[s.id] = (
        typ,
        s.labels.get("review_queue"),
        s.labels.get("review_priority"),
    )
    texts = [v.comparison_text for v in collect_route_views(s, cfg)]
    if len(set(texts)) == 1 and texts[0]:
        counts["production_six_equal"][typ] += 1
        live = classify_sample(s, cfg)
        counts["live_six_equal"][live.type] += 1
        if live.type in {"critical_content_risk", "semantic_risk"} and live.review_priority == "P0":
            counts["live_six_equal_still_p0_conflict"][live.type] += 1
    if typ == "audio_quality_risk":
        live_q = classify_sample(s, cfg)
        counts["baseline_replay"][live_q.type] += 1
        # Counterfactual ceiling only: NOT measured quality or permission to publish.
        s2 = s.model_copy(deep=True)
        s2.quality.update(noise_band="clean", noise_risk=False, dnsmos_status="success")
        counts["if_quality_all_clean"][classify_sample(s2, cfg).type] += 1

manual = [s for s in samples if s.labels.get("review_queue") == "manual_review"]
selected = select_review_batch(
    samples, AnnotationConfig(), priorities=["P0"], queues=["manual_review"]
)
counts["p0_queue_filter_replay"].update(s.labels.get("review_priority") for s in selected)
tab = pq.read_table(
    ROOT / "datasets/stage1/derived/classified_v3_0908-30000.parquet",
    columns=["id", "label_type", "label_review_queue", "label_review_priority"],
)
parquet = {
    r["id"]: (r["label_type"], r["label_review_queue"], r["label_review_priority"])
    for r in tab.to_pylist()
}
result = {
    "counts": {k: dict(v) for k, v in counts.items()},
    "parquet_json_labels_equal": parquet == json_labels,
    "manual_unique_audio_hashes": len({s.sha256 for s in manual}),
    "manual_unique_leakage_groups": len(
        {s.labels.get("leakage_group_id") for s in manual}
    ),
    "historical_audio_quality_risk": sum(
        1 for s in samples if s.labels.get("type") == "audio_quality_risk"
    ),
}
assert result["parquet_json_labels_equal"]
assert result["historical_audio_quality_risk"] == 4971
assert sum(counts["baseline_replay"].values()) == 4971
# Phase 1: priorities∩queues — P0 pack must not leak P1.
assert dict(counts["p0_queue_filter_replay"]) == {"P0": 9915}
# Phase 3: identical six-route texts must not stay P0 model-conflict under live rules.
assert dict(counts["live_six_equal_still_p0_conflict"]) == {}
print(json.dumps(result, ensure_ascii=False, indent=2))
