"""Classify three-way ASR rows, then score a trained model on each split.

Buckets, in order:

- ``non_speech``: human ``gold_kind`` is ``non_speech``. Scored against an empty
  reference. The reported edit distance is the insertion count (多字).
- ``auto_gold``: plain Qwen, GLM, and SenseVoice are identical and equal the
  plain human text.
- ``hardcase``: plain Qwen differs from the plain human text.
- ``other``: speech that matches neither rule. Kept so source rows are not dropped.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from openpyxl import Workbook

from audio_engine.core.transcript_reconcile import levenshtein_ops, plain_transcript_text

BUCKET_NON_SPEECH = "non_speech"
BUCKET_AUTO_GOLD = "auto_gold"
BUCKET_HARDCASE = "hardcase"
BUCKET_OTHER = "other"
BUCKET_ORDER = (
    "overall",
    BUCKET_AUTO_GOLD,
    BUCKET_HARDCASE,
    BUCKET_NON_SPEECH,
    BUCKET_OTHER,
)

_ID_CANDIDATES = ("id", "sample_id")
_QWEN_CANDIDATES = ("qwen_1_text", "qwen_text", "qwen-base_text")
_GLM_CANDIDATES = ("glm_text", "glm_1_text")
_SENSE_CANDIDATES = ("sensevoice_text", "sensevoice_1_text")
_GOLD_TEXT_CANDIDATES = ("label_gold_text", "gold_text", "label_human_gold_text", "human_gold_text")
_GOLD_KIND_CANDIDATES = (
    "label_agg_gold_kind",
    "label_source_gold_kind",
    "label_gold_kind",
    "gold_kind",
)
_SKIP_MODEL_TEXT = {
    "qwen_1_text",
    "qwen_text",
    "qwen-base_text",
    "glm_text",
    "glm_1_text",
    "sensevoice_text",
    "sensevoice_1_text",
    "candidate_text",
    "label_gold_text",
    "gold_text",
    "label_human_gold_text",
    "human_gold_text",
}


@dataclass(frozen=True)
class Prediction:
    """One trained-model transcription file for a named split."""

    split: str
    path: Path
    model_name: str
    text_by_id: dict[str, str]


@dataclass(frozen=True)
class UtteranceScore:
    accuracy: float
    exact: int
    edits: int
    ref_len: int
    insertions: int


def assign_bucket(*, gold_kind: str, qwen: Any, glm: Any, sensevoice: Any, human: Any) -> str:
    """Return the bucket for one row. ``non_speech`` takes priority."""
    if str(gold_kind or "").strip() == BUCKET_NON_SPEECH:
        return BUCKET_NON_SPEECH
    qwen_plain = plain_transcript_text(qwen)
    glm_plain = plain_transcript_text(glm)
    sense_plain = plain_transcript_text(sensevoice)
    human_plain = plain_transcript_text(human)
    if qwen_plain == glm_plain == sense_plain == human_plain:
        return BUCKET_AUTO_GOLD
    if qwen_plain != human_plain:
        return BUCKET_HARDCASE
    return BUCKET_OTHER


def score_hypothesis(bucket: str, human: Any, hypothesis: Any) -> UtteranceScore:
    """Score one hypothesis. ``non_speech`` uses an empty reference."""
    reference = "" if bucket == BUCKET_NON_SPEECH else _text(human)
    ops = levenshtein_ops(reference, _text(hypothesis))
    accuracy = round(max(0.0, 1.0 - float(ops["cer"])), 6)
    return UtteranceScore(
        accuracy=accuracy,
        exact=int(accuracy == 1.0),
        edits=int(ops["total"]),
        ref_len=int(ops["ref_len"]),
        insertions=int(ops["insert"]),
    )


def evaluate(
    transcripts: Path,
    predictions: Iterable[Prediction],
    output_dir: Path,
    gold: Path | None = None,
    *,
    id_column: str | None = None,
) -> dict[str, Any]:
    """Classify ``transcripts``, score ``predictions``, and write parquet plus xlsx."""
    preds = list(predictions)
    if not preds:
        raise ValueError("至少要一份训练后模型的转写 parquet")
    source = pq.read_table(transcripts)
    names = source.schema.names
    resolved_id = id_column or _require_column(names, _ID_CANDIDATES, "id")
    qwen_col = _require_column(names, _QWEN_CANDIDATES, "qwen 转写")
    glm_col = _require_column(names, _GLM_CANDIDATES, "glm 转写")
    sense_col = _require_column(names, _SENSE_CANDIDATES, "sensevoice 转写")
    rows = source.to_pylist()
    _reject_duplicate_ids(rows, resolved_id, transcripts.name)

    gold_text, gold_kind = _resolve_gold(names, rows, resolved_id, gold)
    buckets = [
        assign_bucket(
            gold_kind=gold_kind[row[resolved_id]],
            qwen=row.get(qwen_col),
            glm=row.get(glm_col),
            sensevoice=row.get(sense_col),
            human=gold_text[row[resolved_id]],
        )
        for row in rows
    ]
    paths = [_audio_paths(row.get("audio"), row.get("source_path")) for row in rows]
    classified = _append_columns(
        source,
        {
            "eval_bucket": buckets,
            "eval_gold_text": [gold_text[row[resolved_id]] for row in rows],
            "eval_gold_kind": [gold_kind[row[resolved_id]] for row in rows],
            "vad_segment_path": [item["vad_segment_path"] for item in paths],
            "parent_audio_path": [item["parent_audio_path"] for item in paths],
            "parent_wav_path": [item["parent_wav_path"] for item in paths],
        },
    )

    by_id = {row[resolved_id]: index for index, row in enumerate(rows)}
    split_of: dict[str, str] = {}
    for pred in preds:
        unknown = [sample_id for sample_id in pred.text_by_id if sample_id not in by_id]
        if unknown:
            raise ValueError(f"{pred.path.name} 有 {len(unknown)} 条 id 不在三路转写里")
        for sample_id in pred.text_by_id:
            previous = split_of.get(sample_id)
            if previous is not None and previous != pred.split:
                raise ValueError(f"id {sample_id} 同时出现在切分 {previous} 和 {pred.split}")
            split_of[sample_id] = pred.split

    scored_positions = [index for index, row in enumerate(rows) if row[resolved_id] in split_of]
    scored = classified.take(pa.array(scored_positions, type=pa.int64()))
    model_columns: dict[str, list[Any]] = {
        "eval_split": [split_of[rows[index][resolved_id]] for index in scored_positions]
    }
    models = _models_in_order(preds)
    per_model_scores: dict[str, list[UtteranceScore | None]] = {name: [] for name in models}
    for index in scored_positions:
        sample_id = rows[index][resolved_id]
        bucket = buckets[index]
        human = gold_text[sample_id]
        for name in models:
            hypothesis = _hypothesis(preds, name, split_of[sample_id], sample_id)
            per_model_scores[name].append(
                None if hypothesis is None else score_hypothesis(bucket, human, hypothesis)
            )
    for name in models:
        scores = per_model_scores[name]
        model_columns[f"{name}_text"] = [
            _hypothesis(preds, name, split_of[rows[index][resolved_id]], rows[index][resolved_id])
            for index in scored_positions
        ]
        model_columns[f"{name}_字准率"] = [None if item is None else item.accuracy for item in scores]
        model_columns[f"{name}_exact_correct"] = [None if item is None else item.exact for item in scores]
        model_columns[f"{name}_编辑距离"] = [None if item is None else item.edits for item in scores]
        model_columns[f"{name}_多字"] = [None if item is None else item.insertions for item in scores]
    scored = _append_columns(scored, model_columns)

    summary = _summarize(buckets, gold_text, preds, by_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    classified_path = output_dir / "classified.parquet"
    scored_path = output_dir / "scored.parquet"
    workbook_path = output_dir / "evaluation.xlsx"
    pq.write_table(classified, classified_path)
    pq.write_table(scored, scored_path)
    _write_workbook(
        workbook_path,
        summary=summary,
        transcripts=transcripts,
        gold=gold,
        preds=preds,
        rows=rows,
        buckets=buckets,
        gold_text=gold_text,
        gold_kind=gold_kind,
        paths=paths,
        id_column=resolved_id,
        qwen_col=qwen_col,
        glm_col=glm_col,
        sense_col=sense_col,
        scored_positions=scored_positions,
        split_of=split_of,
        per_model_scores=per_model_scores,
        models=models,
    )
    return {
        "classified": classified_path,
        "scored": scored_path,
        "workbook": workbook_path,
        "rows": len(rows),
        "scored_rows": len(scored_positions),
        "buckets": dict(_count(buckets)),
        "summary": summary,
    }


def load_prediction(split: str, path: Path, model_name: str | None = None) -> Prediction:
    """Read one model parquet. The transcription column is the non-gold ``*_text`` column."""
    schema_names = pq.read_schema(path).names
    sample_id_column = _require_column(schema_names, _ID_CANDIDATES, f"{path.name} id")
    text_column = model_name_from_columns(schema_names, path)[0]
    resolved_name = model_name or text_column[: -len("_text")]
    found: dict[str, str] = {}
    for row in pq.read_table(path, columns=[sample_id_column, text_column]).to_pylist():
        sample_id = row[sample_id_column]
        if sample_id in found:
            raise ValueError(f"{path.name} id 重复: {sample_id}")
        found[sample_id] = _text(row[text_column])
    return Prediction(split=split, path=path, model_name=resolved_name, text_by_id=found)


def model_name_from_columns(names: list[str], path: Path) -> tuple[str, str]:
    """Return ``(text_column, model_name)`` for a trained-model parquet."""
    candidates = [
        column
        for column in names
        if column.endswith("_text") and column not in _SKIP_MODEL_TEXT and not column.startswith("label_")
    ]
    if not candidates:
        raise ValueError(f"{path.name} 没有模型转写列")
    if len(candidates) == 1:
        column = candidates[0]
    else:
        matched = [column for column in candidates if path.name.startswith(column[: -len("_text")])]
        if len(matched) != 1:
            raise ValueError(f"{path.name} 有多个转写列，无法确定: {candidates}")
        column = matched[0]
    return column, column[: -len("_text")]


def parse_pred_argument(value: str) -> tuple[str, Path]:
    """Parse ``split=parquet``."""
    split, separator, raw_path = value.partition("=")
    if not separator or not split.strip() or not raw_path.strip():
        raise ValueError(f"--pred 需要 切分=parquet路径，收到: {value}")
    return split.strip(), Path(raw_path.strip())


def _summarize(
    buckets: list[str],
    gold_text: Mapping[str, str],
    preds: list[Prediction],
    by_id: Mapping[str, int],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[tuple[str, UtteranceScore]]] = defaultdict(list)
    for pred in preds:
        for sample_id, hypothesis in pred.text_by_id.items():
            index = by_id[sample_id]
            bucket = buckets[index]
            scored = score_hypothesis(bucket, gold_text[sample_id], hypothesis)
            pair = (bucket, scored)
            grouped[(pred.split, pred.model_name, bucket)].append(pair)
            grouped[(pred.split, pred.model_name, "overall")].append(pair)
    summary: list[dict[str, Any]] = []
    for split in _unique(pred.split for pred in preds):
        for model_name in _models_in_order(preds):
            for bucket in BUCKET_ORDER:
                pairs = grouped.get((split, model_name, bucket))
                if not pairs:
                    continue
                summary.append(_aggregate(split, model_name, bucket, pairs))
    return summary


def _aggregate(
    split: str,
    model_name: str,
    bucket: str,
    pairs: list[tuple[str, UtteranceScore]],
) -> dict[str, Any]:
    scores = [score for _, score in pairs]
    if bucket == BUCKET_NON_SPEECH:
        char_scores: list[UtteranceScore] = []
        insertion_scores = scores
    elif bucket == "overall":
        char_scores = [score for origin, score in pairs if origin != BUCKET_NON_SPEECH and score.ref_len > 0]
        insertion_scores = [score for origin, score in pairs if origin == BUCKET_NON_SPEECH]
    else:
        char_scores = [score for score in scores if score.ref_len > 0]
        insertion_scores = scores
    ref_chars = sum(score.ref_len for score in char_scores)
    char_edits = sum(score.edits for score in char_scores)
    total = len(scores)
    correct = sum(score.exact for score in scores)
    return {
        "split": split,
        "model": model_name,
        "bucket": bucket,
        "n": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 2) if total else 0.0,
        "edits": sum(score.edits for score in scores),
        "ref_chars": 0 if bucket == BUCKET_NON_SPEECH else ref_chars,
        "char_acc": None if ref_chars == 0 else round((1 - char_edits / ref_chars) * 100, 2),
        "insertions": sum(score.insertions for score in insertion_scores),
    }


def _write_workbook(path: Path, **kwargs: Any) -> None:
    summary: list[dict[str, Any]] = kwargs["summary"]
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "汇总"
    sheet.append(
        ["切分", "模型", "类别", "样本数", "正确数(字准=1)", "准确率", "总编辑距离", "总基准字数", "字准率", "多字"]
    )
    for row in summary:
        char_cell: Any = "" if row["char_acc"] is None else f"{row['char_acc']:.2f}%"
        sheet.append(
            [
                row["split"],
                row["model"],
                row["bucket"],
                row["n"],
                row["correct"],
                f"{row['accuracy']:.2f}%",
                row["edits"] if row["bucket"] != BUCKET_NON_SPEECH else row["insertions"],
                row["ref_chars"],
                char_cell,
                row["insertions"],
            ]
        )

    detail = workbook.create_sheet("明细")
    models: list[str] = kwargs["models"]
    header = [
        "id",
        "eval_split",
        "eval_bucket",
        "eval_gold_text",
        "eval_gold_kind",
        "qwen_text",
        "glm_text",
        "sensevoice_text",
        "vad_segment_path",
        "parent_audio_path",
        "parent_wav_path",
    ]
    for name in models:
        header.extend([f"{name}_text", f"{name}_字准率", f"{name}_exact_correct", f"{name}_编辑距离", f"{name}_多字"])
    detail.append(header)
    rows: list[dict[str, Any]] = kwargs["rows"]
    buckets: list[str] = kwargs["buckets"]
    gold_text: Mapping[str, str] = kwargs["gold_text"]
    gold_kind: Mapping[str, str] = kwargs["gold_kind"]
    paths: list[dict[str, str]] = kwargs["paths"]
    id_column: str = kwargs["id_column"]
    qwen_col: str = kwargs["qwen_col"]
    glm_col: str = kwargs["glm_col"]
    sense_col: str = kwargs["sense_col"]
    split_of: Mapping[str, str] = kwargs["split_of"]
    per_model_scores: Mapping[str, list[UtteranceScore | None]] = kwargs["per_model_scores"]
    preds: list[Prediction] = kwargs["preds"]
    for offset, index in enumerate(kwargs["scored_positions"]):
        row = rows[index]
        sample_id = row[id_column]
        values: list[Any] = [
            sample_id,
            split_of[sample_id],
            buckets[index],
            gold_text[sample_id],
            gold_kind[sample_id],
            _text(row.get(qwen_col)),
            _text(row.get(glm_col)),
            _text(row.get(sense_col)),
            paths[index]["vad_segment_path"],
            paths[index]["parent_audio_path"],
            paths[index]["parent_wav_path"],
        ]
        for name in models:
            scored = per_model_scores[name][offset]
            hypothesis = _hypothesis(preds, name, split_of[sample_id], sample_id)
            if scored is None:
                values.extend([None, None, None, None, None])
            else:
                values.extend([hypothesis, scored.accuracy, scored.exact, scored.edits, scored.insertions])
        detail.append(values)

    note = workbook.create_sheet("说明")
    note.append(["字段/口径", "说明"])
    notes = [
        ("三路转写", str(kwargs["transcripts"])),
        ("人工金标", str(kwargs["gold"]) if kwargs["gold"] else "三路 parquet 自带金标列"),
        ("模型", "；".join(f"{pred.split}:{pred.model_name}={pred.path}" for pred in preds)),
        ("non_speech", "eval_gold_kind=non_speech。计分金标按空串。编辑距离取多字（插入）的绝对值。"),
        ("auto_gold", "plain(qwen)==plain(glm)==plain(sensevoice)==plain(人工金标)。"),
        ("hardcase", "不是 non_speech，且 plain(qwen) 与 plain(人工金标) 不一致。"),
        ("other", "speech 中 Qwen 与金标一致，但三路转写并不一致。保留，不丢行。"),
        ("准确率", "字准率==1 的条数 / 该类别条数。标点差异在 plain 之后不算错。"),
        ("字准率", "speech 与 overall 的 speech 行：1 - 总编辑距离/总基准字数。non_speech 不写字准率，只写多字。"),
        ("overall 字准率", "只累计有基准字的行，不把 non_speech 的多字算进字准率。"),
        ("overall 多字", "non_speech 行的多字之和。"),
        ("源字段", "classified.parquet 与 scored.parquet 保留三路 parquet 的全部原列，并追加路径和分类列。"),
        ("字准实现", "audio_engine.core.transcript_reconcile.levenshtein_ops（zh_asr_v1）"),
    ]
    for key, value in notes:
        note.append([key, value])
    workbook.save(path)


def _resolve_gold(
    names: list[str],
    rows: list[dict[str, Any]],
    id_column: str,
    gold: Path | None,
) -> tuple[dict[str, str], dict[str, str]]:
    text_col = _find_column(names, _GOLD_TEXT_CANDIDATES)
    kind_col = _find_column(names, _GOLD_KIND_CANDIDATES)
    text = {row[id_column]: _text(row.get(text_col)) if text_col else "" for row in rows}
    kind = {row[id_column]: _text(row.get(kind_col)) if kind_col else "" for row in rows}
    if gold is None:
        if text_col is None or kind_col is None:
            missing = []
            if text_col is None:
                missing.append("人工金标文本")
            if kind_col is None:
                missing.append("gold_kind")
            raise ValueError(f"三路 parquet 缺少 {'、'.join(missing)}，请用 --gold 传入人工标注 parquet")
        return text, kind

    gold_names = pq.read_schema(gold).names
    gold_id = _require_column(gold_names, _ID_CANDIDATES, f"{gold.name} id")
    gold_text_col = _require_column(gold_names, _GOLD_TEXT_CANDIDATES, f"{gold.name} 金标文本")
    gold_kind_col = _find_column(gold_names, _GOLD_KIND_CANDIDATES)
    external_text: dict[str, str] = {}
    external_kind: dict[str, str] = {}
    for row in pq.read_table(
        gold,
        columns=[gold_id, gold_text_col] + ([gold_kind_col] if gold_kind_col else []),
    ).to_pylist():
        sample_id = row[gold_id]
        if sample_id in external_text:
            raise ValueError(f"{gold.name} id 重复: {sample_id}")
        external_text[sample_id] = _text(row.get(gold_text_col))
        if gold_kind_col:
            external_kind[sample_id] = _text(row.get(gold_kind_col))
    missing = [row[id_column] for row in rows if row[id_column] not in external_text]
    if missing:
        raise ValueError(f"人工金标缺少 {len(missing)} 条三路 id，例如 {missing[0]}")
    for row in rows:
        sample_id = row[id_column]
        text[sample_id] = external_text[sample_id]
        if external_kind.get(sample_id):
            kind[sample_id] = external_kind[sample_id]
        elif not kind[sample_id]:
            raise ValueError(f"id {sample_id} 没有 gold_kind")
    return text, kind


def _audio_paths(audio: Any, source_path: Any) -> dict[str, str]:
    payload = audio
    if isinstance(payload, str) and payload:
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        payload = {}
    vad = _text(payload.get("resampled_16k") or source_path)
    parent = _text(payload.get("parent_raw"))
    parent_wav = _text(payload.get("parent_resampled_16k") or payload.get("normalized_16k"))
    return {"vad_segment_path": vad, "parent_audio_path": parent, "parent_wav_path": parent_wav}


def _append_columns(table: pa.Table, columns: dict[str, list[Any]]) -> pa.Table:
    for name, values in columns.items():
        if name in table.schema.names:
            index = table.schema.get_field_index(name)
            table = table.remove_column(index)
        table = table.append_column(name, pa.array(values))
    return table


def _hypothesis(preds: list[Prediction], model_name: str, split: str, sample_id: str) -> str | None:
    for pred in preds:
        if pred.model_name == model_name and pred.split == split and sample_id in pred.text_by_id:
            return pred.text_by_id[sample_id]
    return None


def _models_in_order(preds: list[Prediction]) -> list[str]:
    return _unique(pred.model_name for pred in preds)


def _unique(values: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _count(buckets: list[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for bucket in buckets:
        counts[bucket] += 1
    return counts


def _reject_duplicate_ids(rows: list[dict[str, Any]], id_column: str, name: str) -> None:
    seen: set[Any] = set()
    for row in rows:
        sample_id = row[id_column]
        if sample_id in seen:
            raise ValueError(f"{name} id 重复: {sample_id}")
        seen.add(sample_id)


def _require_column(names: list[str], candidates: tuple[str, ...], label: str) -> str:
    found = _find_column(names, candidates)
    if found is None:
        raise ValueError(f"找不到{label}列，候选 {list(candidates)}，实际列 {names}")
    return found


def _find_column(names: list[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
