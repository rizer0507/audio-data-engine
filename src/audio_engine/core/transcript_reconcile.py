from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

_CONTROL_TAG_RE = re.compile(r"<\|.*?\|>")
_ID_COLUMNS = ("id", "sample_id", "audio_id", "utt_id", "文件名", "音频", "音频文件")
_QWEN_COLUMNS = ("qwen_text", "qwen", "qwen_result", "qwen识别结果", "Qwen识别结果")
_SENSEVOICE_COLUMNS = (
    "sensevoice_text",
    "sensevoice",
    "text",
    "result",
    "sentence",
    "识别结果",
)


def clean_control_tags(value: Any) -> Any:
    """Remove every SenseVoice ``<|...|>`` control field from a cell."""
    if not isinstance(value, str):
        return value
    return _CONTROL_TAG_RE.sub("", value).strip()


def normalize_transcript(value: Any) -> str:
    """Normalize transcript text for character-level comparison."""
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(clean_control_tags(value))).casefold()
    return "".join(ch for ch in text if ch.isalnum())


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for index, char_a in enumerate(a, 1):
        current = [index]
        for offset, char_b in enumerate(b, 1):
            current.append(
                min(current[-1] + 1, previous[offset] + 1, previous[offset - 1] + (char_a != char_b))
            )
        previous = current
    return previous[-1]


def levenshtein_ops(reference: str, hypothesis: str) -> dict[str, int | float | None]:
    """Character-level edit ops with ``reference`` as baseline.

    Returns:
      total: overall edit distance
      sub / 错字: substitutions
      delete / 少字: deletions (in hypothesis vs reference)
      insert / 多字: insertions (extra in hypothesis)
      cer: total / max(len(reference), 1)
    """
    ref = normalize_transcript(reference)
    hyp = normalize_transcript(hypothesis)
    if not ref and not hyp:
        return {
            "total": 0,
            "sub": 0,
            "delete": 0,
            "insert": 0,
            "错字": 0,
            "少字": 0,
            "多字": 0,
            "cer": 0.0,
            "ref_len": 0,
            "hyp_len": 0,
        }

    # DP distance matrix for traceback of operation counts.
    rows, cols = len(ref), len(hyp)
    dist = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        dist[i][0] = i
    for j in range(1, cols + 1):
        dist[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dist[i][j] = min(
                dist[i - 1][j] + 1,  # deletion
                dist[i][j - 1] + 1,  # insertion
                dist[i - 1][j - 1] + cost,  # keep / substitute
            )

    i, j = rows, cols
    sub = delete = insert = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dist[i][j] == dist[i - 1][j - 1]:
            i -= 1
            j -= 1
            continue
        if i > 0 and j > 0 and dist[i][j] == dist[i - 1][j - 1] + 1:
            sub += 1
            i -= 1
            j -= 1
            continue
        if j > 0 and dist[i][j] == dist[i][j - 1] + 1:
            insert += 1
            j -= 1
            continue
        # deletion
        delete += 1
        i -= 1

    total = sub + delete + insert
    return {
        "total": total,
        "sub": sub,
        "delete": delete,
        "insert": insert,
        "错字": sub,
        "少字": delete,
        "多字": insert,
        "cer": round(total / max(len(ref), 1), 4),
        "ref_len": len(ref),
        "hyp_len": len(hyp),
    }


def character_similarity(left: Any, right: Any) -> float:
    """Return normalized Levenshtein similarity in the inclusive range 0..1."""
    a, b = normalize_transcript(left), normalize_transcript(right)
    if not a and not b:
        return 1.0
    return round(1 - _levenshtein(a, b) / max(len(a), len(b)), 4)


def _pick_column(frame: pd.DataFrame, requested: str | None, candidates: Iterable[str], label: str) -> str:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"{label}列不存在: {requested!r}; 可用列: {list(frame.columns)}")
        return requested
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"无法自动识别{label}列; 请显式指定。可用列: {list(frame.columns)}")


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    row = dict(record)
    transcripts = record.get("transcripts")
    if isinstance(transcripts, dict):
        sensevoice = transcripts.get("sensevoice")
        if isinstance(sensevoice, dict):
            row.setdefault("sensevoice_text", sensevoice.get("text", ""))
        elif sensevoice is not None:
            row.setdefault("sensevoice_text", sensevoice)
    if not row.get("sensevoice_text"):
        for column in _SENSEVOICE_COLUMNS:
            if row.get(column) is not None:
                row["sensevoice_text"] = row[column]
                break
    return row


def read_sensevoice_results(path: Path) -> pd.DataFrame:
    """Read SenseVoice output from parquet, Excel, JSON, JSONL, or ``.fenp`` JSONL."""
    if not path.is_file():
        raise FileNotFoundError(f"SenseVoice结果文件不存在: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("data", payload.get("results", [payload]))
        frame = pd.DataFrame(payload)
    else:
        records = []
        with path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number} 不是有效的JSON记录: {exc}") from exc
        frame = pd.DataFrame(records)
    return pd.DataFrame([_flatten_record(row) for row in frame.to_dict("records")])


def reconcile_transcripts(
    xlsx_path: Path,
    sensevoice_path: Path,
    output_path: Path,
    *,
    id_column: str | None = None,
    sensevoice_id_column: str | None = None,
    qwen_column: str | None = None,
    sensevoice_column: str | None = None,
    threshold: float = 0.9,
) -> dict[str, Any]:
    """Join Qwen Excel and SenseVoice output, clean tags, compare, and write Excel."""
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold必须在0到1之间，当前为: {threshold}")
    source = pd.read_excel(xlsx_path)
    sensevoice = read_sensevoice_results(sensevoice_path)
    source_id = _pick_column(source, id_column, _ID_COLUMNS, "Excel ID")
    result_id = _pick_column(
        sensevoice, sensevoice_id_column, (source_id, *_ID_COLUMNS), "SenseVoice ID"
    )
    qwen = _pick_column(source, qwen_column, _QWEN_COLUMNS, "Qwen文本")
    sense_text = _pick_column(sensevoice, sensevoice_column, _SENSEVOICE_COLUMNS, "SenseVoice文本")

    if sensevoice[result_id].duplicated().any():
        duplicates = sensevoice.loc[sensevoice[result_id].duplicated(), result_id].head(5).tolist()
        raise ValueError(f"SenseVoice ID不唯一，示例: {duplicates}")

    right = sensevoice[[result_id, sense_text]].rename(
        columns={result_id: "__match_id", sense_text: "sensevoice_raw_text"}
    )
    output = source.merge(right, how="left", left_on=source_id, right_on="__match_id").drop(
        columns="__match_id"
    )
    # Clean all string cells so the delivered workbook contains no SenseVoice control fields.
    output = output.apply(lambda column: column.map(clean_control_tags))
    output["sensevoice_clean_text"] = output["sensevoice_raw_text"].fillna("")
    output["qwen_normalized"] = output[qwen].map(normalize_transcript)
    output["sensevoice_normalized"] = output["sensevoice_clean_text"].map(normalize_transcript)
    output["character_similarity"] = [
        character_similarity(a, b)
        for a, b in zip(output[qwen], output["sensevoice_clean_text"], strict=False)
    ]
    output["asr_consistent"] = (
        output["sensevoice_raw_text"].notna() & (output["character_similarity"] >= threshold)
    )
    output["comparison_reason"] = output["asr_consistent"].map(
        {True: "字符相似度达到阈值", False: "字符相似度低于阈值"}
    )
    output.loc[output["sensevoice_raw_text"].isna(), "comparison_reason"] = "未找到SenseVoice结果"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_excel(output_path, index=False)
    matched = int(output["asr_consistent"].sum())
    return {
        "total": len(output),
        "consistent": matched,
        "inconsistent": len(output) - matched,
        "missing_sensevoice": int(output["sensevoice_raw_text"].isna().sum()),
        "consistent_rate": round(matched / len(output), 4) if len(output) else 0.0,
        "threshold": threshold,
        "output": str(output_path),
    }
