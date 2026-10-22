#!/usr/bin/env python3
# coding=utf-8
"""从 Qwen exploratory 训练集做 tempo 增强并写出新版本（默认 v003 → v004）。

复用 ``audio_engine.core.augmentation`` 的 FFmpeg ``atempo`` 保音高加速逻辑
（与 ``pipelines/augment_batch_speed.yaml`` / ``augmentation.tempo`` 同源）。

约定：
  - 仅增强 **train** 切分；**dev 原样复制**，避免同源增强泄漏到验证集
  - 默认只增强 ``include_kind=speech``；原 train 行全部保留，再追加增强行
  - 增强行继承源样本 ``text`` / ``group_id`` / 标注血缘；``sample_id`` 用稳定派生 id

示例（在能读到 wav 的机器上）::

  python scripts/build_qwen3_trainset_tempo_augment.py \\
    --src-dir ../audio-qwen3-asr-train/data/training_sets/exploratory/fenshen_1015/train_v003 \\
    --out-dir ../audio-qwen3-asr-train/data/training_sets/exploratory/fenshen_1015/train_v004 \\
    --speed-factor 1.2

  # JSONL 里是 /data2 路径、本机挂载不同前缀时：
  python scripts/build_qwen3_trainset_tempo_augment.py \\
    --src-dir .../train_v003 --out-dir .../train_v004 \\
    --path-map /data2/data-cp/lizi/audio-data-engine=/mnt/audio-data-engine \\
    --publish-path-map /mnt/audio-data-engine=/data2/data-cp/lizi/audio-data-engine
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.augmentation.audio_io import (  # noqa: E402
    TempoAudioError,
    duration_tolerance,
    read_audio_info,
    validate_output_wav,
)
from audio_engine.core.augmentation.ffmpeg_atempo import (  # noqa: E402
    atempo_filter_graph,
    probe_ffmpeg,
    run_atempo,
)
from audio_engine.core.augmentation.identity import (  # noqa: E402
    audio_output_path,
    content_sha256,
    derived_sample_id,
    processing_fingerprint,
)
from audio_engine.core.augmentation.params import (  # noqa: E402
    OPERATOR_VERSION,
    factor_tag,
    resolve_tempo_params,
)
from audio_engine.core.manifest import file_sha256  # noqa: E402

DEFAULT_SRC = (
    Path(__file__).resolve().parents[2]
    / "audio-qwen3-asr-train"
    / "data"
    / "training_sets"
    / "exploratory"
    / "fenshen_1015"
    / "train_v003"
)
DEFAULT_OUT = DEFAULT_SRC.parent / "train_v004"
SOURCE_BATCH_TAG = "fenshen_1015_train_v003"


def _now_iso_local() -> str:
    # 与既有 exploratory dataset.yaml 一致：本地 +08:00 墙钟
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def _info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)


def parse_path_maps(raw: list[str] | None) -> list[tuple[str, str]]:
    maps: list[tuple[str, str]] = []
    for item in raw or []:
        if "=" not in item:
            raise ValueError(f"--path-map 格式应为 OLD=NEW，收到: {item!r}")
        old, new = item.split("=", 1)
        if not old:
            raise ValueError(f"--path-map OLD 为空: {item!r}")
        maps.append((old, new))
    return maps


def rewrite_path(path: str, path_maps: list[tuple[str, str]]) -> str:
    text = str(path or "")
    for old, new in path_maps:
        if text.startswith(old):
            return new + text[len(old) :]
    return text


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256_file(path: Path) -> str:
    return file_sha256(path)


def _write_checksums(out_dir: Path, relative_names: list[str]) -> Path:
    lines: list[str] = []
    for name in relative_names:
        p = out_dir / name
        lines.append(f"{_sha256_file(p)}  {name}")
    path = out_dir / "checksums.sha256"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _load_src(src_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    train = _load_jsonl(src_dir / "train.jsonl")
    dev = _load_jsonl(src_dir / "dev.jsonl")
    smap = _load_jsonl(src_dir / "sample_map.jsonl")
    return train, dev, smap


def _index_train_map(smap: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """row_number (1-based, train) -> sample_map row."""
    out: dict[int, dict[str, Any]] = {}
    for row in smap:
        if str(row.get("split") or "") != "train":
            continue
        rn = int(row.get("row_number") or 0)
        if rn < 1:
            continue
        out[rn] = row
    return out


def _augment_one(
    *,
    src_row: dict[str, Any],
    map_row: dict[str, Any],
    local_input: Path,
    product_root: Path,
    params: Any,
    ffmpeg_info: Any,
    source_batch: str,
    publish_maps: list[tuple[str, str]],
) -> dict[str, Any]:
    info = read_audio_info(local_input)
    input_sha = str(info["sha256"])
    filter_graph = atempo_filter_graph(params.speed_factor)
    fp = processing_fingerprint(
        source_batch=source_batch,
        source_id=str(map_row.get("sample_id") or ""),
        input_audio_sha256=input_sha,
        params=params,
        ffmpeg_version=ffmpeg_info.version,
        filter_graph=filter_graph,
    )
    derived_id = derived_sample_id(fp)
    out_path = audio_output_path(product_root, derived_id)

    expected_dur = float(info["duration"]) / float(params.speed_factor)
    dur_tol = duration_tolerance(expected_dur)

    def _validate() -> None:
        validate_output_wav(
            out_path,
            expected_sr=int(info["sample_rate"]),
            expected_channels=int(info["channels"]),
            expected_duration=expected_dur,
            duration_tol=dur_tol,
        )

    if out_path.is_file() and out_path.stat().st_size > 0:
        # 幂等：已有产物直接复用（同 fingerprint → 同路径）
        try:
            _validate()
            meta = {
                "filter_graph": filter_graph,
                "ffmpeg_version": ffmpeg_info.version,
                "cached": True,
            }
        except TempoAudioError:
            meta = run_atempo(
                input_path=local_input,
                output_path=out_path,
                params=params,
                backend=ffmpeg_info,
            )
            _validate()
    else:
        meta = run_atempo(
            input_path=local_input,
            output_path=out_path,
            params=params,
            backend=ffmpeg_info,
        )
        _validate()

    out_sha = content_sha256(out_path)
    publish_audio = rewrite_path(str(out_path.resolve()), publish_maps)
    return {
        "derived_id": derived_id,
        "fingerprint": fp,
        "local_path": str(out_path.resolve()),
        "publish_path": publish_audio,
        "audio_sha256": out_sha,
        "input_sha256": input_sha,
        "filter_graph": meta.get("filter_graph") or filter_graph,
        "ffmpeg_version": meta.get("ffmpeg_version") or ffmpeg_info.version,
        "cached": bool(meta.get("cached")),
        "src_row": src_row,
        "map_row": map_row,
    }


def _write_prepare_py(out_dir: Path, *, command: str) -> None:
    text = f'''#!/usr/bin/env python3
# coding=utf-8
"""Rebuild helper for trial_fenshen_1015_v004 (tempo-augmented from train_v003).

本目录 JSONL/sample_map 由 audio-data-engine 脚本生成；重新生成请在引擎仓执行::

  {command}

dev 切分与 v003 完全一致；train = v003 train + tempo 增强副本（默认仅 speech）。
"""
from __future__ import annotations

raise SystemExit(
    "train_v004 请用 audio-data-engine/scripts/build_qwen3_trainset_tempo_augment.py 重建，"
    "不要从 batch raw 直接 prepare。"
)
'''
    (out_dir / "prepare.py").write_text(text, encoding="utf-8")


def _write_dataset_yaml(
    out_dir: Path,
    *,
    train_n: int,
    dev_n: int,
    speed_factor: float,
    include_kinds: list[str],
    audio_out_root: Path,
    aug_n: int,
    command: str,
    prepare_sha: str,
) -> None:
    kinds = ",".join(include_kinds)
    created = _now_iso_local()
    # 插入冒号时区：+0800 -> +08:00
    if len(created) >= 5 and created[-5] in "+-" and created[-3] != ":":
        created = created[:-2] + ":" + created[-2:]
    body = f"""# exploratory training set for labeling export 2026-10-15 fenshen
schema_version: local_training_dataset_v1
dataset_id: trial_fenshen_1015_v004
version: train_v004
tier: exploratory
status: frozen
created_at: '{created}'
owner: rizer
purpose: 在 v003 文本/切分基础上，对 train 切分做保音高加速（tempo）增强后追加增强样本；dev 不变。
parent_dataset_ref: exploratory/fenshen_1015/train_v003
change_note: >-
  相对 v003：保留全部原 train/dev 行；对 train 中 include_kind∈[{kinds}]
  做 FFmpeg atempo speed_factor={speed_factor} 增强，追加 {aug_n} 条增强行。
  增强音频目录：{audio_out_root.as_posix()}
source_batches:
  - batch_ref: label_20261015_fenshen/v001
    annotation_file: raw/annotations_ASR-标注任务-数字分身-10月15日.parquet
    training_file: raw/training_ASR-标注任务-数字分身-10月15日.parquet
preparation:
  script_ref: prepare.py
  script_sha256: {prepare_sha}
  command: {command}
  annotation_guideline_version: null
  normalization_version: null
  target_encoding_version: qwen3_asr_language_asr_text_v1
  language_policy: speech_fixed_Chinese_empty_None
  audio_conversion: tempo_ffmpeg_atempo_{factor_tag(speed_factor)}_op{OPERATOR_VERSION}
  selection_and_mixing: >-
    继承 v003 样本与切分；train 追加 tempo 增强副本；dev 不增强。
split:
  group_by: parent_pcm_sha256
  seed: 42
  dev_ratio: 0.2
  independence_checked: true
  fixed_eval_overlap_checked: false
files:
  train: train.jsonl
  dev: dev.jsonl
  sample_map: sample_map.jsonl
  checksums: checksums.sha256
statistics:
  train_samples: {train_n}
  dev_samples: {dev_n}
  train_duration_seconds: null
  dev_duration_seconds: null
  augmented_train_samples: {aug_n}
known_limitations:
  - 音频路径为运行机器绝对路径；跨机需 path-map / 新版本重映射。
  - 增强仅覆盖 train；同 group_id 的增强样本不进 dev。
  - 非正式 Release 准入，不可作为晋级证据。
related_experiments: []
"""
    (out_dir / "dataset.yaml").write_text(body, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(
        description="从 exploratory train_v003 做 tempo 增强并生成 train_v004"
    )
    p.add_argument("--src-dir", type=Path, default=DEFAULT_SRC, help="源训练集目录（train_v003）")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="输出训练集目录（train_v004）")
    p.add_argument(
        "--audio-out-root",
        type=Path,
        default=None,
        help="增强 wav 产物根目录（默认：引擎 datasets/stage1/augmented/<tag>/tempo_<factor>/）",
    )
    p.add_argument("--speed-factor", type=float, default=1.2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-in-flight", type=int, default=16)
    p.add_argument("--ffmpeg-bin", type=str, default="ffmpeg")
    p.add_argument(
        "--include-kind",
        action="append",
        dest="include_kinds",
        default=None,
        help="要增强的 include_kind，可重复；默认仅 speech",
    )
    p.add_argument(
        "--path-map",
        action="append",
        default=None,
        help="读入音频路径前缀替换 OLD=NEW（可重复）",
    )
    p.add_argument(
        "--publish-path-map",
        action="append",
        default=None,
        help="写入 JSONL 前对增强路径做 OLD=NEW（可重复）；便于发回 /data2 路径",
    )
    p.add_argument("--limit", type=int, default=0, help="仅增强前 N 条（调试用；0=全部）")
    p.add_argument("--dry-run", action="store_true", help="只统计将增强的条数，不写文件")
    p.add_argument("--force", action="store_true", help="允许覆盖已存在的 out-dir JSONL")
    args = p.parse_args()

    src_dir = args.src_dir.resolve()
    out_dir = args.out_dir.resolve()
    include_kinds = args.include_kinds or ["speech"]
    include_set = {k.strip() for k in include_kinds if k and k.strip()}
    read_maps = parse_path_maps(args.path_map)
    publish_maps = parse_path_maps(args.publish_path_map)

    if not (src_dir / "train.jsonl").is_file():
        raise SystemExit(f"[ERROR] 缺少 train.jsonl: {src_dir}")
    if not (src_dir / "sample_map.jsonl").is_file():
        raise SystemExit(f"[ERROR] 缺少 sample_map.jsonl: {src_dir}")

    params = resolve_tempo_params(
        {
            "speed_factor": args.speed_factor,
            "workers": args.workers,
            "max_in_flight": args.max_in_flight,
            "ffmpeg_bin": args.ffmpeg_bin,
        }
    )
    ffmpeg_info = probe_ffmpeg(params.ffmpeg_bin)
    tag = factor_tag(params.speed_factor)
    product_root = (
        args.audio_out_root.resolve()
        if args.audio_out_root
        else (ROOT / "datasets" / "stage1" / "augmented" / SOURCE_BATCH_TAG / f"tempo_{tag}")
    )

    train_rows, dev_rows, smap_rows = _load_src(src_dir)
    train_map = _index_train_map(smap_rows)
    if len(train_map) != len(train_rows):
        _warn(
            f"train.jsonl 行数={len(train_rows)} 与 sample_map train 行数={len(train_map)} 不一致；"
            "按 min 对齐，缺 map 的行跳过增强"
        )

    jobs: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    missing_audio = 0
    skipped_kind = 0
    for i, row in enumerate(train_rows, start=1):
        m = train_map.get(i)
        if m is None:
            continue
        kind = str(m.get("include_kind") or "").strip()
        if kind not in include_set:
            skipped_kind += 1
            continue
        audio = str(m.get("audio_path") or row.get("audio") or "").strip()
        local = Path(rewrite_path(audio, read_maps))
        if not local.is_file():
            missing_audio += 1
            continue
        jobs.append((i, row, m))

    if args.limit and args.limit > 0:
        jobs = jobs[: args.limit]

    _info(
        f"src={src_dir} train={len(train_rows)} dev={len(dev_rows)} "
        f"augment_candidates={len(jobs)} skipped_kind={skipped_kind} missing_audio={missing_audio}"
    )
    _info(
        f"speed_factor={params.speed_factor} ffmpeg={ffmpeg_info.version} "
        f"product_root={product_root} include_kinds={sorted(include_set)}"
    )

    if args.dry_run:
        _info("dry-run：不写产物")
        return 0

    if out_dir.exists() and any(out_dir.glob("*.jsonl")) and not args.force:
        raise SystemExit(f"[ERROR] out-dir 已有 JSONL，加 --force 覆盖: {out_dir}")

    if missing_audio and not jobs:
        raise SystemExit(
            "[ERROR] 没有任何可读 wav。请确认在服务器上运行，或用 --path-map 映射 JSONL 路径前缀。"
        )

    product_root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []

    def _task(item: tuple[int, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
        _rn, src_row, map_row = item
        audio = str(map_row.get("audio_path") or src_row.get("audio") or "").strip()
        local = Path(rewrite_path(audio, read_maps))
        return _augment_one(
            src_row=src_row,
            map_row=map_row,
            local_input=local,
            product_root=product_root,
            params=params,
            ffmpeg_info=ffmpeg_info,
            source_batch=SOURCE_BATCH_TAG,
            publish_maps=publish_maps,
        )

    workers = max(1, int(params.workers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_task, job): job[2].get("sample_id") for job in jobs}
        done = 0
        for fut in as_completed(futs):
            sid = futs[fut]
            done += 1
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                failures.append((str(sid), f"{exc}"))
                _warn(f"augment failed sample_id={sid}: {exc}")
            if done % 200 == 0 or done == len(futs):
                _info(f"progress {done}/{len(futs)} ok={len(results)} fail={len(failures)}")

    if failures and not results:
        raise SystemExit(f"[ERROR] 全部增强失败，示例: {failures[0]}")
    if failures:
        fail_log = out_dir / "augment_failures.jsonl"
        with fail_log.open("w", encoding="utf-8") as f:
            for sid, err in failures:
                f.write(json.dumps({"sample_id": sid, "error": err}, ensure_ascii=False) + "\n")
        _warn(f"有 {len(failures)} 条失败，已写 {fail_log}")

    # 稳定顺序：按源 train row_number
    results.sort(key=lambda r: int(r["map_row"].get("row_number") or 0))

    new_train: list[dict[str, Any]] = list(train_rows)
    # 先写原 train 的 sample_map（保持原 row_number / sample_id）
    orig_train_maps = [train_map[i] for i in range(1, len(train_rows) + 1) if i in train_map]
    # 若部分缺 map，按 train 行补最小字段
    if len(orig_train_maps) != len(train_rows):
        orig_train_maps = []
        for i, row in enumerate(train_rows, start=1):
            m = train_map.get(i)
            if m is not None:
                orig_train_maps.append(dict(m))
            else:
                orig_train_maps.append(
                    {
                        "split": "train",
                        "row_number": i,
                        "sample_id": f"missing_map_{i}",
                        "audio_path": row.get("audio"),
                        "include_kind": "unknown",
                    }
                )

    rebuilt_train_maps: list[dict[str, Any]] = []
    for i, m in enumerate(orig_train_maps, start=1):
        mm = dict(m)
        mm["split"] = "train"
        mm["row_number"] = i
        rebuilt_train_maps.append(mm)

    for r in results:
        src_row = r["src_row"]
        map_row = r["map_row"]
        new_train.append(
            {
                "audio": r["publish_path"],
                "text": src_row.get("text"),
                "prompt": src_row.get("prompt", ""),
            }
        )
        rn = len(new_train)
        aug_map = {
            "split": "train",
            "row_number": rn,
            "sample_id": r["derived_id"],
            "batch_ref": map_row.get("batch_ref"),
            "source_annotation": map_row.get("source_annotation"),
            "audio_path": r["publish_path"],
            "audio_sha256": r["audio_sha256"],
            "group_id": map_row.get("group_id"),
            "gold_kind": map_row.get("gold_kind"),
            "status": map_row.get("status"),
            "decision": map_row.get("decision"),
            "annotation_type": map_row.get("annotation_type"),
            "noise_kind": map_row.get("noise_kind"),
            "include_kind": map_row.get("include_kind"),
            "augmentation": {
                "type": "tempo",
                "backend": "ffmpeg_atempo",
                "speed_factor": params.speed_factor,
                "preserve_pitch": True,
                "operator_version": OPERATOR_VERSION,
                "filter_graph": r["filter_graph"],
                "ffmpeg_version": r["ffmpeg_version"],
                "source_sample_id": map_row.get("sample_id"),
                "source_audio_sha256": r["input_sha256"],
                "source_row_number": map_row.get("row_number"),
                "fingerprint": r["fingerprint"],
            },
        }
        rebuilt_train_maps.append(aug_map)

    # sample_map：train 在前，dev 在后（与 v003 一致）
    final_smap = rebuilt_train_maps + [
        r for r in smap_rows if str(r.get("split") or "") == "dev"
    ]

    cmd = (
        "python scripts/build_qwen3_trainset_tempo_augment.py "
        f"--src-dir {src_dir.as_posix()} --out-dir {out_dir.as_posix()} "
        f"--speed-factor {params.speed_factor} --audio-out-root {product_root.as_posix()}"
    )
    _write_prepare_py(out_dir, command=cmd)
    prepare_sha = _sha256_file(out_dir / "prepare.py")

    _write_jsonl(out_dir / "train.jsonl", new_train)
    # dev 原样复制字节，避免无谓重写
    shutil.copy2(src_dir / "dev.jsonl", out_dir / "dev.jsonl")
    _write_jsonl(out_dir / "sample_map.jsonl", final_smap)
    _write_dataset_yaml(
        out_dir,
        train_n=len(new_train),
        dev_n=len(dev_rows),
        speed_factor=params.speed_factor,
        include_kinds=sorted(include_set),
        audio_out_root=product_root,
        aug_n=len(results),
        command=cmd,
        prepare_sha=prepare_sha,
    )
    # dataset.yaml 写完后 prepare_sha 已固定；再算 checksums（不含自身）
    _write_checksums(
        out_dir,
        ["dataset.yaml", "train.jsonl", "dev.jsonl", "sample_map.jsonl", "prepare.py"],
    )

    cached_n = sum(1 for r in results if r.get("cached"))
    _info(
        f"done -> {out_dir} train={len(new_train)} "
        f"(orig={len(train_rows)} + aug={len(results)}, cached={cached_n}) "
        f"dev={len(dev_rows)} fail={len(failures)}"
    )
    if failures:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(0)
