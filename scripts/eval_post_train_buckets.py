#!/usr/bin/env python3
"""训练完成后，按三路转写和人工金标分类，再计算训练集、评测集上的字准率和准确率。

分类规则（non_speech 优先）：
  - non_speech：人工 gold_kind 为 non_speech。计分金标按空串，编辑距离取多字绝对值。
  - auto_gold：Qwen、GLM、SenseVoice 的 plain 文本一致，并且等于人工金标。
  - hardcase：plain(Qwen) 与 plain(人工金标) 不一致。
  - other：Qwen 与金标一致，但三路并不一致。保留，不丢行。

准确率 = 字准率等于 1 的条数 / 该类别条数。
字准率 = 1 - 总编辑距离 / 总基准字数，只对有基准字的行计算。
non_speech 不写字准率，汇总里的编辑距离和「多字」都是插入字数。

classified.parquet / scored.parquet 保留三路 parquet 的全部原列，并追加
eval_bucket、金标、VAD 切段路径、父音频路径。

示例::

  python scripts/eval_post_train_buckets.py \\
    --transcripts datasets/stage1/asr/merged_qwen_sensevoice_glm_20261022-label.parquet \\
    --gold tmp/1022/agg_fenshen_accept_empty_no_vad.parquet \\
    --pred train=tmp/1023/qwen-full-train-best_asr_eval_fenshen_1015_v012_train.parquet \\
    --pred dev=tmp/1023/qwen-full-dev-best_asr_eval_fenshen_1015_v012_dev.parquet \\
    --output-dir tmp/1023/post_train_eval

三路 parquet 自己带有金标文本和 gold_kind 时可以不传 --gold。
同一模型在 train / dev 上的转写列名不同时，用 --model-name 合成一个名字。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.post_train_eval import (
    evaluate,
    load_prediction,
    parse_pred_argument,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按 auto_gold / hardcase / non_speech 评测训练后模型")
    parser.add_argument("--transcripts", type=Path, required=True, help="含 qwen、glm、sensevoice 转写的 parquet")
    parser.add_argument("--gold", type=Path, default=None, help="人工金标 parquet；三路文件里没有金标文本时必填")
    parser.add_argument(
        "--pred",
        action="append",
        required=True,
        help="切分=转写parquet，可重复。例如 train=model_train.parquet dev=model_dev.parquet",
    )
    parser.add_argument("--model-name", default=None, help="汇总里的模型名。默认取转写列名去掉 _text")
    parser.add_argument("--id-column", default=None, help="对齐列，默认 id / sample_id")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    predictions = []
    for raw in args.pred:
        split, path = parse_pred_argument(raw)
        if not path.is_file():
            raise SystemExit(f"找不到模型转写: {path}")
        predictions.append(load_prediction(split, path, args.model_name))
    if args.gold is not None and not args.gold.is_file():
        raise SystemExit(f"找不到人工金标: {args.gold}")
    if not args.transcripts.is_file():
        raise SystemExit(f"找不到三路转写: {args.transcripts}")

    result = evaluate(
        args.transcripts,
        predictions,
        args.output_dir,
        args.gold,
        id_column=args.id_column,
    )
    print(f"classified {result['classified']}")
    print(f"scored {result['scored']}")
    print(f"workbook {result['workbook']}")
    print(f"rows {result['rows']} scored {result['scored_rows']} buckets {result['buckets']}")
    for row in result["summary"]:
        char_acc = "" if row["char_acc"] is None else f"{row['char_acc']:.2f}%"
        distance = row["insertions"] if row["bucket"] == "non_speech" else row["edits"]
        print(
            row["split"],
            row["bucket"],
            row["model"],
            f"n={row['n']}",
            f"正确率={row['accuracy']:.2f}%",
            f"字准率={char_acc or '-'}",
            f"编辑距离={distance}",
            f"多字={row['insertions']}",
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
