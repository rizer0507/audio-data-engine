"""ASR 字准 xlsx：type、non_speech 整数总字准、表结构、主题路径。"""

from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from audio_engine.core.manifest import Manifest
from audio_engine.core.sample import Sample


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_asr_xlsx.py"
    spec = importlib.util.spec_from_file_location("eval_asr_xlsx", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sophon_report_selects_tagged_result_and_rejects_missing_transcript(tmp_path):
    mod = _load_script()
    _, (model, alias) = mod.asr_jobs("sophon", "dev", "best", sophon_run_tag="v1-test")
    assert model == "qwen-sophon-best"
    assert alias == "qwen-sophon-dev-best-v1-test"
    output = tmp_path / "sophon.parquet"
    Manifest([Sample(id="empty", source_path="audio.wav", transcripts={alias: {"text": ""}})]).save(output)
    assert mod.load_asr_text(output, alias, require_alias=True) == {"empty": ""}
    with pytest.raises(SystemExit, match=alias):
        mod.load_asr_text(output, f"{alias}-other", require_alias=True)


def test_score_drops_listed_tokens_from_label_and_hypothesis():
    mod = _load_script()
    assert mod.score_pair("喂，哎。", "唉喂")[:2] == (1.0, 1)
    assert mod.score_pair("不需要了", "不需要啦")[:2] == (1.0, 1)
    assert mod.score_pair("什么事儿", "啥事")[:2] == (1.0, 1)
    assert mod.score_pair("你好啊", "您好")[:2] == (1.0, 1)
    assert mod.score_pair("泥马", "你妈")[:2] == (1.0, 1)
    assert mod.score_pair("我的卡", "我地卡")[:2] == (1.0, 1)
    assert mod.score_pair("好吧", "好")[:2] == (1.0, 1)
    assert mod.score_pair("嗯好", "好")[1] == 0
    assert mod.score_pair("好哦", "好")[1] == 0
    accuracy, exact, edits, _ref_len = mod.score_pair("不需要谢谢", "不需要了谢")
    assert exact == 0
    assert edits > 0
    assert accuracy < 1
    assert mod.score_pair("不需要", "我不要")[1] == 0
    assert mod.collapse_abnormal_loops("不用不用") == "不用不用"
    assert mod.collapse_abnormal_loops("不用不用不用") == "不用"
    assert mod.collapse_abnormal_loops("不用" * 10000) == "不用"
    assert mod.score_pair("不用", "不用不用不用不用")[:2] == (1.0, 1)
    assert mod.score_pair("不用不用", "不用不用")[1] == 1
    assert mod.score_pair("不用不用", "不用不用不用")[1] == 0


def test_empty_gold_ignores_filler_and_skips_char_acc():
    mod = _load_script()
    # 空金标：语气词不能把「什么」抹成正确；字准=0，编辑距离=plain 字数。
    assert mod.score_pair("", "什么?") == (0.0, 0, 2, 0)
    assert mod.score_pair("", "什么?哦。") == (0.0, 0, 3, 0)
    assert mod.score_pair("", "") == (1.0, 1, 0, 0)
    assert mod.score_pair("", "【听不清】") == (1.0, 1, 0, 0)
    assert mod.score_pair("", "噪声") == (0.0, 0, 2, 0)

    gold_rows = [
        {
            "id": "speech",
            "type": "auto_gold",
            "label": "不需要",
            "candidate_text": "不需要",
            "gold_text_ann": "不需要",
            "gold_vs_candidate_mismatch": 0,
            "category": "",
            "decision": "accept",
            "gold_kind": "speech",
            "noise_kind": "",
            "exclude_from_training": False,
            "uncertain": "",
            "label_source": "human",
            "wav_path": "a.wav",
            "reference": "不需要",
        },
        {
            "id": "noise",
            "type": "non_speech",
            "label": "",
            "candidate_text": "【听不清】",
            "gold_text_ann": "",
            "gold_vs_candidate_mismatch": 0,
            "category": "",
            "decision": "empty",
            "gold_kind": "non_speech",
            "noise_kind": "",
            "exclude_from_training": True,
            "uncertain": "",
            "label_source": "human",
            "wav_path": "b.wav",
            "reference": "",
        },
    ]
    frames = mod.build_frames(
        gold_rows,
        [
            ("qwen-base", {"speech": "不需要", "noise": "什么?"}),
            ("qwen-full-best", {"speech": "不需要", "noise": ""}),
        ],
        gold_path=Path("gold.parquet"),
        model_paths={
            "qwen-base": Path("base.parquet"),
            "qwen-full-best": Path("best.parquet"),
        },
    )
    accuracy, _pivot, detail, notes = frames
    overall = accuracy[accuracy["scope"] == "overall"].set_index("模型")
    # overall 总字准只含非空金标；空金标误转「什么」不抬高总编辑距离。
    assert overall.loc["qwen-base", "总基准字数"] == 3
    assert overall.loc["qwen-base", "总编辑距离"] == 0
    assert overall.loc["qwen-base", "总字准"] == "100.00%"
    assert overall.loc["qwen-base", "正确率"] == "50.00%"
    assert overall.loc["qwen-full-best", "正确率"] == "100.00%"
    non_speech = accuracy[accuracy["scope"] == "non_speech"].set_index("模型")
    assert non_speech.loc["qwen-base", "正确率"] == "0.00%"
    assert non_speech.loc["qwen-base", "总字准"] == 2
    assert non_speech.loc["qwen-full-best", "正确率"] == "100.00%"
    assert non_speech.loc["qwen-full-best", "总字准"] == 0
    assert detail.loc[detail["id"] == "noise", "qwen-base_字准率"].item() == 0.0
    assert detail.loc[detail["id"] == "noise", "qwen-base_exact_correct"].item() == 0
    assert "空金标不计入" in notes.loc[notes["字段/口径"] == "总字准", "说明"].item()


def test_omni_rejects_over_checkpoint():
    mod = _load_script()
    with pytest.raises(SystemExit):
        mod.main(["omni", "v030", "--split", "dev", "--ckpt", "over"])


def test_ckpt_keeps_base_as_comparison():
    mod = _load_script()
    assert mod.asr_jobs("full", "dev", "best") == [
        ("qwen-base", "qwen-base-dev"),
        ("qwen-full-best", "qwen-full-dev-best"),
    ]
    assert mod.asr_jobs("full", "train", "over") == [
        ("qwen-base", "qwen-base-full"),
        ("qwen-full-over", "qwen-full-train-over"),
    ]
    assert [name for name, _run in mod.asr_jobs("full", "dev", "best-over")] == [
        "qwen-base",
        "qwen-full-best",
        "qwen-full-over",
    ]
    assert mod.asr_jobs("omni", "dev", "best") == [
        ("qwen-base", "qwen-base-dev"),
        ("qwen-omni-best", "qwen-omni-dev-best"),
    ]
    assert mod.asr_jobs("lora-full", "dev", "best-over") == [
        ("qwen-base", "qwen-base-dev"),
        ("qwen-lora-best", "qwen-lora-dev-best"),
        ("qwen-lora-over", "qwen-lora-dev-over"),
        ("qwen-full-best", "qwen-full-dev-best"),
        ("qwen-full-over", "qwen-full-dev-over"),
    ]
    gold_rows = [
        {
            "id": "a",
            "type": "auto_gold",
            "label": "不需要",
            "candidate_text": "不需要",
            "gold_text_ann": "不需要",
            "gold_vs_candidate_mismatch": 0,
            "category": "",
            "decision": "accept",
            "gold_kind": "speech",
            "noise_kind": "",
            "exclude_from_training": False,
            "uncertain": "",
            "label_source": "human",
            "wav_path": "a.wav",
            "reference": "不需要",
        }
    ]
    frames = mod.build_frames(
        gold_rows,
        [
            ("qwen-base", {"a": "不需要"}),
            ("qwen-full-best", {"a": "不用"}),
        ],
        gold_path=Path("gold.parquet"),
        model_paths={
            "qwen-base": Path("base.parquet"),
            "qwen-full-best": Path("best.parquet"),
        },
    )
    accuracy = frames[0]
    assert list(accuracy.columns)[5] == "相对 qwen-base"
    base = accuracy.loc[accuracy["模型"] == "qwen-base"].iloc[0]
    assert base["相对 qwen-base"] == "baseline"
    best = accuracy.loc[accuracy["模型"] == "qwen-full-best"].iloc[0]
    assert best["正确率"] == "0.00%"
    assert str(best["相对 qwen-base"]).startswith("-")


def test_filler_tokens_come_from_yaml():
    mod = _load_script()
    tokens = mod.load_score_drop_tokens()
    assert tokens == mod._SCORE_DROP_TOKENS
    assert "你妈" in tokens and "什么" in tokens and "吧" in tokens
    assert "嗯" not in tokens and "哦" not in tokens
    assert mod.SCORE_DROP_YAML.name == "fenshen_1015_filler_tokens.yaml"


def test_eval_type_overrides_plain_match(tmp_path: Path):
    mod = _load_script()
    import pyarrow as pa
    import pyarrow.parquet as pq

    gold = tmp_path / "gold.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": "kept-hardcase",
                    "source_path": "a.wav",
                    "audio": '{"resampled_16k": "a.wav"}',
                    "qwen_1_text": "不需要。",
                    "label_gold_text": "不需要。",
                    "label_source_gold_kind": "speech",
                    "label_decision": "accept",
                    "label_eval_type": "hardcase",
                    "label_noise_kind": "",
                },
                {
                    "id": "noise",
                    "source_path": "b.wav",
                    "audio": '{"resampled_16k": "b.wav"}',
                    "qwen_1_text": "没有",
                    "label_gold_text": "",
                    "label_source_gold_kind": "non_speech",
                    "label_decision": "empty",
                    "label_eval_type": "non_speech",
                    "label_noise_kind": "background_suspected",
                },
            ]
        ),
        gold,
    )
    rows = mod.load_gold_rows(gold)
    by_id = {row["id"]: row for row in rows}
    assert by_id["kept-hardcase"]["type"] == "hardcase"
    assert by_id["kept-hardcase"]["reference"] == "不需要。"
    assert by_id["noise"]["type"] == "non_speech"
    assert by_id["noise"]["reference"] == ""
    assert by_id["noise"]["noise_kind"] == "background_suspected"


def test_classify_ignores_punctuation_and_prefers_non_speech():
    mod = _load_script()
    assert mod.classify_type("speech", "accept", "喂，哎。", "喂？哎。") == "auto_gold"
    assert mod.classify_type("speech", "accept", "是啊", "不需要") == "hardcase"
    assert mod.classify_type("speech", "empty", "是啊", "不需要") == "non_speech"
    assert mod.classify_type("non_speech", "accept", "", "噪声") == "non_speech"


def test_unclassified_gold_reports_only_overall(tmp_path: Path):
    mod = _load_script()
    import pyarrow as pa
    import pyarrow.parquet as pq

    gold = tmp_path / "gold.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": "speech",
                    "source_path": "a.wav",
                    "audio": '{"resampled_16k": "a.wav"}',
                    "qwen_1_text": "是啊",
                    "label_gold_text": "不需要",
                    "label_source_gold_kind": "speech",
                    "label_decision": "accept",
                    "label_eval_type_split": False,
                },
                {
                    "id": "empty",
                    "source_path": "b.wav",
                    "audio": '{"resampled_16k": "b.wav"}',
                    "qwen_1_text": "噪声",
                    "label_gold_text": "",
                    "label_source_gold_kind": "non_speech",
                    "label_decision": "empty",
                    "label_eval_type_split": False,
                },
            ]
        ),
        gold,
    )
    rows = mod.load_gold_rows(gold)
    assert [row["type"] for row in rows] == ["", ""]
    assert rows[1]["reference"] == ""
    frames = mod.build_frames(
        rows,
        [("qwen-base", {"speech": "不需要", "empty": ""})],
        gold_path=gold,
        model_paths={"qwen-base": Path("base.parquet")},
    )
    accuracy = frames[0]
    assert list(accuracy["scope"]) == ["overall"]
    assert "不做分类" in frames[3].loc[frames[3]["字段/口径"] == "条数", "说明"].item()


def test_workbook_matches_char_acc_layout(tmp_path: Path):
    mod = _load_script()
    gold_rows = [
        {
            "id": "a",
            "type": "auto_gold",
            "label": "喂。",
            "candidate_text": "喂",
            "gold_text_ann": "喂。",
            "gold_vs_candidate_mismatch": 0,
            "category": "semantic_risk",
            "decision": "accept",
            "gold_kind": "speech",
            "noise_kind": "",
            "exclude_from_training": False,
            "uncertain": "",
            "label_source": "human",
            "wav_path": "a.wav",
            "reference": "喂。",
        },
        {
            "id": "b",
            "type": "hardcase",
            "label": "不需要",
            "candidate_text": "是啊",
            "gold_text_ann": "不需要",
            "gold_vs_candidate_mismatch": 1,
            "category": "semantic_risk",
            "decision": "accept",
            "gold_kind": "speech",
            "noise_kind": "",
            "exclude_from_training": False,
            "uncertain": "",
            "label_source": "human",
            "wav_path": "b.wav",
            "reference": "不需要",
        },
        {
            "id": "c",
            "type": "non_speech",
            "label": "",
            "candidate_text": "",
            "gold_text_ann": "",
            "gold_vs_candidate_mismatch": 0,
            "category": "",
            "decision": "empty",
            "gold_kind": "non_speech",
            "noise_kind": "",
            "exclude_from_training": True,
            "uncertain": "",
            "label_source": "human",
            "wav_path": "c.wav",
            "reference": "",
        },
    ]
    models = [
        ("qwen-base", {"a": "喂", "b": "是啊", "c": "噪声"}),
        ("qwen-full-best", {"a": "喂", "b": "不需要", "c": ""}),
        ("qwen-full-over", {"a": "喂。", "b": "不需要", "c": "有人"}),
    ]
    frames = mod.build_frames(
        gold_rows,
        models,
        gold_path=Path("gold.parquet"),
        model_paths={name: Path(f"{name}.parquet") for name, _ in models},
    )
    accuracy, pivot, detail, notes = frames
    assert list(accuracy.columns) == [
        "scope",
        "模型",
        "样本数",
        "正确数(字准=1)",
        "正确率",
        "相对 qwen-base",
        "总编辑距离",
        "总基准字数",
        "总字准",
        "相对 qwen-base总字准",
    ]
    non_speech = accuracy[accuracy["scope"] == "non_speech"].set_index("模型")
    assert non_speech.loc["qwen-base", "总基准字数"] == 0
    assert non_speech.loc["qwen-base", "总字准"] == non_speech.loc["qwen-base", "总编辑距离"]
    assert non_speech.loc["qwen-base", "相对 qwen-base"] == "baseline"
    assert non_speech.loc["qwen-base", "相对 qwen-base总字准"] == "baseline"
    assert non_speech.loc["qwen-base", "正确率"] == "0.00%"
    assert non_speech.loc["qwen-full-best", "总字准"] == 0
    assert str(non_speech.loc["qwen-full-best", "相对 qwen-base总字准"]).startswith("-")
    assert non_speech.loc["qwen-full-best", "正确率"] == "100.00%"
    overall = accuracy[accuracy["scope"] == "overall"].set_index("模型")
    # overall 总编辑距离不含空金标「噪声」的插入字数。
    assert overall.loc["qwen-base", "总编辑距离"] == overall.loc["qwen-full-best", "总编辑距离"] + 3
    assert detail.loc[detail["id"] == "c", "qwen-base_exact_correct"].item() == 0
    assert detail.loc[detail["id"] == "c", "qwen-full-best_exact_correct"].item() == 1
    assert "qwen-base_正确率%" in pivot.columns
    assert pivot.loc[pivot["scope"] == "overall", "标注总条数"].item() == 3
    assert list(detail["id"]) == ["a", "b", "c"]
    assert detail.loc[detail["id"] == "b", "qwen-base_exact_correct"].item() == 0
    assert detail.loc[detail["id"] == "b", "qwen-full-best_exact_correct"].item() == 1
    assert detail.loc[detail["id"] == "a", "qwen-full-over_exact_correct"].item() == 1
    assert bool(detail.loc[detail["id"] == "a", "eval_scored"].item()) is True
    assert list(notes.columns) == ["字段/口径", "说明"]

    out = tmp_path / "evaluation_final_dev_v015_base_vs_best_vs_over.xlsx"
    mod.write_xlsx(out, frames)
    book = pd.ExcelFile(out)
    assert book.sheet_names == ["正确率", "透视对比", "明细", "说明"]


@pytest.mark.parametrize(
    ("mode", "ckpt", "options", "splits"),
    [
        ("full", "all", ("--split", "both"), ("train", "dev")),
        ("lora-full", "best", (), ("dev",)),
        ("lora-full", "best-over", ("--split", "both", "--ckpt", "best-over"), ("train", "dev")),
    ],
)
def test_evaluation_groups_splits_and_preserves_previous_workbooks(
    tmp_path: Path, monkeypatch, mode, ckpt, options, splits
):
    mod = _load_script()
    import pyarrow as pa
    import pyarrow.parquet as pq

    gold_rows = pa.Table.from_pylist(
        [
            {
                "id": "a",
                "source_path": "a.wav",
                "audio": '{"resampled_16k": "vad/a.wav"}',
                "qwen_1_text": "喂。",
                "label_gold_text": "喂",
                "label_source_gold_kind": "speech",
                "label_decision": "accept",
                "label_label_source": "human",
                "label_exclude_from_training": False,
            }
        ]
    )
    theme = "fenshen"
    for split in splits:
        gold = mod.gold_path(tmp_path, theme, split, "v015")
        gold.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(gold_rows, gold)
        for _name, run in mod.asr_jobs(mode, split, ckpt):
            path = mod.asr_parquet(tmp_path, theme, "v015", split, run)
            path.parent.mkdir(parents=True, exist_ok=True)
            sample = Sample(
                id="a",
                source_path="a.wav",
                sha256="a" * 64,
                transcripts={run: {"text": "喂"}},
            )
            Manifest([sample]).save(path)

    run_times = iter((datetime(2026, 11, 10, 14, 30, 12), datetime(2026, 11, 10, 14, 30, 58)))

    class FixedClock:
        @staticmethod
        def now():
            return next(run_times)

    monkeypatch.setattr(mod, "datetime", FixedClock)
    command = [theme, mode, "v015", *options, "--root", str(tmp_path)]
    reports = tmp_path / "datasets" / "stage3" / "asr" / f"eval_{theme}_v015"
    first = reports / "1110_1430"
    second = reports / "1110_1430_2"
    names = {f"{theme}_v015_{split}_{mode}.xlsx" for split in splits}
    assert mod.main(command) == 0
    original = {path.name: path.read_bytes() for path in first.iterdir()}
    assert original.keys() == names
    assert mod.main(command) == 0
    assert {path.name for path in second.iterdir()} == names
    assert {path.name: path.read_bytes() for path in first.iterdir()} == original
    report = second / f"{theme}_v015_dev_{mode}.xlsx"
    detail = pd.read_excel(report, sheet_name="明细")
    assert detail.loc[0, "type"] == "auto_gold"
    assert detail.loc[0, "qwen-full-best_exact_correct"] == 1
    assert detail.loc[0, "wav_path"] == "vad/a.wav"
    if mode == "lora-full":
        assert detail.loc[0, "qwen-lora-best_exact_correct"] == 1
        summary = pd.read_excel(report, sheet_name="正确率")
        overall_models = summary.loc[summary["scope"] == "overall", "模型"].tolist()
        if ckpt == "best":
            assert overall_models == ["qwen-base", "qwen-lora-best", "qwen-full-best"]
        else:
            assert overall_models == [
                "qwen-base", "qwen-lora-best", "qwen-lora-over", "qwen-full-best", "qwen-full-over"
            ]
