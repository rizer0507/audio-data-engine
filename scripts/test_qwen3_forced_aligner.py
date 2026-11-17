#!/usr/bin/env python3
"""Load local Qwen3-ForcedAligner weights and align one audio/transcript pair."""

import argparse
from pathlib import Path


DEFAULT_MODEL_DIR = Path("/data2/data-cp/models/Qwen3-ForcedAligner-0.6b")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--audio", type=Path, required=True, help="Local audio file")
    parser.add_argument("--text", required=True, help="Transcript for this exact audio file")
    parser.add_argument("--language", default="Chinese", help="Qwen language name")
    parser.add_argument("--device", default="cuda:0", help="PyTorch device")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    audio = args.audio.expanduser().resolve()
    text = args.text.strip()
    if not (model_dir / "config.json").is_file():
        parser.error(f"model config not found: {model_dir / 'config.json'}")
    if not audio.is_file():
        parser.error(f"audio file not found: {audio}")
    if not text:
        parser.error("--text must contain a transcript")

    import torch
    from qwen_asr import Qwen3ForcedAligner

    aligner = Qwen3ForcedAligner.from_pretrained(
        str(model_dir),
        dtype=getattr(torch, args.dtype),
        device_map=args.device,
        local_files_only=True,
    )
    result = aligner.align(audio=str(audio), text=text, language=args.language)[0]
    if not result.items:
        raise SystemExit("Alignment returned no timestamp items.")

    print("text\tstart_s\tend_s")
    for item in result.items:
        print(f"{item.text}\t{item.start_time:.3f}\t{item.end_time:.3f}")


if __name__ == "__main__":
    main()
