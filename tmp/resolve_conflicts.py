from __future__ import annotations

import re
import subprocess
from pathlib import Path


def resolve_conflicts_theirs(text: str) -> str:
    pattern = re.compile(
        r"<<<<<<< [^\n]*\n.*?=======\n(.*?)>>>>>>> [^\n]*\n",
        re.DOTALL,
    )
    while "<<<<<<<" in text:
        text = pattern.sub(r"\1", text, count=1)
    marker = re.compile(r"^(<<<<<<< .*|=======|>>>>>>> .*)$")
    return "".join(
        line for line in text.splitlines(keepends=True) if not marker.match(line.rstrip("\n"))
    )


def dedupe_consecutive_blocks(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        removed = False
        for size in range(min(20, len(lines) - i), 1, -1):
            block = lines[i : i + size]
            if i + 2 * size <= len(lines) and lines[i + size : i + 2 * size] == block:
                while i + 2 * size <= len(lines) and lines[i + size : i + 2 * size] == block:
                    i += size
                out.extend(block)
                removed = True
                break
        if not removed:
            out.append(lines[i])
            i += 1
    return "".join(out)


def load_text(root: Path, rel: str) -> str:
    path = root / rel
    for cmd in (["git", "show", f":{rel}"], ["git", "show", f"HEAD:{rel}"]):
        try:
            return subprocess.check_output(cmd).decode("utf-8")
        except subprocess.CalledProcessError:
            continue
    return path.read_text(encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    files = [
        "src/audio_engine/cli/main.py",
        ".gitignore",
        "tests/test_dataset_workflow.py",
        "docs/流水线改进/全自动训练与评测闭环改造方案.md",
    ]
    for rel in files:
        text = load_text(root, rel)
        if "<<<<<<<" not in text:
            print("skip", rel)
            continue
        resolved = dedupe_consecutive_blocks(resolve_conflicts_theirs(text))
        (root / rel).write_text(resolved, encoding="utf-8")
        print("resolved", rel)


if __name__ == "__main__":
    main()
