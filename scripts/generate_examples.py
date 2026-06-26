#!/usr/bin/env python3
"""Generate JSONL examples from the music reasoning task families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from music_reasoning_tasks import get_task, list_tasks


def _write_readable(path: Path, examples: list) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write("PROMPT:\n")
            handle.write(str(example.prompt))
            handle.write("\n\nANSWER:\n")
            handle.write(str(example.answer))
            handle.write("\n\nCOT:\n")
            handle.write(str(example.metadata.get("cot", "")))
            handle.write("\n\n---\n")


def _write_jsonl(path: Path, examples: list) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            row = example.to_dict()
            row["metadata"] = json.dumps(dict(row["metadata"]))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=list_tasks(), help="Task family to generate.")
    parser.add_argument("--level", type=int, default=5, help="Difficulty/distribution level.")
    parser.add_argument("--num-examples", type=int, default=500, help="Number of examples.")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Token budget per generated example.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument("--out-dir", type=Path, default=Path("generated_data"), help="Output directory.")
    parser.add_argument("--readable", action="store_true", help="Also write a human-readable text file.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    task = get_task(args.task)
    examples = task.generate_balanced_batch(
        batch_size=args.num_examples,
        max_tokens=args.max_tokens,
        level=args.level,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.task}_l{args.level}_{args.num_examples}"
    _write_jsonl(args.out_dir / f"{stem}.jsonl", examples)
    if args.readable:
        _write_readable(args.out_dir / f"{stem}_readable.txt", examples)


if __name__ == "__main__":
    main()
