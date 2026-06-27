#!/usr/bin/env python3
"""Generate a balanced music reasoning benchmark for Hugging Face upload.

Default benchmark layout:

    easy      level 0    3200 examples
    moderate  level 3    3200 examples
    hard      level 5    3200 examples

Each split contains 200 examples for every mode of both implemented task
families, giving 1600 pitch/interval examples and 1600 chord/Roman examples.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from music_reasoning_tasks import __version__ as GENERATOR_VERSION
from music_reasoning_tasks import get_task
from music_reasoning_tasks.chord_roman_reasoning import MODE_NAMES as CHORD_ROMAN_MODES
from music_reasoning_tasks.pitch_interval_reasoning import MODE_NAMES as PITCH_INTERVAL_MODES


TASK_FAMILIES = {
    "pitch_interval_reasoning": tuple(PITCH_INTERVAL_MODES),
    "chord_roman_reasoning": tuple(CHORD_ROMAN_MODES),
}
DEFAULT_LEVELS = (0, 3, 5)
DEFAULT_SPLIT_NAMES = {
    0: "easy",
    3: "moderate",
    5: "hard",
}
DEFAULT_BENCHMARK_VERSION = "v0.2.0"
BENCHMARK_RESPONSE_INSTRUCTION = "Return only the requested answer."


def _jsonable(value: Any) -> Any:
    """Convert generated metadata into JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _metadata_json(metadata: Any) -> str:
    """Serialize metadata as a stable JSON string for simple HF schemas."""
    return json.dumps(_jsonable(dict(metadata)), ensure_ascii=False, sort_keys=True)


def _split_name(level: int) -> str:
    return DEFAULT_SPLIT_NAMES.get(level, f"level_{level}")


def _benchmark_prompt(prompt: object) -> str:
    """Add the output-only instruction used for benchmark evaluation."""
    return f"{str(prompt).rstrip()} {BENCHMARK_RESPONSE_INSTRUCTION}"


def _row_from_example(example: Any, *, split: str, level: int, family: str, index: int) -> dict[str, Any]:
    """Create one benchmark row from a generated example."""
    metadata = dict(example.metadata)
    mode = str(metadata.get("mode", ""))
    answer_kind = str(metadata.get("answer_kind", "text"))
    cot = str(metadata.get("cot", ""))
    for redundant_key in ("prompt", "cot", "_time"):
        metadata.pop(redundant_key, None)
    row_id = f"{split}-{family}-{mode}-{index:04d}"
    return {
        "id": row_id,
        "split": split,
        "level": level,
        "difficulty": split,
        "family": family,
        "mode": mode,
        "prompt": _benchmark_prompt(example.prompt),
        "answer": str(example.answer),
        "answer_kind": answer_kind,
        "cot": cot,
        "metadata": _metadata_json(metadata),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _generate_mode_rows(
    *,
    family: str,
    mode: str,
    level: int,
    split: str,
    n_examples: int,
    max_tokens: int,
    deduplicate_prompts: bool,
    seen_prompts: set[tuple[str, str]],
    max_attempts: int,
    start_index: int,
) -> list[dict[str, Any]]:
    """Generate exactly n_examples rows for one family/mode/level cell."""
    task = get_task(family)
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < n_examples:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not generate {n_examples} unique examples for "
                f"{family}/{mode} at level {level} after {max_attempts} attempts."
            )

        example = task.generate_example(level=level, max_tokens=max_tokens, mode=mode)
        prompt_key = (split, _benchmark_prompt(example.prompt))
        if deduplicate_prompts and prompt_key in seen_prompts:
            continue

        metadata_mode = example.metadata.get("mode")
        if metadata_mode != mode:
            raise RuntimeError(f"Expected mode {mode!r}, got {metadata_mode!r}.")

        seen_prompts.add(prompt_key)
        rows.append(
            _row_from_example(
                example,
                split=split,
                level=level,
                family=family,
                index=start_index + len(rows),
            )
        )
    return rows


def _summarize_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return compact counts for one generated split."""
    family_counts = Counter(row["family"] for row in rows)
    mode_counts: dict[str, Counter[str]] = defaultdict(Counter)
    answer_kind_counts = Counter(row["answer_kind"] for row in rows)
    for row in rows:
        mode_counts[row["family"]][row["mode"]] += 1
    return {
        "num_examples": len(rows),
        "families": dict(sorted(family_counts.items())),
        "modes": {
            family: dict(sorted(counter.items()))
            for family, counter in sorted(mode_counts.items())
        },
        "answer_kinds": dict(sorted(answer_kind_counts.items())),
    }


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_dataset_card(
    path: Path,
    *,
    levels: list[int],
    examples_per_mode: int,
    seed: int | None,
    benchmark_version: str,
    generator_version: str,
) -> None:
    """Write a minimal Hugging Face dataset card for the generated benchmark."""
    split_rows = "\n".join(
        f"  - split: {_split_name(level)}\n    path: data/{_split_name(level)}.jsonl" for level in levels
    )
    total_per_split = examples_per_mode * sum(len(modes) for modes in TASK_FAMILIES.values())
    total = total_per_split * len(levels)
    path.write_text(
        f"""---
configs:
- config_name: default
  data_files:
{split_rows}
---

# Procedural Music Reasoning Benchmark

This benchmark was generated with
[`Danila-Pechenev/procedural-music-reasoning`](https://github.com/Danila-Pechenev/procedural-music-reasoning).

Benchmark version: `{benchmark_version}`.
Generator version: `{generator_version}`.

It contains balanced examples from two implemented music reasoning task
families:

- `pitch_interval_reasoning`
- `chord_roman_reasoning`

## Splits

| Split | Generator level | Examples |
|---|---:|---:|
"""
        + "\n".join(f"| `{_split_name(level)}` | {level} | {total_per_split} |" for level in levels)
        + f"""

Total examples: **{total}**.

Each split contains `{examples_per_mode}` examples per mode. With eight modes
per family and two families, this gives `{total_per_split}` examples per split.

## Columns

- `id`: stable row identifier.
- `split`: split name.
- `level`: generator difficulty/distribution level.
- `difficulty`: human-readable difficulty name.
- `family`: task family.
- `mode`: task mode within the family.
- `prompt`: model input.
- `answer`: canonical expected answer.
- `answer_kind`: answer-normalization family.
- `cot`: generator-produced reasoning trace.
- `metadata`: JSON string with symbolic generation metadata.

## Evaluation Protocol

For benchmark evaluation, give the model the `prompt` only and compare its
answer with `answer` using the task scorer. The `cot` field is provided for
inspection, supervised training, and error analysis, but should not be included
in the model prompt during benchmark evaluation.

Every benchmark prompt ends with `{BENCHMARK_RESPONSE_INSTRUCTION}` This
benchmark-only instruction requests the short answer expected by the scorer;
it is not added to examples produced directly by the task generators.

Difficulty levels are distributional. A hard split may still contain some
simple examples, but harder musical features are sampled more often or from a
larger space.

## Versioning

Hugging Face dataset releases should be tagged with the benchmark version. To
load this exact release after upload, use:

```python
from datasets import load_dataset

dataset = load_dataset(
    "dpechenev/music-reasoning-benchmark",
    revision="{benchmark_version}",
)
```

Generation seed: `{seed}`.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=list(DEFAULT_LEVELS),
        help="Generator levels to produce. Defaults to 0 3 5.",
    )
    parser.add_argument(
        "--examples-per-mode",
        type=int,
        default=200,
        help="Number of examples for each family/mode/level cell.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark_data/music_reasoning_benchmark"),
        help="Output directory containing data/, README.md, and summary.json.",
    )
    parser.add_argument("--max-tokens", type=int, default=8192, help="Token budget per generated example.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed. Use -1 for no explicit seed.")
    parser.add_argument(
        "--benchmark-version",
        default=DEFAULT_BENCHMARK_VERSION,
        help=f"Benchmark release version recorded in README.md and summary.json. Defaults to {DEFAULT_BENCHMARK_VERSION}.",
    )
    parser.add_argument(
        "--generator-version",
        default=GENERATOR_VERSION,
        help="Generator package version recorded in README.md and summary.json.",
    )
    parser.add_argument(
        "--max-attempts-multiplier",
        type=int,
        default=80,
        help="Maximum attempts per cell equals examples_per_mode times this value.",
    )
    parser.add_argument(
        "--allow-duplicate-prompts",
        action="store_true",
        help="Allow duplicate prompts inside the same split.",
    )
    parser.add_argument(
        "--write-combined",
        action="store_true",
        help="Also write data/all.jsonl for local analysis. By default, difficulty splits stay separate.",
    )
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed
    if seed is not None:
        random.seed(seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "benchmark_version": args.benchmark_version,
        "generator_version": args.generator_version,
        "response_instruction": BENCHMARK_RESPONSE_INSTRUCTION,
        "levels": args.levels,
        "examples_per_mode": args.examples_per_mode,
        "seed": seed,
        "families": {family: list(modes) for family, modes in TASK_FAMILIES.items()},
        "splits": {},
    }
    all_rows: list[dict[str, Any]] = []

    for level in args.levels:
        split = _split_name(level)
        split_rows: list[dict[str, Any]] = []
        seen_prompts: set[tuple[str, str]] = set()
        print(f"Generating split {split!r} at level {level}...")
        for family, modes in TASK_FAMILIES.items():
            for mode in modes:
                print(f"  {family}/{mode}: {args.examples_per_mode}")
                rows = _generate_mode_rows(
                    family=family,
                    mode=mode,
                    level=level,
                    split=split,
                    n_examples=args.examples_per_mode,
                    max_tokens=args.max_tokens,
                    deduplicate_prompts=not args.allow_duplicate_prompts,
                    seen_prompts=seen_prompts,
                    max_attempts=args.examples_per_mode * args.max_attempts_multiplier,
                    start_index=len(split_rows),
                )
                split_rows.extend(rows)

        _write_jsonl(data_dir / f"{split}.jsonl", split_rows)
        all_rows.extend(split_rows)
        summary["splits"][split] = _summarize_split(split_rows)

    if args.write_combined:
        _write_jsonl(data_dir / "all.jsonl", all_rows)

    summary["total_examples"] = len(all_rows)
    _write_summary(args.out_dir / "summary.json", summary)
    _write_dataset_card(
        args.out_dir / "README.md",
        levels=args.levels,
        examples_per_mode=args.examples_per_mode,
        seed=seed,
        benchmark_version=args.benchmark_version,
        generator_version=args.generator_version,
    )
    print(f"Done. Wrote benchmark to {args.out_dir}")


if __name__ == "__main__":
    main()
