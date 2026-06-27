#!/usr/bin/env python3
"""Evaluate an OpenRouter model on every split of a generated benchmark.

Paste an OpenRouter API key into ``OPENROUTER_API_KEY`` below, then run:

    python scripts/evaluate_openrouter.py openai/gpt-4.1-mini

The evaluator sends benchmark prompts unchanged, scores responses with the
task family's ``score_answer`` implementation, and writes one resumable JSONL
file and one Markdown report with metrics kept separate by difficulty split.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import uuid
from typing import Any, Iterable, Mapping, Sequence

from packaging.version import InvalidVersion, Version

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from music_reasoning_tasks import score_answer


# Keep this placeholder in committed code. Paste your key locally and never
# commit the populated value.
OPENROUTER_API_KEY = "PASTE_YOUR_OPENROUTER_API_KEY_HERE"

DEFAULT_DATASET_REPO = "dpechenev/music-reasoning-benchmark"
DEFAULT_OUTPUT_ROOT = Path("benchmark_results")
DEFAULT_SPLITS = ("easy", "moderate", "hard")
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
FATAL_API_ERROR_MARKERS = (
    "authentication",
    "invalid api key",
    "unauthorized",
    "forbidden",
    "insufficient credits",
    "payment required",
    "model not found",
    "no endpoints found",
    "401",
    "402",
    "403",
    "404",
)


class EvaluationError(RuntimeError):
    """Raised when evaluation cannot continue safely."""


@dataclass
class Metrics:
    """Aggregate scoring, usage, and failure statistics."""

    total: int = 0
    completed: int = 0
    correct: int = 0
    api_errors: int = 0
    scoring_errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def completed_accuracy(self) -> float:
        return self.correct / self.completed if self.completed else 0.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-_") or "model"


def _openrouter_model(model: str) -> str:
    """Return the exact LiteLLM OpenRouter model route."""
    model = model.strip()
    if model.count("/") != 1:
        raise EvaluationError(
            "Use an exact OpenRouter model ID in provider/model form, "
            "for example, openai/gpt-4.1-mini."
        )
    return f"openrouter/{model}"


def _validate_api_key() -> None:
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "PASTE_YOUR_OPENROUTER_API_KEY_HERE":
        raise EvaluationError(
            "Set OPENROUTER_API_KEY near the top of scripts/evaluate_openrouter.py before running evaluation."
        )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    return str(value)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
    return rows


def _load_benchmark(
    dataset_repo: str,
    dataset_revision: str,
    splits: Sequence[str],
    limit_per_split: int | None,
) -> list[dict[str, Any]]:
    """Download benchmark splits from a pinned Hugging Face dataset revision."""
    from datasets import load_dataset

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    try:
        benchmark = load_dataset(dataset_repo, revision=dataset_revision)
    except Exception as exc:
        raise EvaluationError(
            f"Could not load {dataset_repo!r} at revision {dataset_revision!r}: {exc}"
        ) from exc
    for split in splits:
        if split not in benchmark:
            available = ", ".join(sorted(benchmark))
            raise EvaluationError(
                f"Dataset split {split!r} does not exist at revision {dataset_revision!r}. "
                f"Available splits: {available}."
            )
        dataset = benchmark[split]
        split_rows = [dict(row) for row in dataset]
        if limit_per_split is not None:
            split_rows = split_rows[:limit_per_split]
        for row in split_rows:
            missing = {"id", "split", "family", "mode", "prompt", "answer", "metadata"} - row.keys()
            if missing:
                raise EvaluationError(
                    f"Benchmark row in {dataset_repo}/{split} is missing fields: {sorted(missing)}"
                )
            if row["id"] in seen_ids:
                raise EvaluationError(f"Duplicate benchmark row ID: {row['id']}")
            seen_ids.add(row["id"])
            rows.append(row)
    return rows


def _resolve_dataset_commit(dataset_repo: str, dataset_revision: str) -> str:
    """Resolve a branch, tag, or commit to the exact Hub commit being evaluated."""
    from huggingface_hub import HfApi

    try:
        refs = HfApi().list_repo_refs(dataset_repo, repo_type="dataset")
    except Exception as exc:
        raise EvaluationError(
            f"Could not resolve {dataset_repo!r} revision {dataset_revision!r}: {exc}"
        ) from exc
    for ref in [*refs.tags, *refs.branches]:
        if ref.name == dataset_revision:
            return str(ref.target_commit)
    if re.fullmatch(r"[0-9a-fA-F]{7,40}", dataset_revision):
        return dataset_revision.lower()
    raise EvaluationError(
        f"Revision {dataset_revision!r} is not a branch, tag, or commit SHA in {dataset_repo!r}."
    )


def _latest_dataset_version_tag(dataset_repo: str) -> str:
    """Return the highest stable semantic-version tag in a dataset repository."""
    from huggingface_hub import HfApi

    try:
        tags = HfApi().list_repo_refs(dataset_repo, repo_type="dataset").tags
    except Exception as exc:
        raise EvaluationError(f"Could not list tags for {dataset_repo!r}: {exc}") from exc

    versioned_tags: list[tuple[Version, str]] = []
    for tag in tags:
        normalized = tag.name.removeprefix("v")
        try:
            version = Version(normalized)
        except InvalidVersion:
            continue
        if not version.is_prerelease and not version.is_devrelease:
            versioned_tags.append((version, tag.name))
    if not versioned_tags:
        raise EvaluationError(
            f"Dataset {dataset_repo!r} has no stable semantic-version tags. "
            "Pass --revision explicitly."
        )
    return max(versioned_tags)[1]


def _load_latest_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["id"]): row for row in _load_jsonl(path)}


def _is_reusable_result(
    result: dict[str, Any],
    row: dict[str, Any],
    model: str,
    dataset_repo: str,
    dataset_commit: str,
    reasoning_effort: str | None,
    temperature: float,
    max_tokens: int,
) -> bool:
    return (
        result.get("status") == "ok"
        and result.get("model_requested") == model
        and result.get("dataset_repo") == dataset_repo
        and result.get("dataset_commit") == dataset_commit
        and result.get("reasoning_effort") == reasoning_effort
        and _numeric(result.get("temperature", 0.0)) == temperature
        and int(result.get("max_tokens", 1024)) == max_tokens
        and result.get("prompt_sha256") == _prompt_hash(str(row["prompt"]))
        and result.get("expected_answer") == row.get("answer")
    )


def _usage_dict(response: Any) -> dict[str, Any]:
    usage = _jsonable(getattr(response, "usage", None))
    return usage if isinstance(usage, dict) else {}


def _scorable_row(row: dict[str, Any]) -> dict[str, Any]:
    """Restore metadata fields needed when rebuilding a Reasoning Core Problem."""
    scorable = dict(row)
    metadata = row.get("metadata", {})
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    metadata = dict(metadata)
    metadata.setdefault("cot", row.get("cot", ""))
    scorable["metadata"] = metadata
    return scorable


def _numeric(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _result_from_response(
    row: dict[str, Any],
    response: Any,
    *,
    model: str,
    batch_seconds: float,
    batch_id: str | None = None,
    reasoning_effort: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    prediction = str(response).strip()
    usage = _usage_dict(response)
    reported_cost = _numeric(usage.get("cost"), _numeric(getattr(response, "cost", 0.0)))
    result: dict[str, Any] = {
        "id": row["id"],
        "split": row.get("split"),
        "level": row.get("level"),
        "family": row["family"],
        "mode": row["mode"],
        "prompt": row["prompt"],
        "expected_answer": row["answer"],
        "prediction": prediction,
        "status": "ok",
        "score": 0.0,
        "model_requested": model,
        "model_used": str(getattr(response, "model_used", model)),
        "reasoning_effort": reasoning_effort,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt_sha256": _prompt_hash(str(row["prompt"])),
        "usage": usage,
        "cost": reported_cost,
        "reasoning": _jsonable(getattr(response, "reasoning", None)),
        "batch_id": batch_id,
        "batch_seconds": round(batch_seconds, 3),
        "evaluated_at": _utc_now(),
        "error": None,
    }
    try:
        result["score"] = float(score_answer(prediction, _scorable_row(row)))
    except Exception as exc:
        result["status"] = "scoring_error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _error_result(
    row: dict[str, Any],
    *,
    model: str,
    error: Exception,
    batch_seconds: float,
    reasoning_effort: str | None,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "id": row["id"],
        "split": row.get("split"),
        "level": row.get("level"),
        "family": row["family"],
        "mode": row["mode"],
        "prompt": row["prompt"],
        "expected_answer": row["answer"],
        "prediction": "",
        "status": "api_error",
        "score": 0.0,
        "model_requested": model,
        "model_used": None,
        "reasoning_effort": reasoning_effort,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt_sha256": _prompt_hash(str(row["prompt"])),
        "usage": {},
        "cost": 0.0,
        "reasoning": None,
        "batch_id": uuid.uuid4().hex,
        "batch_seconds": round(batch_seconds, 3),
        "evaluated_at": _utc_now(),
        "error": f"{type(error).__name__}: {error}",
    }


def _fatal_api_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in FATAL_API_ERROR_MARKERS)


def _call_rows(
    rows: Sequence[dict[str, Any]],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    num_retries: int,
    timeout: float,
    caching: bool,
    reasoning_effort: str | None,
) -> list[dict[str, Any]]:
    """Call one batch, falling back to individual calls to isolate failures."""
    from litlm import complete

    prompts = [str(row["prompt"]) for row in rows]
    started = time.monotonic()
    completion_kwargs: dict[str, Any] = {
        "model": model,
        "api_key": OPENROUTER_API_KEY,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "num_retries": num_retries,
        "timeout": timeout,
        "caching": caching,
        "show_progress": False,
    }
    if reasoning_effort is not None:
        completion_kwargs["reasoning_effort"] = reasoning_effort
    try:
        responses = complete(prompts, **completion_kwargs)
    except Exception as exc:
        if _fatal_api_error(exc):
            raise EvaluationError(f"OpenRouter request cannot continue: {exc}") from exc
        if len(rows) == 1:
            return [
                _error_result(
                    rows[0],
                    model=model,
                    error=exc,
                    batch_seconds=time.monotonic() - started,
                    reasoning_effort=reasoning_effort,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            ]
        isolated: list[dict[str, Any]] = []
        for row in rows:
            isolated.extend(
                _call_rows(
                    [row],
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    num_retries=num_retries,
                    timeout=timeout,
                    caching=caching,
                    reasoning_effort=reasoning_effort,
                )
            )
        return isolated

    if not isinstance(responses, list):
        responses = [responses]
    if len(responses) != len(rows):
        raise EvaluationError(f"OpenRouter returned {len(responses)} responses for {len(rows)} prompts.")
    batch_seconds = time.monotonic() - started
    batch_id = uuid.uuid4().hex
    return [
        _result_from_response(
            row,
            response,
            model=model,
            batch_seconds=batch_seconds,
            batch_id=batch_id,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for row, response in zip(rows, responses, strict=True)
    ]


def _chunks(items: Sequence[dict[str, Any]], size: int) -> Iterable[Sequence[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _append_results(path: Path, results: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _metrics(results: Iterable[dict[str, Any]]) -> Metrics:
    metrics = Metrics()
    for result in results:
        metrics.total += 1
        status = result.get("status")
        if status == "ok":
            metrics.completed += 1
            metrics.correct += int(_numeric(result.get("score")) >= 1.0)
        elif status == "api_error":
            metrics.api_errors += 1
        elif status == "scoring_error":
            metrics.scoring_errors += 1
        usage = result.get("usage") or {}
        metrics.prompt_tokens += int(_numeric(usage.get("prompt_tokens")))
        metrics.completion_tokens += int(_numeric(usage.get("completion_tokens")))
        metrics.total_tokens += int(_numeric(usage.get("total_tokens")))
        metrics.cost += _numeric(usage.get("cost"), _numeric(result.get("cost")))
    return metrics


def _percentage(value: float) -> str:
    return f"{value:.2%}"


def _format_duration(seconds: float) -> str:
    hours, remainder = divmod(max(0.0, seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {seconds:.1f}s"
    return f"{seconds:.1f}s"


def _recorded_benchmark_seconds(results: Sequence[dict[str, Any]]) -> float:
    """Sum one wall-time duration per recorded concurrent API batch."""
    batches: dict[tuple[str, ...], float] = {}
    for result in results:
        duration = _numeric(result.get("batch_seconds"))
        if duration <= 0:
            continue
        batch_id = result.get("batch_id")
        if batch_id:
            key = ("batch", str(batch_id))
        else:
            # Results generated before batch IDs were added share their
            # timestamp and rounded duration within each concurrent batch.
            key = (
                "legacy",
                str(result.get("evaluated_at", "")),
                str(result.get("batch_seconds")),
            )
        batches[key] = max(duration, batches.get(key, 0.0))
    return sum(batches.values())


def _metric_row(label: str, metrics: Metrics) -> str:
    return (
        f"| {label} | {metrics.total} | {metrics.completed} | {metrics.correct} | "
        f"{_percentage(metrics.accuracy)} | {_percentage(metrics.completed_accuracy)} | "
        f"{metrics.api_errors} | {metrics.scoring_errors} | ${metrics.cost:.6f} |"
    )


def _group_results(
    results: Iterable[dict[str, Any]],
    keys: Sequence[str],
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[tuple(str(result.get(key, "")) for key in keys)].append(result)
    return dict(grouped)


def _metrics_table(
    groups: dict[tuple[str, ...], list[dict[str, Any]]],
    ordered_keys: Sequence[tuple[str, ...]] | None = None,
) -> str:
    lines = [
        "| Group | Total | Completed | Correct | Accuracy | Completed accuracy | API errors | Scoring errors | API cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    keys = ordered_keys if ordered_keys is not None else sorted(groups)
    for key in keys:
        if key in groups:
            lines.append(_metric_row(" / ".join(key), _metrics(groups[key])))
    return "\n".join(lines)


def _generator_version(rows: Sequence[dict[str, Any]]) -> str:
    versions: set[str] = set()
    for row in rows:
        metadata = row.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                continue
        if isinstance(metadata, dict) and metadata.get("_generator_version") is not None:
            versions.add(str(metadata["_generator_version"]))
    return ", ".join(sorted(versions)) or "unknown"


def _split_report_lines(
    split: str,
    results: Sequence[dict[str, Any]],
    max_incorrect_examples: int,
    benchmark_seconds: float,
) -> list[str]:
    """Build one self-contained metrics section for a difficulty split."""
    split_metrics = _metrics(results)
    family_groups = _group_results(results, ("family",))
    mode_groups = _group_results(results, ("family", "mode"))
    incorrect = [row for row in results if row.get("status") == "ok" and _numeric(row.get("score")) < 1.0]
    failures = [row for row in results if row.get("status") != "ok"]
    split_label = split.replace("_", " ").title()
    lines = [
        f"## {split_label} Split",
        "",
        "| Scope | Total | Completed | Correct | Accuracy | Completed accuracy | API errors | Scoring errors | API cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        _metric_row(f"{split_label} split", split_metrics),
        "",
        "`Accuracy` uses all requested rows in this split as the denominator. "
        "`Completed accuracy` excludes API and scoring failures.",
        "",
        "### Results by Task Family",
        "",
        _metrics_table(family_groups),
        "",
        "### Results by Mode",
        "",
        _metrics_table(mode_groups),
        "",
        "### Usage",
        "",
        f"- Recorded benchmark time: `{_format_duration(benchmark_seconds)}`",
        f"- Prompt tokens: `{split_metrics.prompt_tokens:,}`",
        f"- Completion tokens: `{split_metrics.completion_tokens:,}`",
        f"- Total tokens: `{split_metrics.total_tokens:,}`",
        f"- Reported API cost: `${split_metrics.cost:.6f}`",
        "",
        "Usage and cost are summed from values returned by the provider. A provider may omit some fields.",
        "",
        "### Request and Scoring Failures",
        "",
    ]
    if failures:
        lines.extend(
            f"- `{row['id']}` ({row['status']}): {str(row.get('error', '')).replace(chr(10), ' ')}"
            for row in failures
        )
    else:
        lines.append("No API or scoring failures were recorded.")

    lines.extend(["", "### Sample Incorrect Responses", ""])
    if not incorrect:
        lines.append("No incorrect completed responses were recorded.")
    else:
        for row in incorrect[:max_incorrect_examples]:
            lines.extend(
                [
                    f"#### `{row['id']}`",
                    "",
                    f"- **Split:** `{row['split']}`",
                    f"- **Task:** `{row['family']} / {row['mode']}`",
                    "- **Prompt:**",
                    "",
                    "```text",
                    str(row["prompt"]),
                    "```",
                    f"- **Expected:** `{str(row['expected_answer']).replace('`', '\\`')}`",
                    "- **Prediction:**",
                    "",
                    "```text",
                    str(row.get("prediction", "")),
                    "```",
                    "",
                ]
            )
        if len(incorrect) > max_incorrect_examples:
            lines.append(
                f"The report shows {max_incorrect_examples} of {len(incorrect)} incorrect responses. "
                "See `results.jsonl` for every prediction."
            )
    return lines


def _write_report(
    path: Path,
    *,
    splits: Sequence[str],
    dataset_repo: str,
    dataset_revision: str,
    dataset_commit: str,
    generator_version: str,
    model: str,
    reasoning_effort: str | None,
    results: Sequence[dict[str, Any]],
    max_incorrect_examples: int,
    split_benchmark_seconds: Mapping[str, float],
    temperature: float,
    max_tokens: int,
) -> None:
    result_splits = {str(result.get("split")) for result in results}
    unexpected_splits = result_splits - set(splits)
    if unexpected_splits:
        raise EvaluationError(f"Report contains unexpected splits: {sorted(unexpected_splits)}.")

    split_groups = _group_results(results, ("split",))
    lines = [
        "# Music Reasoning Benchmark Evaluation",
        "",
        "## Run Configuration",
        "",
        f"- **Model:** `{model.removeprefix('openrouter/')}`",
        f"- **Reasoning effort:** `{reasoning_effort or 'provider default'}`",
        f"- **Dataset:** [`{dataset_repo}`](https://huggingface.co/datasets/{dataset_repo})",
        f"- **Dataset revision:** `{dataset_revision}`",
        f"- **Dataset commit:** `{dataset_commit}`",
        f"- **Generator version:** `{generator_version}`",
        f"- **Generated at:** `{_utc_now()}`",
        f"- **Temperature:** `{temperature}`",
        f"- **Maximum completion tokens:** `{max_tokens}`",
        "- **Prompt protocol:** benchmark prompts were sent unchanged, without the reference answer or CoT.",
        "",
        "## Results by Difficulty",
        "",
        "| Split | Total | Completed | Correct | Accuracy | Completed accuracy | API errors | Scoring errors | API cost | Benchmark time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[
            (
                f"| {split} | {metrics.total} | {metrics.completed} | {metrics.correct} | "
                f"{_percentage(metrics.accuracy)} | {_percentage(metrics.completed_accuracy)} | "
                f"{metrics.api_errors} | {metrics.scoring_errors} | ${metrics.cost:.6f} | "
                f"{_format_duration(split_benchmark_seconds.get(split, 0.0))} |"
            )
            for split in splits
            for metrics in [_metrics(split_groups.get((split,), []))]
        ],
        "",
    ]
    for split in splits:
        split_results = [result for result in results if result.get("split") == split]
        lines.extend(
            _split_report_lines(
                split,
                split_results,
                max_incorrect_examples,
                split_benchmark_seconds.get(split, 0.0),
            )
        )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Exact OpenRouter model ID, for example, openai/gpt-4.1-mini.")
    parser.add_argument(
        "--dataset-repo",
        default=DEFAULT_DATASET_REPO,
        help=f"Hugging Face dataset repository. Defaults to {DEFAULT_DATASET_REPO}.",
    )
    parser.add_argument(
        "--revision",
        help="Hugging Face tag, branch, or commit. Defaults to the latest stable semantic-version tag.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory for model-specific results. Defaults to {DEFAULT_OUTPUT_ROOT}.",
    )
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS), help="Benchmark splits to evaluate.")
    parser.add_argument("--batch-size", type=int, default=16, help="Concurrent prompts submitted per litlm batch.")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum completion tokens per prompt.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        help="OpenRouter reasoning effort. If omitted, the provider default is used.",
    )
    parser.add_argument("--num-retries", type=int, default=5, help="Retries delegated to LiteLLM per request.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout in seconds per request.")
    parser.add_argument("--cache", action="store_true", help="Enable litlm's local response cache.")
    parser.add_argument(
        "--limit-per-split",
        type=int,
        help="Evaluate only the first N rows of each split. Intended for inexpensive smoke tests.",
    )
    parser.add_argument(
        "--max-incorrect-examples",
        type=int,
        default=30,
        help="Maximum incorrect responses embedded in the Markdown report.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Discard existing results for this model.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    if args.limit_per_split is not None and args.limit_per_split < 1:
        parser.error("--limit-per-split must be at least 1")
    if args.max_incorrect_examples < 0:
        parser.error("--max-incorrect-examples cannot be negative")
    return args


def main() -> None:
    args = _parse_args()
    _validate_api_key()
    model = _openrouter_model(args.model)
    dataset_revision = args.revision or _latest_dataset_version_tag(args.dataset_repo)
    dataset_commit = _resolve_dataset_commit(args.dataset_repo, dataset_revision)
    print(f"Dataset: {args.dataset_repo}@{dataset_revision} ({dataset_commit[:12]})")
    output_dir = (
        args.output_root
        / _model_slug(model.removeprefix("openrouter/"))
        / _model_slug(dataset_revision)
    )
    if args.reasoning_effort is not None:
        output_dir /= f"reasoning-{_model_slug(args.reasoning_effort)}"
    results_path = output_dir / "results.jsonl"
    report_path = output_dir / "report.md"
    if args.overwrite:
        results_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)

    benchmark_rows = _load_benchmark(
        args.dataset_repo,
        dataset_revision,
        args.splits,
        args.limit_per_split,
    )
    split_benchmark_seconds: dict[str, float] = {}
    for split in args.splits:
        split_rows = [row for row in benchmark_rows if row.get("split") == split]
        if not split_rows:
            raise EvaluationError(f"No benchmark rows were loaded for split {split!r}.")
        latest_results = _load_latest_results(results_path)
        pending_rows = [
            row
            for row in split_rows
            if not _is_reusable_result(
                latest_results.get(str(row["id"]), {}),
                row,
                model,
                args.dataset_repo,
                dataset_commit,
                args.reasoning_effort,
                args.temperature,
                args.max_tokens,
            )
        ]
        print(
            f"{split}: evaluating {len(pending_rows):,} of {len(split_rows):,} rows "
            f"({len(split_rows) - len(pending_rows):,} resumed)."
        )

        completed_now = 0
        for batch in _chunks(pending_rows, args.batch_size):
            results = _call_rows(
                batch,
                model=model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                num_retries=args.num_retries,
                timeout=args.timeout,
                caching=args.cache,
                reasoning_effort=args.reasoning_effort,
            )
            for result in results:
                result["dataset_repo"] = args.dataset_repo
                result["dataset_revision"] = dataset_revision
                result["dataset_commit"] = dataset_commit
            _append_results(results_path, results)
            completed_now += len(results)
            print(f"  completed {completed_now:,}/{len(pending_rows):,}", end="\r", flush=True)
        if pending_rows:
            print()

        latest_results = _load_latest_results(results_path)
        split_results = [
            latest_results[str(row["id"])]
            for row in split_rows
            if str(row["id"]) in latest_results
        ]
        split_metrics = _metrics(split_results)
        split_benchmark_seconds[split] = _recorded_benchmark_seconds(split_results)
        print(
            f"{split} accuracy: {_percentage(split_metrics.accuracy)} "
            f"({split_metrics.correct}/{split_metrics.total})"
        )
        print(f"{split} benchmark time: {_format_duration(split_benchmark_seconds[split])}")

    latest_results = _load_latest_results(results_path)
    ordered_results = [
        latest_results[str(row["id"])]
        for row in benchmark_rows
        if str(row["id"]) in latest_results
    ]
    _write_report(
        report_path,
        splits=args.splits,
        dataset_repo=args.dataset_repo,
        dataset_revision=dataset_revision,
        dataset_commit=dataset_commit,
        generator_version=_generator_version(benchmark_rows),
        model=model,
        reasoning_effort=args.reasoning_effort,
        results=ordered_results,
        max_incorrect_examples=args.max_incorrect_examples,
        split_benchmark_seconds=split_benchmark_seconds,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print(f"Results: {results_path}")
    print(f"Report:  {report_path}")


if __name__ == "__main__":
    try:
        main()
    except EvaluationError as exc:
        raise SystemExit(f"Error: {exc}") from exc
