import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/evaluate_openrouter.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("evaluate_openrouter", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
evaluate_openrouter = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = evaluate_openrouter
SCRIPT_SPEC.loader.exec_module(evaluate_openrouter)


def _benchmark_row():
    return {
        "id": "easy-family-mode-0000",
        "split": "easy",
        "family": "family",
        "mode": "mode",
        "prompt": "Prompt",
        "answer": "Answer",
        "metadata": "{}",
    }


@pytest.mark.parametrize(
    ("extra_args", "expected_config"),
    [
        ([], "n64"),
        (["--dataset-config", "n16"], "n16"),
    ],
)
def test_parse_args_selects_dataset_configuration(monkeypatch, extra_args, expected_config):
    monkeypatch.setattr("sys.argv", ["evaluate_openrouter.py", "provider/model", *extra_args])

    args = evaluate_openrouter._parse_args()

    assert args.dataset_config == expected_config


def test_load_benchmark_passes_configuration_to_hugging_face(monkeypatch):
    calls = []

    def fake_load_dataset(repo, config, *, revision):
        calls.append((repo, config, revision))
        return {"easy": [_benchmark_row()]}

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))

    rows = evaluate_openrouter._load_benchmark(
        "owner/benchmark",
        "n16",
        "v0.3.0",
        ["easy"],
        None,
    )

    assert calls == [("owner/benchmark", "n16", "v0.3.0")]
    assert rows == [_benchmark_row()]


def test_resume_requires_matching_dataset_configuration():
    row = _benchmark_row()
    result = {
        "status": "ok",
        "model_requested": "openrouter/provider/model",
        "dataset_repo": "owner/benchmark",
        "dataset_config": "n64",
        "dataset_commit": "abc123",
        "reasoning_effort": None,
        "temperature": 0.0,
        "max_tokens": 1024,
        "prompt_sha256": evaluate_openrouter._prompt_hash(row["prompt"]),
        "expected_answer": row["answer"],
    }

    assert evaluate_openrouter._is_reusable_result(
        result,
        row,
        "openrouter/provider/model",
        "owner/benchmark",
        "n64",
        "abc123",
        None,
        0.0,
        1024,
    )
    assert not evaluate_openrouter._is_reusable_result(
        result,
        row,
        "openrouter/provider/model",
        "owner/benchmark",
        "n16",
        "abc123",
        None,
        0.0,
        1024,
    )
