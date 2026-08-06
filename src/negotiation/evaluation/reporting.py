"""Result normalization, snapshots, and failure taxonomy."""

import hashlib
import json
import math
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


def classify_failure(output: str, error: Exception | None = None) -> str:
    text = f"{type(error).__name__ if error else ''} {error or ''} {output or ''}".lower()
    if "budget" in text:
        return "budget_exhausted"
    if "precision" in text and "passed precision" not in text:
        return "precision_failure"
    if "timeout" in text:
        return "timeout"
    if "ssh" in text or "connection" in text:
        return "infrastructure"
    if "compile" in text or "error:" in text:
        return "compile_failure"
    if error:
        return "runtime_exception"
    return "verification_failure"


def result_record(result, *, problem: str, mode: str, call_budget: int,
                  npu_executions: int, error: Exception | None = None) -> dict:
    if result is None:
        return {
            "problem": problem,
            "mode": mode,
            "passed": False,
            "call_budget": call_budget,
            "llm_calls": call_budget if "budget" in str(error).lower() else None,
            "npu_executions": npu_executions,
            "failure_type": classify_failure("", error),
            "error": f"{type(error).__name__}: {error}",
        }
    raw = asdict(result)
    return {
        "problem": problem,
        "mode": mode,
        "passed": result.passed,
        "call_budget": call_budget,
        "llm_calls": result.total_llm_calls,
        "npu_executions": npu_executions,
        "rounds": result.total_rounds,
        "elapsed_seconds": round(result.execution_time, 3),
        "winner_agent_id": result.winner_agent_id,
        "failure_type": None if result.passed else classify_failure(result.test_output),
        "test_output_excerpt": result.test_output[:1000],
        "result": raw,
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return [round(max(0, centre - margin), 4), round(min(1, centre + margin), 4)]


def summarize(records: list[dict]) -> dict:
    summary = {}
    for mode in sorted({row["mode"] for row in records}):
        rows = [row for row in records if row["mode"] == mode]
        passed = sum(bool(row["passed"]) for row in rows)
        calls = [row["llm_calls"] for row in rows if row.get("llm_calls") is not None]
        summary[mode] = {
            "tasks": len(rows),
            "passed": passed,
            "pass_rate": round(passed / len(rows), 4) if rows else 0.0,
            "pass_rate_wilson_95": wilson_interval(passed, len(rows)),
            "avg_llm_calls": round(sum(calls) / len(calls), 2) if calls else None,
            "avg_npu_executions": round(
                sum(row["npu_executions"] for row in rows) / len(rows), 2
            ) if rows else None,
            "failure_types": {
                failure: sum(row.get("failure_type") == failure for row in rows)
                for failure in sorted({row.get("failure_type") for row in rows if row.get("failure_type")})
            },
        }
    return summary


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def build_snapshot(root: Path, dataset: Path, config: dict) -> dict:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dataset": str(dataset.relative_to(root)),
        "dataset_sha256": file_sha256(dataset),
        "experiment": config,
        "secrets_recorded": False,
    }
