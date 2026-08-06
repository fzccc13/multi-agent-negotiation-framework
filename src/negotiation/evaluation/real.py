"""Real-model and Ascend 310B4 call-budget-capped evaluation."""

import json
from pathlib import Path

from config_ascendc import AGENT_MODELS, DATASET_CONFIG, EXECUTION_CONFIG, EXPERIMENT_CONFIG
from experiment_ascendc import CallBudgetExceeded, ExperimentRunner
from .reporting import build_snapshot, result_record, summarize


def run_real_evaluation(*, num: int, out_dir: str, call_budget: int,
                        max_fix_rounds: int, homogeneous_agent: int,
                        resume: bool = False):
    if not 0 <= homogeneous_agent < EXPERIMENT_CONFIG["N"]:
        raise ValueError("homogeneous_agent is outside the configured Agent range")
    if call_budget < EXPERIMENT_CONFIG["N"]:
        raise ValueError("call_budget must allow at least one proposal per negotiation role")

    root = Path(__file__).resolve().parents[3]
    output = Path(out_dir) / "real"
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw_results.jsonl"
    if raw_path.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite {raw_path}; use --resume or a new --out-dir")

    dataset_path = root / DATASET_CONFIG["dataset_path"]
    all_data = json.loads(dataset_path.read_text(encoding="utf-8"))
    executable = [item for item in all_data if item.get("has_test_scripts")]
    problems = executable[:num]
    if not problems:
        raise RuntimeError("No representative task has executable hardware tests")

    model = AGENT_MODELS[homogeneous_agent]
    snapshot = build_snapshot(root, dataset_path, {
        "evidence_tier": "real_hardware",
        "design": "homogeneous-model protocol ablation under an equal LLM-call cap",
        "model": model["model"],
        "homogeneous_agent": homogeneous_agent,
        "call_budget_per_task_mode": call_budget,
        "max_fix_rounds": max_fix_rounds,
        "tasks": [item["name"] for item in problems],
        "modes": ["baseline", "K=N", "K=1", "K=2"],
        "limitation": "Call cap is matched; token and monetary cost are reported only when provider usage is available.",
    })
    (output / "experiment_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    existing = []
    completed = set()
    if resume and raw_path.exists():
        existing = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line]
        completed = {(row["problem"], row["mode"]) for row in existing}

    EXECUTION_CONFIG["output_dir"] = str(output / "runner_artifacts")
    runner = ExperimentRunner(use_mock_llm=False, simulate=False)
    runner.dataset = problems
    runner.max_fix_rounds = max_fix_rounds
    runner.llm.force_agent_id = homogeneous_agent
    if runner.llm.use_mock:
        raise RuntimeError("Real LLM client initialization failed; refusing to record mock results as real")
    if not runner.connect_npu():
        raise RuntimeError("Ascend 310B4 connection failed")

    records = list(existing)
    try:
        for problem in problems:
            mode_specs = [
                ("baseline", None),
                ("K=N", EXPERIMENT_CONFIG["N"]),
                ("K=1", 1),
                ("K=2", 2),
            ]
            for mode, k in mode_specs:
                if (problem["name"], mode) in completed:
                    continue
                runner.llm.call_count = 0
                runner.llm.call_budget = call_budget
                before_exec = runner.executor.execution_count
                result = None
                error = None
                try:
                    if mode == "baseline":
                        result = runner.run_baseline(problem, single_agent=homogeneous_agent)
                    else:
                        result = runner.run_negotiation(problem, K=k)
                except CallBudgetExceeded as exc:
                    error = exc
                except Exception as exc:
                    error = exc
                row = result_record(
                    result,
                    problem=problem["name"],
                    mode=mode,
                    call_budget=call_budget,
                    npu_executions=runner.executor.execution_count - before_exec,
                    error=error,
                )
                records.append(row)
                with raw_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    finally:
        runner.disconnect_npu()

    report = {
        "evidence_tier": "real_hardware",
        "snapshot": "experiment_snapshot.json",
        "raw_results": "raw_results.jsonl",
        "summary": summarize(records),
        "interpretation_rule": (
            "Report deltas only with the task count, confidence interval, average calls, "
            "and NPU executions. Do not claim a causal gain from fewer than 30 tasks."
        ),
    }
    report_path = output / "summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Real evaluation complete: {report_path}")
