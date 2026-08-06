"""Zero-key protocol demonstration.

The output is structural evidence only. It deliberately contains no pass-rate
or negotiation-gain field because no model or NPU is involved.
"""

import hashlib
import json
from pathlib import Path

from config_ascendc import DATASET_CONFIG, EXPERIMENT_CONFIG
from negotiation.executors.simulated import DeterministicDemoExecutor
from negotiation.protocol import MultiAgentNegotiationFramework


def _dataset() -> list[dict]:
    root = Path(__file__).resolve().parents[3]
    path = root / DATASET_CONFIG["dataset_path"]
    data = json.loads(path.read_text(encoding="utf-8"))
    return [item for item in data if item.get("has_test_scripts")]


def _run_topology(problem: dict, k: int) -> tuple[dict, list[dict]]:
    n = EXPERIMENT_CONFIG["N"]
    framework = MultiAgentNegotiationFramework(
        N=n,
        K=k,
        alpha=EXPERIMENT_CONFIG["alpha"],
        gamma=EXPERIMENT_CONFIG["gamma"],
    )
    calls = {"proposal": 0, "refine": 0, "vote": 0}

    def propose(agent_id, _phase):
        calls["proposal"] += 1
        return (
            f'extern "C" __global__ __aicore__ void {problem["name"]}_a{agent_id}() {{}}'
        )

    def refine(agent, _alive, _weights):
        calls["refine"] += 1
        return agent.solution + f"\n// reviewed in round {framework.round}"

    def vote(agent, alive, top_k, weight_distribution):
        calls["vote"] += 1
        candidates = [other for other in alive if other.agent_id != agent.agent_id]
        candidates.sort(
            key=lambda other: (
                -weight_distribution[other.agent_id]["weight"],
                -weight_distribution[other.agent_id]["history_consistency"],
                other.agent_id,
            )
        )
        return [other.agent_id for other in candidates[:top_k]]

    winner = framework.run(propose, refine, vote)
    executor = DeterministicDemoExecutor()
    interface_ok, _ = executor.execute_test(problem["name"], winner.solution, problem)
    topology = "K=N" if k == n else f"K={k}"
    return {
        "topology": topology,
        "completed": True,
        "winner": winner.agent_id,
        "rounds": framework.round,
        "calls_by_phase": calls,
        "executor_interface_ok": interface_ok,
        "trace_sha256": hashlib.sha256(
            json.dumps(framework.history, sort_keys=True, default=float).encode("utf-8")
        ).hexdigest(),
    }, framework.history


def _write_replay(path: Path, problem: str, topology: str, history: list[dict]):
    lines = [
        f"# Protocol replay: {problem} / {topology}",
        "",
        "> Demo-only trace. It proves protocol execution, not model quality or NPU correctness.",
        "",
    ]
    for item in history:
        lines.extend([
            f"## Round {item.get('round')}",
            f"- Phase: {item.get('phase')}",
            f"- Action: {item.get('action')}",
            f"- Winner: {item.get('winner', '-')}",
            f"- Eliminated: {item.get('eliminated_agent', '-')}",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_demo(num: int = 5, out_dir: str = "artifacts"):
    output = Path(out_dir) / "demo"
    output.mkdir(parents=True, exist_ok=True)
    problems = _dataset()[:num]
    if not problems:
        raise RuntimeError("No executable representative tasks were found")

    records = []
    replay_written = False
    n = EXPERIMENT_CONFIG["N"]
    for problem in problems:
        for k in (n, 1, 2):
            record, history = _run_topology(problem, k)
            record["problem"] = problem["name"]
            records.append(record)
            if not replay_written:
                _write_replay(output / "protocol_replay.md", problem["name"], record["topology"], history)
                replay_written = True

    result = {
        "evidence_tier": "demo_only",
        "performance_claim_allowed": False,
        "note": "No real model or NPU was used; do not derive negotiation gain from this file.",
        "tasks": len(problems),
        "runs": records,
    }
    result_path = output / "protocol_results.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Protocol demo completed: {len(records)} runs")
    print(f"Structural evidence only; no performance gain was calculated: {result_path}")
