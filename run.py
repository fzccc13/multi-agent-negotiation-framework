"""Unified project command line."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _run(*args: str) -> int:
    return subprocess.call([sys.executable, *args], cwd=ROOT)


def dataset_info() -> int:
    path = ROOT / "ascend-ops-dataset" / "final" / "test.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    executable = [row for row in records if row.get("has_test_scripts")]
    print(f"Task records: {len(records)}")
    print(f"Hardware-executable representative tasks: {len(executable)}")
    print("Executable tasks: " + ", ".join(row["name"] for row in executable))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Agent Negotiation Framework")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run the zero-key protocol demo")
    demo.add_argument("--num", type=int, default=5)
    demo.add_argument("--out-dir", default="artifacts")
    real = sub.add_parser("real-eval", help="Run real-model + Ascend 310B4 evaluation")
    real.add_argument("--num", type=int, default=5)
    real.add_argument("--out-dir", default="artifacts")
    real.add_argument("--call-budget", type=int, default=50)
    real.add_argument("--max-fix-rounds", type=int, default=10)
    real.add_argument("--homogeneous-agent", type=int, default=2)
    real.add_argument("--resume", action="store_true")
    sub.add_parser("test", help="Run the automated test suite")
    sub.add_parser("dataset-info", help="Show corpus and executable-subset counts")
    args = parser.parse_args()

    if args.command == "demo":
        return _run("evaluate.py", "--mode", "demo", "--num", str(args.num),
                    "--out-dir", args.out_dir)
    if args.command == "real-eval":
        command = [
            "evaluate.py", "--mode", "real", "--num", str(args.num),
            "--out-dir", args.out_dir, "--call-budget", str(args.call_budget),
            "--max-fix-rounds", str(args.max_fix_rounds),
            "--homogeneous-agent", str(args.homogeneous_agent),
        ]
        if args.resume:
            command.append("--resume")
        return _run(*command)
    if args.command == "test":
        return _run("-m", "pytest", "-q")
    return dataset_info()


if __name__ == "__main__":
    raise SystemExit(main())
