"""Evaluation entry point with explicit evidence tiers.

``demo`` validates protocol flow without claiming model-quality gains.
``real`` runs a call-budget-capped ablation with a real model and Ascend 310B4.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from negotiation.evaluation.demo import run_demo
from negotiation.evaluation.real import run_real_evaluation


def main():
    parser = argparse.ArgumentParser(description="Multi-agent negotiation evaluation")
    parser.add_argument("--mode", choices=("demo", "real"), default="demo")
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--call-budget", type=int, default=50)
    parser.add_argument("--max-fix-rounds", type=int, default=10)
    parser.add_argument("--homogeneous-agent", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo(num=args.num, out_dir=args.out_dir)
        return
    run_real_evaluation(
        num=args.num,
        out_dir=args.out_dir,
        call_budget=args.call_budget,
        max_fix_rounds=args.max_fix_rounds,
        homogeneous_agent=args.homogeneous_agent,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
