# Real evaluation protocol

## Research question

Under a fixed logical LLM-call cap, does the negotiation workflow improve AscendC hardware-validation outcomes over a single-model self-refinement baseline?

## Controlled variables

- Same task set and task order.
- Same underlying model for baseline and every negotiation role (`--homogeneous-agent`).
- Same maximum logical LLM calls per task and mode (`--call-budget`).
- Same CANN environment, Ascend 310B4 device, test scripts, precision threshold, and repair-round cap.
- Fresh output directory, or explicit `--resume` from recorded checkpoints.

## Compared modes

- `baseline`: one selected model generates a solution and repairs it from compiler/runtime feedback.
- `K=N`: five logical roles use full-consensus topology; all roles call the same selected model in this ablation.
- `K=1`: single-expert monitoring topology with the same model behind every role.
- `K=2`: dual-expert monitoring topology with the same model behind every role.

This homogeneous setup isolates workflow effects better than mixing GLM, Qwen, and Kimi. A separate heterogeneous-system experiment may measure the best practical configuration, but must not be interpreted as a pure protocol ablation.

## Resource accounting

Each result records:

- Logical LLM-call cap and actual logical calls.
- NPU compile/run executions.
- Wall-clock duration.
- Repair and negotiation rounds.
- Failure category and raw output excerpt.

Logical calls are not identical to provider requests: an SDK retry can issue another request inside one logical call. Token and monetary comparisons require provider usage metadata and are not yet implemented.

## Success criterion

A task passes only when the representative operator project compiles, executes on Ascend 310B4, and the test output contains `passed Precision` or case-insensitive `test pass`.

## Reporting rule

- Keep `raw_results.jsonl` and `experiment_snapshot.json` with every report.
- Report task count, pass rate, Wilson 95% interval, average logical calls, and average NPU executions together.
- Do not publish a causal “negotiation gain” from fewer than 30 executable tasks.
- Treat infrastructure failures separately and publish both intention-to-test and infrastructure-clean sensitivity summaries when the sample is large enough.

## Current limitation

The public repository contains only five tasks with executable representative projects. Their results are useful for end-to-end verification and failure analysis, not statistical proof. Expanding the executable set is required before adding a quantified gain to a resume.
