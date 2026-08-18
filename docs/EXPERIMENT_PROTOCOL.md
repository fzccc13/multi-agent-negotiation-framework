# Real evaluation protocol

## Research question

Under a fixed logical LLM-call cap, does the negotiation workflow improve AscendC hardware-validation outcomes over a single-model self-refinement baseline?

## Controlled variables

- Same task set and task order.
- Same underlying model for baseline and every negotiation role (`--homogeneous-agent`).
- Same maximum logical LLM calls per task and mode (`--call-budget`).
- Same CANN environment, Ascend 310B4 device, test scripts, precision threshold, and repair-round cap.
- Fresh output directory, or continuation from recorded checkpoints with the CLI checkpoint option.

## Compared modes

- `baseline`: one selected model generates a solution and repairs it from compiler/runtime feedback.
- `K=N`: the protocol enters the Best-1 endgame immediately because the alive-candidate count is already at the threshold.
- `K=1`: Top-K elimination continues until one candidate remains or a candidate crosses the winner threshold.
- `K=2`: Top-K elimination continues until two candidates remain, then switches to the Best-1 endgame.

`K` is a phase-transition threshold, not the number of participating Agents. All five logical roles participate at initialization in every negotiation configuration.

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

The public repository provides a small representative set of executable projects for end-to-end verification and failure analysis. These examples should not be treated as statistical proof. A quantified comparison should be published only after expanding the executable task set and preserving the complete raw artifacts for independent verification.
