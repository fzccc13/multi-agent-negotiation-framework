# Evaluation status

The public zero-key Demo is an engineering validation of the negotiation protocol. It verifies protocol termination, history recording, topology configuration, and executor-interface compatibility. It does not claim model-quality or hardware pass-rate improvements.

For a real hardware experiment, keep the generated artifacts together so that the reported results can be independently checked:

- `experiment_snapshot.json`
- `raw_results.jsonl`
- `summary.json`

Any reported metric should be reproducible from those files and accompanied by the task count, confidence interval, resource usage, and failure breakdown. The current representative hardware examples are intended for end-to-end validation and failure analysis; a quantified comparison requires a larger executable task set and matched experimental conditions.
