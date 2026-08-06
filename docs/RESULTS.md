# Results status

No public real-hardware negotiation-gain result is currently claimed.

The zero-key Demo verifies protocol termination, history recording, topology configuration, and executor-interface compatibility. It intentionally does not output model-quality pass rates.

When a real run is completed, publish the generated files without manually rewriting the numbers:

- `experiment_snapshot.json`
- `raw_results.jsonl`
- `summary.json`

Any README or resume metric must be reproducible from those files. The current five-task executable subset is too small for a causal performance claim; retain results as engineering evidence until at least 30 representative tasks have executable tests.
