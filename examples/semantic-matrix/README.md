# Semantic outcome matrix

This standard-library-only example shows why process success and semantic
outcomes are different evidence. Its explicit cells cover matching behavior, a
typed candidate mismatch, an expected rejection, and an unavailable backend.

Run each case as a typed target or send the cases together through MCP
`capture_and_analyze` in `experiment` mode. Declare the blocks, seed, metric,
estimand, practical threshold, and semantic-oracle argv in that request; no
workspace or configuration file is required.

```console
flameox capture --provider direct -- \
  python semantic_workload.py reference portable float32 contiguous 4 stateless ordinary
```

For durable review, preserve the returned session `analysis_id`. The resulting
manifest records the exact argv, input/output digests, episode timestamp,
coverage, limitations, and native artifact roles. Keep the agent's hypothesis
and interpretation in its own notes and cite the durable `evidence_id`.

The exit code and raw receipt bytes are observed facts. Parsed outcome fields are
derived evidence. A claim that one mismatch explains a broader application
symptom remains an inference requiring a representative experiment.
