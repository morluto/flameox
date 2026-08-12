# Semantic outcome matrix

This standard-library-only example shows how a bounded Flameox experiment keeps
categorical semantic outcomes separate from process success. Its eight explicit
cells cover matching reference/candidate behavior, a typed candidate mismatch,
an expected rejection that counts as a pass, and an intentionally unavailable
backend that stays unsupported.

From this directory, initialize the workspace. The valid workload in
`flameox.toml` is immediately authoritative; there is no approval copy. Then
create and run an investigation:

```console
flameox init
flameox workload show semantic --json
flameox investigations create '{"question":"Does the candidate preserve the declared semantic contract?"}' --json
flameox experiment plan semantic_matrix --investigation <investigation-id> --adapter command --json
flameox experiment run semantic_matrix --investigation <investigation-id> --adapter command --json
```

Use `flameox experiment trial <trial-id> --json` to inspect the first failure.
Its `run_id` leads to `flameox runs show <run-id> --json`; the run manifest names
the authoritative `validation_receipt` artifact, which can be inspected with
`flameox artifacts show <artifact-id> --json`. The MCP equivalents are the
experiment protocol, `flameox://experiments/{experiment_id}/trials`, and the
first-failure trial resource returned by `run_experiment`.

A finding can cite that immutable trial directly:

```console
flameox findings record '{"kind":"semantic_mismatch","title":"Candidate forward output differs","claim":"The candidate differs in the declared mismatch cell.","evidence_level":"derived","confidence":"high","assessment":"supported","evidence":[{"ref_type":"trial","ref_id":"<trial-id>","relation":"supports"}]}' --json
```

The process exit code and raw receipt bytes are observed facts. Flameox parses
the receipt fields and derives validation/trial classifications. Any claim that
the mismatch explains a broader application symptom remains an inference and
needs a representative experiment.

For replay bookkeeping, retain the experiment ID, first-failure trial ID and
factor map, workload definition, source and environment identities, receipt
artifact identity, and any manual commands or edits used to reconstruct the
cell. This example is one authored trace, not evidence of repeated replay
friction; it therefore does not justify an automatic replay API. To expand the
matrix, change `combination_policy` to `cartesian`, remove `combinations`, and
keep `max_trials` at a deliberately reviewed bound.
