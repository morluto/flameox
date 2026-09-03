# Investigations and evidence quality

Flameox supplies bounded evidence and experiment structure. The agent owns the
hypothesis, interpretation, and narrative outside Flameox.

## Investigation loop

```text
symptom → explicit artifact or capture → bounded evidence → hypothesis
        → discriminating experiment → supported, refuted, or inconclusive finding
```

An analysis result identifies the capability/provider, exact input digests,
typed evidence blocks, coverage, truncation, limitations, and an opaque
continuation. These fields distinguish what was observed from what an agent may
infer.

## Profiles and claims

Profiles rank where time or memory was observed. They do not prove causality,
semantic correctness, or an improvement. A confirmatory performance claim needs:

- a representative target and environment;
- a declared metric, unit, and estimand;
- compatible source and provider identities;
- preserved samples and effective requests;
- a practical threshold, not only statistical significance;
- an appropriate semantic oracle.

Coverage and limitations remain part of the evidence. Partial capture, provider
sampling, missing symbols, truncation, unsupported platform behavior, or absent
containment must not be silently promoted to complete evidence.

## Experiments

MCP capture tools accept `single` execution. Capabilities whose analysis intentionally composes
multiple artifacts also accept a discriminated `experiment` request. Single-artifact analyses do
not advertise experiment mode and reject it during runtime preflight before a target starts. The
CLI accepts the same `ExperimentDesign` object through `capture --experiment JSON` when the chosen
capability supports it. An experiment declares cases, blocks, seed, metric, estimand, practical
threshold, and an optional semantic oracle. Cases are bounded and execute through the same broker
as a single capture.

For GPU kernel work, the agent normally compiles and edits with its native coding tools, records
correctness through `analyze_kernel_validation` and `analyze_kernel_compare`, checks hazards with
`capture_sanitizer_failures`, measures representative baseline/candidate cases with
`capture_benchmark_summary` in experiment mode, and profiles only the remaining uncertainty with
`capture_gpu_launches` or `capture_gpu_kernel_metrics`. Flameox preserves the verification evidence;
it does not generate kernels, wrap compilers, or decide which optimization to implement.

The 0.2 runtime accepts `wall_time_ns` and paired `median_difference` or
`mean_difference`. Each non-baseline case is compared with the first declared
case within the same blocks. Failed or oracle-invalid pairs are excluded and
reported as limitations; fewer than three eligible pairs produce a descriptive
estimate without a confidence interval. The practical-threshold decision is
returned as typed comparison evidence, not retained only as request metadata.

Artifact comparison is separate from that experiment result. Capture the representative baseline
and candidate summaries independently, preserve them if they must survive the session, then submit
both sources to the matching `analyze_*_compare` tool. There is no `capture_*_compare` shortcut:
comparison requires explicit artifact identity, while experiment mode owns randomized case order
and repeated measurements within one request.

## Scaling

`benchmark.scaling` is distinct from both workflows. The caller names a declared numeric benchmark
dimension such as `elements`; Flameox groups positive measurements for each compatible benchmark
series, averages repeated samples at each input value, and fits
`measurement = coefficient * input ** exponent` in log space. The result reports the observed
input range, point count, exponent, and goodness of fit. A series with fewer than two distinct
positive input values is explicitly inconclusive rather than falling back to a generic benchmark
summary. The fit describes the measured range; it does not establish asymptotic complexity or a
causal performance change.

## Pytest fixture work

`pytest.fixtures` records fixture setup and finalizer phases without serializing fixture values.
Each invocation retains its fixture name, scope, test identity, worker, phase outcome, and measured
duration. Session-scoped fixtures therefore appear once per xdist worker, rather than being
mistaken for one global invocation. Aggregate work is summed evidence and can overlap across
workers; Flameox reports the maximum accumulated worker work separately and does not label either
quantity as wall-clock critical-path duration. Interrupted runs preserve incomplete invocations
instead of assigning missing finalizers a zero duration.

Randomization and blocking reduce ordering and environmental bias; they do not
make an unrepresentative workload representative. Failed and partial trials are
evidence and must stay visible in the returned episode. A semantic oracle checks
behavioral equivalence; benchmark timing is not an oracle.

## Identity and preservation

`analysis_id` exists only to preserve one result during the current process. It
may expire earlier when the bounded session cache evicts it and must not appear in a durable claim.
`evidence_id` binds the effective request,
provider/input identity, data files, coverage, limitations, and episode time.

Preservation is optional but required for conclusions another person or agent
must reproduce later. The evidence resource exposes the canonical manifest;
native payloads remain local files addressed by digest.

## Comparisons

Comparison handlers accumulate member identities in dictionaries and test each
incoming identity directly against existing keys. They must not rebuild the
accumulated key set for every member. Large derived tables belong in immutable
evidence data files; request-local DuckDB may aggregate them without becoming an
authority.
