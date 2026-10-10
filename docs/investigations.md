# Investigations and evidence quality

Flameox supplies bounded evidence and experiment structure. The agent owns the
hypothesis, interpretation, and narrative outside Flameox.

## Investigation loop

```text
symptom → explicit artifact or capture → bounded evidence → hypothesis
        → discriminating experiment → supported, refuted, or inconclusive finding
```

An analysis result identifies the operation/provider, exact input digests,
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

MCP capture tools run the target once by default. Capabilities whose analysis intentionally composes
multiple artifacts also accept an optional top-level `experiment` design. Single-artifact analyses
do not advertise that field and reject experiments before a target starts. The
CLI accepts the same `ExperimentDesign` object through `capture --experiment JSON` when the chosen
operation supports it. An experiment declares cases, blocks, seed, metric, estimand, practical
threshold, and an optional semantic oracle. Cases are bounded and execute through the same broker
as a single capture.

For GPU kernel work, the agent normally compiles and edits with its native coding tools, records
correctness through `inspect_kernel_validation` and `compare_kernel_validation` analysis, checks hazards with a
`inspect_sanitizer_failures` capture, measures representative baseline/candidate cases with a
`summarize_benchmarks` experiment, and profiles only the remaining uncertainty with `inspect_gpu_launches` or
`inspect_gpu_kernel_metrics`. Flameox preserves the verification evidence;
it does not generate kernels, wrap compilers, or decide which optimization to implement.

The runtime accepts `wall_time_ns` and paired `median_difference` or
`mean_difference`. This metric is the complete invoked capture process's wall time, including
collector startup and overhead. Native per-sample benchmark measurements remain separate evidence.
Each non-baseline case is compared with the first declared case within the same blocks.
Failed or oracle-invalid pairs are excluded and
reported as limitations; fewer than three eligible pairs produce a descriptive
estimate without a confidence interval. `point_estimate_classification` describes only the
observed estimate against the practical margin, with `decision_basis=descriptive_point_estimate`
on the experiment metrics block. It replaces the ambiguous `decision` field. `within_threshold`
does not establish equivalence; a wide interval may still span material improvement and regression.
The deterministic percentile interval is reported separately and is not a calibrated equivalence
test. Semantic correctness still requires the declared oracle.

Artifact comparison is separate from that experiment result. Capture the representative baseline
and candidate summaries independently, preserve them if they must survive the session, then submit
both sources to the named `compare_benchmarks`, `compare_inference`, or
`compare_kernel_validation` tool. Comparison requires explicit artifact identity. An experiment
owns randomized case order and repeated measurements within one request.

## Scaling

`analyze_benchmark_scaling` is distinct from both workflows. The caller names a declared numeric benchmark
dimension such as `elements`; Flameox groups positive measurements for each compatible benchmark
series while retaining all non-axis dimensions, averages repeated samples at each input value,
and fits
`measurement = coefficient * input ** exponent` in log space. The result reports the observed
input range, point count, exponent, and goodness of fit. A series with fewer than two distinct
positive input values is explicitly inconclusive rather than falling back to a generic benchmark
summary. The fit describes the measured range; it does not establish asymptotic complexity or a
causal performance change.

Scaling uses stable positive means to avoid overflow from adding finite samples before averaging.
It omits measurements outside its finite numeric range and reports the omission; a series without
enough remaining points is inconclusive. Benchmark comparison returns a typed `LIMIT_EXCEEDED`
when aggregate totals or derived ratios exceed the finite numeric range. Native exact integer
samples remain unchanged.

Comparison and scaling aggregate native samples into per-series summaries inside their
bounded readers. The sample population may therefore exceed the result-row page size without being
silently clipped. The independent native sample and semantic-series safety ceilings still apply.

## Comparison compatibility

Benchmark and kernel comparisons join only complete semantic identities. Units, dimensions,
timing protocols, devices, dtypes, shapes, scopes, phases, and other declared identity axes are not
pooled merely because a metric name matches. The result reports identities absent from one or more
inputs rather than manufacturing a ratio across them.

Benchmark comparison and scaling share the same semantic series identity:
metric, unit, dimensions, scope, phase, loop count, worker, and variant. Scaling
removes only its requested input dimension from that identity. Trial IDs, block
IDs, order within a block, and worker run indices identify repeated samples;
they remain in native evidence and do not split otherwise compatible series.

Inference exports expose digest-only workload and system identities when their native format
provides enough metadata. Known-different identities fail validation by default. A caller may set
`allow_heterogeneous=true` for an explicitly exploratory ratio; the result labels that comparison
`heterogeneous` and retains the differing axes. Missing identity axes produce `partial`, not a
claim of compatibility. Prompts, generations, endpoints, and raw model or dataset names remain
excluded from normalized evidence. Failed or cancelled AIPerf records may omit token metrics;
unknown token totals remain null, and incomplete token populations do not establish a workload
identity. Successful records still require their token metrics.

## Pytest fixture work

`inspect_pytest_fixtures` records fixture setup and finalizer phases without serializing fixture values.
Each invocation retains its fixture name, scope, test identity, worker, phase outcome, and measured
duration. Session-scoped fixtures therefore appear once per xdist worker, rather than being
mistaken for one global invocation. Aggregate work is summed evidence and can overlap across
workers; Flameox reports the maximum accumulated worker work separately and does not label either
quantity as wall-clock critical-path duration. Interrupted runs preserve incomplete invocations
instead of assigning missing finalizers a zero duration.

Session completion also retains pytest's nonzero `run_finished.exitstatus`. Repeated phase reports
remain an ordered attempt history; a failed attempt followed by a passing attempt is reported as
flaky rather than allowing the later report to erase the failure. Attempts are grouped by
worker and test identity so replicated xdist executions do not become false retries. Collected
test counts describe logical test identities; executed counts include worker replicas. Test
identities longer than the event bound retain a prefix and a digest of the complete identity.

The capture plugin is scoped to the owned pytest invocation and propagated through pytest arguments
to xdist workers; it does not export plugin state that would instrument nested pytest processes.
A completed timing callback proves only that teardown was observed. Fixture success or failure is
combined with pytest's teardown report, and attribution that pytest cannot identify precisely
remains incomplete rather than being reported as a successful finalizer.

Randomization and blocking reduce ordering and environmental bias; they do not
make an unrepresentative workload representative. Failed and partial trials are
evidence and must stay visible in the returned episode. A semantic oracle checks
behavioral equivalence; benchmark timing is not an oracle.

## Identity and preservation

`analysis_id` exists only to preserve or rescue one result during the current process. It
may expire earlier when the bounded session cache evicts it and must not appear in a durable claim.
`evidence_id` binds the effective request,
provider/input identity, data files, coverage, limitations, and episode time.

Preservation is optional but required for conclusions another person or agent
must reproduce later. `inspect_evidence` returns a redacted projection of the canonical manifest inline;
native payloads remain local files addressed by digest.

## Comparisons

Kernel-validation producers retain authority over their declared verdicts, but Flameox checks their
internal consistency. Structurally ambiguous documents, including duplicate semantic measurements,
are rejected. Status or comparator contradictions remain visible as bounded consistency failures and
make the derived summary inconclusive; producer statuses and native JSON remain unchanged. Cases with
no outputs still produce case evidence rows.

Comparison handlers accumulate member identities in dictionaries and test each
incoming identity directly against existing keys. They must not rebuild the
accumulated key set for every member. Large derived tables belong in immutable
evidence data files; request-local DuckDB may aggregate them without becoming an
authority.
