# Investigations and analysis

This document defines how flameox turns captures into falsifiable investigations.
It covers representative workloads, experimental structure, recipe outputs,
comparison validity, and the boundary between exploratory and confirmatory
claims.

Storage fields and provenance are defined in
[Storage and evidence](storage-and-evidence.md). Commands and MCP tools are
defined in [CLI and MCP boundaries](interfaces.md).

## Primary workflows

### CPU hotspot investigation

1. Inspect local capabilities.
2. Plan a capture and show the exact command, tool, expected overhead, and
   required permissions.
3. Run the workload with `py-spy`, `perf`, or another installed adapter.
4. Register the native pprof, Speedscope, or perf artifact.
5. Extract aggregated self and inclusive frame measurements.
6. Return the top source-linked hotspots with coverage and limitations.
7. Preserve a command for opening the native artifact in an existing viewer.

### Scaling investigation

1. Define a workload and an explicit input parameter such as sequence length.
2. Run warm-ups.
3. Measure balanced randomized blocks across variants and inputs.
4. retain raw samples, not just averages;
5. report medians, dispersion, effect sizes, and environmental metadata;
6. fit only declared candidate growth models;
7. identify the source regions whose cost grows with the input;
8. label complexity conclusions as inferred unless the source mechanism is
   independently established.

### Before/after validation

1. Execute the same workload against baseline and candidate source roots,
   installations, or commands supplied by the caller; flameox does not switch the
   working tree between revisions.
2. Validate behavior before accepting performance evidence.
3. Compare benchmark distributions.
4. Compare profiles or traces to explain the change.
5. Report both improvements and regressions.
6. Save a finding whose evidence references are sufficient to reproduce the
   result.

### PyTorch operator and accelerator investigation

1. Capture CPU operators and the available accelerator activity through
   `torch.profiler`.
2. Record whether stacks, shapes, modules, memory, and FLOPs were enabled.
3. Export a Perfetto-compatible trace.
4. Query operator duration, call count, synchronization, idle gaps, repeated
   operators, and memory changes.
5. Keep warm-up and compilation phases separate from steady state.
6. Report profiler overhead and missing accelerator data.

### Memory-growth investigation

1. Capture repeated workload phases using Memray or an appropriate native
   memory tool.
2. distinguish peak allocation, retained allocation, allocation volume, and
   resident memory;
3. compare stacks across phases or runs;
4. identify growth correlated with a workload dimension;
5. retain the Memray binary and link to its supported reports.

### Population-level failure investigation

Core-dump and crash-report ingestion is outside the supported adapter set, but
population-level analysis remains a first-class architectural requirement.

1. Import core dumps, sanitizer reports, or crash summaries.
2. Extract deterministic features such as signal, fault address class, register
   properties, stack signature, modules, build IDs, host attributes, and
   symbolization quality.
3. group and compare crash cohorts;
4. expose representative artifacts from each cluster;
5. preserve multiple plausible clusters instead of forcing one signature;
6. run change-point and dimensional analyses;
7. record interventions and recurrence observations.

## Workload and experiment specification

### Project configuration

Repeatable workloads live in `flameox.toml` at the project root:

```toml
schema_version = 1

[workloads.gae]
argv = ["python", "-m", "benchmarks.gae", "--length", "{length}"]
cwd = "."
timeout_seconds = 300

[workloads.gae.parameters]
length = [4096, 8192, 16384, 32768, 65536]

[workloads.gae.oracle]
strength = "cross_treatment_equivalence"
argv = ["python", "-m", "tests.validate_gae", "--length", "{length}"]

[experiments.gae_scaling]
workload = "gae"
variants = ["baseline", "candidate"]
design = "randomized_complete_blocks"
blocks = 10
primary_metric = "benchmark.wall_time"
polarity = "lower_is_better"
estimand = "median_paired_log_ratio"
practical_threshold = 0.05
confidence_level = 0.95
random_seed = 1984
```

Template substitution is limited to declared scalar parameters. It does not
perform shell interpolation, command substitution, environment expansion, or
path globbing.

### Repetition ordering

Confirmatory before/after comparisons use balanced randomized complete blocks.
Every complete block contains each treatment exactly once and randomizes its
within-block order:

```text
block 1: candidate, baseline
block 2: baseline, candidate
block 3: candidate, baseline
```

This controls temporal drift while preserving a valid paired unit. The seed and
realized order are recorded. An arbitrary randomized sequence is not described
as paired. Fixed-order or incomplete designs remain available for exploratory
work but their limitations and independent unit are explicit.

The experiment declares its primary metric, polarity, estimand, practical
threshold, confidence level, sample or stopping rule, and validation oracle
before confirmatory collection. Adaptive stopping is allowed only when its rule
and statistical method are predeclared. Profiling and operator scans may select
a candidate mechanism, but the confirmation experiment uses fresh measurements.

### Validation oracle

Performance evidence is not sufficient to establish semantic preservation.
Supported validation forms:

- command exits successfully;
- output artifact hashes match;
- structured numeric outputs satisfy declared tolerances;
- an existing test target passes;
- a user-supplied JSON result follows a declared schema.

Oracle strength is explicit:

- `execution_check`: the validation action completed successfully;
- `contract_check`: each treatment independently meets a declared contract;
- `cross_treatment_equivalence`: baseline and candidate outputs are compared
  under a predeclared schema and tolerances.

Only cross-treatment equivalence directly supports an output-preservation
claim. Numeric tolerances, excluded fields, canonicalization, input domain, and
comparison direction are declared before execution. A test pass can still be
valuable without being mislabeled as equivalence proof.

Validation executes without the profiler unless the recipe explicitly requires
instrumented behavior. A candidate that fails validation cannot be described as
a successful optimization. Failed, errored, or unavailable validation remains
visible on the trial and cannot be removed by excluding its performance sample.

### Semantic observations

Some important defects are configuration or algorithmic invariant violations,
not conventional hotspots. The optional SDK supports explicit annotations:

```python
from flameox.sdk import observe, phase

with phase("ppo_epoch"):
    observe("policy.old_log_prob_source", source="rollout")
    observe("policy.clip_fraction", value=clip_fraction)
```

Annotations must serialize to bounded primitive values and may be exported as
Perfetto events or a structured observation artifact. This provides agents with
evidence such as whether a branch executed, which policy snapshot supplied a
value, or how many times an update path ran.

## Analysis recipes

### Recipe contract

A recipe declares:

- required capabilities;
- accepted workload and artifact types;
- capture plans;
- validation requirements;
- deterministic queries;
- comparison rules;
- result schema;
- persisted analysis-record inputs and output digest;
- evidence limitations;
- structured suggested next experiments.

Recipes orchestrate adapters but cannot access MCP or CLI presentation.

### `cpu_hotspots`

Returns:

- top exclusive and inclusive frames;
- source locations;
- bounded callers, callees, and representative stacks for each hotspot;
- percentage of captured samples represented;
- symbolization and sample coverage;
- thread/process filters;
- native artifact and viewer references.

### `scaling`

Returns:

- attempted trials and raw measurement hierarchy by input;
- median and dispersion by input;
- environmental stability indicators;
- candidate model fits, intercepts, residuals, uncertainty, and diagnostics;
- input-correlated frames or operators;
- the range over which the observation is supported;
- explicit warnings against extrapolating beyond measured sizes.

Candidate models are selected by the recipe, not discovered through arbitrary
formula search. Default candidates may include constant, logarithmic, linear,
`n log n`, and quadratic growth. A fit is evidence about observed scaling, not
proof of asymptotic complexity. A recipe may conclude
`indistinguishable` or `inconclusive`; it must not choose a winner solely from
the largest R².

### `compare_run_sets`

Returns:

- compatibility report;
- frozen baseline and candidate run-set definitions;
- attempted, eligible, failed, and paired trial counts;
- metric changes under the predeclared estimand;
- top regressed and improved frames/operators;
- validation status;
- profiler-configuration differences;
- evidence references and limitations.

A pair of run IDs is accepted as shorthand for two one-element run sets but
cannot supply population-level confidence without independent replication.

### `pytorch_operator_breakdown`

Returns:

- operator calls, self time, total time, and device time;
- shapes only when captured;
- CPU/accelerator synchronization indicators;
- compilation and warm-up separation;
- repeated small operations;
- memory allocation summaries;
- trace coverage.

### `memory_growth`

Returns:

- high-water mark, live-at-end allocations, total allocation volume, and RSS
  only where the collector supports each concept;
- allocation volume and counts;
- top retained stacks;
- phase or input-correlated growth;
- native versus Python allocation coverage;
- limitations around RSS and allocator caching.

### `execution_path`

Returns:

- declared source lines and branch arcs that executed;
- counts and dynamic contexts when available;
- explicit semantic observations emitted by the SDK;
- relevant configuration values captured by policy;
- comparison of path or observation changes across two runs;
- gaps where value provenance remains unknown.

Execution-path evidence can disprove claims such as "this branch never ran." It
cannot, without additional observations or source reasoning, prove that a
particular value caused the branch.

### `failure_population`

Returns:

- deterministic groups and their sizes;
- dimensions enriched in each group;
- change points;
- representative artifacts;
- data-quality and symbolization coverage;
- competing hypotheses rather than one forced conclusion.

## Statistical and comparison policy

flameox must preserve raw measurements and avoid presenting one aggregate as the
entire experiment.

Default reporting includes:

- attempted, eligible, excluded, failed, and independent sample counts;
- median;
- minimum and maximum;
- median absolute deviation or another declared robust dispersion measure;
- confidence interval for the declared estimand;
- relative and absolute change;
- practical threshold configured by the workload.

Statistical significance is not sufficient by itself, and failure to reject a
null hypothesis is not evidence of equivalence. Findings include effect size,
practical impact, and a decision from:
`meaningful_improvement`, `meaningful_regression`,
`no_meaningful_difference`, `inconclusive`, or `descriptive_only`.
`no_meaningful_difference` requires a predeclared equivalence or interval
criterion; otherwise an interval crossing the practical threshold is
`inconclusive`. Small samples, incomplete blocks, multimodal distributions,
excessive variance, thermal drift, and background load are reported when
observable.

flameox delegates benchmark collection, calibration, warm-up, and instability
metadata to pyperf through public APIs. It does not use pyperf's private
programmatic comparison implementation. The default paired comparison uses a
specified SciPy bootstrap method over the block-level estimand, such as the
median paired log ratio, with the confidence method, library version, seed, and
complete-pair count recorded. Scaling fits use maintained statistical
libraries, with statsmodels added only when its diagnostics are required. Every
method is named, versioned, fixture-tested, and persisted in the comparison.

Exploratory scans across many metrics or frames record the tested family and
any multiplicity adjustment. Their results remain exploratory. The primary
confirmatory metric is selected before collection rather than after inspecting
the largest improvement.

## Output and evidence quality

Every analytical response includes:

- `schema_version`;
- operation and operation version;
- analysis ID and pinned corpus commit;
- run and artifact references;
- exact experiment/run-set inputs;
- parameters;
- results with units;
- evidence level;
- coverage;
- limitations;
- total and returned counts, truncation state, and stable cursor;
- suggested next experiments when appropriate.

Example:

```json
{
  "schema_version": 1,
  "analysis": "cpu_hotspots",
  "analysis_version": 1,
  "analysis_id": "...",
  "corpus_commit_id": "...",
  "run_id": "...",
  "artifact_id": "sha256:...",
  "coverage": {
    "samples": 5021,
    "symbolized_fraction": 0.97,
    "native_frames": true
  },
  "hotspots": [
    {
      "frame_id": "...",
      "function": "compute_gae",
      "file": "reinforce/gae.py",
      "line": 118,
      "self_fraction": 0.71,
      "evidence_level": "derived"
    }
  ],
  "limitations": [
    "Sampling establishes where CPU time was observed, not semantic correctness."
  ],
  "truncated": false
}
```

Agent-facing prose may summarize this response, but the structured form is the
contract.

Analysis text never presents artifact-derived strings as instructions. It
quotes or labels them as untrusted evidence and strips terminal control
sequences from terminal rendering.
