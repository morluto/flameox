# Storage and evidence contracts

This document defines flameox's authoritative local data, identity, provenance,
publication, and schema contracts. Changes here must preserve rebuildability:
native artifacts, JSON manifests, and committed Parquet generations are
authoritative; DuckDB is derived state.

[README.md](../README.md#scope) defines product scope.
[Acceptance and verification](acceptance.md) records completion criteria and
representative proof. [Runtime safety](runtime-safety.md) defines locking,
recovery, retention, and integrity behavior around this data.

## Workspace

### Discovery

flameox discovers a workspace by walking from the requested working directory
toward the filesystem root and selecting the nearest `.diagnostics` directory.
If none exists, commands that mutate state fail with a remediation suggesting
`flameox init`. Read-only capability discovery does not require initialization.

An explicit `--workspace` overrides discovery. The resolved workspace must be
inside the selected project root unless the user explicitly supplies an
external absolute path through the CLI. MCP tools cannot choose an arbitrary
external workspace, but artifact import explicitly permits the fixed project
root or the system temporary directory as bounded source roots.

In a Git repository, `flameox init` adds `.diagnostics/` to
`.git/info/exclude` when it is not already ignored. It does not edit the
project's tracked `.gitignore`. The shareable `flameox.toml` workload
configuration remains outside `.diagnostics` and may be committed deliberately.

### Directory layout

```text
.diagnostics/
├── workspace.json
├── config.toml
├── write.lock
├── catalog.lock
├── retention.lock
├── corpus/
│   ├── HEAD
│   └── commits/
│       └── <commit-id>.json
├── generations/
│   └── <generation-id>/
│       └── manifest.json
├── artifacts/
│   └── sha256/
│       └── ab/
│           └── abcd.../
│               ├── payload.pftrace
│               └── artifact.json
├── runs/
│   └── <run-id>/
│       ├── manifest.json
│       ├── revisions/
│       └── log.jsonl
├── evidence/
│   ├── runs/generation=<generation-id>/
│   ├── investigations/generation=<generation-id>/
│   ├── hypotheses/generation=<generation-id>/
│   ├── experiments/generation=<generation-id>/
│   ├── variants/generation=<generation-id>/
│   ├── trials/generation=<generation-id>/
│   ├── run_sets/generation=<generation-id>/
│   ├── artifacts/
│   ├── environments/
│   ├── source_states/
│   ├── measurements/
│   ├── frames/
│   ├── frame_measurements/
│   ├── observations/
│   ├── analyses/
│   ├── evidence_refs/
│   ├── findings/
│   └── comparisons/
├── catalog.duckdb
├── staging/
└── quarantine/
```

The split between `artifact.json` and `runs/<run-id>/manifest.json` is
intentional:

- the artifact directory is keyed by the payload's SHA-256 and contains only
  content-level immutable metadata;
- the run manifest contains invocation, environment, workload, and provenance;
- multiple runs may reference the same artifact without mutating it.

`corpus/HEAD` contains one commit ID. A corpus commit is immutable and lists
the exact generation manifests visible to a reader. Each generation manifest
lists exact Parquet paths, hashes, row counts, Arrow schema versions, input
artifacts and runs, extractor or publisher identity, an optional exact-operation
digest for idempotent extraction, and superseded
generations. A file not reachable from the pinned commit is not queryable,
even if it exists beneath `evidence/`.

```json
{
  "schema_version": 1,
  "commit_id": "sha256:...",
  "parent_commit_id": "sha256:...",
  "created_at": "RFC3339 timestamp",
  "generation_manifests": [
    "generations/<generation-id>/manifest.json"
  ],
  "inventory_digest": "sha256:..."
}
```

The commit ID is the digest of canonical commit content excluding
`commit_id`. The generation list is the complete active inventory, not an
instruction to scan the filesystem.

### Workspace identity

`workspace.json` contains:

```json
{
  "schema_version": 1,
  "workspace_id": "uuid4",
  "created_at": "RFC3339 timestamp",
  "project_root": "..",
  "flameox_version": "version at creation"
}
```

`workspace.json` is static workspace identity, not mutable corpus state. The
only publication authority is `corpus/HEAD`; there is no independently updated
generation counter that can disagree with it.

Paths stored in manifests are relative to the workspace or project root where
possible. Moving the project may invalidate cached catalog paths but must not
invalidate manifests or Parquet evidence. `catalog rebuild` resolves paths from
the workspace's current location.

### Configuration

`config.toml` defines policy, not mutable state:

```toml
schema_version = 1

[capture]
default_timeout_seconds = 300
max_artifact_bytes = 4294967296
max_parallel_captures = 2

[privacy]
record_environment_allowlist = ["CUDA_VISIBLE_DEVICES"]
capture_git_diff = false
allow_core_content = false

[execution]
allow_privileged_collectors = false
allowed_working_roots = [".."]
child_environment_allowlist = ["PATH", "CUDA_VISIBLE_DEVICES"]
containment = "required_for_mcp"
network = "deny_when_contained"
max_processes = 256
max_memory_bytes = 17179869184
max_output_bytes = 16777216

[analysis]
default_row_limit = 100
max_row_limit = 1000

[storage]
max_workspace_bytes = 107374182400
min_free_bytes = 2147483648
max_staging_bytes = 17179869184
max_files_per_import = 1
max_rows_per_generation = 100000000
```

Secrets must not be placed in this file. Unknown keys fail validation instead
of being silently ignored.

Configuration has two classes. Security and privacy policy is owned only by
`.diagnostics/config.toml` plus explicit CLI startup policy. A named workload
or MCP request may choose a value only inside that policy; it cannot weaken
allowed roots, environment filtering, containment, network, privilege, output,
process, memory, or storage limits. Operational defaults use this precedence:

1. explicit CLI arguments or MCP server startup arguments;
2. the selected named workload in project `flameox.toml`;
3. `.diagnostics/config.toml`;
4. built-in safe defaults.

MCP tool inputs may select among allowed options but cannot override workspace
security policy. Environment-variable configuration is limited to documented
non-secret process settings such as color and log level; there is no generic
environment-to-configuration mapping.

The project-controlled `flameox.toml` is validated when it is created or loaded.
The current canonical workload definition hash is bound into every plan and
run. Editing the command, working directory, environment, validator, resource
policy, or included configuration changes that hash and invalidates existing
plans before execution. MCP can create and update named workloads, but it never
accepts replacement arguments or shell strings outside the structured definition.
The structured configuration path records the canonical definition in
`flameox.toml`; there is no separate workload approval record or human execution
check.

## Identity and provenance

### Run identity

A run receives a UUID4 before execution. The run ID identifies one attempted
execution or one import, including failed and cancelled attempts. Repeating an
identical command creates a new run because time, machine state, and samples
differ. Runs have `run_type = execution | import`. Every imported artifact
creates a new import run; completed runs are never mutated to attach later
imports.

Run identity is deliberately distinct from:

- `workload_definition_id`: canonical declared workload before parameters;
- `workload_instance_id`: resolved workload plus parameter values;
- `experiment_design_id`: variants, blocks, ordering, sample and stopping rule;
- `measurement_protocol_id`: collector and benchmark configuration;
- `capture_spec_id`: requested capture features and resource policy;
- `validation_spec_id`: oracle and tolerance contract;
- `source_state_id`: exact source content state;
- `environment_id`: immutable environment record;
- `experiment_id`: one execution of an experimental design.

### Artifact identity

An artifact ID is `sha256:<lowercase hex digest>` of the exact payload bytes.
The digest is calculated after the collector closes the staged file. A payload
is copied or safely reflinked into the store; symlinks and mutable hard links
are not used.

Artifact extensions are descriptive and excluded from identity. Content-level
metadata does not include kind, media type, display name, producer, role, or
sensitivity because identical bytes may be registered in different contexts.
Those fields belong to a run-to-artifact registration.

### Environment identity

An environment record is immutable and content-addressed from canonical JSON.
It may include:

- operating system and version;
- kernel;
- architecture;
- CPU model and logical/physical count;
- memory total;
- accelerator model, count, driver, and runtime where available;
- Python implementation and version;
- relevant framework versions;
- collector and extractor versions.

There is no single global compatibility fingerprint. Each comparison declares
the environment dimensions that must match, may differ, or are unknown. The
complete redacted environment record remains available. Hostname, username,
absolute home paths, and arbitrary environment variables are excluded by
default.

### Source identity

Every run receives a `source_state_id`. For a Git working tree, its canonical
input includes:

- repository-relative root;
- `HEAD` commit;
- canonical `git diff --binary HEAD`, including staged changes;
- sorted hashes and paths of relevant untracked inputs;
- submodule commits and dirty states;
- current branch only as descriptive metadata;
- resolved interpreter and executable path;
- executable digest and platform build identity where available: ELF build ID,
  Mach-O UUID, or PDB GUID and age;
- imported module or package versions relevant to the workload.

`capture_git_diff` controls whether diff bytes are retained as a sensitive
artifact; it never controls whether the diff contributes to identity.
`identity_quality = exact | partial | clean` describes completeness. A partial
source identity prevents a comparison from being confirmatory. flameox never
checks out, resets, or changes revisions; callers provide baseline and candidate
as separate roots, installations, or resolved commands.

### Workload identity

A workload definition is the canonicalized declared command template,
parameters, working-directory rule, environment policy, timeout, resource
limits, validation specification, and allowed collectors. A workload instance
adds resolved parameter values, executable, arguments, working directory, and
controlled environment overrides.

Warm-up, repetition, randomization, collector settings, and stopping rules are
separate experiment-design and measurement-protocol identities. Workload
identity does not include source or machine, allowing the same logical workload
to be compared across those dimensions without conflating them.

## Run manifest

The run manifest is the complete provenance record. Required top-level fields:

```json
{
  "schema_version": 1,
  "run_id": "uuid4",
  "run_type": "execution",
  "created_at": "RFC3339 timestamp",
  "started_at": "RFC3339 timestamp",
  "finished_at": "RFC3339 timestamp",
  "project": {},
  "source": {},
  "environment": {},
  "workload": {},
  "collector": {},
  "process": {},
  "artifacts": [],
  "execution_status": "succeeded",
  "capture_status": "registered",
  "validation_status": "passed",
  "limitations": [],
  "limitation_details": []
}
```

### State machine

Independent state machines prevent a later extractor failure from rewriting a
successful workload execution:

```text
execution:  not_applicable | planned | running |
            succeeded | failed | timed_out | cancelled

capture:    pending | running | registered |
            failed | quarantined | cancelled

validation: not_requested | pending | running |
            passed | failed | error | cancelled

generation attempt:
            staged | published | failed | superseded | quarantined
```

Transitions are append-logged as immutable run revisions. `manifest.json` is an
atomic current projection. Terminal execution does not erase successfully
captured artifacts. Extraction and validation can be retried as new attempts
without changing the historical execution result.

An active capture revision records a lease containing process identifier,
process start identity, host boot ID, monotonic heartbeat, wall-clock
observation, and expiry. Recovery never trusts a PID alone. Expired leases are
reconciled with staged output and containment state.

### Command representation

Commands are always represented as:

```json
{
  "argv": ["python", "-m", "benchmarks.gae", "--length", "32768"],
  "cwd": "relative/path",
  "env_overrides": {"CUDA_VISIBLE_DEVICES": "0"},
  "timeout_seconds": 300
}
```

There is no string shell-command form in domain or MCP contracts.

### Process result

Record:

- exit code or terminating signal;
- start and end monotonic timestamps;
- wall time;
- peak RSS when available;
- stdout/stderr artifact references or bounded excerpts;
- timeout and cancellation cause;
- child process cleanup result.

Stdout and stderr are size-limited and redacted according to configuration.

`limitation_details` is the typed additive form of `limitations`. Each detail
has a bounded `source` (`adapter`, `preflight`, `collector`, `artifact`,
`resource`, or `validation`), stable code, and human-readable message. The
string field remains a deterministic, de-duplicated compatibility projection;
older manifests containing only strings remain readable. Runtime resource
summaries preserve sampling interval, minimum free bytes, staging growth, peak
RSS, storage-policy termination, and explicitly unavailable metrics.

Capture publication writes runtime-resource evidence into two dedicated tables:
`runtime_resource_summaries` contains one bounded observation per terminal run,
and `runtime_writable_root_growth` contains one observation per declared
writable root. The latter stores a stable writable-root identity and a
project-relative target path; it never publishes the host storage path. A
short-lived process or a failed sampler publishes a nullable metric plus an
explicit unavailable marker rather than silently substituting zero. Storage
reserve termination is preserved in the summary and in the run's typed
limitations.

## Artifact model

Each content object `artifact.json` contains only storage facts:

```json
{
  "schema_version": 1,
  "artifact_id": "sha256:...",
  "byte_length": 12345,
  "payload_name": "payload.pftrace",
  "integrity": {
    "sha256": "...",
    "hashed_at": "..."
  }
}
```

Each immutable run-artifact registration records display name, media type,
artifact kind, role, producer, producer version, sensitivity, source path
policy, and registration time. Identical payload bytes may have multiple
registrations. Effective sensitivity is the maximum classification across all
registrations and mandatory format floors; a later registration can never make
content less sensitive. Core dumps and source snapshots have a mandatory
`sensitive` floor.

Initial artifact kinds:

- `execution_trace`;
- `sample_profile`;
- `memory_profile`;
- `benchmark_samples`;
- `execution_coverage`;
- `process_output`;
- `validation_output`;
- `core_dump`;
- `sanitizer_report`;
- `source_snapshot`;
- `collector_metadata`;
- `analysis_result`.

Sensitivity levels:

- `normal`: ordinary benchmark or profile data;
- `internal`: may contain paths, symbols, command arguments, or source names;
- `sensitive`: may contain source snapshots, process memory, request data, or
  secrets.

MCP never returns raw artifact bytes. `get_artifact` returns content facts and a
bounded, paginated list of registrations. Supplying `run_id` selects one
registration context and avoids ambiguous producer or sensitivity claims.

## Evidence model and Parquet schemas

### Storage rules

- Every publication creates immutable Parquet files.
- Existing Parquet files are never appended or edited.
- A complete generation is first written under `staging`, validated, and moved
  to immutable final paths.
- Publication is visible only when an atomic corpus commit references the
  generation manifest.
- Files include `schema_version`, `evidence_generation_id`, `published_at`,
  `extractor_name`, and `extractor_version`.
- Arrow schemas are checked in and versioned explicitly. Timestamps use
  `timestamp[us, tz=UTC]`; durations use signed `int64` nanoseconds; addresses
  use `uint64`; non-finite floats are rejected unless a field explicitly
  defines them.
- Durations and byte quantities use integers, not floating-point values.
- Units are explicit.
- High-cardinality, collector-specific details remain in native artifacts unless
  needed by a supported query.
- The current Arrow schema is major 1, minor 5. Schema evolution is additive only for declared nullable fields within a major
  schema version. `union_by_name = true` implements that declared evolution; it
  is not itself the evolution policy.
- Evidence generations written with minor 1.4 remain authoritative and are
  never rewritten. The catalog projects the two new runtime-resource tables as
  empty typed views for those generations, and read results report that the
  resource summary was not published.
- Typed dimension columns are used for comparison-critical fields. A bounded
  map may carry collector-specific descriptive dimensions but cannot determine
  pairing, compatibility, or treatment assignment.

Those common publication columns apply to every evidence table even when they
are omitted from the table-specific lists below. An extraction batch receives
one UUID `evidence_generation_id`. Its manifest records exact file hashes,
sizes, row counts, Arrow schema identities, input corpus commit, input runs and
artifacts, publisher, and superseded generations. Re-extraction publishes a new
generation and a new corpus commit; it does not edit previous files. Catalog
views expose only generations reachable from the pinned commit.

Small generations are periodically compacted with PyArrow `write_dataset` or
DuckDB `COPY` into immutable target segments, normally 64–256 MiB. Compaction
publishes a replacement generation and commit before old segments become
eligible for garbage collection. It never merges Parquet rows with a custom
writer.

### Investigations and experiments

The analytical hierarchy is:

```text
Investigation
└── Hypothesis
    └── Experiment
        ├── Variant
        └── Trial ──► Run
```

`investigations` records the motivating symptom, question, project root,
created time, lifecycle status, and optional parent investigation.

`hypotheses` records a bounded claim, an explicit prediction, a discriminating
condition that could refute it, lifecycle status, and revision. A hypothesis
does not become supported merely because a profile is compatible with it.
Revisions require an expected-revision compare-and-swap just like findings.

`experiments` records the recipe and version, workload definition, design ID,
measurement protocol, validation specification, primary metric, metric
polarity, estimand, practical threshold, confidence level, sample or stopping
rule, random seed, confirmatory or exploratory role, and creation time.

`variants` records the treatment name and exact source state, workload
instance, command/build identity, environment requirements, and parameter
values.

`trials` records every attempted treatment execution:

| Column | Type | Meaning |
|---|---|---|
| `trial_id` | string | UUID |
| `experiment_id` | string | parent experiment |
| `variant_id` | string | assigned treatment |
| `run_id` | string | attempted execution run |
| `block_id` | string nullable | randomized complete block |
| `order_in_block` | integer nullable | execution position |
| `parameter_name` | string nullable | scaling dimension |
| `parameter_value_int` | int64 nullable | exact integral value |
| `parameter_value_float` | double nullable | fractional value |
| `attempt` | integer | retry/attempt number |
| `outcome` | string | succeeded, failed, timed_out, cancelled, oom, invalid |
| `exclusion_reason` | string nullable | predeclared analysis exclusion |
| `validation_status` | string | oracle outcome |
| `oracle_receipt_json` | string nullable | bounded parsed structured receipt |
| `oracle_receipt_artifact_id` | string nullable | authoritative raw receipt identity |

Every attempted trial remains visible, including failures and exclusions.
Analyses report counts and reasons rather than filtering them silently.

Run manifests may also carry an `oracle_receipt` record containing the parsed
producer receipt, authoritative receipt artifact, validation stdout/stderr
artifact identities, same-run diagnostic artifact identities, and
Flameox-observed parsing limitations. Older manifests and evidence generations
omit these additive fields. Parquet remains authoritative across mixed schema
minor versions through name-based union, and DuckDB remains rebuildable.

`run_sets` freeze a cohort for analysis. A run set records its ID, creation
time, pinned corpus commit, normalized selection parameters, ordered run/trial
membership, inclusion and exclusion reasons, and membership digest. Membership
never changes after creation. A new selection produces a new run set even when
it currently resolves to the same members. This prevents later imports from
silently changing a completed comparison.

Multi-run paired comparisons require explicit `trial_id` membership and pair
on the trial's `block_id`. Member order is presentation metadata and never a
statistical pairing key. One-run shorthand may use the collector's independent
sample hierarchy.

### `runs`

One row per run:

| Column | Type | Meaning |
|---|---|---|
| `schema_version` | integer | table schema |
| `run_id` | string | run UUID |
| `created_at` | timestamp | creation time |
| `run_type` | string | execution or import |
| `execution_status` | string | independent execution state |
| `capture_status` | string | independent capture state |
| `validation_status` | string | independent oracle state |
| `workload_definition_id` | string nullable | declared workload |
| `workload_instance_id` | string nullable | resolved workload |
| `measurement_protocol_id` | string nullable | collector/benchmark protocol |
| `environment_id` | string | immutable environment record |
| `source_state_id` | string nullable | exact or partial source identity |
| `collector` | string nullable | adapter name |
| `collector_version` | string nullable | installed version |
| `exit_code` | integer nullable | process result |
| `wall_time_ns` | int64 nullable | total execution time |
| `manifest_path` | string | workspace-relative manifest |

### `artifacts`, `environments`, and `source_states`

One row per run-to-artifact relationship:

| Column | Type | Meaning |
|---|---|---|
| `run_id` | string | producing or importing run |
| `artifact_id` | string | content ID |
| `kind` | string | artifact kind |
| `media_type` | string | representation |
| `byte_length` | uint64 | exact size |
| `sensitivity` | string | access classification |
| `role` | string | primary, log, validation, auxiliary |
| `producer` | string nullable | collector or importer for this run |
| `producer_version` | string nullable | producer version for this run |

The `artifacts` table is one row per registration, not one row per content
object. `registration_id`, display name, registered time, and effective
sensitivity are also required.

`environments` and `source_states` contain immutable, content-addressed records
with typed comparison-critical fields plus a bounded canonical JSON extension.
They include identity quality and missing-field lists so unknown cannot be
mistaken for equal.

### `measurements`

Generic scalar or distribution samples:

| Column | Type | Meaning |
|---|---|---|
| `measurement_id` | string | deterministic row identity |
| `run_id` | string | owning run |
| `artifact_id` | string nullable | source artifact |
| `name` | string | namespaced metric |
| `value_int` | signed integer nullable | exact duration, bytes, or count |
| `value_float` | double nullable | fractional or inherently floating value |
| `unit` | string | `ns`, `bytes`, `count`, `ratio`, etc. |
| `aggregation` | string | sample, total, mean, median, p95, peak |
| `scope` | string | process, thread, operator, device, workload |
| `trial_id` | string nullable | owning attempted trial |
| `worker_id` | string nullable | independent pyperf/process worker |
| `worker_run_index` | integer nullable | run within worker |
| `value_index` | integer nullable | raw value within run |
| `loop_count` | uint64 nullable | operations represented |
| `is_warmup` | boolean | warm-up versus measured value |
| `block_id` | string nullable | randomized experiment block |
| `variant_id` | string nullable | treatment |
| `order_in_block` | integer nullable | treatment order |
| `phase` | string nullable | warmup, compile, steady_state, validation |
| `dimensions` | map<string,string> | bounded analysis dimensions |
| `evidence_level` | string | observed or derived |

Raw hierarchy must be retained when a reported aggregate is based on repeated
measurements. pyperf calibration, warm-ups, workers, runs, values, and loop
counts are not flattened into a single iteration index. Exactly one of
`value_int` and `value_float` is set. Durations, byte quantities, and counts use
`value_int`; ratios and derived fractional statistics use `value_float`.

### `frames`

Deduplicated frame identities:

| Column | Type | Meaning |
|---|---|---|
| `frame_id` | string | stable logical frame identity |
| `language` | string nullable | Python, C++, Rust, etc. |
| `function` | string nullable | symbolized function |
| `module` | string nullable | module or binary |
| `file` | string nullable | normalized source path |
| `line` | integer nullable | source line |
| `column` | integer nullable | source column |
| `address` | uint64 nullable | machine address |
| `build_id` | string nullable | binary build identity |
| `module_relative_address` | uint64 nullable | ASLR-stable offset |
| `inline_chain_id` | string nullable | complete inline context |
| `source_state_id` | string nullable | Python/source identity |
| `artifact_id` | string nullable | required for unstable unsymbolized frames |
| `inlined` | boolean nullable | inline-frame flag |
| `symbolization` | string | complete, partial, absent |

Absolute source paths are normalized to repository-relative paths when safe.
Native frame identity uses build ID, module-relative address, and inline chain.
Python identity uses source state, normalized file, qualified function/code
identity, and first line. Raw addresses do not define cross-run identity.
Unsymbolized frames are artifact-local. The original path and exact stack may
remain in the sensitive native artifact.

### `frame_measurements`

Aggregated profile facts:

| Column | Type | Meaning |
|---|---|---|
| `run_id` | string | run |
| `artifact_id` | string | profile or trace |
| `frame_id` | string | referenced frame |
| `metric` | string | CPU time, samples, allocated bytes, etc. |
| `self_value` | signed integer nullable | exact exclusive value |
| `inclusive_value` | signed integer nullable | exact cumulative value |
| `unit` | string | explicit unit |
| `sample_count` | uint64 nullable | contributing samples |
| `thread_name` | string nullable | optional dimension |
| `process_name` | string nullable | optional dimension |
| `phase` | string nullable | workload phase |

Complete stacks remain in pprof or native trace data. Initial hotspot and
memory recipes must nevertheless expose bounded callers, callees, and
representative stacks through native-format or Perfetto queries. flameox reuses
the pprof mapping/location/function/line model when normalizing cross-profile
stacks instead of inventing an incompatible universal stack schema.

### `observations`

Bounded execution-path and semantic observations:

| Column | Type | Meaning |
|---|---|---|
| `observation_id` | string | deterministic row identity |
| `run_id` | string | owning run |
| `artifact_id` | string nullable | source coverage, trace, or SDK artifact |
| `kind` | string | line_hit, branch_arc, annotation, configuration |
| `name` | string | namespaced observation name |
| `value_json` | string | canonical bounded primitive or object |
| `file` | string nullable | repository-relative source file |
| `line_from` | integer nullable | line or branch origin |
| `line_to` | integer nullable | branch destination |
| `context` | string nullable | test, phase, or coverage context |
| `evidence_level` | string | observed or derived |

`value_json` is size-limited and must validate as JSON containing only bounded
primitive values, lists, and objects. Secrets and arbitrary object
representations are rejected.

Coverage observations represent executed lines and arcs, not hit counts unless
the producer explicitly supplies counts through a supported contract.

### `analyses` and `evidence_refs`

Every persisted recipe invocation creates an immutable analysis record:

| Column | Type | Meaning |
|---|---|---|
| `analysis_id` | string | UUID |
| `recipe` | string | stable recipe name |
| `recipe_version` | string | semantic implementation version |
| `parameters_json` | string | bounded canonical parameters |
| `parameters_digest` | string | exact request identity |
| `corpus_commit_id` | string | pinned snapshot |
| `input_generation_ids` | list<string> | exact normalized inputs |
| `input_run_ids` | list<string> | exact run inputs |
| `input_artifact_ids` | list<string> | exact native inputs |
| `result_digest` | string | structured result identity |
| `result_artifact_id` | string nullable | large result payload |
| `coverage_json` | string | measured coverage |
| `limitations` | list<string> | proof gaps |
| `started_at` | timestamp | invocation start |
| `completed_at` | timestamp nullable | terminal time |

Typed `evidence_refs` connect analyses, hypotheses, findings, comparisons,
runs, artifacts, observations, and generations:

| Column | Type | Meaning |
|---|---|---|
| `owner_type` | string | analysis, finding, or hypothesis |
| `owner_id` | string | owning entity |
| `ref_type` | string | typed referenced entity |
| `ref_id` | string | validated entity identity |
| `relation` | string | supports, contradicts, context, validates |

### `comparisons`

One row per comparison metric:

| Column | Type | Meaning |
|---|---|---|
| `comparison_id` | string | comparison UUID |
| `experiment_id` | string nullable | owning experiment |
| `baseline_run_set_id` | string | frozen baseline cohort |
| `candidate_run_set_id` | string | frozen candidate cohort |
| `metric` | string | compared metric |
| `unit` | string | metric unit |
| `polarity` | string | lower_is_better, higher_is_better, neutral |
| `estimand` | string | exact target statistic |
| `baseline_value_int` | int64 nullable | exact integral estimate |
| `baseline_value_float` | double nullable | fractional estimate |
| `candidate_value_int` | int64 nullable | exact integral estimate |
| `candidate_value_float` | double nullable | fractional estimate |
| `absolute_change_int` | int64 nullable | exact integral change |
| `absolute_change_float` | double nullable | fractional change |
| `relative_change` | double nullable | unitless ratio |
| `effect_size` | double nullable | selected effect measure |
| `confidence_low` | double nullable | interval |
| `confidence_high` | double nullable | interval |
| `confidence_level` | double nullable | declared coverage |
| `method` | string | method and semantic version |
| `random_seed` | uint64 nullable | reproducibility |
| `independent_unit` | string | block, worker, run, etc. |
| `paired` | boolean | paired design |
| `baseline_attempted_n` | uint64 | all attempted baseline trials |
| `baseline_eligible_n` | uint64 | baseline trials eligible for estimand |
| `candidate_attempted_n` | uint64 | all attempted candidate trials |
| `candidate_eligible_n` | uint64 | candidate trials eligible for estimand |
| `complete_pair_n` | uint64 nullable | complete blocks |
| `multiplicity_json` | string nullable | family and adjustment |
| `decision` | string | meaningful_improvement, meaningful_regression, no_meaningful_difference, inconclusive, descriptive_only |
| `validity` | string | valid, exploratory, invalid |
| `mismatches` | list<string> | incompatible dimensions |

Pairwise `baseline_run_id` and `candidate_run_id` inputs are syntactic sugar
that create frozen one-element run sets.

### `findings`

A finding is a durable claim, not merely a log message:

| Column | Type | Meaning |
|---|---|---|
| `finding_id` | string | UUID |
| `revision` | integer | monotonically increasing finding revision |
| `created_at` | timestamp | creation time |
| `kind` | string | hotspot, regression, anomaly, hypothesis, validation |
| `title` | string | short specific description |
| `claim` | string | bounded factual statement |
| `evidence_level` | string | observed, derived, inferred |
| `confidence` | string | high, medium, low, unknown |
| `assessment` | string | unassessed, supported, refuted, inconclusive |
| `lifecycle` | string | active, superseded, retracted |
| `limitations` | list<string> | material proof gaps |
| `next_experiments_json` | string | structured recipe/experiment requests |

Updating a finding requires `expected_revision` compare-and-swap and appends a
new revision; default catalog views select the highest valid revision. A
performance optimization cannot be assessed `supported` without a valid
comparison, declared estimand and practical threshold, a passing
cross-treatment semantic oracle, and no critical identity or environment
mismatch. The mere presence of an observed reference is insufficient.

## DuckDB catalog

### Responsibilities

`catalog.duckdb` provides:

- stable schemas, macros, and query definitions used to construct
  snapshot-local views over Parquet;
- schema and extractor compatibility metadata;
- parameterized analytical queries used by recipes.

Measured, reproducible cached summaries may be added after profiling proves
they improve an important query. The catalog contract does not promise
materialized caches or indexes.

It does not own:

- raw artifacts;
- run manifests;
- the only copy of a finding;
- job state;
- user accounts;
- arbitrary mutable application records.

Every analysis connection is bound to an explicit corpus commit inventory and
creates temporary snapshot views from the exact file lists in that inventory.
Persistent definitions never use unconstrained globs over the mutable
workspace. Empty tables are represented by checked-in typed schema anchors so
a new workspace has valid views. Publishing evidence therefore requires no
catalog rewrite; a new connection can pin the new commit immediately while an
existing connection continues using its old temporary views.

### Rebuild

`flameox catalog rebuild`:

1. pins and validates the current corpus commit and referenced generation
   manifests without holding the catalog lock;
2. holds the retention lock shared while creating and validating a temporary
   catalog from those exact file lists, then releases it;
3. acquires the workspace write lock and exclusive catalog lock;
4. rechecks that `corpus/HEAD` still identifies the pinned commit and that every
   referenced file is still present with the expected metadata;
5. checkpoints and closes the temporary catalog;
6. flushes it according to the local-filesystem durability policy;
7. atomically replaces `catalog.duckdb`;
8. releases locks and allows new read connections.

Existing readers continue using the old catalog until replacement. A rebuild
does not mutate or quarantine authoritative artifacts, manifests, or evidence.
Detected corruption is reported. The explicit `flameox repair` operation,
separate from rebuild, may move recoverable material under the mutation locks
only from a validated, previewable repair plan.

### SQL safety

flameox does not expose arbitrary SQL through MCP or its agent-facing CLI.

Internally:

- SQL text is a version-controlled constant;
- user inputs are bound parameters;
- each concurrent query uses its own read-only connection; a DuckDB connection
  or cursor is never shared across tasks;
- permitted workspace Parquet paths are configured before locking connection
  configuration;
- external access outside those paths, extension installation, autoload,
  community extensions, secrets, and attachments are disabled;
- memory, threads, temporary-directory use, and query time are bounded;
- result rows, fields, nested values, and text lengths are capped;
- every result records the query name and query version.

The exact DuckDB release is pinned and tested because configuration names and
the interaction between external-access restrictions and `read_parquet` can
change. Catalog validation accepts only known views and macros and rejects
unexpected attachments, extensions, or secrets. Blocking queries run outside
the event loop with a dedicated connection; cancellation invokes DuckDB
interruption before joining the worker.

Advanced users can open the local catalog with DuckDB's own CLI. That process
does not honor flameox's catalog or retention locks and remains outside flameox's
safety and concurrency contract.

### Query result budgets

Every query accepts or derives:

- maximum rows;
- maximum rows, fields, nested values, and text lengths;
- timeout;
- sort order;
- optional cursor.

If truncated, the result includes `truncated=true`, the applied limit, and a
stable continuation cursor when the query supports pagination. A cursor binds
the query version, normalized filters, ordering, last sort key, and corpus
commit ID. It is opaque to callers. A cursor from a different commit fails
explicitly rather than silently skipping or duplicating rows.

## Extensibility and schema evolution

### Domain schemas

Every manifest, MCP result, and Parquet table carries an integer
`schema_version`.

- Arrow schemas have explicit major and minor versions;
- declared additive nullable fields increment the minor version;
- changed meaning, unit, identity, or required fields does;
- readers support the current version and a declared compatibility window;
- migration produces new derived files and never rewrites raw artifacts;
- MCP result schema changes require contract tests.

### Extractor versions

An extractor version changes whenever output semantics change. Repeating the
same operation identity over the same native artifact reuses its active
generation. A changed extractor/tool/configuration identity creates a new
generation and supersedes older derivations without deleting them immediately.

### Adapter compatibility

Adapters declare supported producer versions. Unknown newer versions are
validated conservatively. An adapter must not parse a format it cannot identify
and then emit apparently valid evidence.
