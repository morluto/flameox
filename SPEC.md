# Flamo: Local Runtime Evidence CLI and MCP Server

Status: proposed implementation specification
Audience: maintainers, contributors, agent-tool authors, and reviewers
Working language: Python 3.12+
MCP SDK baseline: `mcp==2.0.0b2`

## 1. Executive summary

Flamo is a permanently local command-line application and Model Context
Protocol server that helps coding agents investigate runtime behavior. It does
not attempt to replace profilers, debuggers, benchmark harnesses, or tracing
systems. It coordinates those tools, preserves their native artifacts, extracts
compact structured evidence, compares observations across runs, and ties every
claim back to reproducible inputs.

The product exists to make investigations like the following repeatable:

- discover that an apparently vectorized algorithm contains an expensive
  Python reverse scan;
- show how the behavior scales over representative input sizes;
- connect the hotspot to exact source and configuration;
- distinguish a performance problem from a semantic bug;
- validate that a proposed change preserves outputs and improves performance;
- analyze a population of crashes or profiles without forcing them into one
  root-cause narrative.

Flamo is not a generic AI bug finder. It is an evidence system for agents.
Deterministic collectors and query engines establish what happened. The agent
uses that evidence to form hypotheses, choose discriminating experiments, and
explain conclusions.

The durable architecture is:

```text
existing collectors
  py-spy · pyperf · torch.profiler · Memray · perf · Perfetto
       │
       ▼
native immutable artifacts
  pprof · Perfetto trace · Memray capture · benchmark JSON · core dump
       │
       ├──────────────► Perfetto Trace Processor
       │                 detailed queries within one trace
       │
       ▼
normalized Parquet evidence
       │
       ▼
DuckDB
  pinned corpus snapshots · comparisons · scaling · cohorts · findings
       │
       ├──────────────► `flamo` CLI
       └──────────────► local MCP server
```

DuckDB is the long-term local analytical engine. Parquet and native artifacts
are authoritative. Immutable generation manifests and an atomically published
corpus commit define which Parquet files belong to a readable snapshot.
`catalog.duckdb` contains rebuildable views and may contain measured,
reproducible caches. There is no PostgreSQL service and no SQLite application
database.

The unit of execution is a run. The unit of experimental reasoning is not.
Flamo models an investigation containing hypotheses and experiments; each
experiment contains variants and attempted trials, and each trial references
one run. This preserves randomized blocks, repetitions, input grids, failed or
timed-out attempts, validation outcomes, and the distinction between an
exploratory profile and confirmatory performance evidence.

## 2. Problem statement

Profilers expose valuable information, but using them effectively requires
several decisions that agents currently make poorly:

1. choosing the right collector for the symptom;
2. capturing a representative and reproducible workload;
3. controlling warm-up, environmental noise, and profiler overhead;
4. converting tool-specific output into evidence small enough to reason about;
5. comparing runs without accidentally comparing incompatible environments;
6. preserving the raw artifact so a conclusion can be audited;
7. separating observations from hypotheses;
8. running a follow-up experiment that could disprove the leading explanation.

Most existing tools optimize either collection or human visualization. They
rarely provide a shared local corpus that an agent can query across benchmark
runs, profiles, traces, memory captures, and failures.

Source inspection alone is also insufficient. A configuration interaction may
be semantically wrong without appearing in a CPU profile. Conversely, a slow
path may be obvious in a profile but harmless at realistic sizes. The system
must support source, runtime, workload, environment, and validation evidence as
one investigation.

## 3. Goals

Flamo must:

1. remain completely local and usable without a daemon or external service;
2. expose the same diagnostic domain operations through Python, CLI, and MCP;
3. preserve native collector output without lossy mandatory conversion;
4. make each artifact content-addressed and immutable;
5. make capture provenance explicit and reproducible;
6. normalize only the evidence needed for cross-run analysis;
7. support representative before/after and scaling comparisons;
8. produce structured results that fit within an agent context window;
9. let agents drill from a finding to measurements, frames, runs, and raw
   artifacts;
10. distinguish observed, derived, and inferred statements;
11. serialize local corpus mutations while allowing safe concurrent reads;
12. rebuild the entire DuckDB catalog from manifests and Parquet evidence;
13. operate safely when invoked by an agent, especially around command
   execution, raw SQL, core dumps, environment variables, and large outputs;
14. make unsupported capabilities and proof gaps explicit;
15. be extensible through adapters without placing collector logic in the MCP
   transport;
16. represent investigations, hypotheses, experimental designs, variants,
   attempted trials, and validation oracles explicitly;
17. make every multi-query analysis operate against one immutable corpus
   snapshot;
18. preserve failed, timed-out, cancelled, and out-of-memory trials so analyses
   cannot silently select only successful observations;
19. record the exact source, executable, build, environment, measurement
   protocol, and analysis inputs behind every result.

## 4. Non-goals

The initial product will not:

- implement a sampling profiler;
- implement a trace database or timeline renderer;
- decode private Memray, Nsight, or vendor formats;
- replace Perfetto, pprof, Speedscope, Memray reports, or debugger UIs;
- continuously monitor production services;
- upload artifacts or telemetry;
- provide team accounts, permissions, or remote synchronization;
- claim root cause from correlation alone;
- automatically modify source code;
- expose unrestricted shell or process execution through MCP;
- expose unrestricted DuckDB or Perfetto SQL through either interface;
- normalize every field from every native format;
- make a single universal event schema;
- reimplement native trace formats or statistics already exposed through
  maintained public APIs;
- automatically delete artifacts;
- require PyTorch, CUDA, Memray, or any profiler in the base installation;
- promise identical collector availability across operating systems;
- claim that named workloads are sandboxed unless an active containment backend
  actually enforces that boundary;
- treat statistical non-significance as proof of equivalence.

## 5. Design principles

### 5.1 Native evidence first

The native artifact is the most authoritative representation of a capture.
Normalized evidence is a query accelerator and shared vocabulary, not a
replacement. Every normalized row must link to an artifact, run, and extractor
version.

### 5.2 Facts before interpretation

Tool output must use three evidence levels:

- `observed`: directly emitted or measured by a collector;
- `derived`: deterministically calculated from observed evidence;
- `inferred`: a hypothesis or interpretation that may require another
  experiment.

An inferred claim must never be represented as an observed fact.

### 5.3 Comparisons are valid only under declared conditions

Every comparison reports the environment and workload fields that match, differ,
or are unknown. Flamo may refuse a comparison when a mismatch invalidates the
claim. `--force` can display an exploratory comparison but cannot relabel it
valid.

### 5.4 Task-shaped APIs

Agents should call `compare_run_sets` or `analyze_scaling`, not construct
arbitrary SQL. Internally, Flamo uses reviewed parameterized SQL and
collector-specific queries. This keeps results compact and prevents DuckDB
features from becoming an unintended local file-access or code-execution
interface.

### 5.5 Rebuildable derived state

Deleting `catalog.duckdb` must not delete evidence. `flamo catalog rebuild`
recreates it from immutable manifests and Parquet partitions.

### 5.6 Capture outside the write lock

Profiling and benchmarks can take minutes. They run concurrently in isolated
staging directories. Only registration and evidence publication require the
short-lived workspace write lock.

### 5.7 No silent fallback

If native stacks, GPU events, symbols, shapes, or memory records were requested
but unavailable, the result states that explicitly. Flamo must not silently
substitute a weaker collector and present the result as equivalent.

### 5.8 Experimental structure is evidence

Warm-ups, worker processes, randomized block order, treatment assignment,
failed attempts, exclusion reasons, validation results, and stopping rules are
part of the evidence. They must not be flattened into an unlabeled list of
samples. Profile-guided discovery is exploratory; a fresh, predeclared
benchmark and semantic oracle provide confirmatory evidence.

### 5.9 One analysis, one corpus snapshot

The corpus is append-only. A reader pins one immutable corpus commit before its
first query and uses the exact file inventory in that commit for every query in
the analysis. Files that exist on disk but are absent from the pinned inventory
are invisible. Publication becomes visible only by atomically advancing the
corpus `HEAD`.

### 5.10 Safe composition over custom infrastructure

Flamo prefers maintained public interfaces: Perfetto Trace Processor for
supported trace and profile formats, PyArrow datasets for Parquet publication,
DuckDB for analytical SQL, pyperf for benchmark collection, SciPy for declared
bootstrap calculations, and collector-supported readers for native artifacts.
Private APIs and parsing human-readable reports are not product contracts.

## 6. Primary workflows

### 6.1 CPU hotspot investigation

1. Inspect local capabilities.
2. Plan a capture and show the exact command, tool, expected overhead, and
   required permissions.
3. Run the workload with `py-spy`, `perf`, or another installed adapter.
4. Register the native pprof, Speedscope, or perf artifact.
5. Extract aggregated self and inclusive frame measurements.
6. Return the top source-linked hotspots with coverage and limitations.
7. Preserve a command for opening the native artifact in an existing viewer.

### 6.2 Scaling investigation

1. Define a workload and an explicit input parameter such as sequence length.
2. Run warm-ups.
3. Measure balanced randomized blocks across variants and inputs.
4. retain raw samples, not just averages;
5. report medians, dispersion, effect sizes, and environmental metadata;
6. fit only declared candidate growth models;
7. identify the source regions whose cost grows with the input;
8. label complexity conclusions as inferred unless the source mechanism is
   independently established.

### 6.3 Before/after validation

1. Execute the same workload against baseline and candidate source roots,
   installations, or commands supplied by the caller; Flamo does not switch the
   working tree between revisions.
2. Validate behavior before accepting performance evidence.
3. Compare benchmark distributions.
4. Compare profiles or traces to explain the change.
5. Report both improvements and regressions.
6. Save a finding whose evidence references are sufficient to reproduce the
   result.

### 6.4 PyTorch operator and accelerator investigation

1. Capture CPU operators and the available accelerator activity through
   `torch.profiler`.
2. Record whether stacks, shapes, modules, memory, and FLOPs were enabled.
3. Export a Perfetto-compatible trace.
4. Query operator duration, call count, synchronization, idle gaps, repeated
   operators, and memory changes.
5. Keep warm-up and compilation phases separate from steady state.
6. Report profiler overhead and missing accelerator data.

### 6.5 Memory-growth investigation

1. Capture repeated workload phases using Memray or an appropriate native
   memory tool.
2. distinguish peak allocation, retained allocation, allocation volume, and
   resident memory;
3. compare stacks across phases or runs;
4. identify growth correlated with a workload dimension;
5. retain the Memray binary and link to its supported reports.

### 6.6 Population-level failure investigation

This is a later adapter set but a first-class architectural requirement.

1. Import core dumps, sanitizer reports, or crash summaries.
2. Extract deterministic features such as signal, fault address class, register
   properties, stack signature, modules, build IDs, host attributes, and
   symbolization quality.
3. group and compare crash cohorts;
4. expose representative artifacts from each cluster;
5. preserve multiple plausible clusters instead of forcing one signature;
6. run change-point and dimensional analyses;
7. record interventions and recurrence observations.

## 7. System architecture

### 7.1 Package boundaries

```text
src/flamo/
├── domain/                  # contracts and invariants
├── application/             # transport-independent use cases
├── investigations/          # hypotheses, experiments, variants, and trials
├── adapters/                # collector integrations
│   ├── pyperf/
│   ├── py_spy/
│   ├── torch_profiler/
│   ├── memray/
│   ├── perf/
│   ├── coverage/
│   └── perfetto/
├── evidence/
│   ├── models/              # evidence contracts
│   ├── storage/             # Parquet readers and publication
│   └── provenance/          # run, source, environment, workload identity
├── catalog/                 # DuckDB views and parameterized queries
├── execution/               # safe subprocess and process-tree management
│   ├── broker.py             # only subprocess creation boundary
│   ├── containment.py        # optional platform containment backends
│   └── workers.py            # bounded extractor/query workers
├── recipes/                 # task-shaped investigations
├── analysis/
│   ├── comparisons/         # compatibility and before/after analysis
│   ├── scaling/             # measured growth analysis
│   └── findings/            # evidence-backed claims
├── cli/                     # Typer presentation only
├── mcp/                     # pinned MCP SDK v2 adapter only
└── sdk/                     # optional in-process annotations
```

Dependencies point inward:

```text
CLI ─┐
     ├─► application services ─► domain
MCP ─┘          │                 ▲
                ├─► adapters ─────┤
                ├─► workspace ────┤
                └─► analysis ─────┘
```

`flamo.domain` must not import `mcp`, `typer`, DuckDB, Perfetto, PyTorch, or
collector packages. Domain models are ordinary Pydantic models and enums.

### 7.2 Process model

The CLI is a short-lived process. The MCP server is a long-lived local process
using stdio by default. Both call the same application services.

Transport composition selects a named execution policy. The CLI uses
`trusted_local`; MCP uses `approved_agent`. Application services receive that
policy rather than a transport boolean, and plans bind the selected policy for
execution-time revalidation.

Collectors run as child processes or explicitly selected in-process adapters.
External commands are always executed as argument arrays with `shell=False`.
On Linux, Flamo uses a cgroup v2 or systemd scope when available so cancellation,
timeouts, and resource limits apply to descendants even if they create a new
process group. A process-group fallback is reported as degraded containment,
not as an equivalent guarantee.

`flamo.execution` owns subprocess creation through
`asyncio.create_subprocess_exec` behind a single broker. Collectors, validators,
active capability probes, symbolizers, Perfetto, viewers, and extractor workers
must use that broker; adapters do not call subprocess APIs directly. This gives
command validation, environment construction, containment, quotas, output
budgets, cancellation, timeouts, and descendant cleanup one implementation.
Heavy or artifact-facing extractors run in bounded worker processes. Small
trusted metadata operations may use an AnyIO worker thread.

The MCP SDK's AnyIO cancellation is translated at the application boundary into
the execution service's cancellation signal. Cleanup that awaits subprocess or
publication work runs in a bounded shielded cancellation scope and re-raises
the original cancellation afterward.

The MCP server is not a background system daemon. It starts and stops with the
MCP host.

### 7.3 Python and dependency baseline

Required:

- Python 3.12 or newer;
- `mcp==2.0.0b2`;
- Pydantic 2;
- DuckDB;
- PyArrow;
- Typer;
- AnyIO;
- Portalocker for cross-process shared and exclusive file locks;
- structured logging using the standard library or a minimal compatible layer.

The MCP beta pin is intentional. It must remain exact until a deliberate SDK
upgrade is validated. All SDK imports and conversions live under `flamo.mcp`.
The domain service API must be testable without importing MCP.

The project uses the v2 `MCPServer` API, not the v1
`mcp.server.fastmcp.FastMCP` compatibility surface. The exact dependency is
declared in `pyproject.toml`, and the resolved beta is committed in `uv.lock`.

Optional extras group collector dependencies:

```text
flamo[python]       pyperf and Python profiling integrations
flamo[memory]       Memray integration
flamo[torch]        in-process PyTorch integration
flamo[stats]        SciPy and optional statsmodels analyses
flamo[dev]          test, lint, type-check, and fixture tooling
flamo[all]          all non-vendor optional integrations
```

System executables such as `perf`, `py-spy`, `trace_processor_shell`,
`llvm-symbolizer`, Bubblewrap, GDB, or LLDB are detected at runtime and are not
silently downloaded. Trace Processor is either an explicitly provisioned
binary whose version and digest are recorded, or a deliberately bundled,
platform-specific package. Its convenience network download path is disabled.

### 7.4 Application services

Transport-independent services define the callable product surface:

```python
class WorkspaceService(Protocol):
    async def status(self) -> WorkspaceStatus: ...
    async def validate(self, request: ValidateWorkspaceRequest) -> ValidationResult: ...
    async def initialize(self, request: InitializeWorkspaceRequest) -> WorkspaceStatus: ...


class CaptureService(Protocol):
    async def plan(self, request: CaptureRequest) -> CapturePlan: ...
    async def execute_plan(self, plan_id: str) -> RunResult: ...
    async def import_artifact(self, request: ImportRequest) -> ImportResult: ...


class EvidenceService(Protocol):
    async def list_runs(self, query: RunQuery) -> Page[RunSummary]: ...
    async def get_run(self, run_id: str) -> RunDetail: ...
    async def list_findings(self, query: FindingQuery) -> Page[FindingSummary]: ...
    async def get_finding(self, finding_id: str) -> FindingDetail: ...
    async def get_artifact(
        self, artifact_id: str, run_id: str | None = None
    ) -> ArtifactDetail: ...


class AnalysisService(Protocol):
    async def hotspots(self, request: HotspotRequest) -> HotspotResult: ...
    async def scaling(self, request: ScalingRequest) -> ScalingResult: ...
    async def compare(self, request: CompareRunSetsRequest) -> ComparisonResult: ...
    async def execution(self, request: ExecutionPathRequest) -> ExecutionPathResult: ...


class ExperimentService(Protocol):
    async def create_investigation(
        self, request: CreateInvestigationRequest
    ) -> Investigation: ...
    async def record_hypothesis(
        self, request: RecordHypothesisRequest
    ) -> Hypothesis: ...
    async def plan(self, request: ExperimentRequest) -> ExperimentPlan: ...
    async def execute(self, plan_id: str) -> ExperimentResult: ...
```

CLI commands render these models. MCP handlers validate SDK inputs, call these
services, and put the same domain models inside a transport envelope. CLI JSON
returns the domain model directly. Services raise typed domain errors rather
than Typer exits or MCP error objects.

### 7.5 Platform policy

The storage, query, CLI, and MCP layers are platform-neutral. Linux is the
initial first-class capture platform because `perf`, native symbols,
accelerator tooling, cgroup containment, and core dumps matter to the target
investigations.

macOS and Windows may expose adapters whose upstream collectors support them,
but capability reports must identify degraded behavior. Platform support is
claimed per adapter and feature, never from the fact that the Python package
installs.

## 8. Workspace

### 8.1 Discovery

Flamo discovers a workspace by walking from the requested working directory
toward the filesystem root and selecting the nearest `.diagnostics` directory.
If none exists, commands that mutate state fail with a remediation suggesting
`flamo init`. Read-only capability discovery does not require initialization.

An explicit `--workspace` overrides discovery. The resolved workspace must be
inside the selected project root unless the user explicitly supplies an
external absolute path through the CLI. MCP tools cannot choose an arbitrary
external workspace.

In a Git repository, `flamo init` adds `.diagnostics/` to
`.git/info/exclude` when it is not already ignored. It does not edit the
project's tracked `.gitignore`. The shareable `flamo.toml` workload
configuration remains outside `.diagnostics` and may be committed deliberately.

### 8.2 Directory layout

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
artifacts and runs, extractor or publisher identity, and superseded
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

### 8.3 Workspace identity

`workspace.json` contains:

```json
{
  "schema_version": 1,
  "workspace_id": "uuid4",
  "created_at": "RFC3339 timestamp",
  "project_root": "..",
  "flamo_version": "version at creation"
}
```

`workspace.json` is static workspace identity, not mutable corpus state. The
only publication authority is `corpus/HEAD`; there is no independently updated
generation counter that can disagree with it.

Paths stored in manifests are relative to the workspace or project root where
possible. Moving the project may invalidate cached catalog paths but must not
invalidate manifests or Parquet evidence. `catalog rebuild` resolves paths from
the workspace's current location.

### 8.4 Configuration

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
allow_mcp_ad_hoc_commands = false
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
2. the selected named workload in project `flamo.toml`;
3. `.diagnostics/config.toml`;
4. built-in safe defaults.

MCP tool inputs may select among allowed options but cannot override workspace
security policy. Environment-variable configuration is limited to documented
non-secret process settings such as color and log level; there is no generic
environment-to-configuration mapping.

The project-controlled `flamo.toml` is untrusted until a human approves its
canonical workload definition hash through the CLI. Editing the command,
working directory, environment, validator, resource policy, or included
configuration revokes that approval. MCP can select an approved workload but
cannot approve one.

## 9. Identity and provenance

### 9.1 Run identity

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

### 9.2 Artifact identity

An artifact ID is `sha256:<lowercase hex digest>` of the exact payload bytes.
The digest is calculated after the collector closes the staged file. A payload
is copied or safely reflinked into the store; symlinks and mutable hard links
are not used.

Artifact extensions are descriptive and excluded from identity. Content-level
metadata does not include kind, media type, display name, producer, role, or
sensitivity because identical bytes may be registered in different contexts.
Those fields belong to a run-to-artifact registration.

### 9.3 Environment identity

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

### 9.4 Source identity

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
source identity prevents a comparison from being confirmatory. Flamo never
checks out, resets, or changes revisions; callers provide baseline and candidate
as separate roots, installations, or resolved commands.

### 9.5 Workload identity

A workload definition is the canonicalized declared command template,
parameters, working-directory rule, environment policy, timeout, resource
limits, validation specification, and allowed collectors. A workload instance
adds resolved parameter values, executable, arguments, working directory, and
controlled environment overrides.

Warm-up, repetition, randomization, collector settings, and stopping rules are
separate experiment-design and measurement-protocol identities. Workload
identity does not include source or machine, allowing the same logical workload
to be compared across those dimensions without conflating them.

## 10. Run manifest

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
  "limitations": []
}
```

### 10.1 State machine

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

### 10.2 Command representation

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

### 10.3 Process result

Record:

- exit code or terminating signal;
- start and end monotonic timestamps;
- wall time;
- peak RSS when available;
- stdout/stderr artifact references or bounded excerpts;
- timeout and cancellation cause;
- child process cleanup result.

Stdout and stderr are size-limited and redacted according to configuration.

## 11. Artifact model

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

## 12. Evidence model and Parquet schemas

### 12.1 Storage rules

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
- Schema evolution is additive only for declared nullable fields within a major
  schema version. `union_by_name = true` implements that declared evolution; it
  is not itself the evolution policy.
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

### 12.2 Investigations and experiments

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

Every attempted trial remains visible, including failures and exclusions.
Analyses report counts and reasons rather than filtering them silently.

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

### 12.3 `runs`

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

### 12.4 `artifacts`, `environments`, and `source_states`

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

### 12.5 `measurements`

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

### 12.6 `frames`

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

### 12.7 `frame_measurements`

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
representative stacks through native-format or Perfetto queries. Flamo reuses
the pprof mapping/location/function/line model when normalizing cross-profile
stacks instead of inventing an incompatible universal stack schema.

### 12.8 `observations`

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

### 12.9 `analyses` and `evidence_refs`

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

### 12.10 `comparisons`

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

### 12.11 `findings`

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

## 13. DuckDB catalog

### 13.1 Responsibilities

`catalog.duckdb` provides:

- stable schemas, macros, and query definitions used to construct
  snapshot-local views over Parquet;
- schema and extractor compatibility metadata;
- parameterized analytical queries used by recipes.

Measured, reproducible cached summaries may be added after profiling proves
they improve an important query. The initial catalog does not promise
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

### 13.2 Rebuild

`flamo catalog rebuild`:

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
Detected corruption is reported. The explicit `flamo repair` operation,
separate from rebuild, may move recoverable material under the mutation locks
only from a validated, previewable repair plan.

### 13.3 SQL safety

Flamo does not expose arbitrary SQL through MCP or its agent-facing CLI.

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
- result rows and bytes are capped;
- every result records the query name and query version.

The exact DuckDB release is pinned and tested because configuration names and
the interaction between external-access restrictions and `read_parquet` can
change. Catalog validation accepts only known views and macros and rejects
unexpected attachments, extensions, or secrets. Blocking queries run outside
the event loop with a dedicated connection; cancellation invokes DuckDB
interruption before joining the worker.

Advanced users can open the local catalog with DuckDB's own CLI. That process
does not honor Flamo's catalog or retention locks and remains outside Flamo's
safety and concurrency contract.

### 13.4 Query result budgets

Every query accepts or derives:

- maximum rows;
- maximum serialized bytes;
- timeout;
- sort order;
- optional cursor.

If truncated, the result includes `truncated=true`, the applied limit, and a
stable continuation cursor when the query supports pagination. A cursor binds
the query version, normalized filters, ordering, last sort key, and corpus
commit ID. It is opaque to callers. A cursor from a different commit fails
explicitly rather than silently skipping or duplicating rows.

## 14. Perfetto integration

Perfetto Trace Processor is the authoritative query engine for detailed
temporal traces.

The adapter may use:

- the Perfetto Python API;
- `trace_processor_shell` as a subprocess;
- custom versioned PerfettoSQL packages;
- structured trace summaries.

The Python API must be configured with a locally installed or deliberately
packaged Trace Processor binary. Its convenience download behavior is disabled;
Flamo never fetches a binary during an agent operation.

Initial Perfetto support is import and query, not unspecified system-wide
capture. Trace Processor is the preferred ingestion path for formats it
supports, including Perfetto protobuf traces, Chrome JSON traces, pprof
profiles, and `perf.data`. Flamo records the Trace Processor binary version and
SHA-256 with every extraction or query.

Initial query modules:

```text
flamo.cpu.hotspots
flamo.cpu.run_queue_delay
flamo.process.wall_breakdown
flamo.threads.blocked_time
flamo.operations.longest
flamo.operations.repeated
flamo.pytorch.operator_summary
flamo.pytorch.cpu_gpu_sync
flamo.accelerator.idle_gaps
flamo.memory.timeline_summary
```

Each query module declares:

- supported trace kinds and required tables;
- parameters and defaults;
- output schema;
- unit and polarity of every metric;
- expected query complexity;
- known blind spots;
- module version.

Flamo does not copy the entire Perfetto trace into Parquet. It extracts only
cross-run summaries and source-linked measurements required by supported
recipes.

## 15. Adapter system

### 15.1 Interface

Adapters implement:

```python
class Adapter(Protocol):
    name: str

    async def probe(self, context: ProbeContext) -> CapabilityReport: ...
    async def plan(self, request: CaptureRequest) -> CapturePlan: ...
    async def capture(
        self,
        plan: CapturePlan,
        execution: ExecutionContext,
    ) -> CaptureResult: ...
    async def extract(
        self,
        artifact: ArtifactRef,
        context: ExtractionContext,
    ) -> EvidenceBatch: ...
    async def validate(
        self,
        artifact: ArtifactRef,
    ) -> ArtifactValidation: ...
```

`plan` is side-effect free and returns the exact executable, arguments,
permissions, estimated overhead, output types, and limitations. `capture` may
only execute an approved plan. `extract` reads immutable artifacts and writes
only to staging.

### 15.2 Discovery

Built-in adapters are registered explicitly. Third-party adapters use Python
entry points under `flamo.adapters`. Entry points execute Python code and are
therefore part of the trusted computing base. Only built-ins load by default.
A third-party plugin requires explicit CLI approval by distribution name,
version, and package identity, and its approval is revoked when that identity
changes. Loading an entry point does not by itself authorize collector
execution.

Each adapter has independent:

- domain tests;
- executable probes;
- fixture artifacts;
- extractor version;
- supported platform matrix;
- optional dependencies.

### 15.3 Initial adapters

#### `pyperf`

Use for controlled Python benchmarks. Reuse calibration, warm-ups, worker
processes, machine metadata, instability checks, and JSON output. Preserve
pyperf JSON as a native artifact and load it through public
`BenchmarkSuite`/benchmark value and metadata APIs. Flamo does not import
pyperf's private `_compare` helpers or parse human-readable comparison output.
pyperf owns calibration, warm-ups, workers, runs, and values within a benchmark
trial; Flamo's experiment blocks and treatment randomization sit above that
hierarchy and do not duplicate it.

#### `py-spy`

Use for out-of-process Python CPU sampling and attach workflows. Prefer a
machine-readable Chrome trace for initial extraction and feed it to Perfetto
Trace Processor. py-spy currently emits flamegraph, raw, Speedscope, and Chrome
trace formats; the adapter must not claim it emits pprof. Record whether native
frames, subprocesses, idle threads, and GIL state were captured. Speedscope or
raw output may be preserved for existing viewers, but Flamo does not implement
a bespoke sampled-stack parser when Trace Processor can ingest the chosen
format.

#### `torch.profiler`

Use for operator-level CPU and accelerator activity. Record all enabled
features because stacks, shapes, modules, FLOPs, and memory change overhead and
evidence completeness. Separate compilation/warm-up from steady state.

The adapter has three explicit capability tiers:

- trace import for an existing Chrome/Perfetto-compatible export;
- whole-entrypoint launcher mode for a declared Python script or module, with no
  promise of application-specific phase separation;
- SDK/recipe mode for user-instrumented steps, phases, schedules, and semantic
  annotations.

Arbitrary non-Python commands cannot be transparently wrapped by
`torch.profiler`; the adapter must report that limitation instead of pretending
to provide operator evidence. `with_modules` is treated as a TorchScript-only
capability unless current upstream behavior proves otherwise; FLOP coverage is
operator-limited. Scheduled captures must register every exported cycle or
explicitly state that only the final cycle was retained.

#### `memray`

Use supported Memray readers and reporters. Do not decode its private binary
format. Extract peak, retained, and allocation-volume summaries while
preserving the original capture. High-level statistics may use Memray's
supported JSON reporter; stack evidence uses its public `FileReader`. Results
name the exact concept measured: high-water mark, live allocations at end,
temporary allocation count, total allocation volume, tracked heap, or RSS.
They also record aggregated versus all-allocation mode and all outputs produced
by follow-fork capture.

#### `perf`

Use Linux perf through separate explicit modes: `record`, `stat`, `sched`,
off-CPU, and system-wide capture. Preserve `perf.data` and feed supported data
to Trace Processor. Record kernel restrictions, scope, privilege,
symbolization coverage, and build identities.

#### `perfetto`

Import Perfetto-compatible traces and run versioned PerfettoSQL queries.
Explicit, platform-specific system capture may be added later as a separately
scoped mode.

#### `coverage`

Use coverage.py's supported data APIs for Python line, branch, and dynamic
context evidence. Preserve the native coverage data and extract bounded
repository-relative line and arc sets. This adapter answers whether a path
executed; it does not claim why the path executed or what values flowed through
it.

### 15.4 Later adapters

- GDB/LLDB and elfutils for core metadata, with user init files, autoload, and
  implicit debuginfod disabled;
- AddressSanitizer, UndefinedBehaviorSanitizer, and ThreadSanitizer raw reports
  with narrow producer-versioned extraction rather than a claimed universal
  JSON schema;
- `rr` recording references;
- Nsight Systems supported Arrow, JSONL, SQLite, or Parquet exports;
- `rocprofv3` Perfetto-compatible exports;
- heaptrack or platform-native heap profiles;
- VizTracer or Python monitoring integrations when ordered call evidence is
  necessary and Perfetto annotations are insufficient;
- OpenTelemetry Profiles once its profile signal is stable enough for a
  compatibility commitment.

## 16. Workload and experiment specification

### 16.1 Project configuration

Repeatable workloads live in `flamo.toml` at the project root:

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

### 16.2 Repetition ordering

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

### 16.3 Validation oracle

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

### 16.4 Semantic observations

Some important defects are configuration or algorithmic invariant violations,
not conventional hotspots. The optional SDK supports explicit annotations:

```python
from flamo.sdk import observe, phase

with phase("ppo_epoch"):
    observe("policy.old_log_prob_source", source="rollout")
    observe("policy.clip_fraction", value=clip_fraction)
```

Annotations must serialize to bounded primitive values and may be exported as
Perfetto events or a structured observation artifact. This provides agents with
evidence such as whether a branch executed, which policy snapshot supplied a
value, or how many times an update path ran.

## 17. Analysis recipes

### 17.1 Recipe contract

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

### 17.2 `cpu_hotspots`

Returns:

- top exclusive and inclusive frames;
- source locations;
- bounded callers, callees, and representative stacks for each hotspot;
- percentage of captured samples represented;
- symbolization and sample coverage;
- thread/process filters;
- native artifact and viewer references.

### 17.3 `scaling`

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

### 17.4 `compare_run_sets`

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

### 17.5 `pytorch_operator_breakdown`

Returns:

- operator calls, self time, total time, and device time;
- shapes only when captured;
- CPU/accelerator synchronization indicators;
- compilation and warm-up separation;
- repeated small operations;
- memory allocation summaries;
- trace coverage.

### 17.6 `memory_growth`

Returns:

- high-water mark, live-at-end allocations, total allocation volume, and RSS
  only where the collector supports each concept;
- allocation volume and counts;
- top retained stacks;
- phase or input-correlated growth;
- native versus Python allocation coverage;
- limitations around RSS and allocator caching.

### 17.7 `execution_path`

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

### 17.8 `failure_population`

Returns:

- deterministic groups and their sizes;
- dimensions enriched in each group;
- change points;
- representative artifacts;
- data-quality and symbolization coverage;
- competing hypotheses rather than one forced conclusion.

## 18. Statistical and comparison policy

Flamo must preserve raw measurements and avoid presenting one aggregate as the
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

Flamo delegates benchmark collection, calibration, warm-up, and instability
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

## 19. CLI specification

### 19.1 General behavior

The executable is `flamo`. Commands produce readable terminal output by
default and structured JSON with `--json`. CLI JSON is the domain result model;
MCP wraps that same model in its transport envelope.

Common options:

```text
--workspace PATH
--project-root PATH
--json
--quiet
--log-level LEVEL
--timeout SECONDS
```

Exit codes:

- `0`: successful operation;
- `1`: operation completed with a negative result such as failed validation;
- `2`: invalid arguments or configuration;
- `3`: required capability unavailable;
- `4`: capture or external process failed;
- `5`: artifact or corpus integrity failure;
- `6`: comparison invalid;
- `7`: lock or concurrency failure;
- `8`: operation cancelled or timed out;
- `9`: safety or policy refusal.

### 19.2 Workspace commands

```text
flamo init
flamo status
flamo capabilities [--refresh]
flamo config show
flamo validate
flamo workload approve <name>
```

`init` creates only local files and never installs collectors. `capabilities`
reports installed tools, versions, supported modes, permissions, and remediation
commands.

### 19.3 Capture commands

```text
flamo capture plan <adapter> [options] -- <argv...>
flamo capture run <adapter> [options] -- <argv...>
flamo import <path> [--kind KIND]
```

`capture plan` is side-effect free. `capture run` prints the plan before
execution unless `--json` is used, in which case the plan is part of the result.
Every import creates a new import run.

### 19.4 Workload commands

```text
flamo workload list
flamo workload show <name>
flamo workload run <name> [parameter overrides]
flamo investigations create <structured-input>
flamo investigations list [filters]
flamo investigations show <investigation-id>
flamo hypotheses record <structured-input>
flamo hypotheses show <hypothesis-id>
flamo experiment plan <name> [parameter overrides]
flamo experiment run <name> [parameter overrides]
flamo experiment show <experiment-id>
```

### 19.5 Analysis commands

```text
flamo analyze hotspots <run-or-artifact>
flamo analyze scaling <experiment-or-run-set>
flamo analyze compare <baseline-run-set> <candidate-run-set>
flamo analyze pytorch <run-or-artifact>
flamo analyze memory <run-or-artifact>
flamo analyze execution <run-or-artifact> [--compare RUN]
flamo analyze failures [filters]
flamo analyze record <structured-input>
flamo analyze record-comparison <structured-input>
```

### 19.6 Evidence commands

```text
flamo runs list [filters]
flamo runs show <run-id>
flamo artifacts list [filters]
flamo artifacts show <artifact-id>
flamo findings list [filters]
flamo findings show <finding-id>
flamo findings record <structured-input>
flamo evidence get <typed-reference>
flamo measurements query [curated filters]
flamo stacks callers <run-or-artifact> <frame-id>
flamo stacks callees <run-or-artifact> <frame-id>
flamo stacks examples <run-or-artifact> <frame-id>
flamo trace window <artifact-id> --start NS --end NS
flamo open <artifact-id>
```

Drill-down commands are bounded and use reviewed query families. They report
total and returned counts, truncation, coverage, and stable keyset cursors.
There is no arbitrary SQL or free-form PerfettoSQL command.

`open` prints the appropriate installed viewer command by default.
`flamo open --launch` executes it explicitly. It never launches a browser when
`--json` is active.

### 19.7 Catalog and recovery

```text
flamo catalog validate
flamo catalog rebuild
flamo recover
flamo repair <structured-plan>
flamo gc --dry-run
flamo gc --apply
flamo gc --purge <trash-manifest>
```

Garbage collection is always dry-run by default. Apply may move only eligible
staging, generations, caches, and unreferenced artifacts to recoverable trash.
It reports exact paths, retention roots, and recoverability before mutation.
Purge is a separate destructive action against a specific expired trash
manifest.

### 19.8 MCP

```text
flamo mcp serve [--init]
flamo mcp inspect
```

`serve` uses stdio exclusively. `--init` performs the additive workspace
initialization before protocol startup and never approves project-controlled
workloads. A network transport is outside the initial product contract.

## 20. MCP server specification

### 20.1 SDK and transport

The server uses the official Python SDK pinned to `mcp==2.0.0b2`. The pin is
required because v2 is a prerelease and unpinned resolution may select an
incompatible stable v1 release.

The supported transport is stdio. The server:

- writes protocol messages only to stdout;
- writes diagnostics to stderr or MCP logging;
- initializes workspace and read services through SDK lifespan state;
- returns an explicit MCP result envelope with concise text and one structured
  payload;
- reports progress for long captures and analyses;
- propagates cancellation through the execution broker and containment backend;
- closes DuckDB and Perfetto resources on shutdown.

Handlers translate SDK inputs and outputs but contain no diagnostic logic:

```python
from typing import Annotated

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp_types import CallToolResult, TextContent, ToolAnnotations

server = MCPServer("flamo", lifespan=flamo_lifespan)


@server.tool(
    annotations=ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )
)
async def compare_run_sets(
    request: CompareRunSetsRequest,
    ctx: Context[AppContext],
) -> Annotated[CallToolResult, ToolPayload[ComparisonResult]]:
    services = ctx.request_context.lifespan_context.services
    try:
        result = await services.comparisons.compare(request)
        payload = ToolPayload.ok(result)
        return CallToolResult(
            content=[TextContent(type="text", text=result.short_summary())],
            structured_content=payload.model_dump(mode="json"),
        )
    except DomainError as error:
        payload = ToolPayload.failed(error.to_detail())
        return CallToolResult(
            content=[TextContent(type="text", text=error.concise_message)],
            structured_content=payload.model_dump(mode="json"),
            is_error=True,
        )
```

`ToolPayload[T]` is an object-root schema with `schema_version`, `ok`,
`result | null`, and `error | null`. Domain errors use `isError=true` with that
structured payload. Invalid protocol requests and missing tools remain JSON-RPC
errors. Returning bare Pydantic models is forbidden because this SDK beta
duplicates them into text and structured content; uncaught domain exceptions
are forbidden because their structured detail is lost.

The concrete v2 SDK context, annotations, content blocks, and lifespan types are
confined to `flamo.mcp`. The server calls synchronous `server.run()` for stdio.
It does not depend on experimental MCP background-task APIs.

### 20.2 Tool design

Tools are few, task-shaped, and versioned through their result schemas. Every
registration supplies explicit `ToolAnnotations`.

- Query, status, and analysis tools set `read_only_hint=True`,
  `destructive_hint=False`, `idempotent_hint=True`, and
  `open_world_hint=False`.
- Read-only analysis results are deterministic for their pinned inputs and do
  not publish evidence. `record_analysis` and `record_comparison` are explicit
  additive operations that materialize typed provenance.
- `plan_capture` and `plan_experiment` are read-only, idempotent, and
  closed-world with the same annotation values.
- Imports and finding revisions are mutating, additive where applicable,
  non-idempotent, and closed-world:
  `read_only_hint=False`, `destructive_hint=False`,
  `idempotent_hint=False`, `open_world_hint=False`.
- Capture and experiment execution are mutating, non-idempotent, potentially
  destructive, and open-world:
  `read_only_hint=False`, `destructive_hint=True`,
  `idempotent_hint=False`, `open_world_hint=True`, because an uncontained
  workload may modify files or use the network.

Initial tools:

#### `initialize_workspace`

Additive and idempotent. Initializes only the MCP server's fixed project root.
It cannot select an external path or approve workloads. Hosts may instead start
the server using `flamo mcp serve --init`.

#### `workspace_status`

Read-only. Returns workspace identity, validation state, catalog state, storage
usage, active captures, and warnings.

#### `list_capabilities`

Read-only. Returns adapter capabilities, installed versions, required
permissions, unavailable features, and remediation. The default call performs
only passive inspection. Active executable probes are separately requested,
bounded, and executed through the subprocess broker; merely listing
capabilities does not run a project-controlled binary found on `PATH`.

#### `plan_capture`

Read-only. Default inputs are an approved `workload_name`, declared scalar
parameter overrides, adapter, and requested features. Returns the exact
resolved plan, expected artifacts, overhead, permissions, limits, containment
state, warnings, request digest, and a short-lived `plan_id`. The ordinary
schema does not advertise `argv` or `cwd`. If explicitly enabled, ad-hoc MCP
capture uses separately named `plan_ad_hoc_capture` and
`execute_ad_hoc_capture_plan` tools with consequential-action annotations.

Plan IDs are 256-bit opaque random values held only in MCP-process memory. They
are bound to workspace ID, workload approval hash, resolved executable and
identity, arguments, working directory, child-environment policy and overrides,
adapter/version/capabilities, requested features, validator, source state,
limits, containment decision, and policy generation. Expiry uses monotonic
time. Plans are atomically single-use, consumed before process creation,
bounded in count, invalid after restart, and rejected if any bound input changes.
A request digest is audit evidence, not an authorization boundary.

#### `execute_capture_plan`

Mutating and command-executing. Executes an unexpired `plan_id`, streams
progress, observes cancellation, and returns a completed or failed run record.
Every bound identity, capability, approval, and policy input is rechecked
immediately before execution. It never accepts a shell string or replacement
arguments.

#### `plan_experiment` and `run_experiment`

Planning is read-only over an approved named workload and experiment
definition. Execution runs the predeclared variants and blocks, registers every
attempted trial, validates outputs, and returns the experiment plus initial
analysis references.

#### `create_investigation` and `record_hypothesis`

Mutating, additive structured operations. Hypothesis revision requires
`expected_revision`, an explicit prediction, and a discriminating condition.
Corresponding bounded list/get tools allow agents to resume an investigation
without relying on prior conversation context.

#### `import_artifact`

Mutating. Imports a file under allowed roots, computes its identity, validates
the selected or detected kind, creates a new import run, and optionally extracts
evidence.

#### `list_runs`

Read-only, filtered, sorted, and paginated. It cannot return unbounded
manifests.

#### `get_run`

Read-only. Returns one run manifest summary, artifact references, validation,
and limitations.

#### `get_artifact`

Read-only metadata. Returns artifact identity, kind, size, sensitivity,
integrity, supported analyses, and paginated registrations. An optional
`run_id` selects one contextual registration. It does not return binary
content.

#### `analyze_hotspots`

Read-only. Runs the appropriate profile or trace query and returns bounded
source-linked hotspots.

#### `analyze_scaling`

Strictly read-only over an existing experiment or frozen run set. It never
executes missing trials. Work requiring collection uses `plan_experiment` and
`run_experiment`.

#### `compare_run_sets`

Read-only. Returns compatibility, metric differences, frame/operator changes,
validation, attempted/failure counts, estimand, and limitations. Pairwise run
IDs are accepted as one-element run-set shorthand.

#### `record_analysis` and `record_comparison`

Mutating and additive. These execute the same curated deterministic recipes as
the read-only tools, then persist an `AnalysisRecord`, result digest, exact
input corpus commit, typed evidence references, coverage, and limitations.
They do not accept arbitrary SQL or caller-supplied result bodies.

#### `analyze_pytorch`

Read-only over an existing trace. Returns operator and accelerator summaries.

#### `analyze_memory`

Read-only over an existing memory artifact or run.

#### `analyze_execution`

Read-only over coverage, trace annotations, configuration observations, and SDK
observations. Returns bounded source-path evidence and, optionally, differences
between two runs.

#### `record_finding`

Mutating. Stores a structured claim with typed evidence references. Updating
requires `expected_revision`. It validates reference existence, assessment
requirements, and evidence-level consistency.

#### `list_findings` and `get_finding`

Read-only, filtered, bounded, and paginated finding retrieval.

#### `get_evidence` and bounded drill-down tools

Read-only operations retrieve a typed evidence reference, curated measurement
query, hotspot callers/callees/representative stacks, or a bounded trace
time-window/event neighborhood. They return stable cursors, total/returned
counts, coverage, and limitations. They do not accept raw SQL.

#### `validate_workspace`

Read-only by default. Checks manifests, artifact hashes on request, Parquet
schemas, references, and catalog freshness. Repairs require the explicit CLI
repair command; MCP never repairs.

### 20.3 MCP resources

Resources provide stable, bounded representations:

```text
flamo://runs/{run_id}
flamo://artifacts/{artifact_id}
flamo://findings/{finding_id}
flamo://investigations/{investigation_id}
flamo://hypotheses/{hypothesis_id}
flamo://experiments/{experiment_id}
flamo://run-sets/{run_set_id}
flamo://analyses/{analysis_id}
flamo://comparisons/{comparison_id}
flamo://observations/{observation_id}
flamo://schemas/{schema_name}/{version}
```

Resources return JSON or text summaries. Large native artifacts are represented
by metadata and local handles, not injected into model context. Template
resources declare `mime_type="application/json"`, percent-encode identifiers,
and resolve services through a server-local lifespan closure. In the exact
`2.0.0b2` wheel, template-handler `Context` is reconstructed by Pydantic and
loses its private request state, so `ctx.request_context` raises at runtime.
Resource handlers therefore omit `Context`; the MCP adapter stores the active
lifespan value in a closure for the duration of `server.run()`. Tool handlers
continue using injected `ctx.request_context.lifespan_context`. A contract test
must fail if a future SDK change invalidates either path, and the workaround is
removed on upgrade when template context is proven functional.

Mutable workspace and capability views remain tools rather than static
resources because this SDK beta does not inject lifespan context into static
resource handlers. Tool results should include MCP `ResourceLink` blocks for
addressable runs, artifacts, findings, analyses, and comparisons when useful.

### 20.4 No MCP prompts initially

Investigation recipes belong in executable domain logic and tool descriptions,
not MCP prompt templates. Prompts can be added only when they express a stable
human-facing workflow that cannot be represented by structured tools.

### 20.5 Progress

Long operations report named phases:

```text
planning
warming_up
capturing
validating_artifact
extracting
publishing
analyzing
completed
```

Progress is one monotonic stream for the entire request; it never resets at a
phase boundary. A default operation uses fixed phase work units, for example
`0/8 planning` through `8/8 completed`. Measurable sub-work may occupy a fixed
interval. Unknown-duration work reports phase transitions and uses MCP logging
for elapsed time rather than inventing percentages. Reporting is best-effort:
`ctx.report_progress()` is a no-op when the client supplied no progress token.

### 20.6 Cancellation

An incoming MCP cancellation cancels the handler's AnyIO scope. Cleanup must
therefore be explicitly shielded and bounded:

```python
try:
    return await execution.execute_capture_plan(plan)
except anyio.get_cancelled_exc_class():
    with anyio.CancelScope(shield=True):
        with anyio.fail_after(cleanup_timeout):
            await execution.cancel_containment(plan.run_id)
            await runs.publish_cancelled(plan.run_id)
    raise
```

Cleanup sends a graceful signal, waits a bounded interval, terminates the
entire containment unit or process tree, retains complete validated artifacts,
quarantines incomplete artifacts, publishes terminal lifecycle revisions, and
releases staging resources and locks. It then re-raises cancellation. Startup
recovery handles process death before shielded cleanup completes. Cancellation
must not leave an active lease indefinitely.

### 20.7 Structured errors

Every tool returns the same object-root transport envelope. For an expected
domain failure, `ok=false`, `result=null`, `error` contains:

```json
{
  "code": "CAPABILITY_UNAVAILABLE",
  "message": "Native stack capture is unavailable on this platform.",
  "retryable": false,
  "details": {},
  "remediation": [
    "Install py-spy or select Python-only stack capture."
  ],
  "run_id": null
}
```

Initial codes:

- `WORKSPACE_NOT_FOUND`;
- `WORKSPACE_INVALID`;
- `CAPABILITY_UNAVAILABLE`;
- `INVALID_CAPTURE_PLAN`;
- `EXECUTION_REFUSED`;
- `PROCESS_FAILED`;
- `PROCESS_TIMEOUT`;
- `PROCESS_CANCELLED`;
- `ARTIFACT_TOO_LARGE`;
- `ARTIFACT_INTEGRITY_FAILED`;
- `ARTIFACT_PARSE_FAILED`;
- `EVIDENCE_SCHEMA_MISMATCH`;
- `COMPARISON_INVALID`;
- `QUERY_BUDGET_EXCEEDED`;
- `WRITE_LOCK_TIMEOUT`;
- `SENSITIVE_ARTIFACT_REFUSED`;
- `INTERNAL_ERROR`.

Tracebacks are logged locally but not returned by default.

## 21. Concurrency and atomicity

### 21.1 Operations

Operations fall into three classes:

- `read`: pinned manifests, commit inventories, Parquet, artifact metadata, and
  analytical queries;
- `capture`: long-running external collection into a unique staging directory;
- `commit`: short mutation publishing manifests, artifacts, generations, and a
  new corpus commit;
- `retention`: compaction, trash movement, purge, and repair that may make old
  paths unavailable.

Multiple reads and captures may run concurrently. Commits are serialized.

### 21.2 Workspace write lock

Portalocker controls `.diagnostics/write.lock` with an exclusive advisory lock.
Both CLI and MCP honor it. Portalocker is used instead of a custom lock because
it exposes cross-platform shared and exclusive lock modes and bounded waits.

The lock covers:

- allocating or transitioning persistent run state;
- publishing an artifact into the content store;
- publishing immutable generation manifests and corpus commits;
- updating or replacing `catalog.duckdb`;
- retention mutations.

It does not cover workload execution, profiling, extraction into staging, or
read-only analysis.

`.diagnostics/catalog.lock` is separate:

- analytical readers hold a shared catalog lock only while a DuckDB connection
  uses `catalog.duckdb`;
- catalog rebuild holds the exclusive catalog lock;
- ordinary evidence publication does not rebuild the catalog; new analyses pin
  the new commit and resolve its explicit inventory;
- workspace commits never wait for long-running analytical readers unless they
  also request catalog replacement.

`.diagnostics/retention.lock` prevents garbage collection or repair from
removing a file used by a pinned reader. Analyses that may open artifacts or
Parquet hold it shared for the operation; GC, purge, and repair hold it
exclusive.

The global lock order is workspace write lock, then retention lock, then
catalog lock when multiple exclusive locks are needed. Readers acquire
retention before shared catalog. Code never acquires the workspace write lock
while holding a read connection. An analysis releases its connection and
retention lock before recording a finding or performing another commit.

### 21.3 Commit protocol

1. Capture in `.diagnostics/staging/<operation-id>/`.
2. Close producer descriptors; validate, size, and hash payloads.
3. Acquire the write lock, publish content-addressed objects and run-artifact
   registrations, append lifecycle revisions, and release the lock.
4. Extract from immutable objects into a staged generation using checked-in
   Arrow schemas.
5. Validate row budgets, referential integrity, Parquet hashes, and row counts.
6. Write and validate the immutable generation manifest containing exact
   inputs and output files.
7. Acquire the write lock.
8. Pin and recheck current `corpus/HEAD`, run revisions, and supersession
   preconditions.
9. Move all generation files and its manifest to final immutable paths.
10. Write an immutable corpus commit whose inventory is the previous commit
    plus the new/superseding generation.
11. Flush required files and directories under the filesystem durability
    policy.
12. Atomically replace `corpus/HEAD` as the final visibility step.
13. Publish current run projections and release the lock.

Publishing raw artifacts before extraction means an extractor crash cannot lose
a valid, expensive capture. Extraction can be retried without executing the
workload again.

If the process crashes before `HEAD` advances, any new immutable files are
orphans and invisible. Recovery may resume or quarantine staging and reports
unreachable final files without guessing that they are committed. If `HEAD`
advances, the referenced generation is complete by construction.
Content-addressed publication, generation placement, and commit creation are
idempotent.

This protocol assumes a supported local filesystem with advisory locks and
same-volume atomic replace. Initialization validates those assumptions.
Durability requires flushing files before the directory entry or `HEAD`
replacement. Platforms without equivalent semantics are reported as degraded;
their precise crash guarantees are tested and documented rather than assumed.

### 21.4 Reader behavior

Readers:

- never read from staging;
- pin `corpus/HEAD` once and resolve only files in that commit's immutable
  inventory;
- use read-only DuckDB connections;
- keep the shared retention lock for all file-backed work in the analysis;
- remain unaffected by newer commits appearing between queries;
- include the corpus commit ID and snapshot timestamp in results;
- retry once when a catalog replacement invalidates a connection.

Within the MCP server, a process-local asynchronous read/write coordination
primitive prevents catalog replacement while active readers hold connections.
Cross-process catalog coordination uses the shared/exclusive catalog lock.

## 22. Security and privacy

### 22.1 Local does not mean harmless

The MCP host may be controlled by an agent. Flamo must not turn a diagnostic
tool into unrestricted command, filesystem, or secret access.

The default threat model protects against malformed inputs, accidental
confused-deputy behavior, stale plans, symlink races, resource exhaustion, and
untrusted artifact contents. A named workload is still arbitrary same-user
code. Without an active containment backend it can read the user's accessible
files, mutate the project or `.diagnostics`, access inherited credentials, and
use the network. Flamo reports that boundary honestly rather than calling a
named workload safe.

### 22.2 Command execution

- Accept only argument arrays.
- Never invoke a shell.
- Resolve executables through a documented policy.
- By default, allow MCP execution only for named workloads declared in
  `flamo.toml` and approved through the CLI by canonical definition hash.
- Require `allow_mcp_ad_hoc_commands = true` for an MCP client to plan an
  arbitrary argument array.
- Bind MCP execution to the single-use plan invariants in section 20.
- Restrict working directories to configured roots.
- reject NUL bytes and invalid path encodings;
- apply time and artifact-size limits;
- preserve a previewable capture plan;
- treat privileged collectors as disabled by default;
- use Bubblewrap plus cgroup/systemd-scope containment on Linux when policy
  requires it, hiding `.diagnostics`, limiting writable roots, constructing a
  minimal environment, applying CPU/memory/process limits, and controlling
  network access;
- refuse MCP execution when required containment is unavailable; otherwise
  report the process-group fallback as degraded;
- never execute commands embedded in imported artifacts.

### 22.3 Environment collection

Child environment and recorded environment use separate allowlists. The broker
constructs a minimal child environment rather than forwarding the host
environment. Dangerous loader and interpreter controls such as `LD_PRELOAD`,
`LD_LIBRARY_PATH`, `DYLD_*`, `PYTHONPATH`, debugger init variables, and
credential variables are excluded unless a human allows a specific named
workload to receive them. Recorded metadata uses a second, normally narrower
allowlist. Names indicating tokens, passwords, secrets, keys, credentials, or
cookies are excluded even from broad patterns unless explicitly allowed by
local policy.

### 22.4 Filesystem access

- Canonicalize and validate every requested root.
- Reject traversal outside configured roots in MCP.
- For imports, open beneath an approved directory without following symlinks,
  require a regular file with an acceptable link count, reject
  devices/FIFOs/sockets and mutable hard-link sources, stream from that same
  descriptor into staging under a size budget, compare `fstat` identity and
  size before and after, then hash the staged destination.
- Do not retain mutable hard links or source references.
- use restrictive permissions for `.diagnostics`;
- mark sensitive artifacts explicitly;
- treat extracted strings, source names, log text, and artifact metadata as
  untrusted data that may contain terminal escapes or prompt injection;
- return no stdout/stderr excerpt over MCP by default; expose only metadata and
  an explicitly bounded local reference.

Workspace, staging, import, file-count, row-count, string-length, nesting,
stdout/stderr, temporary-disk, memory, CPU, and process quotas are enforced
before or while consuming input. Publication checks remaining free space.

### 22.5 DuckDB

Raw SQL is absent from both Flamo interfaces. Internal connections use the
allowlisted-path, locked-configuration, extension, attachment, secret, memory,
thread, and temporary-file restrictions in section 13. Parameterized query APIs
select from known views only.

### 22.6 Core dumps and process memory

Core dumps may contain credentials, source data, user content, and encryption
keys. They require explicit local configuration and are never exposed as MCP
binary resources. Default analyses return only bounded structural metadata.
Future debugger extraction runs in a bounded worker with GDB `-nx`, autoload
and debuginfod disabled, or the LLDB equivalent with user initialization
disabled.

### 22.7 Network behavior

The Flamo control process performs no network calls during capture or analysis.
Capability remediation may print installation documentation but does not fetch
it. Symbol-server or debuginfod access is disabled unless explicitly enabled
in local configuration and invoked through the CLI. Child workloads may use
the network unless an active containment backend denies it; this is displayed
in every capture plan and result.

### 22.8 Retention and recovery safety

GC and repair use the retention lock described in section 21 and never remove
files beneath active readers. Applying GC first moves eligible material to a
workspace trash area and writes a recovery manifest. A separate explicit purge
removes expired trash. MCP exposes neither operation. Shared content remains
retained while any reachable run registration or pinned corpus commit requires
it.

## 23. Output and evidence quality

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

## 24. Capability detection

Capability detection is divided into passive and active probes. Passive probes
inspect installed distributions, explicitly configured executable paths, and
platform state without executing project-controlled code. Active probes execute
through the broker and require an explicit request. Results are cached only for
the MCP process lifetime unless `refresh=true`. A report contains:

- adapter name and version;
- executable or import location;
- supported capture modes;
- supported artifact formats;
- platform and architecture;
- permission status;
- kernel or driver restrictions;
- optional features such as native frames, GPU activity, or symbols;
- probe timestamp;
- remediation.

A capability may be:

- `available`;
- `degraded`;
- `unavailable`;
- `permission_required`;
- `unsupported_platform`;
- `unknown`.

Probe commands must be bounded and expected to be side-effect free, but active
execution is still labeled as execution. An executable found only through a
project-controlled `PATH` is not run during passive discovery.

## 25. Catalog and artifact integrity

### 25.1 Validation levels

`flamo validate` supports:

- `quick`: manifests, referenced paths, schemas, and sizes;
- `standard`: quick plus Parquet reads and catalog consistency;
- `full`: standard plus rehashing every raw artifact.

MCP uses quick validation at startup and exposes standard validation on request.
Full validation is a CLI operation because it may read many gigabytes.
Validation never mutates evidence. Repair is a separate explicit CLI operation.

### 25.2 Quarantine

Unparseable or partially written files move to `quarantine` with:

- original staged path;
- reason;
- expected and actual format;
- originating run;
- recovery suggestion.

Quarantine is recoverable and never automatically deleted.

### 25.3 Garbage collection

No automatic retention policy exists. `gc --dry-run` identifies:

- abandoned staging directories;
- unreferenced content-addressed artifacts;
- generations unreachable from `HEAD`, retained analysis/run-set snapshots, or
  the recovery window;
- superseded rebuildable catalog caches.

`gc --apply` requires explicit CLI invocation and moves eligible objects to
recoverable trash with a manifest. `gc --purge` is a separate explicit,
destructive CLI action after the recovery window. MCP does not expose either.
Corpus commits referenced by an analysis, comparison, finding, or run set are
retention roots and cannot be pruned while that record remains active.

## 26. Extensibility and schema evolution

### 26.1 Domain schemas

Every manifest, MCP result, and Parquet table carries an integer
`schema_version`.

- Arrow schemas have explicit major and minor versions;
- declared additive nullable fields increment the minor version;
- changed meaning, unit, identity, or required fields does;
- readers support the current version and a declared compatibility window;
- migration produces new derived files and never rewrites raw artifacts;
- MCP result schema changes require contract tests.

### 26.2 Extractor versions

An extractor version changes whenever output semantics change. Re-extraction
from the same native artifact creates new Parquet partitions or a new evidence
generation and supersedes older derivations without deleting them immediately.

### 26.3 Adapter compatibility

Adapters declare supported producer versions. Unknown newer versions are
validated conservatively. An adapter must not parse a format it cannot identify
and then emit apparently valid evidence.

## 27. Observability of Flamo itself

Flamo writes structured local logs containing:

- operation ID;
- run ID when applicable;
- phase;
- adapter;
- elapsed time;
- bounded error details;
- lock wait duration;
- query name and duration;
- rows and bytes returned.

Logs must not contain raw environment dumps, core contents, or unrestricted
stdout/stderr.

`flamo status` reports:

- workspace and catalog validity;
- storage by artifact kind;
- stale or quarantined runs;
- active capture count;
- last catalog rebuild;
- extractor versions;
- capability warnings.

## 28. Testing strategy

### 28.1 Unit tests

- canonical identity and hashing;
- Pydantic contract validation;
- independent lifecycle transitions and run lease recovery;
- balanced block generation and trial accounting;
- path and command safety;
- comparison compatibility;
- evidence-level rules;
- query parameter construction;
- statistical estimands, bootstrap reproducibility, equivalence decisions, and
  incomplete-pair behavior;
- schema evolution.

### 28.2 Adapter fixture tests

Each adapter includes small, legally redistributable fixture artifacts. Tests
validate extraction independently of collector availability. Fixtures cover:

- complete and partial symbols;
- malformed and truncated artifacts;
- empty profiles;
- multiple processes and threads;
- recursive and inlined frames;
- unknown producer versions.

### 28.3 Integration tests

- initialize and rebuild a workspace;
- capture a deterministic local workload;
- cancel and time out a containment unit or degraded process tree;
- deduplicate identical artifact bytes across runs;
- preserve distinct sensitivity and provenance registrations for identical
  bytes;
- run a randomized complete-block experiment and compare frozen run sets;
- delete and rebuild `catalog.duckdb`;
- recover every interrupted commit boundary;
- prove an unpublished generation remains invisible;
- run CLI JSON and MCP calls against the same expected domain result;
- start the real stdio MCP server and perform initialize, list, call, resource
  read, progress, and cancellation operations;
- assert that server stdout contains only valid JSON-RPC messages.

Contract snapshots cover every MCP input schema, output envelope schema,
annotations object, structured success, and structured `isError` result.

### 28.4 Concurrency tests

- concurrent read-only analyses;
- concurrent captures committing in either order;
- CLI commit while MCP reads;
- two external CLI commits contending for the lock;
- catalog rebuild while a read is active;
- reader pinned to an old commit while new evidence publishes and GC waits;
- cancellation during spawn, capture, validation, extraction, write-lock
  acquisition, DuckDB query, and corpus publication;
- crash injection after every commit-protocol step.

### 28.5 Security tests

- shell metacharacters remain literal arguments;
- path traversal and symlink escapes fail;
- import symlink-swap, FIFO, device, hard-link, growth, and truncation races
  fail without blocking;
- raw SQL cannot enter MCP;
- external DuckDB access is unavailable through internal queries;
- unexpected DuckDB attachments, extensions, secrets, and catalog objects fail
  validation;
- secrets are redacted from environment and logs;
- sensitive artifacts cannot be read as resources;
- artifact-derived terminal escapes and instruction-shaped text remain quoted
  untrusted data;
- output and artifact budgets are enforced;
- imported files cannot trigger execution;
- changed workload approval hashes and replayed or expired plan IDs fail;
- required containment failure refuses MCP execution;
- escaped descendants are terminated under the Linux containment backend.

### 28.6 Performance tests

Flamo's own benchmarks cover:

- catalog startup with 10, 1,000, and 100,000 runs;
- file counts and compaction behavior at those scales;
- common cohort and comparison queries;
- artifact hashing throughput;
- Parquet extraction and publication;
- bounded MCP serialization;
- catalog rebuild.

Performance tests must distinguish collector overhead from Flamo orchestration
overhead.

### 28.7 Golden investigations

The initial acceptance corpus should include three end-to-end investigations:

1. a Python reverse scan whose time grows linearly with sequence length;
2. a configuration interaction where execution observations reveal a disabled
   algorithmic safeguard even when the CPU profile is unremarkable;
3. a retained-memory or repeated-allocation regression.

Each golden investigation must produce a hypothesis, evidence references, a
prediction, discriminating validation, analysis record, and before/after
run-set result. The reverse-scan case measures 32K–128K sequence lengths,
preserves the pyperf worker hierarchy, and demonstrates that the semantic
oracle fails when the implementation is deliberately perturbed. The
configuration-interaction case requires semantic observations rather than
inferring correctness from CPU time.

## 29. Initial implementation scope

The first complete vertical slice includes:

- workspace initialization and validation;
- immutable artifact objects and contextual registrations;
- investigations, hypotheses, experiments, variants, trials, independent run
  lifecycles, and exact source/environment identity;
- generation manifests and atomic corpus commits;
- minimal Parquet publication for runs, trials, measurements, comparisons,
  analyses, typed evidence references, and findings;
- rebuildable DuckDB catalog;
- pyperf JSON import through public APIs;
- run-set comparison with a declared paired estimand and oracle;
- CLI commands for import, inspection, experiment analysis, findings, catalog
  rebuild, and recovery.

Subsequent vertical slices add, in order:

- the subprocess broker, Linux containment, approved named-workload capture,
  cancellation, and randomized experiment execution;
- py-spy Chrome trace import/capture through Perfetto, hotspot drill-down, and
  the reverse-scan golden investigation;
- stdio MCP over the already proven services, including plans, annotations,
  envelopes, progress, resources, and cancellation;
- coverage/semantic observations and the configuration-interaction golden
  investigation;
- PyTorch, Memray, and additional Perfetto query families independently;
- compaction and further scale optimization when the 100,000-run benchmarks
  show a need.

The product remains one permanently local CLI/MCP application. These are
implementation slices, not hosted deployment phases. Each slice is
end-to-end, testable, and preserves forward-compatible domain contracts.

The completed initial product includes:

- all vertical slices above;
- pyperf adapter;
- py-spy adapter;
- Perfetto import and Trace Processor queries;
- torch.profiler import or SDK capture;
- Memray import and supported extraction;
- coverage.py import or capture for execution-path evidence;
- `cpu_hotspots`, `scaling`, `compare_run_sets`, `pytorch_operator_breakdown`,
  `memory_growth`, and `execution_path` recipes;
- CLI commands for those operations;
- stdio MCP using `mcp==2.0.0b2`;
- structured findings;
- catalog rebuild and crash recovery.

The initial version does not require core-dump capture, GPU vendor profilers, a
custom UI, background jobs, network transport, arbitrary SQL, or automatic
installation of system tools.

## 30. Suggested implementation order

0. Compatibility spikes: verify exact MCP beta envelopes, errors, annotations,
   lifespan, progress and cancellation; verify the pinned DuckDB path-security,
   empty-schema, interruption, and catalog-replacement behavior; round-trip
   representative collector fixtures.
1. Domain and storage slice: artifact/run/experiment models, generation commit
   protocol, minimal Parquet/DuckDB catalog, pyperf import, run-set comparison,
   and CLI.
2. Execution slice: approved named workloads, subprocess broker, containment,
   quotas, capture leases, cancellation, validation, and atomic publication.
3. Hotspot slice: py-spy Chrome traces through pinned Trace Processor,
   source-linked drill-down, and the reverse-scan golden investigation.
4. MCP slice: `mcp==2.0.0b2` stdio adapter over proven application services and
   real-client contract tests.
5. Semantic slice: coverage.py, bounded observations, configuration-interaction
   golden investigation, and hypothesis/finding workflows.
6. Specialized evidence slices: PyTorch, Memray, and additional Perfetto
   analysis, each independently fixture-tested.
7. Scale hardening: compaction, caches, and additional indexes only when
   representative corpus benchmarks justify them.

MCP follows proven services because it is a thin transport, not an alternate
implementation. Compatibility spikes settle unstable external APIs before
domain code depends on them.

## 31. Acceptance criteria

The initial release is acceptable when:

1. deleting `catalog.duckdb` and running `flamo catalog rebuild` preserves all
   queryable evidence at the same corpus commit;
2. two identical payloads from different runs occupy one artifact object while
   retaining distinct registration provenance and the maximum sensitivity;
3. crash injection at every publication boundary never makes a partial
   generation visible through `corpus/HEAD`;
4. concurrent captures can execute while commits remain serialized;
5. CLI JSON validates as the domain result and MCP's `result` field contains
   that same model inside a stable success/error envelope;
6. no MCP tool accepts a shell command or unrestricted SQL;
7. cancellation during every awaited phase performs bounded shielded cleanup,
   interrupts DuckDB where applicable, terminates contained descendants, and
   leaves terminal lifecycle revisions;
8. comparison results operate on frozen run sets, preserve all attempted
   trials, identify identity/environment mismatches, and refuse invalid proof;
9. every persisted analysis and finding has typed references to existing
   immutable evidence and the exact pinned corpus commit, while ephemeral
   read-only results carry their pinned commit without acquiring the write
   lock;
10. the three golden investigations can be reproduced from a fresh workspace;
11. profiler and extractor limitations appear in results instead of logs alone;
12. native artifacts can be opened with existing ecosystem viewers;
13. a full workspace integrity check detects altered artifact bytes;
14. ordinary read-only analyses do not acquire the workspace write lock;
15. the Flamo control process makes no network requests during operations and
   capture results truthfully report whether child network access was
   contained;
16. every confirmatory optimization result records a primary metric, estimand,
   practical threshold, confidence method, independent unit, paired count,
   source identity, and passing cross-treatment oracle;
17. the real stdio server emits only protocol messages on stdout and passes
   initialize, list, call, resource, progress, structured-error, and
   cancellation contract tests;
18. an agent can move from a bounded hotspot or anomaly result to its callers,
   representative stack, measurement hierarchy, analysis provenance, run,
   artifact registration, and native viewer command without raw SQL;
19. a 100,000-run synthetic corpus meets recorded startup, query, file-count,
   and rebuild budgets after compaction;
20. stale cursors, modified workload approvals, replayed plans, partial source
   identity, incomplete blocks, and failed validation are surfaced explicitly
   rather than silently weakened.

## 32. Explicit decisions

The following choices are settled by this specification:

- Flamo is permanently local.
- Python is the implementation language.
- The MCP SDK is pinned to `mcp==2.0.0b2`.
- stdio is the supported MCP transport.
- DuckDB is the local cross-run analytical engine.
- Parquet is the normalized evidence format.
- native artifacts remain authoritative.
- corpus commits and generation manifests define atomic analytical snapshots.
- Perfetto Trace Processor handles detailed temporal trace analysis.
- there is no PostgreSQL or SQLite database.
- the DuckDB catalog is rebuildable.
- writes are serialized; captures and reads may run concurrently.
- runs, trials, experiments, and investigations are distinct domain entities.
- content identity is separate from contextual artifact registration.
- confirmatory comparisons operate on frozen run sets and preserve failed
  attempts.
- named workloads require CLI approval and are not described as sandboxed
  without active containment.
- MCP does not expose unrestricted command execution, raw SQL, deletion, or
  sensitive artifact content.
- the CLI and MCP share one domain and application layer; MCP adds a transport
  envelope.
- third-party adapters are disabled until explicitly approved through the CLI.
- existing profilers and viewers are reused rather than reimplemented.

## 33. Deferred decisions

These choices should be made only with implementation evidence:

- whether the persistent DuckDB file materially improves startup versus
  constructing an in-memory catalog over Parquet;
- whether pprof or Perfetto should be preferred when a collector genuinely
  supports both and the required queries are equivalent;
- whether retaining source snapshots is worth the sensitivity cost; dirty
  source identity itself is never optional;
- which additional normalized stack representation is needed for
  population-level cross-profile analysis;
- which macOS and Windows adapters can meet the same evidence contract;
- whether OpenTelemetry Profiles is stable enough to become an accepted import
  or export contract;
- whether detached background captures are necessary. The default remains
  synchronous progress and cancellation;
- which platform-specific Trace Processor provisioning method gives the best
  auditable installation experience. Runtime download remains disallowed.

## 34. Reference infrastructure

The design intentionally builds on:

- [Perfetto Trace Processor](https://perfetto.dev/docs/analysis/trace-processor)
  and [trace summaries](https://perfetto.dev/docs/analysis/trace-summary) for
  headless temporal analysis;
- [DuckDB's Parquet support](https://duckdb.org/docs/stable/data/parquet/overview)
  for embedded analytical queries over portable evidence;
- [Apache Arrow and PyArrow datasets](https://arrow.apache.org/docs/python/parquet.html)
  for explicit schemas, Parquet publication, and compaction;
- [pprof](https://github.com/google/pprof/blob/main/doc/README.md) for sampled
  profile representation and comparison;
- [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html) for
  operator, accelerator, stack, shape, and memory evidence;
- [Memray](https://bloomberg.github.io/memray/) for Python and native allocation
  capture;
- [coverage.py](https://coverage.readthedocs.io/) for Python line, branch, and
  dynamic-context execution evidence;
- [pyperf](https://pyperf.readthedocs.io/) and
  [Airspeed Velocity](https://asv.readthedocs.io/) for reliable benchmarks and
  performance history;
- [SciPy statistics](https://docs.scipy.org/doc/scipy/reference/stats.html) for
  declared bootstrap intervals and
  [statsmodels](https://www.statsmodels.org/) when scaling diagnostics require
  its fitted-model APIs;
- [Portalocker](https://portalocker.readthedocs.io/) for cross-process shared
  and exclusive local file locking;
- [Bubblewrap](https://github.com/containers/bubblewrap) and Linux cgroup/systemd
  scopes for optional enforced workload containment;
- the
  [official Python MCP SDK v2.0.0b2](https://github.com/modelcontextprotocol/python-sdk/tree/v2.0.0b2)
  for the local protocol adapter.

These projects remain separate dependencies or installed tools. Flamo's value
is the evidence lifecycle and investigation workflow that connects them.
