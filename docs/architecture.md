# Architecture

This document explains flameox's process model, package boundaries, dependencies,
and platform policy. Data contracts live in
[Storage and evidence](storage-and-evidence.md); runtime invariants live in
[Runtime safety](runtime-safety.md).

## System overview

flameox is a permanently local command-line application and Model Context
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

flameox is not a generic AI bug finder. It is an evidence system for agents.
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
       ├──────────────► `flameox` CLI
       └──────────────► local MCP server
```

DuckDB is the long-term local analytical engine. Parquet and native artifacts
are authoritative. Immutable generation manifests and an atomically published
corpus commit define which Parquet files belong to a readable snapshot.
`catalog.duckdb` contains rebuildable views and may contain measured,
reproducible caches. There is no PostgreSQL service and no SQLite application
database.

The unit of execution is a run. The unit of experimental reasoning is not.
flameox models an investigation containing hypotheses and experiments; each
experiment contains variants and attempted trials, and each trial references
one run. This preserves randomized blocks, repetitions, input grids, failed or
timed-out attempts, validation outcomes, and the distinction between an
exploratory profile and confirmatory performance evidence.

## Design principles

### Native evidence first

The native artifact is the most authoritative representation of a capture.
Normalized evidence is a query accelerator and shared vocabulary, not a
replacement. Every normalized row must link to an artifact, run, and extractor
version.

### Facts before interpretation

Tool output must use three evidence levels:

- `observed`: directly emitted or measured by a collector;
- `derived`: deterministically calculated from observed evidence;
- `inferred`: a hypothesis or interpretation that may require another
  experiment.

An inferred claim must never be represented as an observed fact.

### Comparisons are valid only under declared conditions

Every comparison reports the environment and workload fields that match, differ,
or are unknown. flameox may refuse a comparison when a mismatch invalidates the
claim. `--force` can display an exploratory comparison but cannot relabel it
valid.

### Task-shaped APIs

Agents should call `compare_run_sets` or `analyze_scaling`, not construct
arbitrary SQL. Internally, flameox uses reviewed parameterized SQL and
collector-specific queries. This keeps results compact and prevents DuckDB
features from becoming an unintended local file-access or code-execution
interface.

### Rebuildable derived state

Deleting `catalog.duckdb` must not delete evidence. `flameox catalog rebuild`
recreates it from immutable manifests and Parquet partitions.

### Capture outside the write lock

Profiling and benchmarks can take minutes. They run concurrently in isolated
staging directories. Only registration and evidence publication require the
short-lived workspace write lock.

### No silent fallback

If native stacks, GPU events, symbols, shapes, or memory records were requested
but unavailable, the result states that explicitly. flameox must not silently
substitute a weaker collector and present the result as equivalent.

### Experimental structure is evidence

Warm-ups, worker processes, randomized block order, treatment assignment,
failed attempts, exclusion reasons, validation results, and stopping rules are
part of the evidence. They must not be flattened into an unlabeled list of
samples. Profile-guided discovery is exploratory; a fresh, predeclared
benchmark and semantic oracle provide confirmatory evidence.

### One analysis, one corpus snapshot

The corpus is append-only. A reader pins one immutable corpus commit before its
first query and uses the exact file inventory in that commit for every query in
the analysis. Files that exist on disk but are absent from the pinned inventory
are invisible. Publication becomes visible only by atomically advancing the
corpus `HEAD`.

### Safe composition over custom infrastructure

flameox prefers maintained public interfaces: Perfetto Trace Processor for
supported trace and profile formats, PyArrow datasets for Parquet publication,
DuckDB for analytical SQL, pyperf for benchmark collection, SciPy for declared
bootstrap calculations, and collector-supported readers for native artifacts.
Private APIs and parsing human-readable reports are not product contracts.

## System architecture

### Package boundaries

```text
src/flameox/
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

`flameox.domain` must not import `mcp`, `typer`, DuckDB, Perfetto, PyTorch, or
collector packages. Domain models are ordinary Pydantic models and enums.

### Process model

The CLI is a short-lived process. The MCP server is a long-lived local process
using stdio by default. Both call the same application services.

Transport composition selects a named execution policy. The CLI uses
`trusted_local`. MCP's default `capture_mode='auto'` runs the named workload
directly and records the absence of enforced descendant containment;
`capture_mode='managed'` binds `approved_agent` and refuses a plan without the
required containment when the project policy requires it. Application services
receive that policy rather than a transport boolean, and plans bind the selected
policy for execution-time revalidation.

Collectors run as child processes or explicitly selected in-process adapters.
External commands are always executed as argument arrays with `shell=False`.
The default trusted-local agent path runs directly without a containment backend.
The managed policy can use a cgroup v2 or systemd scope on Linux so cancellation,
timeouts, and resource limits apply to descendants even if they create a new
process group; when that policy is not available, planning refuses rather than
claiming equivalent guarantees.

`flameox.execution` owns subprocess creation through
`asyncio.create_subprocess_exec` behind a single broker. Collectors, validators,
active capability probes, symbolizers, Perfetto, viewers, and extractor workers
must use that broker; adapters do not call subprocess APIs directly. This gives
command validation, environment construction, containment, quotas, output
budgets, cancellation, timeouts, and descendant cleanup one implementation.
Heavy or artifact-facing extractors run in bounded worker processes. Small
trusted metadata operations may use an AnyIO worker thread.

Runtime evidence extends this boundary without adding a second execution model:
file-imported OTLP is normalized by an application service into bounded
Parquet tables, and every user-visible broker run can publish privacy-limited
process snapshots around cleanup. Lifecycle queries are curated DuckDB
operations over those tables; they do not expose SQL or materialize an agent
interaction graph. Native ddmin invokes the declared predicate through the
same broker. Reduction plans are Flameox-owned and do not run an arbitrary
reducer workload or expose a reducer socket protocol.

The MCP SDK's AnyIO cancellation is translated at the application boundary into
the execution service's cancellation signal. Cleanup that awaits subprocess or
publication work runs in a bounded shielded cancellation scope and re-raises
the original cancellation afterward.

The MCP server is not a background system daemon. It starts and stops with the
MCP host.

### Python and dependency baseline

Required:

- Python 3.12 or newer;
- `mcp==2.0.0` and `mcp-types==2.0.0`;
- Pydantic 2;
- DuckDB;
- PyArrow;
- Typer;
- AnyIO;
- Portalocker for cross-process shared and exclusive file locks;
- structured logging using the standard library or a minimal compatible layer.

The MCP packages remain exact-pinned as a matched SDK/protocol-model release.
All SDK imports and conversions live under `flameox.mcp`. The domain service
API must be testable without importing MCP.

The project uses the stable v2 `MCPServer` API, not the v1
`mcp.server.fastmcp.FastMCP` compatibility surface. The exact dependencies are
declared in `pyproject.toml`, and the resolved release is committed in `uv.lock`.

Optional extras group collector dependencies:

```text
flameox[python]       pyperf and Python profiling integrations
flameox[memory]       Memray integration
flameox[torch]        in-process PyTorch integration
flameox[stats]        SciPy and optional statsmodels analyses
flameox[dev]          test, lint, type-check, and fixture tooling
flameox[all]          all non-vendor optional integrations
```

The MCP setup actions are the agent-facing installers. `start_capability_setup`
accepts an explicit managed adapter enum and idempotency key, returns a durable
operation, and is followed by `get_capability_setup` or
`cancel_capability_setup`.
`prepare_adapter` records an exact installed third-party identity, and
`prepare_workload_dependencies` installs only Python distributions already
declared by a named workload. Selected extras are recorded in the workspace
capability manifest so a later runtime upgrade does not silently remove them.
None of these actions runs a workload.

System and privileged executables such as `perf`, `llvm-symbolizer`, Bubblewrap,
GDB, or LLDB are detected at runtime and are not provisioned by FlameOx. The
non-privileged Trace Processor is the exception: `start_capability_setup` stages
a pinned platform-specific binary under `.diagnostics/tools`, verifies its
bounded version command, and records the configured path. Permission
requirements such as `perf_event_open` remain explicit.

### Application services

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

### Platform policy

The storage, query, CLI, and MCP layers are platform-neutral. Linux is the
primary first-class capture platform because `perf`, native symbols,
accelerator tooling, cgroup containment, and core dumps matter to the target
investigations.

macOS and Windows may expose adapters whose upstream collectors support them,
but capability reports must identify degraded behavior. Platform support is
claimed per adapter and feature, never from the fact that the Python package
installs.

## Reference infrastructure

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
  [official Python MCP SDK v2.0.0](https://github.com/modelcontextprotocol/python-sdk/tree/v2.0.0)
  for the local protocol adapter.

These projects remain separate dependencies or installed tools. flameox's value
is the evidence lifecycle and investigation workflow that connects them.
