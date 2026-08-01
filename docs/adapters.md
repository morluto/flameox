# Adapters and capabilities

Adapters connect flameox to maintained collectors and native formats. They own
capability probes, capture planning, extraction, validation, compatibility, and
viewer handoff. They do not own workspace publication, analysis policy, or MCP
transport behavior.

## Perfetto integration

Perfetto Trace Processor is the authoritative query engine for detailed
temporal traces.

The adapter may use:

- the Perfetto Python API;
- `trace_processor_shell` as a subprocess;
- custom versioned PerfettoSQL packages;
- structured trace summaries.

The Python API must be configured with a locally installed or deliberately
packaged Trace Processor binary. Its convenience download behavior is disabled;
flameox never fetches a binary during an agent operation.

Initial Perfetto support is import and query, not unspecified system-wide
capture. Trace Processor is the preferred ingestion path for formats it
supports, including Perfetto protobuf traces, Chrome JSON traces, pprof
profiles, and `perf.data`. flameox records the Trace Processor binary version and
SHA-256 with every extraction or query.

Initial query modules:

```text
flameox.cpu.hotspots
flameox.cpu.run_queue_delay
flameox.process.wall_breakdown
flameox.threads.blocked_time
flameox.operations.longest
flameox.operations.repeated
flameox.pytorch.operator_summary
flameox.pytorch.cpu_gpu_sync
flameox.accelerator.idle_gaps
flameox.memory.timeline_summary
```

Each query module declares:

- supported trace kinds and required tables;
- parameters and defaults;
- output schema;
- unit and polarity of every metric;
- expected query complexity;
- known blind spots;
- module version.

flameox does not copy the entire Perfetto trace into Parquet. It extracts only
cross-run summaries and source-linked measurements required by supported
recipes.

## Adapter system

### Interface

Approved third-party adapters implement the public v1 contract exported from
`flameox.domain`:

```python
class AdapterV1(Protocol):
    name: str
    api_version: Literal[1]

    async def probe(self, context: AdapterProbeContext) -> AdapterProbeResult: ...
    async def plan(self, request: AdapterPlanRequest) -> AdapterExecutionPlan: ...
    async def validate(self, artifact_path: str, declaration):
        ...
    async def extract(self, artifact_path: str, declaration):
        ...
```

`plan` is side-effect free and returns a command prefix, declared relative
artifact paths, permissions, estimated overhead, output types, limitations,
and extractor version. Flameox appends `--` and the already declared workload
command; an adapter cannot replace the workload argv, cwd, environment, or
execution policy. Flameox owns capture execution, containment, quotas,
cancellation, artifact registration, and publication. `validate` receives the
declared staging artifact, while `extract` receives the immutable registered
artifact. Extraction summaries are bounded, versioned, and linked to that
exact input artifact in `adapter_extractions`.

### Discovery

Built-in adapters are registered explicitly. Third-party adapter support uses
Python entry points under `flameox.adapters`. Entry points execute
Python code and are therefore part of the trusted computing base. Only
built-ins load by default. A third-party plugin requires explicit CLI approval
by distribution name, version, and package identity, and its approval is
revoked when that identity changes. Loading an entry point does not by itself
authorize collector execution.

Each adapter has independent:

- domain tests;
- executable probes;
- fixture artifacts;
- extractor version;
- supported platform matrix;
- optional dependencies.

### Supported adapters

#### `pyperf`

Use for controlled Python benchmarks. Reuse calibration, warm-ups, worker
processes, machine metadata, instability checks, and JSON output. Preserve
pyperf JSON as a native artifact and load it through public
`BenchmarkSuite`/benchmark value and metadata APIs. flameox does not import
pyperf's private `_compare` helpers or parse human-readable comparison output.
pyperf owns calibration, warm-ups, workers, runs, and values within a benchmark
trial; flameox's experiment blocks and treatment randomization sit above that
hierarchy and do not duplicate it.

#### `python-startup`

Use for repeated startup of a declared Python script or module. Each sample
uses a fresh interpreter. Repeated wall/RSS samples run without import
instrumentation; one separate `-X importtime` process preserves the raw trace
and package-grouped module counts and costs in the native JSON artifact. The
first wall sample is labeled
`uncontrolled_initial`, not cold: flameox does not drop operating-system
caches. Later samples are warm process restarts with a fresh interpreter and a
potentially populated filesystem cache.

On POSIX systems with `wait4`, peak RSS comes from the terminated child's
`ru_maxrss`; the extractor records `wait4_ru_maxrss` as the measurement
backend. Other platforms use bounded psutil polling and record
`psutil_polling`, which may miss a very short-lived peak.

Package summaries add module self time and retain the maximum cumulative time
reported for a module in that top-level package. Summing cumulative import
times would double-count nested imports. Experiments publish
`python_startup.wall_time`, `python_startup.peak_rss`, and package dimensions
through ordinary measurements, so existing identity and compatibility rules
govern base/candidate comparisons.

#### `pytest`

Use for declared `pytest` or `python -m pytest` workloads. A small plugin writes
an append-only JSONL event stream while the suite runs. Public pytest hooks
provide collection, setup/call/teardown reports, fixture setup timing, outcomes,
and interruption events. With local xdist, fixture and test-start events are
flushed immediately to bounded per-worker sidecars. After each clean shutdown
or crash, the controller validates the event schema, type, and worker identity,
then appends accepted events to the authoritative primary JSONL artifact.
Controller hooks also record scheduler strategy, worker creation, readiness,
collection, clean shutdown, and crashes.

`pytest.time_to_first_failure.observed` uses the worker report stop time;
`pytest.time_to_first_failure.reported` uses controller receipt time and better
approximates when a parallel failure becomes actionable. Stable xdist hooks do
not expose an exact per-test controller queue timestamp, so the adapter records
execution start and reports that limitation instead of inferring queue delay.
If a run times out after the event stream exists, flameox registers the partial
artifact and marks the run timed out; collected tests without a phase report
remain explicitly unexecuted. A forcefully terminated controller can still lose
sidecar events it had not recovered; the primary artifact never treats an
unregistered sidecar as authoritative evidence.

#### `py-spy`

Use for out-of-process Python CPU sampling and attach workflows. Prefer a
machine-readable Chrome trace for normalized extraction and feed it to Perfetto
Trace Processor. Supported py-spy versions emit flamegraph, raw, Speedscope,
and Chrome trace formats; the adapter must not claim they emit pprof. Record
whether native frames, subprocesses, idle threads, and GIL state were captured.
Speedscope or raw output may be preserved for existing viewers, but flameox does
not implement a bespoke sampled-stack parser when Trace Processor can ingest
the chosen format.

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

Perf readiness is an active permission check, not executable version discovery.
The broker runs a fixed Flameox-owned `perf record` probe in a temporary
staging directory with an explicit output path, bounded timeout, output, and
resource policy. Build-ID cache and uncontrolled output side effects are
disabled. A successful probe reports granted permission; a recognized
`perf_event_open` or kernel-policy denial reports `permission_required` with
remediation; other failures remain `degraded` with bounded diagnostics. Passive
capability discovery therefore reports that an active probe is required, and a
capture plan must bind and recheck the active report immediately before running
the workload.

#### `perfetto`

Import Perfetto-compatible traces and run versioned PerfettoSQL queries.
Platform-specific system capture is outside the supported adapter set and
requires a separately scoped mode.

#### `coverage`

Use coverage.py's supported data APIs for Python line, branch, and dynamic
context evidence. Preserve the native coverage data and extract bounded
repository-relative line and arc sets. This adapter answers whether a path
executed; it does not claim why the path executed or what values flowed through
it.

### Candidate adapters

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

## Capability detection

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

Declared workflow details expose bounded, sorted adapter options for
capture-capable built-ins and approved third-party adapters. Each option
includes capability status, planning disposition, required preflight mode,
permissions, supported modes and formats, features, limitations, and
remediation. The list is capped at 64 entries and reports its total and
truncation state.
