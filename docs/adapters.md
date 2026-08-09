# Adapters and capabilities

Adapters connect flameox to maintained collectors and native formats. They own
capability probes, capture planning, extraction, validation, compatibility, and
viewer handoff. They do not own workspace publication, analysis policy, or MCP
transport behavior.

The durable adapter lifecycle is:

```text
probe → plan → execute through broker → validate → preserve → extract
```

An adapter probes the external capability and produces a side-effect-free plan.
Flameox executes that plan through its canonical broker, validates and preserves
the declared native artifacts, and only then runs bounded extraction against the
immutable inputs. This keeps tool-specific commands and formats behind typed
contracts while containment, provenance, publication, and analysis policy remain
owned by the application services.

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

## OTLP trace import

The optional `trace` extra includes the official `opentelemetry-proto` types.
File imports accept explicit `application/x-protobuf`, `application/protobuf`,
or `application/json` media types. Binary payloads use the generated
`ExportTraceServiceRequest`; JSON uses the standard protobuf JSON parser with
unknown fields rejected. Resource, scope, span, event, and link rows preserve
OpenInference attributes as ordinary OTLP attributes and record malformed
lengths, timestamps, duplicates, and dropped counts as limitations.

There is no live OTLP receiver, format sniffing, compressed transport envelope,
or provider-specific attribute mapping in this version.

## Toxiproxy transport-fault boundary

The typed `fault_experiments` configuration, standard-library control client,
and pinned 2.12.0 asset stager establish the transport-fault boundary. The
stager accepts only allowlisted release assets and verifies their SHA-256
receipt; unsupported platforms are reported as unavailable. Proxies are
loopback-only and the configuration requires a declared workload endpoint
parameter.

`flameox fault plan`, `flameox fault run`, and `flameox fault show` use a
broker-owned Toxiproxy lease. Each trial creates a loopback proxy, runs the
baseline through that proxy without an active toxic, applies one typed
treatment, and injects only the declared endpoint parameter. Configuration,
tool receipt, ports, logs, process snapshots, cleanup outcome, and oracle
result remain attached to the workload run. Workload containment and sidecar
containment are recorded independently: the workload follows its selected
execution policy, while the pinned sidecar is restricted to a managed process
group and loopback scope. Remote upstreams, arbitrary endpoint injection,
malformed application messages, and semantic-event activation remain
unsupported.

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
Python entry points under `flameox.adapters`. Entry points execute Python code
and are therefore part of the trusted computing base. A third-party plugin
requires explicit approval by adapter, distribution name, version, and package
identity. Agents use the MCP `prepare_adapter` tool, which records agent-created
provenance; the CLI remains available for local manual administration. Approval
is revoked when that identity changes. Loading an entry point does not by itself
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

#### `flameox.benchmark-samples.v1`

Producer-neutral accelerator benchmarks may be imported as bounded JSON with
raw warm-up and measured samples. Every series declares its exact unit,
measurement clock, synchronization method, loop count, phase, device/stream,
trial/block/variant identity, and bounded workload dimensions. Extraction
publishes one observed measurement per raw sample; it does not replace samples
with a precomputed mean. Device-event measurements with missing or unknown
synchronization remain available with an explicit limitation.

```json
{
  "schema_version": "flameox.benchmark-samples.v1",
  "producer": "decode-benchmark",
  "producer_version": "git:abc123",
  "benchmarks": [{
    "name": "decode.token_latency",
    "unit": "ns",
    "measurement_clock": "cuda_event",
    "synchronization": "event_synchronize",
    "device": {"type": "cuda", "index": 0, "stream": "7"},
    "warmups": [45000],
    "samples": [42100, 41900, 42400]
  }]
}
```

Host monotonic time measures the host-side interval. A CUDA/HIP event measures
device-stream progress and requires the declared event or stream/device
synchronization before its result is comparable; Flameox preserves these clocks
as different measurement semantics. Repeating extraction for the same artifact
and extractor identity reuses the existing normalized generation.

#### `torch.profiler`

Use for operator-level CPU and accelerator activity. Record all enabled
features because stacks, shapes, modules, FLOPs, and memory change overhead and
evidence completeness. Separate compilation/warm-up from steady state.

The adapter has three explicit capability tiers:

- trace import for an existing Chrome/Perfetto-compatible export;
- whole-entrypoint launcher mode for a declared Python script, module, or inline
  `python -c` program, with no
  promise of application-specific phase separation;
- SDK/recipe mode for user-instrumented steps, phases, schedules, and semantic
  annotations.

Capture plans bind `mode`, activities, shapes, memory, stacks, FLOPs, module
hierarchy, and the full bounded schedule. Whole-entrypoint mode rejects a
schedule because an external launcher cannot infer iteration boundaries. SDK
mode requires an explicit schedule and an approved workload using
`flameox.sdk.torch_profiler()`; the yielded session exposes `step()` and a
trace-visible `phase()` range. These are parsed as separate option variants, so
the launcher never receives a nullable schedule paired with a mode flag. Every
active cycle is exported to a distinct
planned filename and registered with a cycle role. Missing, extra, empty, or
overwritten cycle outputs fail the native-output publication gate.
Normalize a multi-cycle run by calling `extract_perfetto` once per exact trace
artifact ID; cycle roles survive normalization and keep regions separate.

Arbitrary non-Python commands cannot be transparently wrapped by
`torch.profiler`; the adapter must report that limitation instead of pretending
to provide operator evidence. Declared inline programs run through a stable
synthetic filename and remain bound to the workload's exact argv; callers do
not need to create a checkout script. `with_modules` is treated as a TorchScript-only
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
the workload. A denied probe is reported as `permission_required`, which is
unusable for sampling until the host's `perf_event_open` policy is changed;
the report includes the observed kernel restriction and a remediation to grant
the required event access before refreshing capabilities.

Workloads that declare `nvcc` receive a separate bounded CUDA toolkit preflight.
It compiles a tiny header probe and distinguishes an installed compiler from a
usable development toolkit. Missing `cuda_runtime.h` is recorded as
`environment_blocked` with the compiler diagnostic and a remediation to install
the CUDA development headers. Flameox does not build the workload during this
check.

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

#### `nsight.systems` structured export

The maintained import-first adapter supports official Nsight Systems SQLite
exports registered with producer `nsight.systems`. It never parses
`.nsys-rep` and does not require `nsys` to be installed during extraction.
Produce the supported input outside Flameox with, for example,
`nsys export --type sqlite --output report.sqlite report.nsys-rep`, then import
`report.sqlite` as an execution trace with producer `nsight.systems`.
The current compatibility family requires `CUPTI_ACTIVITY_KIND_RUNTIME` and
`CUPTI_ACTIVITY_KIND_KERNEL`; `StringIds`, `NVTX_EVENTS`, and
`CUPTI_ACTIVITY_KIND_DRIVER`, `CUPTI_ACTIVITY_KIND_MEMCPY`, and
`CUPTI_ACTIVITY_KIND_MEMSET` are versioned optional evidence. Table/column
identity is fingerprinted, unknown required schemas fail explicitly, and the
source SQLite artifact remains authoritative. Extraction preserves nanosecond
timestamps, CUDA runtime/driver APIs, graph launches, kernels, correlation IDs,
device, context, stream, NVTX, memcpy, and memset fields when present. NVTX containment is
reported as a derived temporal association, not causality.

The compatibility suite includes a vendor-produced SQLite export from Nsight
Systems `2025.5.2.266-255236693005v0` in
`tests/fixtures/nsight_systems/nsight-2025.5.2.sqlite`, pinned by its SHA-256
digest and exercised through the same import and analysis path as a user
export. The fixture also covers the 2025.5 graph-trace table: when a producer
already emits `cudaGraphLaunch` runtime rows, the adapter avoids double
counting the corresponding graph-trace rows; older compatible exports can use
the graph-trace table as the graph-launch source. The fixture is hardware- and
timestamp-specific, so the test asserts event classes and identity coverage,
not exact timestamps.

#### `flameox.kernel-validation.v1`

Producer-neutral kernel-correctness evidence is imported as strict JSON conforming to the
`flameox.kernel-validation.v1` schema. The published JSON Schema lives at
`src/flameox/schemas/kernel-validation-v1.schema.json` and is generated from the same Pydantic
model used for validation. The artifact kind is `validation_output`; import does not require a
GPU, CUDA, or a kernel runner.

The contract binds producer and reference identity, bounded case inputs, seed and device,
declared metrics and tolerances, output and case outcomes, up to eight representative failures,
and coverage limitations. Supported metrics are `max_abs_error`, `max_rel_error`, `mse`, `rmse`,
`psnr`, and `cosine_similarity`. Lower-is-better metrics require `<=`; higher-is-better metrics
require `>=`. Metric, output, case, and aggregate statuses are checked for consistency. Passing
the document requires complete coverage; non-finite values and incomplete coverage without a
limitation are rejected.

Extraction publishes additive schema-minor-9 tables `kernel_validation_cases` and
`kernel_validation_metrics`. Native JSON remains authoritative, extracted tables are rebuildable,
and repeated extraction reuses the existing generation. Fixtures are project-owned synthetic JSON
documents generated in `tests/adapters/test_kernel_validation.py` under the project MIT license.

Observed claims are the declared case outcomes, metric values and thresholds, coverage flag, and
representative failure coordinates. Aggregate statuses are derived. The extractor makes no inferred
correctness claim beyond the declared metrics, tolerances, and coverage.

#### `compute-sanitizer`

The maintained adapter runs NVIDIA Compute Sanitizer around a declared workload and preserves its
XML report as `sanitizer_report`. It supports the official `memcheck`, `racecheck`, `initcheck`, and
`synccheck` tools and fixes output to `--xml --save <path>`. Strict options cover launch skip/count,
target-process scope and bounded filter, kernel name, demangling, a distinct finding exit code, and
an optional project-relative suppression file whose SHA-256 digest enters capture-plan identity.
Suppression files must be regular, non-linked, project-contained files; arbitrary flags are refused.

Compatibility family `compute-sanitizer.xml.2026.v1` was observed with Compute Sanitizer 2026.2.1.
NVIDIA does not publish a stable XML XSD, so extraction is version-bounded and unknown record shapes
or tags become limitations. XML parsing runs in an isolated bounded worker using `defusedxml`, caps
host stacks at 64 frames, and normalizes source paths. A configured finding exit plus parsed records
is a completed failed validation, while other nonzero exits, missing or malformed reports, and
timeouts remain failed attempts with preserved partial evidence.

Linux and Windows are supported; GPU access and a CUDA toolkit installation are required and are
not provisioned by Flameox. Overhead depends on the selected sanitizer tool and input. A clean report
covers only the selected tool, launches, processes, and filters and does not prove numerical
correctness.

The live fixture `tests/fixtures/compute_sanitizer/kernel_probe.cu` is project-owned MIT-licensed
CUDA C++. The optional live test compiles it with `nvcc -lineinfo` and checks both in-bounds and
out-of-bounds captures. Deterministic tests use synthetic XML and cover clean, memory, API,
sanitizer, malformed, truncated, unknown, and oversized reports. Observed claims come from XML;
classification and path normalization are deterministic derivations; no root cause is inferred.

+#### `nvbench`

The NVBench integration targets the JSON schema and JSON-binary behavior verified at
`NVIDIA/nvbench@c18488992e313240166f588b9ee4da3e0de76004`. NVBench is linked into each benchmark
executable, so Flameox runs the declared benchmark directly and injects exactly one official
output mode: `--jsonbin <path>` to preserve JSON plus binary samples, or `--json <path>` for
JSON-only output. Existing JSON output arguments are rejected instead of being overridden.

Provider import reads the bounded JSON first and selects only the `file/sample_times` and
`file/sample_freqs` sidecars it declares. Relative paths must remain inside the JSON bundle;
unknown file encodings, missing declarations, traversal, symlinks, hard links, duplicate paths,
count/byte mismatches, and bundles over 100 members or the workspace byte quotas fail explicitly.
The current binary encoding is little-endian float32, with element counts serialized by NVBench
as decimal strings. JSON and sidecars remain unchanged and authoritative. Extraction publishes
raw timing samples in seconds and frequency samples in hertz through the existing measurements
table; derived benchmark conclusions are not synthesized.

Deterministic fixtures are generated in `tests/adapters/test_nvbench.py` from the pinned upstream
shape and contain only Flameox-authored metadata and float32 samples under the project MIT
license. They cover nested sidecar paths, strict encodings, partial process output, integrity and
quota failures, and idempotent extraction. A local build of the pinned official
`nvbench.example.cpp17.stream` benchmark ran through the managed adapter on CUDA 13.3 and an
`sm_86` RTX 3060. The live test preserved the JSON, timing and frequency float32 sidecars, and
then published their samples idempotently. That build used GCC 15, which is newer than NVBench's
published GCC 7–14 compatibility range, so it is observed local compatibility rather than a
supported-version claim. Linux and Windows CUDA hosts are expected upstream platforms; GPU
access, the benchmark's own CUDA compatibility, and workload-dependent benchmark overhead remain
operator responsibilities. Flameox owns containment, quotas, cancellation, and artifact
registration; NVBench owns the native bytes.

Observed claims are benchmark/state identity, producer/schema versions, device index, declared
sample type and count, and preserved float32 samples. Derived claims are stable measurement IDs,
SI-unit labeling from the two documented hints, and bounded row counts. Inferred claims: none —
samples alone do not establish correctness, representativeness, or a performance improvement.

#### `triton.compiler` and `cute.compiler`

Compiler capture uses only upstream dump controls. Triton support is verified against Triton
`3.7.1` (`triton-lang/triton@f797708c0626e5f9840ca5b0a98790e2c7cb09ad`) with
`TRITON_DUMP_DIR`, `TRITON_KERNEL_DUMP`, and optional `TRITON_REPRODUCER_PATH`. CuTe DSL controls
and retained kinds were verified against CUTLASS `4.6.2`
(`NVIDIA/cutlass@6c65a175668952f09bcbf66cb97a8de1b734b4a0`) using
`CUTE_DSL_DUMP_DIR` and the documented `CUTE_DSL_KEEP` tokens. Conflicting workload environment
values are rejected unless identical. Flameox never deletes or rewrites a compiler cache.

The wrappers run the named workload directly, inventory only allowlisted native files inside the
staging root, enforce the shared 100-member and byte quotas, and atomically write a strict
`flameox.kernel-build.v1` manifest even when compilation fails. TTIR, TTGIR, LLVM IR, PTX,
AMDGCN, SASS, CUBIN, HSACO, metadata, debug IR, and reproducers remain immutable native
artifacts; Flameox does not parse or translate them. The manifest records deterministic inventory
stages, artifact integrity, producer/workload identity, cache status when known, normalized
official environment values, diagnostics, and limitations. Import verifies declared size and
digest through the shared bounded bundle importer and registers the result in the existing
`ArtifactPipeline` service, enabling the existing comparison path rather than a second compiler
pipeline store. Managed captures register that same pipeline directly. Provider dumps do not
declare predecessor lineage, so Flameox leaves predecessor edges unset rather than deriving them
from filenames or directories.

A local `sm_86` run with Triton `3.7.1` produced and registered real compiler dumps through the
managed adapter. Deterministic process tests cover both wrappers, failed compilation, missing
output, environment conflicts, CuTe IR retention, absent lineage, containment, links, and
reproducer handling. CUTLASS/CuTe DSL `4.6.2` was installed from its official CUDA 13 wheel, and
the pinned official Ampere `call_bypass_dlpack.py` GEMM ran through the managed adapter on the
same `sm_86` GPU. It verified its output against PyTorch and preserved MLIR, PTX, and SM86 CUBIN
artifacts. Both integrations require a workload that actually invokes the matching compiler and
a compatible accelerator/toolchain; compile and dump overhead is workload- and cache-dependent.
The normal execution policy owns containment, staging, timeout, cancellation, and quotas.

Observed claims are compiler exit status, preserved filenames/bytes/digests, producer version,
and declared dump environment. Derived claims are deterministic inventory ordering, normalized
staging paths, manifest outcome, and pipeline identity. Cache
status remains `unknown` unless the producer supplies evidence. Inferred claims: none — an
available stage does not prove semantic correctness, optimal code generation, or a cache hit.

### Candidate adapters

- GDB/LLDB and elfutils for core metadata, with user init files, autoload, and
  implicit debuginfod disabled;
- AddressSanitizer, UndefinedBehaviorSanitizer, and ThreadSanitizer raw reports
  with narrow producer-versioned extraction rather than a claimed universal
  JSON schema;
- `rr` recording references;
- Nsight Systems Arrow, JSONL, or Parquet exports beyond the maintained SQLite
  subset;
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

For missing managed providers, the report also contains a typed setup action.
Agents should pass its adapter name to `start_capability_setup` with a stable
idempotency key, poll the durable result, and then refresh the capability list
before planning. This
covers the published `execution`, `memory`, `cpu`, `test`, and `torch` extras.
It does not install host tools or change permissions. A fallback adapter must be
chosen only after the agent has shown the requested capability's evidence
limitation in its plan.

Probe commands must be bounded and expected to be side-effect free, but active
execution is still labeled as execution. An executable found only through a
project-controlled `PATH` is not run during passive discovery.

Declared workflow details expose bounded, sorted adapter options for
capture-capable built-ins and approved third-party adapters. Each option
includes capability status, planning disposition, required preflight mode,
permissions, supported modes and formats, features, limitations, and
remediation. The list is capped at 64 entries and reports its total and
truncation state.
