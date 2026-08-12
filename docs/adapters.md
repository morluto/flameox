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

Use for declared `pytest` or `python -m pytest` workloads. The maintained
`pytest-reportlog` JSONL format is authoritative for native `CollectReport`,
`TestReport`, warning, and session records. Flameox's small plugin adds only
namespaced evidence pytest does not own: fixture setup samples, the run origin,
controller receipt time, and collected node IDs. Consumers ignore unknown
report types and keys as required by the producer format. Ordinary pytest and
xdist own collection, scheduling, sharding, and report transport; Flameox does
not maintain a parallel worker-event or sidecar protocol.

`pytest.time_to_first_failure.observed` uses the worker report stop time;
`pytest.time_to_first_failure.reported` uses controller receipt time and better
approximates when a parallel failure becomes actionable. Stable xdist hooks do
not expose an exact per-test controller queue timestamp, so the adapter records
execution start and reports that limitation instead of inferring queue delay.
If a run times out after the event stream exists, flameox registers the partial
artifact and marks the run timed out; collected tests without a phase report
remain explicitly unexecuted. A forcefully terminated worker can only
contribute reports already delivered to pytest; the primary artifact never treats an
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

Producer-neutral kernel-correctness evidence is imported as strict JSON
conforming to the `flameox.kernel-validation.v1` schema. The published JSON
Schema lives at `src/flameox/schemas/kernel-validation-v1.schema.json` and is
generated from the same Pydantic model used for validation, so import and
schema publication cannot drift. The artifact kind is `validation_output`;
import does not require a GPU, CUDA, or any kernel runner.

The document binds producer and reference-implementation identity, per-case
inputs (dtype, shape, role, seed, device), declared metric definitions and
tolerances, per-output and per-case observed metrics, pass/fail/inconclusive/
unsupported outcomes, bounded representative failures, and coverage
limitations. Supported metrics are `max_abs_error`, `max_rel_error`, `mse`,
`rmse`, `psnr`, and `cosine_similarity`; each carries a comparator (`<=` for
lower-is-better, `>=` for higher-is-better), threshold, unit, and status. The
model enforces three aggregate-verdict invariants: a metric status cannot
contradict its value and threshold; an output status cannot contradict its
metrics; and the document status cannot contradict its case outcomes or
declared coverage completeness. Non-finite values, unknown schema versions,
ambiguous aggregates, and incomplete coverage without a stated limitation are
rejected explicitly. Error metrics and their tolerances must be nonnegative,
cosine similarity must remain in `[-1, 1]`, and inconclusive or unsupported
metrics carry a limitation instead of numeric comparison fields. Failed outputs
require at least one bounded, substantive representative failure; non-failed
outputs reject representative failures. JSON validation is strict, so strings
and booleans are not coerced into numeric contract fields. Extraction returns a
bounded summary of document, case, output, and metric limitations.

Extraction publishes two evidence tables:
`kernel_validation_cases` (case/output identity, status, dimensions, inputs,
device, seed, representative failures, limitations) and
`kernel_validation_metrics` (per-output metric name, value, comparator,
threshold, unit, status, limitation). Both are schema minor 9 additions; the
raw artifact remains authoritative and rows are rebuildable. Repeating
extraction for the same artifact and extractor identity reuses the existing
normalized generation. The trial-level `flameox.oracle-receipt.v1` remains the
decisive verdict and may reference this artifact by diagnostic role; the
detailed artifact does not duplicate benchmark samples, run provenance, or
experiment structure.

Ownership: `tests/adapters/test_kernel_validation.py`, lane `adapters`,
markers `unit`. Fixtures are synthetic JSON documents generated in-test; no
vendor-produced artifact is required. The test suite covers exact agreement,
tolerance-bound agreement, numerical failure with bounded representative
failures, non-finite rejection, contradictory aggregate rejection, unknown
schema version rejection, and idempotent re-extraction.

Observed claims: per-case status, per-metric value and threshold, declared
tolerances, coverage completeness flag, representative failure coordinates.
Derived claims: aggregate output status from metric statuses, aggregate case
status from output statuses, document status from case outcomes and coverage.
Inferred claims: none — the extractor does not infer correctness beyond the
declared metrics and tolerances.

#### `compute-sanitizer`

The maintained capture adapter runs NVIDIA Compute Sanitizer around an already
declared workload and preserves the XML report as a `sanitizer_report`
artifact. The adapter supports the four official tools — `memcheck`,
`racecheck`, `initcheck`, and `synccheck` — and emits XML through `--xml` with
`--save`. It does not require a separate import converter; existing XML reports
may also be imported directly with producer `compute-sanitizer`.

Official output: `compute-sanitizer --tool <tool> --xml --save <path> -- <workload>`.
The adapter binds tool choice, `launch_skip`, `launch_count`,
`target_processes` (`application-only` or `all`), `target_processes_filter`,
`kernel_name`, `demangle`, `finding_exit_code`, and an optional
project-relative suppression file whose SHA-256 digest is recorded in the
capture plan. Suppression files must be regular, non-linked, project-contained
files; absolute paths, `..` traversal, and symlinks are rejected. Arbitrary
flag injection is refused — only the declared options are accepted.

Platform: Linux and Windows. Permissions: GPU access and the CUDA toolkit;
the executable is detected at runtime and not provisioned by Flameox. Overhead:
GPU instrumentation overhead whose exact cost depends on the selected sanitizer
tool. Managed containment is currently refused because the containment backend
does not yet bind a bounded NVIDIA device set into its private `/dev`.
Trusted-local capture remains available on a trusted GPU host and records that
choice. The sanitizer wraps the workload argv and Flameox owns process
execution, quotas, and cancellation. The adapter preserves artifacts on
nonzero exit because sanitizer findings produce a nonzero exit code by design.

Local evidence: Compute Sanitizer `2026.2.1` was observed locally. The
compatibility family is `compute-sanitizer.xml.2026.v1`; extraction is
version-bounded because NVIDIA does not publish a stable XSD for the XML
format. A missing, invalid, legacy, or future producer version makes an
otherwise clean report inconclusive rather than assigning it to the verified
2026 family. The parser runs in a bounded subprocess worker
(`flameox.workers.compute_sanitizer`) behind the canonical broker and uses
`defusedxml` rather than implementing XML entity defenses locally. It rejects
DTD and entity declarations, requires a `ComputeSanitizerOutput` root element,
and reports unknown record elements or XML tags as explicit limitations rather
than silently accepting them. Host stacks are truncated to 64 frames. Source
paths are normalized to project-relative form; external paths are reported as
`<external>/<basename>`. The extractor distinguishes `clean` (zero findings,
no limitations), `findings` (one or more records), and `inconclusive`
(limitations without records). A clean report covers only the selected tool,
launches, processes, and filters; it does not prove numerical correctness. An
oracle receipt may reference the sanitizer report by diagnostic role, but a
clean sanitizer run never proves numerical equivalence.

Fixture provenance and licensing: `tests/fixtures/compute_sanitizer/kernel_probe.cu`
is a Flameox-authored CUDA C++ source file under the project MIT license. It
compiles with `nvcc -lineinfo` into a small probe that performs an in-bounds or
out-of-bounds global memory write depending on a runtime argument. No
vendor-produced XML fixture is committed; XML fixtures are synthetic and
generated in-test. The live test (`tests/adapters/test_compute_sanitizer_live.py`)
compiles the probe with `nvcc`, runs `compute-sanitizer --tool memcheck` around
it, imports the XML, and extracts findings; it skips when `compute-sanitizer`
or `nvcc` is absent.

Ownership: `tests/adapters/test_compute_sanitizer.py` is owned by `adapters`,
lane `adapters`, markers `unit`. `tests/adapters/test_compute_sanitizer_live.py`
is owned by `compute-sanitizer-live`, lane `adapters`, markers `integration`,
`optional`, `process`, `serial`, `requires_compute_sanitizer`.

Observed claims: error kind, level, message, memory space, access size,
direction, error class, function, source path, line, PC, thread/block indices,
and host stack frames — all read directly from the XML record. Derived claims:
classification (`memory_access`, `race`, `uninitialized_memory`,
`synchronization`, `api_error`, `sanitizer_error`, `unknown`) is derived from
the record's kind, message, and error text; project-relative path normalization
is a deterministic derivation from the reported path. Inferred claims: none —
the extractor does not infer the root cause of a memory error or claim
correctness beyond the retained findings.

#### `nvbench`

The NVBench integration targets the JSON schema and JSON-binary behavior verified at
`NVIDIA/nvbench@c18488992e313240166f588b9ee4da3e0de76004`. NVBench is linked into each benchmark
executable, so Flameox runs the declared benchmark directly and injects exactly one official
output mode: `--jsonbin <path>` to preserve JSON plus binary samples, or `--json <path>` for
JSON-only output. Existing JSON output arguments are rejected instead of being overridden.

Provider import reads the bounded JSON first and selects only the `file/sample_times` and
`file/sample_freqs` sidecars it declares. Relative paths must remain inside the JSON bundle;
unknown file encodings, missing declarations, traversal, symlinks, hard links, duplicate paths,
count/byte mismatches, and bundles over 100 members or the workspace byte quotas fail explicitly.
The provider document is snapshotted through the same no-follow artifact boundary before parsing,
is limited to 16 MiB, and must declare the verified JSON schema major 1. Repeated references to
the same filename, sample hint, and count are decoded once; conflicting repetitions are rejected.
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
from filenames or directories. Manifest parsing is capped at 1 MiB and starts from a no-follow
immutable snapshot. Pipeline compatibility includes the declared workload and device identities;
if the compiler version is unavailable, comparison remains `unknown` rather than claiming
compatibility.

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

#### `nsight.compute`

The maintained adapter preserves official `.ncu-rep` and `.ncu-repz` reports as immutable
`kernel_profile` artifacts. Extraction runs in the isolated artifact worker and exclusively uses
the `ncu_report` Python interface shipped with the detected Nsight Compute installation. NVIDIA
owns the native report format and reader; Flameox neither decodes the binary format nor adds a
runtime PyPI dependency. The schema fingerprint binds the selected report-interface digest and
report version to the observed metric and section identities. Roofline evidence is published only
when a metric, rule,
or section in the report explicitly identifies roofline data; Flameox does not synthesize a
roofline or bottleneck conclusion.

Capture requires exactly one named section set or 1–32 exact section identifiers, with an
optional bounded kernel name, launch skip from 0 to 1,000,000, launch count from 1 to 1,000,000,
and replay mode `kernel`, `application`, `range`, or `app-range`. Regex sections, arbitrary flags,
external section directories, and source import are not accepted. Linux and Windows are declared
capture platforms; a supported NVIDIA GPU, driver, Nsight Compute installation, and
performance-counter permission are required. Kernel replay can impose substantial,
workload-dependent overhead. The standard execution policy owns process containment, staging,
quotas, timeout, and
cancellation, while `ncu` owns the native report bytes. Flameox never changes privileges. NVIDIA's
`ERR_NVGPUCTRPERM` maps to `permission_required` with remediation rather than a privilege attempt.

Normalization is bounded by the workspace generation row quota, the worker's serialized-output
budget, a 120-second worker timeout, 1,000 ranges, 10,000 actions, and separate metric and
observation budgets. The response budget reserves a 64 KiB envelope and conservatively allows
16 KiB per normalized row. Numeric metrics enter `measurements`; string attributes, rules and
result tables, section identities, and source/SASS/PTX references enter observations. Source
bodies are not copied out of the report. Provenance records the SHA-256 identity of the selected
`ncu_report.py`; changing that reader creates a distinct extraction generation. Unknown metric
value kinds, truncated collections, and exceptions from optional official-interface access
become explicit limitations. Missing optional methods degrade
gracefully; corrupt reports, a missing official interface, and invalid required interface results
fail extraction with bounded recovery guidance.

Local compatibility evidence used Nsight Compute `2026.2.1` and its installed, vendor-produced
`extras/samples/instructionMix/sobelFloat.ncu-rep`. The sample was read successfully through that
installation's `ncu_report` interface and is not redistributed or committed by Flameox; it remains
covered by the locally installed NVIDIA product's license. Deterministic tests use a
Flameox-authored fake interface under the project MIT license to exercise type handling, bounds,
corruption, and provenance. The local driver initially produced the official
`ERR_NVGPUCTRPERM` diagnostic, proving the `permission_required` mapping. After an administrator
enabled non-admin counter access, the managed adapter profiled the pinned official NVBench stream
benchmark with the `basic` set on the `sm_86` RTX 3060, preserved the `.ncu-rep`, and extracted
actions and numeric metrics through the installed official interface. Both the denied capability
path and successful live counter capture are therefore observed on this host.

Observed claims are report/range/action names exposed by the official interface, kernel and device
identity attributes, numeric and string metric values and units, section and rule identities,
result tables, and source/SASS/PTX references. Derived claims are bounded row normalization,
counts, schema fingerprint, and the explicit-identity test for `roofline_present`. Inferred claims:
none — metric presence or magnitude does not by itself prove causality, correctness, a bottleneck,
or an optimization opportunity.

#### `rocprofv3`

The maintained Linux capture adapter invokes the official ROCprofiler-SDK CLI
with `--output-format pftrace`, a Flameox-owned `-d` staging directory, and the
fixed `-o rocprofv3` basename. It accepts only the documented `--hip-trace`,
`--kernel-trace`, `--memory-copy-trace`, `--memory-allocation-trace`,
`--scratch-memory-trace`, and `--marker-trace` domains; at least one must be
enabled. The resulting `rocprofv3_results.pftrace` remains an immutable
`execution_trace` artifact with producer `rocprofv3`. Flameox does not load the
raw ROCprofiler SDK or decode PFTrace: extraction uses the existing bounded
Perfetto worker and accelerator recipe.

Compatibility floor: ROCm 6.2 / ROCprofiler-SDK 0.4 is the documented minimum
for rocprofv3 HIP tracing and PFTrace output. The complete option set above and
the `_results.pftrace` naming convention were verified against current official
ROCm 7.x documentation. An older CLI that rejects a selected domain is reported
as a failed attempt with its stdout, stderr, process identity, and any non-empty
PFTrace retained; Flameox does not silently substitute another domain.

Platform and requirements: Linux, a supported AMD GPU and ROCm installation,
and workload access to the host GPU device nodes (normally `/dev/kfd` and
`/dev/dri`). Flameox never changes device permissions or privileges. Tracing
overhead depends on the enabled domains and workload event rate. The normal
execution policy owns containment, quotas, timeout, cancellation, and the
bounded output directory; rocprofv3 owns the native PFTrace bytes.

Fixture provenance and proof gap: `tests/adapters/test_rocprofv3.py` verifies
strict options and exact argv construction. `tests/adapters/test_rocprofv3_integration.py`
creates a Flameox-authored fake CLI under the project MIT license and verifies output
naming and partial-artifact preservation, but its sentinel output is not presented as
a valid PFTrace. The project-owned
`project-owned-rocm-shaped-perfetto.json` fixture is a valid Perfetto-compatible
Chrome trace containing HIP-runtime-shaped and kernel events. It covers import,
bounded Perfetto extraction, and accelerator summarization, but it was not
produced by rocprofv3 and therefore does not establish native PFTrace
compatibility. No AMD host or vendor-produced trace is in scope, so rocprofv3
capture compatibility remains fixture/process-simulation backed.

Perfetto extraction inherits the workspace row quota, worker timeout, curated
standard-table queries, truncation reporting, and malformed-trace errors.
Unsupported or absent ROCm events degrade to incomplete standard summaries;
Flameox does not add a second parser or infer missing activity. Observed claims
are the selected domains, process result, native artifact identity, and events
returned by Trace Processor. Derived claims are bounded normalized slices,
counts, durations, and accelerator summaries. Inferred claims: none — neither a
trace nor an absent event establishes causality, correctness, or a bottleneck.

### Candidate adapters

- GDB/LLDB and elfutils for core metadata, with user init files, autoload, and
  implicit debuginfod disabled;
- AddressSanitizer, UndefinedBehaviorSanitizer, and ThreadSanitizer raw reports
  with narrow producer-versioned extraction rather than a claimed universal
  JSON schema;
- `rr` recording references;
- Nsight Systems Arrow, JSONL, or Parquet exports beyond the maintained SQLite
  subset;
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
