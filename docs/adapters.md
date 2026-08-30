# Adapters and producer contracts

An adapter connects one maintained producer format to Flameox evidence. It owns
discovery, compatibility qualification, capture options, native artifact
registration, bounded extraction, and explicit limitations. It does not own
process creation, storage transactions, or analysis policy.

## Rules

Every adapter must:

- preserve the native artifact unchanged;
- validate and extract the same immutable CAS payload rather than a mutable output path;
- bind producer, package, executable, workload, and environment identity;
- publish property-defining effective options, defaults, bounds, filters, and
  target scope as run semantics rather than relying on artifact bytes to carry
  them;
- execute through the shared broker and consume a planned executable binding;
- parse in a bounded isolated worker when native libraries or large artifacts
  should not enter the application process;
- normalize only documented evidence with explicit units and provenance;
- distinguish unavailable, incompatible, partial, empty, and complete evidence;
- refuse unknown required producer versions or structures;
- publish limitations rather than silently falling back;
- keep provider-specific tests beside that provider and satisfy shared
  conformance behavior.

For third-party adapters, capture snapshots each declared output through the
canonical no-follow import boundary before invoking adapter code. Validation and
extraction receive that immutable payload path. The validation receipt binds the
artifact ID and byte length to the exact adapter package and validator version;
the package identity is rechecked before both phases. A passing validation of a
mutable staging pathname is not authority for bytes imported later.

Passive discovery reports candidates. It is not execution authorization.
Planning binds the selected candidate through `ExecutableResolver` under the
same contract used by execution.

Internal workload-side collectors execute authorization-bound source directly,
never `python -m flameox.collectors...` from the workload environment or a
mutable staged launcher. Their content identity is part of the reviewed plan
and durable run semantics and is rechecked before execution. The workload
interpreter owns workload packages such as pyperf, pytest, and
pytest-reportlog. Collector code remains bound to the reviewed plan and its
recorded source identity.

Capture and extraction results may project a bounded subset of these run
semantics alongside status, limitations, and artifact references. The projection
is for immediate interpretation; the run remains authoritative after the result
is no longer available.

Compute Sanitizer extraction projects the effective tool, launch bounds,
target-process scope and filters, kernel filter, and suppression content digest.
That projection is also recorded in the normalized provenance observation, while
the run semantic digest remains the identity authority. Imported reports expose
capture scope as unavailable instead of inferring it from identical XML bytes.
The default diagnostic probe captures one launch. `launch_count=0` is an explicit
unlimited opt-in. Launch bounds include setup and reference CUDA work before the
target, so timeout recovery recommends target-only workloads and provider kernel
or process filters rather than merely retrying the same plan.

## Producer ownership

Use maintained formats and models wherever they exist:

| Evidence | Authoritative producer or reader | Flameox-owned semantics |
| --- | --- | --- |
| Python benchmark samples | pyperf JSON | run linkage, comparison eligibility |
| Python startup/import timing | `pyperf command` | declared startup metric identity |
| pytest reports | pytest-reportlog, pytest, pytest-xdist | fixture samples, controller receipt time, collected IDs |
| sampled Python stacks | py-spy formats | bounded frame/measurement normalization |
| allocations | Memray reader | memory concepts and run provenance |
| Python execution | coverage.py data API | bounded contexts and line/branch evidence |
| temporal traces | Perfetto Trace Processor | curated queries and bounded rows |
| PyTorch operators | torch.profiler Chrome trace | operator/accelerator summaries |
| Linux profiles | perf data/script | bounded stacks and limitations |
| NVIDIA system traces | official Nsight Systems SQLite export | qualified table extraction and correlation |
| NVIDIA kernel metrics | `ncu_report.py` | selected metric normalization |
| NVIDIA correctness findings | Compute Sanitizer XML | typed findings and launch correlation |
| GPU benchmark samples | NVBench JSON and declared sidecars | bundle integrity and metric provenance |
| ROCm traces | ROCprofiler Perfetto-compatible output | qualified trace semantics |
| Inference benchmark | AIPerf models, vLLM/SGLang structured exports | protocol identity and prompt-free request metrics |
| HTTP | HTTPX policy transport | loopback/readiness limits and typed errors |
| managed tools | checked-in upstream asset manifests | digest-before-execution and installed-byte receipts |
| test-case reduction | ShrinkRay 26.7.8.0 CLI | predicate authority, tri-state receipts, final revalidation |
| NVIDIA identity | nvidia-ml-py 13.610.43 / NVML | stable device identity and API-enum topology |
| Apple identity | system_profiler, sysctl, sw_vers | Metal, GPU, memory, and macOS build identity |
| Apple Metal traces | xcrun xctrace | immutable native bundle and bounded TOC export |

Parsing human diagnostic text, guessing units from field names, or maintaining a
parallel copy of a provider schema is not an adapter contract.

Static-analysis imports follow the same ownership rule. Flameox preserves a
supported SARIF 2.1.0 report unchanged, records import scope and limitations on
the run, and publishes bounded source candidates in a separate generation.
Provider extensions remain in the native report. A candidate is analyzer output,
not a Flameox Finding or runtime confirmation.

NVIDIA identity runs in a dedicated module-only provider environment. Its typed
worker uses read-only NVML queries and keeps UUID and PCI identity distinct from
the observed NVML index. Apple identity is collected only when declared and
never includes hardware serial numbers. Native `.trace` directories are imported
as sensitive, link-free tar artifacts; the installed `xctrace` must recognize
the Metal System Trace template and export a bounded table of contents. Unknown
event-table schemas remain unavailable rather than being scraped heuristically.

## Reduction provider

ShrinkRay is a managed provider, not a capture adapter or a control-process
library. The `flameox.shrinkray.offline-v1` profile fixes its public CLI,
parallelism, seed, candidate passing, local history, and disabled formatter,
restart, Python reducer, language model, proxy, credential, and download paths.

ShrinkRay proposes candidates and records accepted history. A small Flameox
bridge alone owns the declared predicate command. Each invocation validates and
hashes the candidate beneath the operation root, applies repetitions and
timeouts, and atomically records `interesting`, `not_interesting`, or
`unresolved`. Exit 101 is passed as `--also-interesting`, so unresolved
candidates are recorded but cannot become the reduction basis. After ShrinkRay
returns, the worker rechecks the actual staged final file and the application
rechecks it independently before registering a final result.

Results preserve native history, candidate receipts, bounded logs, provider
identity, sizes, and limitations. Minimality is always `not_claimed`. Planning
binds one exact run registration rather than accepting a duplicate artifact/run
identity pair. A changed final or best-known candidate is registered back onto
that run with ShrinkRay producer identity and inherited sensitivity. UTF-8
process output retains its text or JSON media type only after validation;
everything else uses the explicit `reduced_candidate` kind and reports that the
source provider format has not been requalified.

Managed binary adapters keep tool-specific extraction and compatibility probes,
but they do not own download or attestation policy. A checked-in immutable asset
value names the upstream manifest revision, exact URL, size, archive digest, and
installed executable digest. The shared acquisition helper authenticates those
facts before an adapter may execute bytes. Complete upstream assets are retained
in content-addressed workspace storage; resumable partials remain staging state,
not trusted artifacts. Reuse re-hashes the installed file
against the checked-in executable digest; changing both the binary and its local
receipt therefore cannot manufacture trust. Perfetto values are transcribed
from its generated `trace_processor_shell` manifest, and Toxiproxy values are
qualified from its exact release archives.

## Core adapters

### pyperf and Python startup

`pyperf` owns benchmark process isolation, warm-ups, calibration, samples, and
metadata. Flameox stores its JSON and publishes individual values; it does not
replace samples with a precomputed mean. Startup/import measurements use
`pyperf command`, not a private process-timing loop. Its closed startup profile
uses five workers with one value, one loop, and no warm-ups, so every retained
wall sample represents exactly one fresh command execution. The native pyperf
JSON is authoritative for wall time and `command_max_rss`; a separate raw
`-X importtime` stderr artifact is authoritative for import costs. Missing RSS
metadata remains missing rather than being replaced by another backend.
Confirmatory comparison uses preserved samples and declared paired estimands.

The general `pyperf` capture adapter is explicitly a fresh-process command
benchmark. Each value includes interpreter startup, imports, framework
initialization, input construction, and other entrypoint setup. It is suitable
for CLI, startup, and end-to-end process questions, but not for the latency of
an already-prepared framework operation. Its durable semantics record
`process_scope=fresh_process_invocation`. For a workload that declares an
accelerator identity, the plan also returns `alternative_action`: a validated
`plan_capture` call for `benchmark-samples`. That route remains workload-owned:
the workload must emit synchronized raw samples. PyTorch workloads can instead
instrument an already-prepared callable with `torch.benchmark`; choosing
`pyperf` is a deliberate process-scope measurement, not a substitute for
device timing.

### PyTorch operation benchmarks

`torch.benchmark` is the narrow in-process counterpart for an already-prepared
PyTorch callable. The workload performs setup, lazy initialization, and oracle
validation before it calls `flameox.sdk.torch_benchmark()`. Flameox delegates
host wall-time collection, warm-up, thread-pool selection, and accelerator
synchronization to `torch.utils.benchmark.Timer`, then retains its bounded
per-loop samples through the canonical `flameox.benchmark-samples.v1` artifact
and normalized `benchmark-samples` extraction path.

```python
from flameox.sdk import torch_benchmark

# Construct inputs and validate the result before timing.
torch_benchmark("gae.step", lambda: chunked_gae(rewards, values))
```

`min_run_time_seconds`, `max_samples`, and `num_threads` are plan-bound
options; the capture deadline remains the workload deadline. `max_samples`
bounds retained evidence after the provider's blocked autorange measurement.
When `cuda_event_timing` is selected, Flameox records an additional
`<metric>.cuda_event` series with `measurement_clock=cuda_event` and explicit
device/stream identity. It remains a distinct device-time metric: Timer's
host-observed latency is never replaced or combined with CUDA event time.
CUDA-event collection requires CUDA and fails explicitly when unavailable.

For non-Torch or bespoke accelerator producers, use `benchmark-samples` and
declare the clock, synchronization, warmups, loop count, scope, device, and
dimensions. A metric name cannot mix host and device clocks, so consumers never
silently combine incompatible samples.

### pytest

The unmodified pytest-reportlog JSONL artifact is the source for `CollectReport`,
`TestReport`, warning, and session records. The small Flameox plugin adds only
namespaced values that upstream reports do not provide: individual fixture setup
samples, run origin, controller receipt time, and collected node IDs. It does
not maintain an ordinary report serializer, xdist scheduler, worker report
transport, or recovery sidecar.

Extraction allowlists supported fields, bounds lines and rows, tolerates a
truncated final record as explicitly partial evidence, and ignores unknown
report types as required by reportlog. A worker crash can contribute only
reports already delivered to pytest; the result records that limitation.

### py-spy, Memray, perf, and coverage

These adapters preserve their native artifact and use the producer-supported
reader or export. Frame and allocation rows retain source artifact and run
identity. Missing native symbols, thread identity, contexts, or native frames
stay unavailable. Flameox does not infer them from display text.

Coverage uses the control interpreter's bundled `coverage.py` reader, so its
reader is a core Flameox dependency and is qualified before capture planning.
The exact workload interpreter is queried for the producer version; a captured
run registers `producer="coverage"` with that version. Extraction requires
that identity and a producer version in `coverage>=7.14,<8` before constructing
or reading `CoverageData`. The native database remains unchanged when the
producer is unsupported, and the error directs the caller to recapture with a
supported workload package.

The normalized generation keeps the same provenance chain: its publisher is
`coverage` (the coverage.py reader) and its publisher version is the control
reader version, while its input artifact ID points back to the run registration.
This separates provider-owned SQLite bytes from run-scoped producer identity
and reader provenance. Checked-in provider-generated fixtures exercise
coverage.py 7.14.2 and 7.15.2 through the public `CoverageData` API; they are
read offline by the normal adapter tests.

Memray capture binds the producer version discovered from the declared workload
interpreter. Planning requires a verified managed reader for that exact version,
so an unavailable reader fails before capture overhead and returns a typed
`start_capability_setup` action. Extraction runs that reader outside the control
process and records producer version, reader version and environment, and the
extractor profile separately. Native reader acceptance—not package-major
comparison—is the format compatibility boundary.

Memray has two capture scopes. `whole_entrypoint` is the default and invokes
the provider CLI around the declared Python entrypoint. It includes imports,
setup, and warm-up, so it answers whole-process allocation questions. `sdk`
uses the maintained `memray.Tracker` API around exactly one workload-owned
operation. The single capture plan/run contract binds the region name,
declared warm-up count, provider version, workload and callable identity,
process/thread scope, native-stack and Python-allocator choices, generated
output path, timeout, and artifact limits. The durable run semantics carry the
property-defining subset (`mode`, `process_scope=workload_process`,
`thread_scope=all_threads`, `warmup_count`, and the declared region); workload,
source, output, timeout, and limit ownership stays on the plan/run rather than
being copied into a second Memray schema.

For a warm operation, perform setup and warm-up before the context, then use
the exact planned name:

```python
from flameox.sdk import memray_region

for _ in range(2):
    warm_up_model()
with memray_region("steady_step"):
    run_one_step()
```

Set `warmup_count=2` in the SDK Memray options when the plan uses the two
warm-up calls above. The count is declared run semantics; Flameox does not
guess or count arbitrary calls made outside its SDK context. Whole-entrypoint
captures do not accept a warm-up count because their measured interval starts
at process entry.

The SDK owns the tracker lifecycle and the plan-authorized `.bin` path. Memray
records every thread in the workload process while that context is active; it is
therefore a precise time window, not a thread filter. Forked children are not
tracked: Flameox currently exposes no `follow-fork` switch, so child-process
allocations are a stated limitation in both capture scopes. A nested,
concurrent, or repeated SDK region fails before Flameox starts another tracker
and directs the caller to make a fresh plan with one region. Missing, repeated,
overlapping, and never-closed region lifecycles become typed run validation
errors from the bounded SDK observations. Calling no context leaves the
required native output absent and the capture fails; if Memray has already
written a valid `.bin`, stdout/stderr, or process snapshot before a workload or
validation failure, Flameox preserves that evidence.

The SDK writes its existing bounded semantic observations immediately before
and after the tracker context. They state whether an already-loaded Torch
runtime reported CUDA initialized at each boundary; `null` means Torch was not
loaded or could not report that state. This distinguishes observed runtime state
from the plan's declared warm-up boundary without making an unsupported claim
about allocations that occurred before tracking. Native traces and Python
allocator events are off by default because both increase capture volume and
overhead; enable them explicitly when their extra evidence is needed.

Memray extraction snapshots its complete complexity budget in the durable
operation request: input bytes, provider records, unique frames, stack depth,
aggregate rows, unique call edges, representative stacks, output bytes, wall
time, and worker RSS. The isolated reader
streams each provider record set and retains a bounded heap ordered by allocated
bytes, with provider order as the deterministic tie-breaker. Normalization then
visits those largest provider aggregates first, so frame and aggregate limits
retain contributors from the highest-value records rather than an arbitrary
prefix. This is a bounded projection, not a replacement for the native profile.
Direct caller/callee edges and representative stacks are normalized as bounded
navigation evidence for `memory.high_watermark`, `memory.retained_end`,
`memory.allocated`, and `memory.temporary`. The views retain Memray's provider
meanings: high-watermark records contributed to the allocation peak, retained-end
records were still allocated when tracking ended, allocation-volume records cover
all positive allocation events, and temporary records were allocated and freed
with at most the configured number of other allocations between those events.
The extraction default is Memray's threshold of one; zero means that no other
allocation may intervene. The threshold is an allocation-event distance, not a
duration or byte-size cutoff, and is recorded with the extracted temporary-byte
total.
Aggregated Memray captures do not expose allocation-volume or temporary record
streams. Flameox publishes their peak and retained evidence with those optional
views explicitly unavailable instead of rejecting the capture.
Memray reports stacks leaf-first; Flameox stores them root-to-leaf, preserves
repeated and recursive frames, and weights edges and stacks in bytes with the
provider allocation count alongside that weight. It does not invent native
frames or a Python/native distinction that the selected provider stack API does
not expose. Native and hybrid stacks remain available in the immutable profile
when the capture contains them.

The result reports records and record bytes seen versus selected, dropped stack
frames, aggregates, edges, and representative-stack weight, published row
counts, and output bytes.
Capture a narrower profile or raise applicable workspace budgets before starting
a new extraction when complete normalized coverage is required; the immutable
native profile remains the authority.

Memray frame paths are interpreted from preserved capture provenance. Relative
provider filenames resolve lexically against the captured workload cwd, never
the extractor process cwd. Paths contained by the workspace project root are
stored project-relative and carry the run's source-state identity. Synthetic,
external, or escaping relative paths remain partial and are not promoted to
project source.

Memray count metrics keep provider concepts separate. `memory.allocation_operations`
and `memory.allocated_bytes` come from the version-qualified structured stats
computation used by Memray's stats reporter; deallocation events are therefore
not counted as allocations. `memory.capture_records` preserves the distinct raw
header record count. Pre-aggregated captures, for which Memray does not support
stats computation, publish the raw record count without inventing allocation or
total-volume values.

### Node/V8 CPU and sampling heap profiles

The Node adapters invoke the declared Node executable with `--cpu-prof` or
`--heap-prof`, preserving the native `.cpuprofile` or `.heapprofile` output even
when the workload exits nonzero. Node 20.16+ and 22.4+ are required. Extraction
runs in the bounded profile worker and streams the native JSON rather than
loading it into the control process. CPU tree nodes are visited once; heap
inclusive values include descendant allocations. Frame identity uses normalized
source coordinates, with a script disambiguator only when no URL is available.
Unresolved or synthetic frames remain partial symbolization.

### Perfetto and OTLP

Temporal windows have one provider-neutral application contract. They query
normalized trace events first, so an extracted trace does not require its native
reader for each bounded drill-down. Perfetto-compatible artifacts that have not
been normalized fall back to Trace Processor with versioned, parameterized SQL
and row/time/string budgets. Flameox publishes only evidence needed by supported
recipes rather than copying an entire trace into Parquet.

OTLP JSON or protobuf is an import-first producer format. Resource, scope, span,
event, and link rows retain typed attributes and parentage. Operation-window
analysis is derived from these rows; malformed nesting or missing timestamps is
reported, not repaired heuristically.

### torch.profiler

The native compressed Chrome trace remains authoritative and is passed through
the Perfetto path. Normalization keeps CPU operators, accelerator events,
correlation, shapes, stacks, and memory only when the trace supplies them.
Profiler capture is diagnostic; it does not substitute for an unprofiled
benchmark comparison.

Flameox exposes two capture modes. `whole_entrypoint` profiles the declared
Python entrypoint with CPU and CUDA-if-available activities, while shapes,
memory, stacks, FLOPs, and modules are all opt-in. Its run semantics retain the
selected activities and options, and the plan states their expected overhead.
It is still an application-wide trace: Flameox cannot infer setup, compilation,
warm-up, steady state, or a `record_function` region from an unmodified
workload. `sdk` is the deliberate narrow mode. The workload enters
`flameox.sdk.torch_profiler()` and supplies explicit `step()` boundaries; its
declared schedule determines the expected trace files, which are registered as
the native evidence.

When a Torch capture times out or exceeds process, resource, or artifact bounds,
Flameox offers one lower-overhead replan with enabled high-cardinality options
disabled. If they are already disabled, the result says that retrying the same
whole-entrypoint plan cannot narrow it and directs the agent to SDK/workload
instrumentation. The launcher preserves bounded scalar provider diagnostics with
the run: `artifact_size_bytes`, `event_count`, and `active_duration_us` when the
completed provider exposes event timing. The duration is the span of provider
events, not a claim about a warm-up or operation region; unsupported fields are
omitted. Process duration, RSS, staging growth, and output limits remain in the
existing process/resource owners, while normalized operator counts remain in
the extraction owner.

## Accelerator adapters

### Nsight Systems

The generic `nsight.systems` adapter profiles any declared workload. It preserves
the native `.nsys-rep` and asks the same provider invocation to emit an official
SQLite export, avoiding a second export workflow. Effective trace domains,
capture range, range-end behavior, CUDA graph granularity, process scope, and
disabled CPU sampling and symbol resolution remain run semantics. The defaults
trace CUDA, NVTX, and OS runtime activity across the application process tree;
callers must opt into the risky pre-exec fork interval, system-wide CUDA tracing,
or range-delimited capture. Graph-level CUDA graph tracing is the conservative
default; node detail is explicit.

The maintained import and capture paths accept NVIDIA's official SQLite export.
Required tables are qualified before extraction; unknown required schemas fail.
The native report remains the forward-compatible authority while normalized rows
preserve CUDA runtime, graph, kernel, stream, correlation, and timing identity.

Export product and schema identity are observed from the export metadata and
checked against any asserted producer version. Only an explicit metadata
allowlist is published; host, user, command, and path metadata remain in the
native export. NVTX marks, ranges, and domain metadata preserve their event type,
including valid rows whose end timestamp is null. Bounded trace windows expose
these normalized events without requiring `nsys` or Trace Processor.

The repository contains one vendor-produced qualification fixture at
`tests/fixtures/nsight_systems/nsight-2025.5.2.sqlite`; its adjacent README binds
producer version, workload, host, and digest. That fixture supports only the
documented schema family, not every Nsight release.

`.nsys-rep` capture artifacts remain opaque until exported by the installed
`nsys` tool. Flameox does not parse the private report container.
Automated Nsight captures explicitly disable symbol downloads so provider
finalization cannot silently consume the capture deadline. The bounded response
reports workload, provider-finalization, and SQLite-export durations separately;
symbol resolution is recorded as disabled in the reviewed plan and result.

### Nsight Compute

Capture uses reviewed metric sets and preserves `.ncu-rep`. Extraction delegates
to NVIDIA's installed `ncu_report.py`; its exact reader identity becomes part of
the extraction generation. UINT64 values remain unsigned/decimal-safe. Missing
metrics, replay limitations, and restricted counter permissions stay explicit.

Guided analysis is deliberately a second, bounded projection. Extraction calls
the documented `rule_results_as_dicts()` interface and persists typed
`rule_identifier`, `section_identifier`, `rule_message`, `speedup_estimation`,
and `focus_metrics` with the action and range location in the native report.
Provider message and estimate labels are retained separately from Flameox's
documented interpretation. Extraction also records the provider-reported rule
section identifiers in `profile.extraction`, independently of whether a bounded
typed finding is emitted. `analyze nsight-compute` and
`analyze_nsight_compute` read those persisted facts and pinned run semantics;
they never reopen or decode `.ncu-rep`. The native report remains the immutable
authority for deeper provider inspection or re-extraction.

The analysis projection owns roofline coverage from those persisted section
identifiers. It prioritizes provider-estimated impact, reports target
qualification and extraction bounds, and returns a recapture selection when
coverage is incomplete. Without a recorded kernel filter, that selection leaves
the filter unset and asks for the intended target; it never adopts an incidental
observed action. Re-extract the unchanged report to publish typed facts.

Local qualification covers only versions and devices exercised by the live
lane. A package or executable probe alone does not establish compatibility.

### Compute Sanitizer

Flameox requests the producer's structured XML and derives finding categories
from typed fields, not human diagnostic prose. Unsupported schema families fail
rather than being guessed. Unknown record elements remain limitations. The result
separates tool failure, application failure, and reported findings.

### NVBench

NVBench capture is available only for a declared NVBench executable. A workload
opts into that contract with `execution_protocol = "nvbench"`; the declaration
is not inferred from a Python or shell command. Planning then runs the exact
resolved executable with the bounded, non-benchmarking `--version` probe and
requires NVBench identity before adding any benchmark flags. Passive inspection
can report that the adapter implementation is installed, but it cannot establish
workload compatibility. A failed probe, passive-only planning, or a changed
executable is refused and requires a new plan.

NVBench imports one JSON document plus only the sidecars it declares. Paths must
stay beneath the bundle root and pass encoding, type, size, digest, symlink, and
hard-link checks. Timing and frequency sidecars remain unchanged. Normalization
retains benchmark, device, axis, sample, unit, and metric identity; it does not
pool unrelated axes.

### Kernel build and validation

Triton and CuTe compiler capture preserves exact compiler outputs and binds them
to the workload definition, instance, toolchain, and device. For Triton,
planning resolves the `triton` distribution through the declared workload
interpreter and records its canonical distribution name, version, full RECORD
content digest, and interpreter digest. Execution repeats that probe immediately
before launch; a changed distribution invalidates the plan. Kernel-validation
receipts distinguish exact agreement, tolerance metrics, failure, and
unavailable backends. Compile success is not semantic correctness.

Triton dump directories are evidence groups, not one flattened compiler pass.
Triton creates one dump directory for each source-hash compilation and writes
that compilation's stage files there. Flameox therefore preserves every native
file under its immediate parent directory and creates one sibling
`ArtifactPipeline` per group. Each pipeline has deterministic local lineage
(`TTIR → TTGIR → LLIR → PTX → CUBIN → SASS` when those files exist); it never
implies an order or predecessor across groups. JSON and `.metadata` files are
preserved when the provider emits them, but they are never compiler stages or a
source of selection claims.

Managed Triton capture installs the current provider listener
`triton.knobs.autotuning.listener` in the declared root Python script or module.
The listener is the semantic authority for an observed multi-configuration
decision: function identity, opaque tuning-key digest, selected configuration,
bounded candidate configurations and timing vectors, cache-hit status, and the
fresh-tuning duration. Candidate timing values follow Triton's `do_bench`
milliseconds convention. Cache hits retain provider timings but have no fresh
tuning duration. String-valued configuration entries and tuning-key values are
digested rather than copied into normalized evidence.

The hook does not expose a dump directory, compiler cache path, cache hash, or
pipeline identity. Flameox therefore does not attach a selection to a native
candidate group and does not inspect cache files to fill that gap. Empty evidence
means no multi-configuration decision was observed in the root interpreter; a
workload may have no autotune, one effective configuration, a child-process
autotune, or a later hook replacement. Re-run the declared workload where the
decision occurs in the root interpreter, then use
`query_triton_autotune_selections` or
`flameox://triton-autotune/<run-id>/selections` for cursor-bounded evidence.

The same managed wrapper uses Triton's current
`triton.knobs.compilation.listener`. It records no cache path, cache key, or
cache contents. Instead, the provider callback supplies the target for both a
fresh compile and a cache hit. Flameox requires that target to agree with every
emitted PTX `.target` directive, the exact plan-bound Triton distribution, and
the observed CUDA environment. PTX version is retained when the provider emits
PTX. An explicit `target` option is cross-compilation intent only: it must agree
with the provider-reported target, but it may differ from the execution GPU. If
the callback or authoritative CUDA identity is unavailable, the native evidence
is still preserved, but its pipeline is managed-partial and exact comparison is
rejected with explicit re-capture guidance. Conflicting listener, PTX, plan, or
environment identity rejects the capture rather than assigning a target to the
wrong dump group.

The kernel-build document is a strict provenance record containing the provider
discriminator, grouped artifact paths, media types, sizes, digests, and explicit
attachments. It intentionally excludes tool version, workload,
target, environment, bounds, status, and limitations. Those are durable run
semantics in the authoritative run manifest and plan. `ArtifactPipeline` keeps
only group-local lineage plus qualified compiler and target identities; it
resolves all run semantics through `run_id`. Imported native dumps therefore
remain identity-unverified until a managed run supplies those semantics; Flameox
does not reconstruct them from the provider document.

When a workload emits `flameox.kernel-validation.v2`, use the dedicated
registration operation with the producing execution run and its reviewed
revision. Flameox validates the immutable document before appending it to that
run, derives producer metadata from the document, and keeps the run's existing
workload, environment, source, and execution identities authoritative. Extraction
then publishes bounded case and metric rows against the same run. It does not
manufacture an import run or accept caller-supplied identity copies.

### ROCprofiler

ROCprofiler v3 uses its Perfetto-compatible output and the common trace path.
The committed project-owned fixture proves normalized schema behavior; it is not
vendor-host qualification. Live ROCm compatibility remains unclaimed until the
native lane runs on a supported AMD host.

## Inference benchmarks and profiling

Flameox coordinates AIPerf 0.12, `vllm bench serve`,
`python -m sglang.benchmark.serving`, torch.profiler, and Nsight Systems. It
does not implement a load generator, scheduler, model server, or profiler.

Servers and scenarios are typed `flameox.toml` declarations. Managed servers
name a workload and run through a broker-owned lease. Existing-local servers are
restricted to loopback health/model probes and remain exploratory because their
complete command, model, scheduler, and cache state is outside Flameox authority.

Planning qualifies one executable binding and its supported command interface
before allocating an output directory or issuing a plan. A qualified plan owns
that binding and reviewed argv; an unavailable or incompatible launcher returns
`CAPABILITY_UNAVAILABLE` with recovery instead of a durable unusable plan.
Launch revalidates the exact executable identity without a second provider
discovery. The public plan is a preview; execution consumes its opaque
server-owned capability.

AIPerf records are parsed through maintained provider models with bounded
conversation/turn correlation. vLLM and SGLang use their structured aggregate
or record exports. Prompt bodies, generated text, and provider error messages
remain only in sensitive native artifacts; agent-facing evidence contains token
counts, timing, schedule, safe error types/codes, and declared cache fields.

SGLang support is limited to generic serving-benchmark evidence through the
current canonical `sglang.benchmark.serving` interface. Flameox does not own a
SGLang or Slime rollout HTTP replay protocol: request dumps, lifecycle events,
cancellation, sessions, and rollout coordination remain a declared external
workload or typed imported evidence path. Flameox preserves those native inputs
and outputs as immutable artifacts and records the workload's bounded run
semantics inline.

Each benchmark has one canonical run. Imported provider files retain their own
artifact/source-run provenance while normalized rows publish under the canonical
run. Partial files from failure or cancellation are preserved when valid.

Profiling requires a successful, compatible, unprofiled measurement run.
torch.profiler traces follow the Perfetto path. Nsight captures preserve
`.nsys-rep` and require the official SQLite export before extraction. A profile
is diagnostic and remains linked to the measurement run it represents.

Confirmatory inference comparison requires immutable model/tokenizer identity,
managed server command identity, compatible accelerator and protocol identity,
complete request population, and an appropriate passing oracle. A per-run oracle
cannot prove cross-treatment equivalence.

## Third-party adapters

Third-party entry points are discovered without becoming trusted automatically.
Preparation records an exact installed distribution and entry-point identity.
Execution remains subject to the broker, filesystem, artifact, schema, and
privacy boundaries. Unknown code is not made safe merely by returning a valid
Pydantic model.

## Capability states and qualification

Capability reporting separates:

- package or executable absent;
- installed but incompatible;
- available under the current host and policy;
- blocked by permission, driver, platform, containment, or dependency;
- available only for import, capture, or analysis;
- unqualified live behavior.

Compatibility claims require a representative native fixture or live/provider
lane. Fake processes prove orchestration, not producer compatibility. Every
qualification record states producer version, platform/device when relevant,
artifact origin, exercised behavior, and remaining blind spots.

Candidate formats belong in an issue until they have a maintained reader,
bounded artifact contract, useful evidence semantics, and representative
validation. Integration count is not a product goal.

## Runtime-resource metric admission

The runtime-resource comparison catalog is a reviewed registry, not a projection
over arbitrary evidence-table columns. Every admitted scalar records its exact
scope, unit, collection backend, aggregation, unavailable-evidence behavior,
cross-run compatibility fields, confirmatory eligibility, and limitations. The
workload validator and comparison service consume that same registry, so adding
a configuration name alone cannot accidentally make a column comparable.

The current catalog contains only:

| Metric | Scope and aggregation | Compatibility identity |
| --- | --- | --- |
| `runtime_resource.peak_rss_bytes` | Maximum sampled sum of RSS for the broker-owned workload process tree and recursively discovered descendants | sampling interval and `psutil_recursive_polling` backend |
| `runtime_resource.minimum_free_bytes` | Minimum sampled free bytes on the filesystem containing workspace staging | sampling interval; the `shutil.disk_usage` backend is fixed by the registry |
| `runtime_resource.staging_growth_bytes` | Nonnegative before/after growth of the bounded staging tree, excluding declared writable-root growth | fixed bounded-tree producer contract |

Unavailable observations make that run ineligible; they are never converted to
zero. Peak RSS remains a sampled observation, not a lifetime maximum, and may
miss short-lived descendants. Minimum free space includes unrelated filesystem
activity. Staging growth can miss files created and removed between its two
observations, and zero is outside the current positive log-ratio estimand.

An expansion must change the registry and add behavioral evidence for all of the
fields above. A different accounting scope (for example cgroups, `wait4`, or an
allocation profiler) is a different metric contract, not a fallback backend for
peak RSS. Writable-root growth remains outside the scalar catalog until runs can
bind the same root identity. Metrics without a defensible independent unit or
compatibility predicate must set confirmatory eligibility to false.
