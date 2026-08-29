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
| Inference replay | AIPerf models, vLLM/SGLang structured exports | protocol identity and prompt-free request metrics |
| HTTP | HTTPX policy transport | loopback/readiness limits and typed errors |
| managed tools | checked-in upstream asset manifests | digest-before-execution and installed-byte receipts |
| test-case reduction | ShrinkRay 26.7.8.0 CLI | predicate authority, tri-state receipts, final revalidation |
| NVIDIA identity | nvidia-ml-py 13.610.43 / NVML | stable device identity and API-enum topology |
| Apple identity | system_profiler, sysctl, sw_vers | Metal, GPU, memory, and macOS build identity |
| Apple Metal traces | xcrun xctrace | immutable native bundle and bounded TOC export |

Parsing human diagnostic text, guessing units from field names, or maintaining a
parallel copy of a provider schema is not an adapter contract.

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
identity, sizes, and limitations. Minimality is always `not_claimed`.

Managed binary adapters keep tool-specific extraction and compatibility probes,
but they do not own download or attestation policy. A checked-in immutable asset
value names the upstream manifest revision, exact URL, size, archive digest, and
installed executable digest. The shared acquisition helper authenticates those
facts before an adapter may execute bytes. Reuse re-hashes the installed file
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

## Accelerator adapters

### Nsight Systems

The maintained import path accepts NVIDIA's official SQLite export from
`nsys export --type=sqlite`. Required tables are qualified before extraction;
unknown required schemas fail. The source SQLite stays authoritative and
normalized rows preserve CUDA runtime, graph, kernel, stream, correlation, and
timing identity.

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

### Nsight Compute

Capture uses reviewed metric sets and preserves `.ncu-rep`. Extraction delegates
to NVIDIA's installed `ncu_report.py`; its exact reader identity becomes part of
the extraction generation. UINT64 values remain unsigned/decimal-safe. Missing
metrics, replay limitations, and restricted counter permissions stay explicit.

Local qualification covers only versions and devices exercised by the live
lane. A package or executable probe alone does not establish compatibility.

### Compute Sanitizer

Flameox requests the producer's structured XML and derives finding categories
from typed fields, not human diagnostic prose. Missing, invalid, older, or newer
schema families fail rather than being guessed. Unknown record elements remain
limitations. The result separates tool failure, application failure, and
reported findings.

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
to the workload definition, instance, toolchain, and device. Kernel-validation
receipts distinguish exact agreement, tolerance metrics, failure, and
unavailable backends. Compile success is not semantic correctness.

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

## Inference replay and profiling

Flameox coordinates AIPerf 0.12, `vllm bench serve`, SGLang benchmark entry
points, torch.profiler, and Nsight Systems. It does not implement a load
generator, scheduler, model server, or profiler.

Servers and scenarios are typed `flameox.toml` declarations. Managed servers
name a workload and run through a broker-owned lease. Existing-local servers are
restricted to loopback health/model probes and remain exploratory because their
complete command, model, scheduler, and cache state is outside Flameox authority.

Plans bind the configuration, resolved provider and server executables, model
and tokenizer revisions, schedule, request population, endpoint, output root,
deadline, profiler options, and oracle. The public plan is a preview; execution
consumes its opaque server-owned capability.

AIPerf records are parsed through maintained provider models with bounded
conversation/turn correlation. vLLM and SGLang use their structured aggregate
or record exports. Prompt bodies, generated text, and provider error messages
remain only in sensitive native artifacts; agent-facing evidence contains token
counts, timing, schedule, safe error types/codes, and declared cache fields.

Each replay has one canonical run. Imported provider files retain their own
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
