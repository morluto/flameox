# Adapters and producer contracts

An adapter connects one maintained producer format to Flameox evidence. It owns
discovery, compatibility qualification, capture options, native artifact
registration, bounded extraction, and explicit limitations. It does not own
process creation, storage transactions, or analysis policy.

## Rules

Every adapter must:

- preserve the native artifact unchanged;
- bind producer, package, executable, workload, and environment identity;
- execute through the shared broker and consume a planned executable binding;
- parse in a bounded isolated worker when native libraries or large artifacts
  should not enter the application process;
- normalize only documented evidence with explicit units and provenance;
- distinguish unavailable, incompatible, partial, empty, and complete evidence;
- refuse unknown required producer versions or structures;
- publish limitations rather than silently falling back;
- keep provider-specific tests beside that provider and satisfy shared
  conformance behavior.

Passive discovery reports candidates. It is not execution authorization.
Planning binds the selected candidate through `ExecutableResolver` under the
same contract used by execution.

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

Parsing human diagnostic text, guessing units from field names, or maintaining a
parallel copy of a provider schema is not an adapter contract.

## Core adapters

### pyperf and Python startup

`pyperf` owns benchmark process isolation, warm-ups, calibration, samples, and
metadata. Flameox stores its JSON and publishes individual values; it does not
replace samples with a precomputed mean. Startup/import measurements use
`pyperf command`, not a private process-timing loop. Confirmatory comparison uses
preserved samples and declared paired estimands.

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

### Perfetto and OTLP

Perfetto-compatible traces are queried through Trace Processor with versioned,
parameterized SQL and row/time/string budgets. Flameox publishes only evidence
needed by supported recipes rather than copying an entire trace into Parquet.

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
