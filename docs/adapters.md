# Provider adapters

Adapters translate explicit native artifacts or staged capture outputs into
typed evidence blocks. They do not discover projects, open repositories,
publish evidence, install providers, or own job lifecycle.

## Registry contract

Each capability registry entry owns:

- the capability descriptor and stable ID;
- accepted source modes and formats;
- a strict validation-equivalent argument model;
- compatibility rules derived from declared artifact formats;
- capture and analysis handlers;
- model-visible selection guidance, bounded examples, limits, overhead, and limitations.

One immutable capture-provider contract owns each provider's argument model, declared artifact
roles and formats, and selection description. Generated MCP schemas, pre-execution compatibility
checks, and capture argument validation consume that same contract so their format claims cannot
drift.

Adding or removing a format is complete only when these surfaces agree:

1. the capability's accepted formats;
2. the capture-provider contract and invocation builder, when capture is supported;
3. explicit-path suffix detection;
4. analysis dispatch into the provider or isolated worker;
5. generated MCP analysis and capture schemas;
6. the support table below.

Tests exercise explicit-path analysis and, when applicable, capture through the public runtime.
They also assert that the capture-contract and invocation-builder registries have identical provider
IDs. A parser or worker that is not reachable through a declared capability is either intentionally
internal and documented as such, or an incomplete registration—not latent support.

Capability tools remain discoverable when a provider is absent. Capture validates the selected
provider's package, executable, platform, version, permission, and required external resources as
part of the attempted operation, before workload execution. An unavailable provider returns a typed
error with either an exact `prepare_providers` retry action or external host guidance.

## Evidence and capture support

The registry covers CPU profiles, memory profiles, benchmarks, inference
exports, execution traces, GPU launches and kernel metrics, correctness and
sanitizer failures, coverage, static candidates, and bounded generic previews.
Artifact analysis and typed capture are separate contracts:

| Evidence family | Explicit artifact analysis | Typed capture provider |
| --- | --- | --- |
| CPU profiles | Node/V8, cProfile pstats, py-spy Speedscope, perf collapsed stacks, perf data | `node-cpu-profile`, `py-spy`, `perf` |
| Memory profiles | Memray, Node/V8 sampling heap profiles | `memray`, `node-heap-profile` |
| Benchmarks | pyperf, benchmark samples, PyTorch samples, NVBench | `pyperf`, `benchmark-samples`, `nvbench` |
| Execution traces | Perfetto/Chrome, PyTorch, OTLP, ROCprof PFTrace, xctrace, Nsight Systems | `torch-profiler`, `rocprofv3`, `xctrace`, `nsight-systems` |
| GPU and kernels | Nsight Systems, Nsight Compute, Compute Sanitizer, Triton, kernel validation | `nsight-systems`, `nsight-compute`, `compute-sanitizer` |
| Reliability | pytest events, observations, coverage.py | `pytest`, `observations`, `coverage` |
| Static candidates | SARIF | — |
| Inference exports | AIPerf, vLLM, SGLang, Mooncake | — |
| Generic previews | JSON, JSONL, CSV, text, Parquet | `direct` process output |

An em dash means callers provide an explicit artifact path; it does not mean
the format is unsupported. The offline inference readers omit prompts,
generations, error text, endpoints, tools, payloads, and prefix-hash values.
Pytest capture runs an explicit `python -m pytest` target with a request-bound,
bounded event plugin.

`failures.summary` scans the complete bounded pytest event artifact for aggregate outcomes, but its
table projects only failed, errored, interrupted, and unexecuted identities. Failed collection
reports retain their collector identity; successful collection events, passing tests, and skipped
tests cannot consume the diagnostic row budget.

Support is honest rather than substitutive. A missing Trace Processor does not
turn a Perfetto request into a JSON preview; a missing `ncu` does not become an
empty kernel report.

## Input and output ownership

Handlers receive resolved paths, exact digests, explicit format/producer hints,
provider arguments, and effective limits. They return typed observed/derived
blocks, staged native outputs, coverage, truncation, and limitations. They never
look up runs or mutable records.

Semantic selection happens before bounding. A provider must filter for the requested projection
before applying row, byte, or worker limits, and `rows_observed`, `complete`, and truncation must
describe that same projected population. A bounded prefix of unrelated native rows is not evidence
that the requested activity is absent. OTLP time windows apply their span predicate before resource
and scope rows consume the normalization budget; only owning context for retained spans is emitted.

Projection identity includes every non-axis dimension that can distinguish a series, including
device, dtype, variant, scope, and worker identity where applicable. Providers must not blend those
series. When a composite native label is successfully decomposed into named dimensions, the raw
label must not remain as a redundant identity that changes with the selected axis.

Numeric aggregation requires compatible semantic units. Providers normalize convertible units to
one declared output unit before pooling and reject incompatible dimensions; for example, seconds
and nanoseconds may become seconds, while bytes and elapsed time cannot share a CPU-hotspot total.
Rows and metrics expose the normalized unit so consumers never have to infer it from the format.

Native selectors such as table families, event categories, and field names require a representative
artifact or authoritative format fixture. Unmatched selectors stay incomplete or explicitly limited;
they must not silently become complete negative evidence. Adding a projection also requires a
regression check for sibling projections that share its reader or dispatch path.

Raw parser safety limits are independent from normalized output limits. In particular, V8 node and
sample ceilings bound native traversal, while the hotspot row limit bounds the ranked aggregate
projection. A partial SARIF document is likewise incomplete even when every parsed result fits in
the returned table.

Native references are validated before aggregation. Sample-to-node, edge-to-frame, parent-to-child,
and similar identifiers must resolve inside the same artifact unless the format explicitly defines
an external reference. A syntactically valid but unresolved identifier is corrupt evidence, not an
ignorable sample.

The runtime projects provider blocks into the RFC 8785 JSON domain before
bounding or preservation. Native integers outside JSON's interoperable safe
integer range are represented as exact decimal strings; they are never rounded
or allowed to fail later during canonicalization.

Native inputs remain byte-for-byte unchanged. Capture outputs stay in bounded
session scratch until explicit preservation. Large derived tables are immutable
files inside an evidence bundle, not rows in a durable database.

When capture-time preservation is requested, native outputs are published independently of
downstream normalization success. A decoder failure is retained as a typed analysis failure next
to authoritative native bytes rather than causing the capture to disappear.

## Isolation and conversion

Unsafe or heavy decoders use the typed isolated-worker protocol and
descriptor-bound reads. Workers receive only declared inputs and budgets.
Conversions are cached for the server session by input digest and provider
version.

Nsight Systems prefers `parquetdir` export. Flameox does not read or create a
Nsight SQLite export. Memray aggregation streams bounded records or uses
session-local in-memory DuckDB; it does not create a temporary SQLite database.

## Provider setup boundary

An explicit `flameox setup --provider ...` invocation prepares the exact
version-pinned uvx environment returned in its launcher. The invocation's
provider list is complete; Flameox retains no provider inventory or setup receipt.
System and vendor packages remain externally installed. Setup creates no
project state and is not a prerequisite for explicit-path analysis.
Nsight Compute capture requires both the `ncu` executable and NVIDIA's vendor-shipped
`extras/python/ncu_report.py` reader. Flameox resolves both before workload execution and returns
external setup guidance if the installation is incomplete.

Provider ownership is explicit during capture. External collectors installed
with Flameox, such as py-spy, execute from the launched uvx environment rather
than ambient request `PATH`. In-process collectors, including coverage.py and
Memray, must be installed in the declared workload interpreter; capture probes
that interpreter before execution and never substitutes Flameox's interpreter.

Pyperf command capture preserves multiline typed argv through a metadata-safe launcher because
pyperf rejects newline characters in its display metadata. The launcher executes the original argv
without shell parsing, but its startup is included in each command measurement; exact Flameox
capture provenance remains authoritative for the requested argv.
