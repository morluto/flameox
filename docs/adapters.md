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
| Memory profiles | Memray | `memray` |
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
