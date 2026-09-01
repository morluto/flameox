# Provider adapters

Adapters translate explicit native artifacts or staged capture outputs into
typed evidence blocks. They do not discover projects, open repositories,
publish evidence, install providers, or own job lifecycle.

## Registry contract

Each capability registry entry owns:

- the capability descriptor and stable ID;
- accepted source modes and formats;
- a strict validation-equivalent argument model;
- provider probes and compatibility rules;
- capture and analysis handlers;
- bounded examples, limits, overhead, and limitations.

Discovery probes packages, executables, platform, supported version, permission,
and required external resources without mutating the host. Missing dependencies
are successful availability results with remediation outside Flameox. Invoking
an unavailable provider is a typed tool error.

## Evidence and capture support

The registry covers CPU profiles, memory profiles, benchmarks, inference
exports, execution traces, GPU launches and kernel metrics, correctness and
sanitizer failures, coverage, static candidates, and bounded generic previews.
Artifact analysis and typed capture are separate contracts:

| Evidence family | Explicit artifact analysis | Typed capture provider |
| --- | --- | --- |
| CPU profiles | Node/V8, py-spy Speedscope, perf collapsed stacks, perf data | `node-cpu-profile`, `py-spy`, `perf` |
| Memory profiles | Memray | `memray` |
| Benchmarks | pyperf, benchmark samples, PyTorch samples, NVBench | `pyperf`, `benchmark-samples`, `nvbench` |
| Execution traces | Perfetto/Chrome, PyTorch, OTLP, ROCprof PFTrace, xctrace, Nsight Systems | `torch-profiler`, `rocprofv3`, `xctrace`, `nsight-systems` |
| GPU and kernels | Nsight Systems, Nsight Compute, Compute Sanitizer, Triton, kernel validation | `nsight-systems`, `nsight-compute`, `compute-sanitizer` |
| Reliability | pytest events, observations, coverage.py | `pytest`, `observations`, `coverage` |
| Static candidates | SARIF | — |
| Inference exports | AIPerf, vLLM, SGLang, Mooncake | — |
| Generic previews | JSON, JSONL, CSV, text, Parquet | — |

An em dash means callers provide an explicit artifact path; it does not mean
the format is unsupported. The offline inference readers omit prompts,
generations, error text, endpoints, tools, payloads, and prefix-hash values.
Pytest capture runs an explicit `python -m pytest` target with a request-bound,
bounded event plugin.

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

## Isolation and conversion

Unsafe or heavy decoders use the typed isolated-worker protocol and
descriptor-bound reads. Workers receive only declared inputs and budgets.
Conversions are cached for the server session by input digest and provider
version.

Nsight Systems prefers `parquetdir` export. Flameox does not read or create a
Nsight SQLite export. Memray aggregation streams bounded records or uses
session-local in-memory DuckDB; it does not create a temporary SQLite database.

## Provider setup boundary

An explicit `flameox setup --provider ...` invocation replaces the persistent
uv tool environment with the complete selected Python extra set. System and
vendor packages remain externally installed. Setup creates no project state and
is not a prerequisite for explicit-path analysis.
