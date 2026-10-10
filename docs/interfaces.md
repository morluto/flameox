# Interfaces

CLI and MCP are thin transports over `AnalysisRuntime`. They do not own storage,
provider behavior, or lifecycle state.

## MCP catalog

The low-level Python MCP SDK server owns protocol initialization, framing, transports, progress,
and MCP types. A declarative Flameox tool registry projects strict input and output schemas and
dispatches thin handlers over the shared `AnalysisRuntime`; Flameox does not implement a custom MCP
protocol. The catalog contains 26 named analysis tools, 20 named capture tools, and five lifecycle
tools. Analysis tools are read-only and idempotent. Capture tools execute a typed argv and are
annotated as effects; they never masquerade as reads. Lifecycle tools prepare providers or manage
immutable evidence.

Analysis tool names, in capability-registry order, are:

```text
summarize_trace                 inspect_trace_call_graph
summarize_pytorch_trace         summarize_trace_operations
summarize_trace_lifecycle       inspect_trace_window
rank_cpu_hotspots               inspect_cpu_callers
rank_allocation_hotspots        rank_retained_memory
summarize_benchmarks            analyze_benchmark_scaling
compare_benchmarks              summarize_inference
compare_inference               inspect_gpu_launches
inspect_gpu_kernel_metrics      inspect_triton_autotune
inspect_sanitizer_failures      inspect_kernel_validation
compare_kernel_validation       summarize_failures
inspect_pytest_fixtures         summarize_coverage
inspect_performance_candidates  preview_artifact
```

Capture tools exist for capabilities with a compatible provider:

```text
capture_trace_summary           capture_trace_call_graph
capture_trace_pytorch           capture_trace_operations
capture_trace_lifecycle         capture_trace_window
capture_cpu_hotspots            capture_cpu_callers
capture_memory_hotspots         capture_memory_retained
capture_benchmark_summary       capture_benchmark_scaling
capture_gpu_launches            capture_gpu_kernel_metrics
capture_triton_autotune         capture_sanitizer_failures
capture_failures_summary        capture_pytest_fixtures
capture_coverage_summary        capture_artifact_preview
```

The five lifecycle tools are `prepare_providers`, `preserve_evidence`, `rescue_evidence`,
`query_evidence`, and `inspect_evidence`. There are no gateway tools, opaque capability selectors,
or compatibility aliases.
`flameox mcp inspect` lists compact names and annotations; `--capability CAPABILITY_ID` returns
that capability's direct analysis/capture examples, capability-field schema, and compatible
provider-field schemas. `--tool TOOL_NAME` shows one exact MCP input/output schema, and `--full`
shows the complete catalog. CLI results omit process-local `analysis_id`
because it cannot survive command exit. CLI inspection reports bounded tool/capability choices;
an unsupported declared artifact format returns the capability's accepted formats before provider
decoding.

`rescue_evidence` accepts one live session analysis and an agent-selected explicit absolute path
below an existing parent to a distinct new directory. It stages the normal immutable evidence
format beside that destination, publishes it with a filesystem rename, and returns the
`FLAMEOX_DATA_DIR` restart/reconnect handoff. It does not repair or modify the configured repository,
change the active runtime store, release the session handle, or make the alternate store available to
the active server. Repeating the same rescue request during the live session validates the published
evidence and returns the original handoff. Publication follows ordinary
local filesystem path semantics; Flameox does not impose a workspace or repository-root policy on
the selected parent.

The one-shot CLI mirrors this lifecycle with `analyze --rescue-to ABSOLUTE_PATH` and
`capture --rescue-to ABSOLUTE_PATH`. It requires a destination that does not exist and preflights
its parent before decoding or executing the workload, then returns the same rescue handoff in
`rescued`. The `--preserve` and `--rescue-to` options are mutually exclusive so the publication
destination is unambiguous.

Each analysis tool exposes `sources`, its capability-specific typed fields, optional `continuation`,
`limits`, and `page_size` at the top level. Each capture tool exposes `target`, a discriminated `provider`
object with `kind` and that collector's typed fields, capability-specific fields, `preserve`,
`limits`, and `page_size`; only capabilities with multi-source analysis expose `experiment`. No request wrapper,
opaque `options` bag, or capability selector appears in MCP arguments. Strict validation applies
source cardinality, format compatibility, provider compatibility, and experiment support against
the domain registries before runtime execution. Invalid combinations use Flameox's structured
failure contract, not raw Pydantic diagnostics. Capability-specific schemas are part of
`tools/list`, with optional CLI discovery for compact views.

Trace-window bounds accept exact integers or canonical decimal strings from zero through
`9223372036854775807` nanoseconds; the exclusive end must be positive and greater than the start.
Epoch timestamps beyond JSON's interoperable integer range are preserved and returned in replay
arguments as exact decimal strings. Native readers receive integer bounds without rounding.

Both tool families accept typed `limits` that can lower the server's input-byte, traversal, worker,
process-output, memory, and provenance ceilings. Omitted limits inherit server policy. `page_size`
is shorthand for `limits.max_rows`; when both are supplied, they must agree. Continuation handoffs
carry the effective limits so exact replay retains the original bounds. Field descriptions are
part of the public MCP contract. Shared source, target, provider,
and experiment descriptions are declared on their owning Pydantic models so CLI validation,
runtime validation, and every generated capability tool use the same semantics. Transport-only
fields such as continuations and preservation handles are described at the MCP boundary.

Capture tools execute the target once by default. When present, `experiment` contains cases, blocks,
seed, metric, estimand, threshold, and an optional oracle. Experiment-capable requests also accept
null. There is no separate execution-mode field. Experiment behavior and interpretation are defined
in [investigations and evidence quality](investigations.md).

Every tool returns its complete bounded JSON result inline in a text block and keeps the same
result in `structuredContent`. Content-only clients receive evidence rows, execution provenance,
coverage, limitations, and exact continuation or recovery actions directly, without reading a
separate manifest first. Preserved results return an evidence ID for later inspection or replay.

Failures likewise return their complete structured details inline and in `structuredContent`.
Capture-complete failures are ordinary typed product states, not error-side-channel data. A
`partial` result keeps its analysis handle, typed capture executions, aggregate and workload
status, analysis failure, and executable recovery action at the top level. Already-preserved
captures inspect their existing evidence directly; unpreserved captures offer preservation first.
Recovery offers saved-input reanalysis only when native analysis inputs were retained. A failure's
`details.analysis_source_count` distinguishes replayable inputs from diagnostics-only captures,
which require fixing the collector before a new capture.
Workload failures remain observed evidence. `retryable` and `unavailable` are also composable
non-error states;
invalid requests, terminal infrastructure failures, and failures with no trustworthy result set
`isError=true`. Pages are selected by row count; complete rows, execution provenance, and
continuation arguments are returned without a response-byte ceiling or a second compaction pass.

On the 2026 protocol, SDK cache hints mark the static tool catalog as public for one hour.
Older negotiated protocol revisions omit these fields.

For analysis and capture results, `next_page` contains the exact named analysis tool and complete
arguments for the next call. It includes ordered live path sources or preserved evidence sources,
capability-specific fields, page size, and continuation. Callers do not reconstruct state from
prose. Capture continuation always names the corresponding analysis tool and reads captured native
artifacts; it never reruns the workload. `query_evidence` pagination instead names `query_evidence`
and returns its exact next query arguments. Preserving a live paginated analysis may release its
scratch paths, so `preserve_evidence` returns a refreshed evidence-backed `next_page` that
supersedes the earlier live-path handoff. `rescue_evidence` does the same for the alternate store
that becomes active after reconnecting.

For example, a bounded artifact preview uses a named tool and flattened arguments:

```json
{
  "sources": [
    {"kind": "path", "path": "/absolute/path/to/output.log", "format": "text"}
  ],
  "text_fragment_chars": 1024,
  "page_size": 100
}
```

If the response is partial, do not copy its continuation token into a newly assembled request.
Submit `next_page.tool` with `next_page.arguments` unchanged. This preserves source order, typed
analysis fields, identity checks, and the original page size.

Each capability declaration also owns its accepted source cardinality. The named tool's `sources`
schema carries that exact range before resolving paths or starting capture. Single-artifact
summaries require exactly one source, comparison operations require at least two, and only
intentional aggregations accept a larger bounded collection.

Capability-specific fields appear directly at the tool's top level, defaulting according to the
shared capability model. A capability may require fields such as the start and end bounds for
`trace.window`; transport validation returns their field paths and accepted values where
applicable. Unknown fields are rejected, and pstats CPU metrics use a closed vocabulary in the
tool schema. Its path-source `format` field enumerates the capability's accepted formats.
An incompatible declared format returns a typed validation failure before path resolution or
provider decoding. When omitted, the runtime detects the format where it is unambiguous.

For example, a single Nsight Compute capture for kernel metrics has this argument shape:

```json
{
  "target": {
    "argv": ["python", "kernel.py"],
    "cwd": "/absolute/path/to/project"
  },
  "provider": {"kind": "nsight-compute", "launch_count": 1},
  "preserve": true
}
```

Analysis and capture remain separate tools even when they return the same evidence envelope. MCP
annotations describe a whole tool, so combining read-only artifact analysis and target execution
behind a mode flag would conceal a material effect change. Provider choice stays inside each capture
tool because it is a typed implementation choice for one evidence question. Its schema enumerates
compatible provider kinds and their exact fields; admission rejects incompatible
capability/provider pairs before execution.

`inspect_evidence` accepts an exact `evidence_id` and returns a validated, redacted projection of
its canonical manifest inline. It includes ordered `analysis_sources`, logical source selectors,
artifact identities, coverage, and capture/analysis status. Pass `analysis_sources` unchanged to a
named analysis tool to replay native artifacts. The projection omits argv, environment values,
working directories, and host paths. The local CLI `evidence show` command is the explicit
full-provenance view. Missing or corrupt evidence uses the ordinary structured tool-failure contract.

MCP exposes no resources, resource templates, evidence URIs, or resource-link content blocks.
Clients that previously read an evidence URI should call `inspect_evidence` with its evidence ID.
Preservation, rescue, and inventory results now return IDs without URI fields.

Every tool advertises a compact output schema for its stable result envelope. Provider-specific
metrics and rows remain open JSON values. Success uses structured content directly, without an
`ok/result/error` wrapper. Tool failures set `isError=true` and carry a stable code, message,
retryability, optional field path and accepted values, and a typed next action. The low-level
adapter validates requests itself so malformed arguments also use this structured result contract.

Failure messages and details are protocol data. MCP handlers project typed `RuntimeFailure` values
but never serialize arbitrary exception text: filesystem exceptions may contain host paths, and
dependency exceptions may contain argv or environment-derived values. Unexpected failures use a
stable operation-specific summary; cancellation remains a control path and is re-raised.

`prepare_providers` and capture tools are open-world. Preparation uses request-owned bounded
subprocesses and an overall deadline; cancellation settles the installer and its descendants.
py-spy is prepared as a pinned standalone collector and becomes available in the same live session.
Bindings are activated only after the complete request succeeds, including any server preparation;
failure, timeout, or cancellation leaves prior session bindings unchanged. Preparation forwards
the safe uv controls `UV_OFFLINE`, `UV_CACHE_DIR`, `UV_PYTHON_DOWNLOADS`, and `UV_NO_CONFIG`.
Repeating preparation reuses its verified binding without installation. The returned launcher still
names the complete requested server provider set; preparing another provider does not remove
existing session bindings. No durable provider inventory, project state, or job is created.

Server-import requirements are compared with the active release and installed dependency versions.
`activation_status` is `ready`, `restart_required`, `unknown`, or `not_applicable`. `next_action` is
null for ready or host-only requests. Otherwise `reconnect_mcp` explicitly says whether reconnection
is required or conditional and tells the caller to preserve needed session analyses first. An
unknown identity does not establish a required restart. A prepared server environment is verified
with its version command; preparation does not replace active imports.

Managed IDs remain `aiperf`, `memray`, `otlp`, `perfetto`, `py-spy`, and `torch`. External host tools,
drivers, permissions, and workload-interpreter requirements are reported separately and remain
unverified by preparation. Both structured results and inline text carry those handoffs. No
system package manager or privilege elevation is invoked. Perfetto still requires an externally
installed Trace Processor. The deadline defaults to 1,800 seconds and accepts 1 through 3,600;
MCP preparation failures use bounded path-free diagnostics, while CLI setup retains local stderr.

## Sources and limits

The strict source union is:

```text
PathSource     {kind: "path", path, format?, producer?, expected_sha256?}
EvidenceSource {kind: "evidence", evidence_id, artifact_role? OR artifact_selector?}
```

Continuations are opaque integrity cursors bound to the request and exact input
digests. They contain no authority, credentials, or artifact data and are not
an authentication boundary: a caller already authorized to submit the analysis
can choose which of its rows to request. They can cross process boundaries when their immutable
inputs remain available, so an analysis of explicit paths can resume in a later CLI invocation.
`flameox analyze --evidence EVIDENCE_ID` loads a preserved record's ordered analysis sources
directly. Tokens bind ordered content digests, formats,
producer identities, arguments, and limits, independently of storage paths and publication roles.
Paginated CLI analysis returns an executable `next_page.argv` for its explicit paths or evidence
record. After preserving a CLI capture, execute the same field. A rescued handoff also includes
the exact `next_page.environment` needed to open its alternate evidence store. An unpreserved CLI capture
sets continuation to null and reports the exact preservation or rescue rerun because its scratch is
released at exit. Scratch can be released immediately after preserved evidence is available.
A changed input cannot reuse a continuation. Tokens issued by older path-bound implementations
must be restarted with a fresh analysis. Preview `offset` counts logical rows: text lines, JSONL
records, CSV data records, Parquet records, and projected JSON entries.

For oversized text lines, `artifact.preview` accepts the top-level
`text_fragment_chars` field (1–4,096 decoded characters per fragment). It requires text sources,
counts fragments instead of lines, and is bound into continuation identity. Start a fresh page when
switching modes. Fragment rows preserve line and fragment positions; decoding replaces invalid
UTF-8, so fragments are not byte-exact slices. Each row has one-based `line`, zero-based `fragment`,
`text`, and `line_terminated`, which is true only when that fragment ends in LF. LF and CR are
retained; a final line without LF remains unterminated even at end-of-file. Original native bytes
remain unchanged. The reader uses bounded line reads.

JSON preview traverses the document once in document order. A root array yields its elements;
a root scalar yields one value row. At the root object, arrays yield section rows, scalar fields
yield key/value rows, and nested objects yield key/type summaries. Object keys are literal strings,
so a key containing a dot is not confused with a nested path. Pagination can stop before the end
of the document; only a complete preview has validated JSON through end-of-file.

Decoded offsets must be integers within the available bounded population. Negative offsets and
offsets at or beyond the end fail with `INVALID_INPUT`; they never use Python slicing semantics or
produce empty complete evidence.

Projection providers may expose a bounded prefix when their native reader cannot
resume safely. Such results keep `coverage.complete=false`, identify
`truncation.reason=provider_limit`, and do not emit a continuation after the last
retrievable row. A continuation therefore always names a consumable next page;
it never promises access beyond a provider's declared projection bound. Callers can narrow the
semantic query or reduce a recapture in that terminal case; preservation cannot recover rows the
provider never returned.

CLI capture takes the target as a positional argv after `--`; unknown Flameox options before
that separator are rejected. Paths are resolved and validated by the runtime, so missing or
cyclic paths and malformed preview artifacts produce typed failures. `--format` applies only to
path inputs; combining it with `--evidence` is rejected because preserved sources retain their format.

Requests may lower analysis row limits, decoder timeout and RSS limits, process-output limits, and
durable-provenance limits. Durable provenance includes captured argv and execution metadata.
Requests cannot raise server limits.

`query_evidence` returns 1-200 manifests per page. Its cursor is bound to both the immutable
inventory snapshot and the original filters; callers resume by repeating those filters unchanged.
Creation bounds accept timezone-aware RFC3339 strings, and the upper bound must not precede the
lower bound. Advertised JSON Schema types are enforced before execution, including rejection of
numeric strings and booleans for integer fields, except the explicit decimal-string form of
trace-window bounds. Integral JSON numbers such as `2.0` remain valid.

## Capture

Capture console retention and preservation semantics are defined in
[storage and evidence](storage-and-evidence.md). MCP and CLI expose the same
`target.console_output` choice (`diagnostics` by default or `full`).

A direct target contains an argv array, an existing absolute cwd, and at most 32 bounded environment
overrides after experiment-case overrides are merged. Provider fields appear directly beside
`provider.kind` inside that capture tool's discriminated provider object; capability fields appear
directly at the tool's top level. The runtime models remain the validation authority, and admission
validates those projected typed fields before execution. Shell command strings are not accepted.

`target.budget` controls workload execution independently of analysis limits:

```json
{"budget": {"timeout_seconds": 600, "max_memory_bytes": 8589934592}}
```

Both fields default to null. The workload budget's scope, observations, and limits are described in
[runtime safety](runtime-safety.md).

Provider output formats are compared with the requested capability before scratch allocation or
execution. Statically incompatible pairs fail with the declared formats and compatible capture
providers. Capture then binds every real invocation, resolves the cwd and executables, and validates
aggregate scratch and durable provenance capacity before allocating one request-owned scratch
directory or starting a workload. There is no separate plan or preflight authority: the operation
validates and executes the same bound invocations.

Callers may request preservation as part of capture. Once native collection succeeds, requested
preservation publishes the native artifacts even if immediate analysis fails; the result then
reports separate capture execution state and a typed `analysis_failure`. An analysis failure is
never converted into empty successful evidence.

An `experiment` runs 2–16 cases across 1–100 blocks. Flameox evaluates
`wall_time_ns` with a paired `median_difference` or `mean_difference`. The result's estimate and
classification are descriptive; experiment limits and interpretation are defined in
[investigations and evidence quality](investigations.md). Work remains owned by this request and
supports progress and cancellation.

Capture `outcome` is computed from every execution and retains exact success/failure counts.
MCP error classification consumes that outcome, and every execution record remains available in
the structured result. Each execution identifies whether `returncode` belongs to the workload or
collector, retains separate collector and workload executable SHA-256 identities, and leaves
`workload_returncode` null for wrapped captures. The compatibility `executable_sha256` field identifies
the invoked collector. Exit ownership is declared by each invocation builder: self-reporting workloads retain
their observed exit even when they use a provider other than `direct`. A usable profile does not
prove workload success. When retained, preserved stdout, stderr, and profiles
are individually selectable from `inspect_evidence`.
The first declared case is the baseline. A case inherits the target argv when it omits `argv`, and
its environment overrides the target environment. Each block randomizes case order from the
declared seed. The semantic oracle runs after every successful capture in that case environment;
`FLAMEOX_CAPTURE_STDOUT` and `FLAMEOX_CAPTURE_STDERR` identify the invoked capture process's
console files. Wrapping collectors such as pyperf may emit summary-only console output. Use
workload-owned result files or another independent semantic check when that collector suppresses
workload output. A nonzero exit excludes the corresponding case-block observation from paired
comparison.

Comparison capabilities consume explicit artifacts; they do not capture their inputs. A caller captures
representative baseline and candidate summaries separately, preserves them when durable provenance
is needed, and supplies at least two sources with capability `benchmark.compare`,
`inference.compare`, or `kernel.compare`. Capture admission rejects these capabilities: experiment
capture reports the declared cases' effect but does not create the
case-grouped native inputs required by artifact comparison.

## Stable failure codes

The transport distinguishes invalid input, unavailable providers,
missing or changed input, unsupported format, decode failure, execution failure,
cancellation, limit exceeded, expired session analysis, missing evidence,
repository I/O failure, repository corruption, and unsupported repository
format. An unavailable managed provider identifies `prepare_providers` and the exact provider list
needed for a retry; an unavailable system provider returns external setup guidance.

## CLI

The command surface is:

```text
flameox setup
flameox mcp serve [--limits JSON]
flameox mcp inspect
flameox analyze CAPABILITY_ID [PATH...] [OPTIONS]
flameox capture [OPTIONS] -- <argv...>
flameox evidence query|show|location
```

Analysis options include `--evidence`, `--arguments`, `--continuation`, `--limits`, and either
`--preserve` or `--rescue-to`. Capture requires `--provider` and accepts `--capability`, `--cwd`,
`--capture-arguments`, `--analysis-arguments`, `--console-output`, `--workload-budget`,
`--experiment`, `--limits`, and either `--preserve` or `--rescue-to`.

`--limits` validates the existing `RequestLimits` JSON contract and sets startup
bounds for analysis and storage in that invocation. Its `timeout_seconds` and
`max_memory_bytes` protect conversions and analysis workers, not capture targets.
For example, `flameox capture --workload-budget
'{"max_memory_bytes":8589934592}' ...` requests an 8 GiB workload process-tree
budget without changing decoder protection. Unspecified analysis limits retain
their defaults and hard contract
maxima still apply. MCP tools accept `limits` to lower these ceilings and `page_size` to lower the
row bound. Raising a server ceiling requires restart or reconnect with new `--limits`.
No workspace configuration is created.
Paginated `flameox evidence query` output also carries an executable `next_page.argv` with the
original filters, page size, and snapshot-bound cursor.

`setup` detects supported coding agents and uses one multi-select prompt to choose which global MCP
client configurations to update. It preserves unrelated JSON or TOML content and writes stdio
configuration that launches the exact running Flameox release through `uvx` on Python 3.12.
Changed clients must restart or reconnect. Non-interactive setup requires explicit `--client`
targets or `--all`; `--yes` never converts detection into consent, and `--dry-run` reports the exact
paths and actions without mutation. Repeated `--provider` options declare the complete Python
provider set for the exact version-pinned uvx environment used by the saved launcher.
OpenCode `opencode.jsonc` files retain their comments and unrelated settings while setup creates or
updates the `mcp.flameox` entry. Ambiguous duplicate JSONC keys and configuration nesting beyond
parser limits are rejected before provider preparation or writes.
`--timeout-seconds` accepts 1 through 3,600 and defaults to 1,800. Resolver,
download, and compatibility failures retain uvx's complete stderr. System and
vendor providers receive external installation guidance. Setup does not create a persistent global
tool, durable operation, project state, or MCP setup endpoint. Other CLI commands use the same
explicit paths, capture working directories, and user-level evidence store as MCP.
When another `flameox` executable on `PATH` reports a different version, setup emits a non-fatal
advisory in human and JSON output. It never upgrades or removes that independently managed CLI.
