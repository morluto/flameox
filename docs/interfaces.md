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

Analysis tool names, in operation-registry order, are:

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

Capture tool names combine `capture_and_` with the analysis task name. Available capture tools are:

```text
capture_and_summarize_trace
capture_and_inspect_trace_call_graph
capture_and_summarize_pytorch_trace
capture_and_summarize_trace_operations
capture_and_summarize_trace_lifecycle
capture_and_inspect_trace_window
capture_and_rank_cpu_hotspots
capture_and_inspect_cpu_callers
capture_and_rank_allocation_hotspots
capture_and_rank_retained_memory
capture_and_summarize_benchmarks
capture_and_analyze_benchmark_scaling
capture_and_inspect_gpu_launches
capture_and_inspect_gpu_kernel_metrics
capture_and_inspect_triton_autotune
capture_and_inspect_sanitizer_failures
capture_and_summarize_failures
capture_and_inspect_pytest_fixtures
capture_and_summarize_coverage
capture_and_preview_artifact
```

The five lifecycle tools are `prepare_providers`, `preserve_evidence`, `rescue_evidence`,
`query_evidence`, and `inspect_evidence`. There are no gateway tools, opaque operation selectors,
or compatibility aliases.
`flameox mcp inspect` lists compact names and annotations; `--tool TOOL_NAME` returns
one exact MCP input/output schema, its operation examples, and compatible provider-field schemas.
`--full`
shows the complete catalog. CLI results omit process-local `analysis_id`
because it cannot survive command exit. CLI inspection reports bounded tool/operation choices;
an unsupported declared artifact format returns the operation's accepted formats before provider
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

Each analysis tool exposes `sources`, its operation-specific typed fields, optional `continuation`,
`limits`, and `page_size` at the top level. Each capture tool exposes `target`, a discriminated `provider`
object with `kind` and that collector's typed fields, operation-specific fields, `preserve`,
`limits`, and `page_size`; only operations with multi-source analysis expose `experiment`. No request wrapper,
opaque `options` bag, or operation selector appears in MCP arguments. Strict validation applies
source cardinality, format compatibility, provider compatibility, and experiment support against
the domain registries before runtime execution. Invalid combinations use Flameox's structured
failure contract, not raw Pydantic diagnostics. Operation-specific schemas are part of
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
runtime validation, and every generated operation tool use the same semantics. Transport-only
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
operation-specific fields, page size, and continuation. Callers do not reconstruct state from
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

Each operation declaration also owns its accepted source cardinality. The named tool's `sources`
schema carries that exact range before resolving paths or starting capture. Single-artifact
summaries require exactly one source, comparison operations require at least two, and only
intentional aggregations accept a larger bounded collection.

Operation-specific fields appear directly at the tool's top level, defaulting according to the
shared operation model. A operation may require fields such as the start and end bounds for
`inspect_trace_window`; transport validation returns their field paths and accepted values where
applicable. Unknown fields are rejected, and pstats CPU metrics use a closed vocabulary in the
tool schema. Its path-source `format` field enumerates the operation's accepted formats.
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
operation/provider pairs before execution.

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
EvidenceSource {kind: "evidence", evidence_id, artifact_selector?}
```

Continuations are opaque integrity cursors bound to the request and exact input
digests. They contain no authority, credentials, or artifact data and are not
an authentication boundary: a caller already authorized to submit the analysis
can choose which of its rows to request. They can cross process boundaries when their immutable
inputs remain available, so an analysis of explicit paths can resume in a later CLI invocation.
`flameox analyze OPERATION --evidence EVIDENCE_ID` loads a preserved record's ordered analysis sources
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

For oversized text lines, `preview_artifact` accepts the top-level
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
Native row fields named `input_sha256`, or `section` in JSON section rows, remain under
`value` when they collide with preview metadata. CSV rejects duplicate headers and ragged
records rather than silently discarding or inventing values. Preview rows use the same canonical
value projection as typed providers: integers outside the interoperable JSON range and nonfinite
native numeric values become strings before inline output or preservation. Native JSON parser
limits still apply; YAJL can replace unpaired escaped surrogates during JSON decoding, while
JSONL rejects invalid Unicode. Native bytes remain authoritative.

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
`provider.kind` inside that capture tool's discriminated provider object; operation fields appear
directly at the tool's top level. The runtime models remain the validation authority, and admission
validates those projected typed fields before execution. Shell command strings are not accepted.

`target.budget` controls workload execution independently of analysis limits:

```json
{"budget": {"timeout_seconds": 600, "max_memory_bytes": 8589934592}}
```

Both fields default to null. The workload budget's scope, observations, and limits are described in
[runtime safety](runtime-safety.md).

Provider output formats are compared with the requested operation before scratch allocation or
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
`workload_returncode` null for wrapped captures. Exit ownership is declared by each invocation
builder: self-reporting workloads retain their observed exit even when they use a provider other
than `direct`. A usable profile does not
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

Comparison operations consume explicit artifacts; they do not capture their inputs. A caller captures
representative baseline and candidate summaries separately, preserves them when durable provenance
is needed, and supplies at least two sources with operation `compare_benchmarks`,
`compare_inference`, or `compare_kernel_validation`. Capture admission rejects these operations: experiment
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
flameox update [--client CLIENT] [--version VERSION] [--check|--dry-run]
flameox mcp serve [--limits JSON]
flameox mcp inspect
flameox analyze OPERATION [PATH...] [OPTIONS]
flameox capture [OPTIONS] -- <argv...>
flameox evidence query|show|location
```

Analysis options include `--evidence`, `--arguments`, `--continuation`, `--limits`, and either
`--preserve` or `--rescue-to`. Capture requires `--provider` and accepts `--operation`, `--cwd`,
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
client configurations to update. Detection labels choices without preselecting them. Interactive
setup shows the resolved paths and launcher and asks before preparing dependencies or writing files;
`--yes` approves explicit selections. Prompts require terminal input, output, and stderr, and JSON
mode stays non-interactive. It preserves unrelated JSON or TOML content and writes stdio
configuration that launches the exact running Flameox release through `uvx` on Python 3.12.
Changed clients must restart or reconnect. Non-interactive setup requires explicit `--client`
targets or `--all`; `--yes` never converts detection into consent, and `--dry-run` reports the exact
paths and actions without mutation. Repeated `--provider` options declare the complete Python
provider set for the exact version-pinned uvx environment used by the saved launcher.
Setup, detection, and update follow nonempty `CLAUDE_CONFIG_DIR`, `CODEX_HOME`, and `GEMINI_CLI_HOME`
overrides. OpenCode follows `OPENCODE_CONFIG_DIR`, then `OPENCODE_CONFIG`, then the global directory
under `XDG_CONFIG_HOME` (default `~/.config`). Within a selected directory, an existing
`opencode.jsonc` wins over `opencode.json`; the global directory also admits `config.json`.
Relative override paths are relative to the setup/update working directory. These commands edit
the selected user configuration; they do not establish whether project or administrator policy
overrides that registration in a running client.
OpenCode JSON/JSONC files retain their comments and unrelated settings while setup creates or
updates the `mcp.flameox` entry. Ambiguous duplicate JSON or JSONC keys and configuration nesting beyond
parser limits are rejected before provider preparation or writes.
`--timeout-seconds` accepts 1 through 3,600 and defaults to 1,800. Resolver,
download, and compatibility failures retain bounded local uvx stderr. Setup verifies the base
release even without optional providers, checking distribution version, extras, and tool catalog
before writing configuration. Setup and update share the broker execution boundary, including
output limits and descendant cleanup. System and
vendor providers receive external installation guidance. Setup does not create a persistent global
tool, durable operation, project state, or MCP setup endpoint. Other CLI commands use the same
explicit paths, capture working directories, and user-level evidence store as MCP.
When another `flameox` executable on `PATH` reports a different version, setup emits a non-fatal
advisory in human and JSON output. It never upgrades or removes that independently managed CLI.

`update` changes existing setup-owned MCP registrations. Without `--client`, it targets every
configured Flameox client; it does not add registrations for merely detected clients. It reads
the selected configurations before checking PyPI for the latest stable release. `--version`
selects an exact published version, including prereleases and intentional rollback. Automatic
selection never downgrades a newer configured release. `--check` and `--dry-run` report paths,
old versions, and planned requirements without preparing environments or writing configuration.
No startup or MCP request performs a background update check.

Before changing any pin, update prepares each selected release and provider-extra environment
through `uvx`, verifies its distribution version and advertised extras, and checks that its CLI
exposes a tool catalog. Preparation uses the subprocess broker's output, deadline, and descendant
cleanup bounds, with one overall
`--timeout-seconds` budget (1–3,600, default 1,800). Client-specific uv, proxy, certificate, home,
and cache environment settings participate in preparation. The latest-version check uses a
separate ten-second network timeout and a 1 MiB metadata bound; an explicit version skips it.
Named-index uv password/token environment variables cannot pass the broker's credential policy;
update rejects them explicitly with manual update guidance before changing pins.

Setup and updated launchers disable implicit uv configuration and source overrides, so project
`uv.toml` files cannot change their package resolution. Custom launcher commands are rejected with
manual setup guidance. Provider extras, client environment, disabled state, server limits, unrelated settings, and TOML/JSONC
comments are retained. Preparation failure leaves all client configurations unchanged; publication
checks for intervening configuration edits and replaces each file atomically. Multiple client
files are not one transaction: a later write failure can leave earlier clients updated.
Changed clients must restart or reconnect; running sessions continue on their current release.

Updates do not mutate an independently installed PATH CLI or the evidence store. Manage a
`uv tool` CLI with `uv tool upgrade flameox`; an exact install constraint must be replaced with
`uv tool install 'flameox==VERSION'` when selecting a different version. Other package-manager
installs remain managed by those package managers. Published uvx environments pin the Flameox
release and requested extras, but do not use the repository's `uv.lock` to pin all transitive
dependencies. Rollback resolves the selected release again and does not promise identical
transitive dependency bytes.
