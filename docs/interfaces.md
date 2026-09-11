# Interfaces

CLI and MCP are thin transports over `AnalysisRuntime`. They do not own storage,
provider behavior, or lifecycle state.

## MCP catalog

The catalog exposes task-shaped tools for client-side tool search. Flameox does not add a second
search/inspect protocol in front of its operations. A caller that knows the evidence question can
invoke its tool directly; an unfamiliar caller relies on the MCP client's ordinary tool search and
then receives the selected tool's complete schema.

There are exactly 50 tools:

| Group | Count | Examples | Effect |
| --- | ---: | --- | --- |
| Existing-artifact analysis | 26 | `analyze_cpu_hotspots`, `analyze_cpu_callers`, `analyze_gpu_launches`, `analyze_benchmark_compare`, `analyze_pytest_fixtures`, `preview_artifact` | Read-only and idempotent. |
| Capture and immediate analysis | 20 | `capture_cpu_hotspots`, `capture_cpu_callers`, `capture_triton_autotune`, `capture_gpu_launches`, `capture_benchmark_summary`, `capture_pytest_fixtures`, `capture_process_output` | Executes typed argv; not read-only or idempotent. |
| Evidence lifecycle | 4 | `prepare_providers`, `preserve_evidence`, `rescue_evidence`, `query_evidence` | Prepare an explicit uvx environment or manage immutable evidence. |

For CLI-side discovery, `flameox mcp inspect --summary` returns compact records containing each
tool's name, description, required top-level inputs, and effect class. After selecting a tool,
`flameox mcp inspect --tool TOOL_NAME` returns its complete input and output schemas. The unfiltered
command remains the exact complete catalog. Unknown capability, capture-provider, and tool names
return the requested value, bounded valid choices, and the exact discovery command to run next.
An unsupported artifact format similarly returns the detected or declared format, the capability's
accepted formats, and its exact analysis-tool name before provider decoding begins.

`rescue_evidence` accepts one live session analysis and an explicit absolute path below an existing,
symlink-free parent to a distinct new directory. It anchors publication to an open parent
directory, publishes the normal immutable evidence format there, and returns the
`FLAMEOX_DATA_DIR` restart/reconnect handoff. It does not repair or modify the configured repository,
change the active runtime store, release the session handle, or expose the alternate store through
the active server's resource template. Repeating the same rescue request during the live session
validates the published evidence and returns the original handoff. Secure descriptor-anchored
rescue publication currently requires a POSIX host with `/proc/self/fd` or `/dev/fd`; Windows
returns `UNAVAILABLE_CAPABILITY` before analysis or workload execution.

The one-shot CLI mirrors this lifecycle with `analyze --rescue-to ABSOLUTE_PATH` and
`capture --rescue-to ABSOLUTE_PATH`. It requires a destination that does not exist and preflights
its parent before decoding or executing the workload, then returns the same rescue handoff in
`rescued`. The `--preserve` and `--rescue-to` options are mutually exclusive so the publication
destination is unambiguous.

The capability registry generates the analysis and capture tools through the Python MCP SDK 2.0
registration API. The SDK derives each top-level input schema directly from the registered callable.
A generated analysis tool has `sources`, capability-specific typed `options`, optional lowered
`limits`, and an optional `continuation`. A generated capture tool has `target`, a discriminated
`provider` union containing only compatible capture providers, capability-specific typed `options`,
an explicit execution model, optional lowered `limits`, and optional `preserve`. Capabilities that
can analyze the multiple artifacts produced by paired cases expose the `single` or `experiment`
union; single-artifact analyses expose only `single`. There is no extra request envelope and there
are no free-form provider or analysis argument objects.

Field descriptions are part of the public MCP contract. Shared source, target, limit, provider,
and experiment descriptions are declared on their owning Pydantic models so CLI validation,
runtime validation, and every generated capability tool use the same semantics. Transport-only
fields such as continuations and preservation handles are described at the MCP boundary.

Successful calls keep the complete validated result in `structuredContent`. Their text block is a
short compatibility summary with the capability, completion or truncation state, session handle,
and next action; it does not serialize the evidence tables a second time. Content-only clients can
still identify the outcome and recovery path, while structured clients retain the authoritative
bounded evidence. Preserved results also return a resource link.

For a paginated MCP capture result, `continuation_sources` contains ready-to-submit ordered path
sources while session scratch remains live, or evidence sources when the capture was preserved.
The summary names the matching analysis tool and directs the caller to reuse those sources, the
original options and limits, and the returned continuation without rerunning the workload. This
handoff keeps continuation work read-only and preserves the captured workload identity.

Each capability declaration also owns its accepted source cardinality. MCP encodes that range in
the generated `sources` schema, and the runtime checks the same range before resolving paths or
starting capture. Single-artifact summaries require exactly one source, comparison operations
require at least two, and only intentional aggregations accept a larger bounded collection.

`options` is optional when the capability model can be constructed entirely from documented
defaults, including operations whose option model is empty. It remains required when omission
would leave the request incomplete, such as the start and end bounds for `trace.window`. Unknown
fields are rejected rather than ignored, and pstats CPU metrics use a closed vocabulary in the
generated schema.

For example, a single Nsight Compute capture for kernel metrics has this argument shape:

```json
{
  "target": {
    "argv": ["python", "kernel.py"],
    "cwd": "/absolute/path/to/project"
  },
  "provider": {"kind": "nsight-compute", "options": {"launch_count": 1}},
  "options": {},
  "execution": {"kind": "single"},
  "preserve": true
}
```

Analysis and capture remain separate tools even when they return the same evidence envelope. MCP
annotations describe a whole tool, so combining read-only artifact analysis and target execution
behind a mode flag would conceal a material effect change. Provider choice stays inside a capture
tool because it is a typed implementation choice for one evidence question; incompatible providers
cannot be represented by that tool's schema.

There is one resource template:

```text
flameox://evidence/{evidence_id}
```

`resources/list` is empty. `resources/read` returns a redacted, digest-bound projection of the
canonical manifest with its own versioned media type. It omits argv, environment values, working
directories, and host paths. The local CLI `evidence show` command is the explicit full-provenance
view. A missing resource is a protocol error.

Every tool advertises a compact output schema for its stable result envelope. Provider-specific
metrics and rows remain open JSON values. Success uses structured content directly, without an
`ok/result/error` wrapper. Tool failures set `isError=true` and carry a stable code, message, and
details. MCP SDK argument-validation errors occur before tool execution and therefore use the
protocol error shape rather than the tool's output schema.

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
unverified by preparation. Both structured results and text summaries carry those handoffs. No
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
After preserving a CLI capture, use its structured `continuation_handoff` with `--evidence` and
repeat the original options, limits, and returned continuation. An unpreserved one-shot CLI capture
sets continuation to null and reports the exact preservation or rescue rerun because its scratch is
released at exit. Scratch can be released immediately after preserved evidence is available.
A changed input cannot reuse a continuation. Tokens issued by older path-bound implementations
must be restarted with a fresh analysis. Preview `offset` counts logical rows: text lines, JSONL
records, CSV data records, Parquet records, and projected JSON entries.

For oversized text lines, `preview_artifact` also accepts
`options: {"text_fragment_chars": 1024}` (1–4,096 decoded characters per fragment).
This opt-in mode requires text files and counts fragment rows instead of lines;
start a fresh page when switching modes. Rows carry one-based `line`, zero-based
`fragment` within that line, `text`, and `line_terminated` (true only for a fragment
ending in LF). Text retains LF and CR characters. An unterminated final line stays
`line_terminated=false` even when coverage is complete. UTF-8 decoding replaces
invalid byte sequences; fragments and offsets are not byte-exact slices. Original
native bytes remain unchanged and can be preserved with the analysis handle.
Continuations bind fragment size as well as the native source identity. Lower
fragment sizes can accommodate tighter result-byte limits. The default remains
whole-line preview with its existing offsets.

The fragment reader uses Python's bounded
[text `readline(size)`](https://docs.python.org/3.12/library/io.html#io.TextIOBase.readline)
and explicit LF newline handling; it does not accumulate a whole oversized line.

JSON preview traverses the document once in document order. A root array yields its elements;
a root scalar yields one value row. At the root object, arrays yield section rows, scalar fields
yield key/value rows, and nested objects yield key/type summaries. Object keys are literal strings,
so a key containing a dot is not confused with a nested path. Pagination can stop before the end
of the document; only a complete preview has validated JSON through end-of-file.

Decoded offsets must be integers within the available bounded population. Negative offsets and
offsets at or beyond the end fail with `INVALID_INPUT`; they never use Python slicing semantics or
produce empty complete evidence. Continuation tests cover wrong-request, changed-input, negative,
non-integer, and beyond-end cases.

Projection providers may expose a bounded prefix when their native reader cannot
resume safely. Such results keep `coverage.complete=false`, identify
`truncation.reason=provider_limit`, and do not emit a continuation after the last
retrievable row. A continuation therefore always names a consumable next page;
it never promises access beyond a provider's declared projection bound. MCP summaries direct that
terminal case toward a narrower semantic query or a reduced recapture; preservation cannot recover
rows the provider never returned.

Requests may lower startup row, result-byte, decoder timeout/RSS, output-byte, and durable
provenance-byte limits. Durable provenance bounds the captured argv and execution
metadata retained for explicit preservation.
They cannot raise them.

`query_evidence` returns 1-200 manifests per page. Its cursor is bound to both the immutable
inventory snapshot and the original filters; callers resume by repeating those filters unchanged.

## Capture

Capture retains bounded console diagnostics by default, with
explicit omission counts. Full console retention is reserved for process-output
evidence, semantic-oracle inputs, or an explicit caller request; only that mode
requires disk backing. Requesting preservation makes selected evidence durable
but does not silently request full logs. Set `target.console_output` to `full`
(CLI: `--console-output full`) for explicit retention; the default is `diagnostics`.
An oracle's own output remains diagnostic unless `full` is explicitly selected.
Diagnostics retain at most 4,096 bytes per stream, lowered by the provenance
budget, with UTF-8 replacement decoding. `console_diagnostics` reports observed,
retained, and omitted byte counts and stream completeness; full-output metadata
is reported under `output_streams`. Omitted bytes cannot be recovered later. See
[workload resources and evidence bounds](workload-resource-policy.md).

A direct target contains an argv array, an existing absolute cwd, and at most 32 bounded environment
overrides after experiment-case overrides are merged. Provider fields live in the capture tool's
typed provider union, and analysis fields live in its capability-specific `options` model. Shell
command strings are not accepted.

`target.budget` controls workload execution independently of analysis limits:

```json
{"budget": {"timeout_seconds": 600, "max_memory_bytes": 8589934592}}
```

Both fields default to null: no Flameox workload deadline or RSS cap. An explicit
time budget applies separately to each capture invocation and each semantic
oracle; it is not an experiment-wide deadline. RSS covers the sampled process
tree, including a collector, and is best-effort rather than an OS quota. Timeout
and memory termination report the configured workload budget, including oracle
failures. Request cancellation and descendant cleanup remain active without a
budget. Client timeouts, native-tool limits, and operating-system limits still
apply. Decoder limits are not workload ceilings and cannot be raised through
`target.budget`.

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

Experiment mode adds 2-16 cases, 1-100 blocks, a seed, metric, estimand,
practical threshold, and optional semantic-oracle argv. Version 0.2 evaluates
`wall_time_ns` with a paired `median_difference` or `mean_difference`, reports
eligible blocks and a deterministic percentile interval when at least three
blocks survive capture/oracle validation, and classifies the effect against the
declared threshold. Work is not detached; the request receives progress and owns
cancellation.
The experiment's `point_estimate_classification` is descriptive; its `decision_basis` is explicit
on the metrics block. It does not claim confidence-qualified improvement or equivalence.

Capture `outcome` is computed from every execution before diagnostic compaction and retains exact
success/failure counts. MCP error classification consumes that outcome even when no execution
diagnostics fit inline. Each execution identifies whether `returncode` belongs to the workload or
collector, retains separate collector and workload executable SHA-256 identities, and leaves
`workload_returncode` null for wrapped captures. The compatibility `executable_sha256` field identifies
the invoked collector. Exit ownership is declared by each invocation builder: self-reporting workloads retain
their observed exit even when they use a provider other than `direct`. A usable profile does not
prove workload success. When retained, preserved stdout, stderr, and profiles
are individually selectable from the evidence resource.
The first declared case is the baseline. A case inherits the target argv when it omits `argv`, and
its environment overrides the target environment. Each block randomizes case order from the
declared seed. The semantic oracle runs after every successful capture in that case environment;
`FLAMEOX_CAPTURE_STDOUT` and `FLAMEOX_CAPTURE_STDERR` identify its captured files, and a nonzero
exit excludes the corresponding case-block observation from paired comparison.

Comparison tools consume explicit artifacts; they do not capture their inputs. A caller captures
representative baseline and candidate summaries separately, preserves them when durable provenance
is needed, and supplies at least two sources to `analyze_benchmark_compare`,
`analyze_inference_compare`, or `analyze_kernel_compare`. Flameox does not advertise
`capture_*_compare`: experiment capture reports the declared cases' effect but does not create the
case-grouped native inputs required by artifact comparison.

## Stable failure codes

The transport distinguishes invalid input, unavailable providers,
missing or changed input, unsupported format, decode failure, execution failure,
cancellation, limit exceeded, expired session analysis, missing evidence,
repository I/O failure, repository corruption, and unsupported repository
format. An unavailable managed provider identifies `prepare_providers` and the exact provider list
needed for a retry; an unavailable system provider returns external setup guidance.

## CLI

The retained surface is:

```text
flameox setup
flameox mcp serve [--limits JSON]
flameox mcp inspect
flameox analyze [--limits JSON] [--continuation TOKEN] [--preserve]
flameox capture [--limits JSON] [--workload-budget JSON] [--experiment JSON] [--preserve] -- <argv...>
flameox evidence query|show|location
```

`--limits` validates the existing `RequestLimits` JSON contract and sets startup
bounds for analysis and storage in that invocation. Its `timeout_seconds` and
`max_memory_bytes` protect conversions and analysis workers, not capture targets.
For example, `flameox capture --workload-budget
'{"max_memory_bytes":8589934592}' ...` requests an 8 GiB workload process-tree
budget without changing decoder protection. Unspecified analysis limits retain
their defaults and hard contract
maxima still apply. MCP request limits may only lower these startup bounds; no
tool can raise them or reconfigure the server. No workspace configuration is created.

`setup` detects supported coding agents and uses one multi-select prompt to choose which global MCP
client configurations to update. It preserves unrelated JSON or TOML content and writes stdio
configuration that launches the exact running Flameox release through `uvx` on Python 3.12.
Changed clients must restart or reconnect. Non-interactive setup requires explicit `--client`
targets or `--all`; `--yes` never converts detection into consent, and `--dry-run` reports the exact
paths and actions without mutation. Repeated `--provider` options declare the complete Python
provider set for the exact version-pinned uvx environment used by the saved launcher.
OpenCode `opencode.jsonc` files retain their comments and unrelated settings while setup creates or
updates the `mcp.flameox` entry.
`--timeout-seconds` accepts 1 through 3,600 and defaults to 1,800. Resolver,
download, and compatibility failures retain uvx's complete stderr. System and
vendor providers receive external installation guidance. Setup does not create a persistent global
tool, durable operation, project state, or MCP setup endpoint. Other CLI commands use the same
explicit paths, capture working directories, and user-level evidence store as MCP.
When another `flameox` executable on `PATH` reports a different version, setup emits a non-fatal
advisory in human and JSON output. It never upgrades or removes that independently managed CLI.
