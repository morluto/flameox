# Interfaces

CLI and MCP are thin transports over `AnalysisRuntime`. They do not own storage,
provider behavior, or lifecycle state.

## MCP catalog

The catalog exposes task-shaped tools for client-side tool search. Flameox does not add a second
search/inspect protocol in front of its operations. A caller that knows the evidence question can
invoke its tool directly; an unfamiliar caller relies on the MCP client's ordinary tool search and
then receives the selected tool's complete schema.

There are exactly 47 tools:

| Group | Count | Examples | Effect |
| --- | ---: | --- | --- |
| Existing-artifact analysis | 26 | `analyze_cpu_hotspots`, `analyze_cpu_callers`, `analyze_gpu_launches`, `analyze_benchmark_compare`, `analyze_pytest_fixtures`, `preview_artifact` | Read-only and idempotent. |
| Capture and immediate analysis | 18 | `capture_cpu_hotspots`, `capture_gpu_launches`, `capture_benchmark_summary`, `capture_pytest_fixtures`, `capture_process_output` | Executes typed argv; not read-only or idempotent. |
| Evidence lifecycle | 3 | `prepare_providers`, `preserve_evidence`, `query_evidence` | Prepare an explicit uvx environment or manage immutable evidence. |

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

`prepare_providers` and the capture tools are open-world. Preparation resolves the exact package
requirement through `uvx`, verifies that environment by running Flameox's version command, and
returns the same requirement in a global, version-pinned MCP launcher. Capture tools execute an
explicit caller-supplied target that may itself access external services. The request names the
complete managed
provider set; Flameox keeps no installed-provider inventory or setup receipt. System profilers,
drivers, device access, and OS permissions remain external requirements. Flameox returns guidance
for them but does not invoke a system package manager or elevate privileges. Preparation creates no
project state,
durable job, or plan, and it cannot add packages to the currently running server. A result with
a non-null `next_action` identifies the typed `reconnect_mcp` handoff, explains that the current
server is unchanged, and directs the agent to reconnect with the returned launcher before retrying
the capture. Host-only preparation returns `next_action: null`. Managed provider IDs are `aiperf`,
`memray`, `otlp`, `perfetto`, `py-spy`, and `torch`; every call declares the complete desired set. A
provider such as Perfetto may be both prepared Python support and an external host Trace Processor
requirement. Preparation waits up to 1,800 seconds by default; callers may set `timeout_seconds`
from 1 through 3,600. A uvx failure returns its complete stderr in `SETUP_FAILURE`.

## Sources and limits

The strict source union is:

```text
PathSource     {kind: "path", path, format?, producer?, expected_sha256?}
EvidenceSource {kind: "evidence", evidence_id, artifact_role?}
```

Continuations are opaque integrity cursors bound to the request and exact input
digests. They contain no authority, credentials, or artifact data and are not
an authentication boundary: a caller already authorized to submit the analysis
can choose which of its rows to request. They can cross process boundaries, so
a CLI invocation can resume a previous page. A changed input cannot reuse a
continuation.

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

Requests may lower startup row, result-byte, timeout, output-byte, and durable
provenance-byte limits. Durable provenance bounds the captured argv and execution
metadata retained for explicit preservation.
They cannot raise them.

`query_evidence` returns 1-200 manifests per page. Its cursor is bound to both the immutable
inventory snapshot and the original filters; callers resume by repeating those filters unchanged.

## Capture

A direct target contains an argv array, an existing absolute cwd, and at most 32 bounded environment
overrides after experiment-case overrides are merged. Provider fields live in the capture tool's
typed provider union, and analysis fields live in its capability-specific `options` model. Shell
command strings are not accepted.

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
flameox mcp serve|inspect
flameox analyze [--continuation TOKEN] [--preserve]
flameox capture [--experiment JSON] [--preserve] -- <argv...>
flameox evidence query|show
```

`setup` detects supported coding agents and uses one multi-select prompt to choose which global MCP
client configurations to update. It preserves unrelated JSON or TOML content and writes stdio
configuration that launches the exact running Flameox release through `uvx` on Python 3.12.
Changed clients must restart or reconnect. Non-interactive setup requires explicit `--client`
targets or `--all`; `--yes` never converts detection into consent, and `--dry-run` reports the exact
paths and actions without mutation. Repeated `--provider` options declare the complete Python
provider set for the exact version-pinned uvx environment used by the saved launcher.
An existing OpenCode `opencode.jsonc` remains untouched because rewriting it as JSON would discard
comments; setup reports the file that needs a manual Flameox entry instead.
`--timeout-seconds` accepts 1 through 3,600 and defaults to 1,800. Resolver,
download, and compatibility failures retain uvx's complete stderr. System and
vendor providers receive external installation guidance. Setup does not create a persistent global
tool, durable operation, project state, or MCP setup endpoint. Other CLI commands use the same
explicit paths, capture working directories, and user-level evidence store as MCP.
When another `flameox` executable on `PATH` reports a different version, setup emits a non-fatal
advisory in human and JSON output. It never upgrades or removes that independently managed CLI.
