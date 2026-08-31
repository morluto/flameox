# CLI and MCP boundaries

The CLI and MCP server are transports over the same application services. They
share domain models, validation, execution authority, persistence, and errors.
Neither transport may invent a second behavior contract.

This document describes workflow and trust semantics. Do not maintain a second
catalog of every flag or JSON field here:

```console
flameox --help
flameox <command> --help
flameox mcp inspect --project-root . --json
```

Those commands expose the authoritative interface for the installed version.

## Workspace selection

Every invocation operates on one explicit workspace or fixed project root.

- CLI `--workspace` names `.diagnostics` directly.
- CLI `--project-root` discovers `.diagnostics` beneath that root.
- `mcp serve --project-root PATH` fixes the root for the server lifetime.
- MCP initialization may create `.diagnostics` only beneath that fixed root.

The server does not initialize its launch directory implicitly or accept an
arbitrary workspace path from each tool call. Call `workspace_status`; if it
reports `WORKSPACE_NOT_FOUND`, call `initialize_workspace`, then repeat status.

## CLI contract

Global `--json` emits one domain result on stdout. Human progress and local
diagnostics do not contaminate JSON output. `--quiet` suppresses successful
human output; typed failures retain a nonzero exit code. `--timeout` bounds the
owning asynchronous operation but does not abandon cleanup.

Commands are grouped by task: workspace, capture, workload, experiment,
inference, import/extract, analysis, evidence lookup, integrity, and retention.
Mutating commands make mutation explicit. In particular:

- `capture plan` never runs the measured workload; an adapter may run an explicitly
  documented, bounded compatibility probe under the requested execution policy;
- `capture execute` consumes one previously reviewed plan token and may require its
  public plan identity; it accepts no replacement execution fields;
- `capture run` plans then consumes one server-owned plan;
- `open` prints a viewer plan unless `--launch` is supplied;
- `validate` never repairs;
- `gc` is a dry run unless `--apply` is supplied;
- permanent purge names one expired trash manifest;
- catalog rebuild changes only derived DuckDB state.

CLI JSON is the validated domain model. Scripts should bind to a released
Flameox version because domain changes are explicit and unknown fields are not
silently accepted.

## MCP transport

The MCP server uses stdio. Stdout is reserved for protocol messages; logs go to
stderr or local diagnostic files. It exposes tools and resource templates, but
no prompts.

Tool inputs are generated from strict SDK/Pydantic models. Unknown keys, invalid
unions, coercion-dependent values, malformed paths, and unbounded strings are
rejected at the boundary. Domain failures return `isError=true` with a stable
typed error projection and, when recovery exists, one canonical action with
the exact MCP tool name and validated arguments.

Extractor prerequisites use the same error contract across formats. An unknown
run is `RUN_NOT_FOUND`; an existing run without the required evidence is
`ARTIFACT_NOT_FOUND`; bytes that exist but cannot be decoded are
`ARTIFACT_PARSE_FAILED`. Missing-input details name the required artifact kinds
and compatible capture adapters or import producers. Recovery points to capture
or import and is never marked safe to repeat until that prerequisite state has
changed.

Tool annotations describe side effects:

- read-only tools do not mutate durable state;
- configuration tools modify only their declared project/runtime contract;
- plan tools do not execute a measurement workload, but may perform bounded
  compatibility probes declared by the adapter contract;
- execute tools consume approved intent;
- idempotent start tools reconnect when called with the same key and intent.

MCP does not expose arbitrary commands, SQL, arbitrary native artifact bytes,
viewer launch, garbage collection, purge, or source editing. `preview_artifact`
is the narrow exception for textual process and validation output: it requires
explicit byte and line bounds, enforces effective sensitivity and UTF-8 validity,
and returns a continuation offset without exposing a payload path. The CLI
provides the equivalent `artifacts preview` operation.

Tool results use a hybrid response boundary. They return the bounded run
semantics needed to interpret the immediate outcome, such as effective mode and
scope, status, limitations, coverage, and truncation. They use native
`ResourceLink`s and typed references for native artifacts, normalized evidence
generations, and other larger durable records. Artifact identifiers remain
provenance references; they are never the sole carrier of run meaning.
Structured references use one `{kind, resource_id, uri}` shape, while the
matching native `ResourceLink` remains available to hosts that route content
blocks directly. Artifact resources expose content identity and byte length,
not content-store filenames or native bytes.

Inline fields are response projections, not an alternate persistence model.
Property-defining execution semantics remain authoritative in the run manifest,
and selected findings or summaries remain authoritative in their durable records
when they must survive the response. A transport may inline a bounded subset for
agent usability without duplicating or replacing that state.

Feature-specific tool names, fields, and limits are intentionally omitted here.
Use `mcp inspect` for the installed interface and [Adapters and producer
contracts](adapters.md) for producer behavior. Across all tools, imports preserve
native bytes, captures preserve effective run semantics, queries stay bounded,
and recovery actions name the next valid operation.

## Agent workflow

The normal capture path is:

```text
workspace_status
  └─ initialize_workspace, only when absent
workload_configuration_status
  └─ configure_workload, only when a declaration is needed
list_declared_workflows
get_declared_workflow
list_capabilities(adapter=...)
  ├─ start_capability_setup → get_capability_setup
  ├─ prepare_adapter
  └─ prepare_workload_dependencies
plan_capture
  ├─ execute_capture_plan          short operation
  └─ start_detached_capture        long operation
       └─ get_detached_capture
get_run
  └─ bounded extract/analyze
       └─ get_evidence → record_analysis/record_finding
```

Capability responses distinguish passive discovery from execution binding.
Setup installs one allowlisted provider per version-addressed environment; it
never changes the running MCP environment. Receipts bind the installed package
and environment identity. Workload dependency preparation only reports missing
packages and never changes the workload environment. None of these steps runs
the workload.

`capture_mode="auto"` uses trusted-local execution and records the containment
limitation. `capture_mode="managed"` requests the stronger project policy and
fails if it cannot be supplied.

## Plans and retries

Plan previews contain audit identity and reviewable intent. They are not
execution authority. Execution accepts the opaque `plan_token`; SQLite returns
the complete issued intent and consumes it atomically. An optional expected plan
ID detects a caller reviewing one plan and presenting another token.

Plans are short-lived authorization receipts, not durable run manifests. They
retain only the finalized typed values needed to review, execute, and revalidate
one launch. The authorization digest comes from that plan, not a second field
list. Durable semantics move to the run manifest when execution begins.

A consumed synchronous capture plan is not retryable. If the response is lost,
inspect durable run and operation state rather than submitting the token again.
Detached capture and capability setup are retryable only with the same
idempotency key and same request digest. A changed intent requires a new key.
Managed setup reports bounded transfer progress. It reports bytes as resumable
only after a checkpoint is committed and never exposes download credentials or
request headers.

Configuration, workload, executable, provider, oracle, and environment identity
are revalidated before launch. Stale intent fails and must be replanned.

## Inference workflow

Inference servers and scenarios are typed `flameox.toml` declarations. The MCP
flow is:

```text
configure_inference_server / configure_inference_scenario
list_inference_configurations
plan_inference_scenario
run_inference_scenario(plan_token, expected_plan_id)
list_inference_requests / query_measurements
```

Managed servers are named workloads. Existing-local targets are loopback-only
and exploratory because Flameox cannot bind their complete server state. A
diagnostic profile requires a successful compatible unprofiled measurement run:

```text
plan_inference_profile(measurement_run_id=...)
run_inference_profile(plan_token, expected_plan_id)
```

Provider output remains prompt-free on agent-facing surfaces; sensitive native
exports stay local as artifacts.

Normalized inference-request rows carry one typed `outcome` object. They do
not duplicate that fact as flattened `success`, `cancelled`, or error columns.

## Evidence reads and snapshots

List operations are bounded and cursor-paginated. A cursor binds the query,
ordering, and snapshot; it cannot be reused for a different request. Analysis
and evidence resolution pin one corpus snapshot before the first lookup. A
cursor keeps that snapshot even if the workspace publishes newer evidence.
Changing the query invalidates the cursor.

MCP resources are bounded JSON projections, not alternate authorities. Current
templates cover runs, artifacts, pipelines, investigations, hypotheses,
findings, experiments and trials, analyses, comparisons, and other durable
records advertised by `mcp inspect`. Artifact resources never contain native
bytes.

A bounded projection may include small normalized summaries, prioritized
findings, coverage, or next-step guidance when useful for the current task. It
links to the authoritative run, analysis, normalized generation, or native
artifact for deeper inspection. Large result sets and native evidence are not
repeated in every tool response merely because an inline representation is
possible.

An empty result and unavailable evidence are different states. Results carry
coverage, limitation, or recovery information when the requested evidence was
not extracted, unsupported, outside the snapshot, or truncated. Recovery points
to the next operation that can change that state.

## Progress and cancellation

Known-duration work reports monotonic completed/total progress. Unknown-duration
phases report phase changes without invented percentages. Progress delivery is
non-authoritative: a logging or notification failure cannot change the operation
result.

Client cancellation propagates into the owning operation. If cleanup is still
running, the tool returns a pollable `cancelling` state. Detached operations can
be inspected after the client disconnects.

## Errors

Errors identify a stable code, safe message, retryability, and bounded details.
Recovery guidance is executable: a tool action contains only `tool_name` and
complete validated `arguments`; manual and external actions contain their
canonical instructions.

Sensitive paths, environment values, child output, prompts, generated text, and
artifact contents are excluded from agent-facing diagnostics. Local native
artifacts retain their declared sensitivity and are inspected through explicit
local workflows.

## Setup and distribution

`npx flameox@latest setup` is the guided client-registration path. It previews
approved config changes and installs a persistent versioned Python runtime.
`npx flameox upgrade` resolves the latest bootstrap before updating that runtime.
Connected clients launch the installed runtime directly; they do not perform a
network-dependent install on every MCP start.

The official MCP Registry name is `io.github.morluto/flameox`. Registry clients
may launch a selected PyPI release through `uvx`. Setup, registry discovery, and
source development are distribution choices; all start the same stdio server
and application services.
