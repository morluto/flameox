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

- `capture plan` is side-effect-free with respect to workload execution;
- `capture run` plans then consumes one server-owned plan;
- `open` prints a viewer plan unless `--launch` is supplied;
- `validate` never repairs;
- `gc` is a dry run unless `--apply` is supplied;
- permanent purge names one expired trash manifest;
- catalog rebuild changes only derived DuckDB state.

CLI JSON is the validated domain model, not a separately versioned response
shape. Scripts should still bind to a released Flameox version because domain
schema evolution is explicit, not indefinitely permissive.

## MCP transport

The MCP server uses stdio. Stdout is reserved for protocol messages; logs go to
stderr or local diagnostic files. It exposes tools and resource templates, but
no prompts.

Tool inputs are generated from strict SDK/Pydantic models. Unknown keys, invalid
unions, coercion-dependent values, malformed paths, and unbounded strings are
rejected at the boundary. Domain failures return `isError=true` with a stable
typed error projection and, when recovery exists, exact next-tool arguments.

Tool annotations describe side effects:

- read-only tools do not mutate durable state;
- configuration tools modify only their declared project/runtime contract;
- plan tools do not execute a workload;
- execute tools consume approved intent;
- idempotent start tools reconnect when called with the same key and intent.

MCP does not expose arbitrary commands, SQL, native artifact bytes, viewer
launch, garbage collection, purge, or source editing.

Tool results use a hybrid response boundary. They return the bounded run
semantics needed to interpret the immediate outcome, such as effective mode and
scope, status, limitations, coverage, and truncation. They use resource links
and typed references for native artifacts, normalized evidence generations, and
other larger durable records. Artifact identifiers remain provenance references;
they are never the sole carrier of run meaning.

Inline fields are response projections, not an alternate persistence model.
Property-defining execution semantics remain authoritative in the run manifest,
and selected findings or summaries remain authoritative in their durable records
when they must survive the response. A transport may inline a bounded subset for
agent usability without duplicating or replacing that state.

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
get_run → extract/analyze → get_evidence → record_analysis/record_finding

V8 profile imports use `extract_node_cpu_prof` or `extract_node_heap_prof` after
the corresponding native artifact has been preserved.
```

Capability responses distinguish passive discovery from execution binding.
Setup installs only allowlisted managed providers into version-addressed provider
environments; it never adds packages to the running MCP server. Provider setup
installs core Flameox plus that provider—not a Flameox “all extras” environment—
so mutually incompatible profilers and reducers remain usable. Third-party entry
points need an exact package-identity approval. Workload dependency preparation is deliberately
inspection-only: it queries the exact Python interpreter bound to the workload and
reports missing distributions without changing that environment or Flameox's own
runtime. None of these steps runs the workload.

`import_xctrace` and `flameox import-xctrace` are the same application operation.
They preserve a native `.trace` directory as a sensitive immutable package with
its bounded `xctrace` table-of-contents export. They do not record a workload,
install Xcode, or expose arbitrary XPath queries.

Reduction is a separate task-shaped workflow:

```text
list_capabilities(adapter="shrinkray")
  └─ start_capability_setup, when unavailable
plan_reduction(original_artifact_id, predicate_workload, input_format, limits)
execute_reduction(plan_id)
get_reduction(reduction_id)
```

The MCP request schema describes this stable capability; it does not expose
ShrinkRay flags or arbitrary predicate commands. Planning resolves the named
predicate workload and exact managed provider, and execution refuses changed
provider, bridge, predicate, or artifact identity.

`capture_mode="auto"` uses trusted-local execution and records the containment
limitation. `capture_mode="managed"` requests the stronger project policy and
fails if it cannot be supplied.

## Plans and retries

Plan previews contain audit identity and reviewable intent. They are not
execution authority. Execution accepts the opaque `plan_token`; SQLite returns
the complete issued intent and consumes it atomically. An optional expected plan
ID detects a caller reviewing one plan and presenting another token.

A consumed synchronous capture plan is not retryable. If the response is lost,
inspect durable run and operation state rather than submitting the token again.
Detached capture and capability setup are retryable only with the same
idempotency key and same request digest. A changed intent requires a new key.

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

## Evidence reads and snapshots

List operations are bounded and cursor-paginated. A cursor binds the query,
ordering, and snapshot; it cannot be reused for a different request. Analysis
and evidence resolution pin one corpus snapshot before the first lookup.

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
not extracted, not supported, outside the snapshot, or truncated by a budget.

## Progress and cancellation

Known-duration work reports monotonic completed/total progress. Unknown-duration
phases report phase changes without invented percentages. Progress delivery is
non-authoritative: a logging or notification failure cannot change the operation
result.

Client cancellation propagates into the owning application operation. The
operation cancels and awaits children, records truthful terminal state, and
finishes bounded cleanup before returning cancellation. Detached operations can
be inspected after client disconnect.

## Errors

Errors identify a stable code, safe message, retryability, and bounded details.
Recovery guidance is executable: it names the next tool and complete validated
arguments rather than asking an agent to reconstruct intent from prose.

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
