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
scope, status, limitations, coverage, and truncation. They use resource links
and typed references for native artifacts, normalized evidence generations, and
other larger durable records. Artifact identifiers remain provenance references;
they are never the sole carrier of run meaning.

`register_kernel_validation` is the explicit post-run handoff for correctness
evidence produced by a workload. It requires a succeeded run and its exact
reviewed revision, returns the linked run and artifact identities inline, and
links to both resources. When the run has one artifact pipeline, registration
also derives and links a new immutable pipeline generation containing the
validation stage. Runs with multiple pipelines require an explicit pipeline ID;
Flameox never guesses which lineage owns the evidence. `extract_kernel_validation`
remains idempotent and normalizes the registered document without moving run
semantics into its bytes.

`import_artifact` keeps free-form producer metadata separate from validated run
semantics. Omitting `profile` creates a generic import even when `producer` is
declared. `profile="py-spy-chrometrace"` validates the staged native event shape
before assigning the py-spy adapter identity; rejection returns a failed run
that still references the immutable bytes. The import receipt returns the
bounded semantic projection inline and links to the authoritative run and
artifact resources.
`qualify_artifact_import` applies the same validation to an artifact already
owned by a run, so recovery never depends on retaining the original source path
and does not duplicate content-addressed bytes.

Inline fields are response projections, not an alternate persistence model.
Property-defining execution semantics remain authoritative in the run manifest,
and selected findings or summaries remain authoritative in their durable records
when they must survive the response. A transport may inline a bounded subset for
agent usability without duplicating or replacing that state.

Capture responses may also return a bounded `recovery` action. For a timed-out
Compute Sanitizer probe, unlimited plans receive an executable one-launch replan;
already-bounded plans require a changed target-only workload or provider filter
instead of suggesting that the same capture be repeated.

MCP capture receipts inline the safe semantic projection—semantic identity,
adapter identity, mode, process scope, bounds, filters, and explicit unavailable
fields—and link to the bounded run resource and any automatically registered
artifact pipelines. `list_artifact_pipelines` provides cursor-paginated discovery
by run, producer, schema, or source artifact; `get_artifact_pipeline` returns one
pipeline plus bounded compatible candidates. Pipeline comparison accepts a
pipeline ID or a run ID only when that run resolves to exactly one pipeline.
Typed MCP output schemas version
these response contracts; individual success envelopes and receipts do not add
an ornamental schema-version field.

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
points need an exact package-identity approval. From an editable Flameox checkout,
setup builds one wheel into workspace-owned content-addressed storage and binds its
wheel digest, bounded source-tree identity, revision, and dirty state in the provider
receipt. Released installations resolve Flameox and the provider into a preserved,
content-addressed requirements lock and require artifact hashes during installation;
setup never points a provider environment at a mutable checkout. Workload dependency
preparation is deliberately
inspection-only: it queries the exact Python interpreter bound to the workload and
reports missing distributions without changing that environment or Flameox's own
runtime. None of these steps runs the workload.

Adapter availability and workload compatibility are separate facts. In
particular, an NVBench workload declares `execution_protocol = "nvbench"` and
active planning qualifies the exact executable with `--version` before provider
flags are constructed. The capture plan exposes that qualification inline because
it is run-scoped interpretation and authorization evidence, not an artifact.

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

Plans are short-lived authorization receipts, not durable run manifests and not
an additional evidence schema. They retain only the finalized typed values needed
to review, execute, and revalidate one launch. The authorization digest is derived
from that finalized plan instead of a separately maintained field list. Durable
semantics move to the run manifest when execution begins.

A consumed synchronous capture plan is not retryable. If the response is lost,
inspect durable run and operation state rather than submitting the token again.
Detached capture and capability setup are retryable only with the same
idempotency key and same request digest. A changed intent requires a new key.
Managed capability setup projects bounded transfer progress inline: received and
expected bytes, elapsed time, and durable safe-resume availability. In-flight response
bytes are not reported as resumable until their authenticated checkpoint is committed. It never exposes asset
URLs, validators, or request headers through operation status.

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
`get_trace_window` is provider-neutral: it reads normalized temporal evidence
when available and uses a native Perfetto reader only as a fallback. Its inline
events carry the provider and interpretation fields needed for the window;
native traces and full normalized generations remain referenced evidence.

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
Workspace status and validation return the typed `catalog.rebuild` action when
the disposable catalog is missing or unreadable. MCP exposes that action as
`rebuild_catalog`. A new corpus HEAD does not make the catalog stale: each read
constructs transient views from its exact pinned immutable corpus inventory.

## Progress and cancellation

Known-duration work reports monotonic completed/total progress. Unknown-duration
phases report phase changes without invented percentages. Progress delivery is
non-authoritative: a logging or notification failure cannot change the operation
result.

Client cancellation propagates into the owning application operation. A cancellation
tool durably records the request and waits up to 250 milliseconds for immediate cleanup;
if cleanup is still active, it returns the pollable `cancelling` state instead of blocking.
Detached operations can be inspected after client disconnect.

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
