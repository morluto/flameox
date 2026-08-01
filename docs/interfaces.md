# CLI and MCP boundaries

## Local client setup

`npx flameox setup` is the supported interactive path for connecting local MCP
clients. The npm package and Python distribution share an exact release
version. npm supplies a thin launcher and the upstream `jsonc-parser` editor;
the Python application service owns discovery, planning, runtime installation,
verification, locking, activation, rollback, and structured results. The same
setup service is available from `flameox setup` for standard JSON and TOML
clients.

`npx flameox upgrade` is the non-interactive npm shorthand for updating the
detected clients to the bootstrap package's matching Python runtime. It keeps
the same managed-runtime verification and atomic launcher activation as setup.

The first run requires an explicit multi-select. Detection preselects detected
clients for connection, while the user can adjust the selection before applying
it. Every mutation has a complete preview and a confirmation whose default is
no. Non-interactive application requires both an explicit target (`--codex`,
another client flag, `--all`, `--refresh`, `--rollback`, or `--verify`) and
`--yes`; `--dry-run --json` returns the plan
without mutation.

Interactive setup is also the lifecycle entry point for connecting or
disconnecting clients, updating the active runtime, rolling back to an installed
version, and verification. Verification checks the managed runtime version and
real MCP handshake, then confirms that every configured client still contains
the exact active executable and fixed launcher arguments. It is read-only and
fails on configuration drift instead of repairing it implicitly.

Client launchers contain the absolute path of a verified managed runtime and
the fixed arguments `mcp serve --project-root .`. Setup never passes `--init`.
Each MCP client therefore binds flameox's project root to the client's launch
directory. Client setup is complete when the launcher is verified; each checkout
still requires deliberate project initialization through `flameox init`,
`flameox mcp serve --init`, or the MCP `initialize_workspace` tool after the
client's launch directory has been verified. A fresh server reports
`WORKSPACE_NOT_FOUND` until that project step is complete.

Setup serializes mutations with a user-local lock. Before applying, it compares
every config with the exact bytes used for the preview. It writes a recovery
journal before the first config change, uses atomic replacement, and restores
all original files after a partial failure or on the next invocation after a
crash. Existing comments, unrelated servers, permissions, and formatting are
preserved where the native format supports it. Malformed or non-UTF-8 configs
are refused rather than replaced.

The CLI and MCP server expose the same application services to different trust
contexts. The CLI may offer explicit local administration and destructive
recovery operations. MCP exposes bounded, task-shaped operations suitable for
an agent and does not expose arbitrary commands, raw SQL, deletion, or
sensitive artifact bytes.

The transport does not own domain behavior. Results, validation, cancellation,
provenance, and error semantics belong to the shared application layer.

## CLI specification

### General behavior

The executable is `flameox`. Commands produce readable terminal output by
default and structured JSON with `--json`. CLI JSON is the domain result model;
MCP wraps that same model in its transport envelope.

Common options:

```text
--workspace PATH
--project-root PATH
--json
--quiet
--log-level LEVEL
--timeout SECONDS
```

Exit codes:

- `0`: successful operation;
- `1`: operation completed with a negative result such as failed validation;
- `2`: invalid arguments or configuration;
- `3`: required capability unavailable;
- `4`: capture or external process failed;
- `5`: artifact or corpus integrity failure;
- `6`: comparison invalid;
- `7`: lock or concurrency failure;
- `8`: operation cancelled or timed out;
- `9`: safety or policy refusal.

### Workspace commands

```text
flameox init
flameox status
flameox capabilities [--refresh]
flameox config show
flameox validate
flameox workload approve <name>
```

`init` creates only local files and never installs collectors. `capabilities`
reports installed tools, versions, supported modes, permissions, and remediation
commands.

### Capture commands

```text
flameox capture plan <adapter> [options] -- <argv...>
flameox capture run <adapter> [options] -- <argv...>
flameox import <path> [--kind KIND]
```

`capture plan` is side-effect free. `capture run` prints the plan before
execution unless `--json` is used, in which case the plan is part of the result.
Every import creates a new import run.

### Workload commands

```text
flameox workload list
flameox workload show <name>
flameox workload run <name> [parameter overrides]
flameox investigations create <structured-input>
flameox investigations list [filters]
flameox investigations show <investigation-id>
flameox hypotheses record <structured-input>
flameox hypotheses show <hypothesis-id>
flameox experiment plan <name> [parameter overrides]
flameox experiment run <name> [parameter overrides]
flameox experiment show <experiment-id>
flameox experiment trial <trial-id> [--experiment-id <experiment-id>]
```

### Analysis commands

```text
flameox analyze hotspots <run-or-artifact>
flameox analyze scaling <experiment-or-run-set>
flameox analyze compare <baseline-run-set> <candidate-run-set>
flameox analyze pytorch <run-or-artifact>
flameox analyze memory <run-or-artifact>
flameox analyze execution <run-or-artifact> [--compare RUN]
flameox analyze failures [filters]
flameox analyze record <structured-input>
flameox analyze record-comparison <structured-input>
```

### Evidence commands

```text
flameox runs list [filters]
flameox runs show <run-id>
flameox artifacts list [filters]
flameox artifacts show <artifact-id>
flameox findings list [filters]
flameox findings show <finding-id>
flameox findings record <structured-input>
flameox evidence get <typed-reference>
flameox evidence summarize [bounded evidence selections] [--format json|markdown]
flameox measurements query [curated filters]
flameox stacks callers <run-or-artifact> <frame-id>
flameox stacks callees <run-or-artifact> <frame-id>
flameox stacks examples <run-or-artifact> <frame-id>
flameox trace window <artifact-id> --start NS --end NS
flameox open <artifact-id>
```

Drill-down commands are bounded and use reviewed query families. They report
total and returned counts, truncation, coverage, and stable keyset cursors.
There is no arbitrary SQL or free-form PerfettoSQL command.

`open` prints the appropriate installed viewer command by default.
`flameox open --launch` executes it explicitly. It never launches a browser when
`--json` is active.

### Catalog and recovery

```text
flameox catalog validate
flameox catalog rebuild
flameox recover [--quarantine QUARANTINE_ID]
flameox repair [PLAN.json]
flameox gc [--dry-run | --apply | --purge TRASH_MANIFEST | --restore TRASH_MANIFEST]
```

Garbage collection is always dry-run by default. Apply may move only eligible
staging, generations, caches, and unreferenced artifacts to recoverable trash.
It reports exact paths, retention roots, and recoverability before mutation.
Purge is a separate destructive action against a specific expired trash
manifest.

### MCP

```text
flameox mcp serve [--init]
flameox mcp inspect
```

`serve` uses stdio exclusively. `--init` performs the additive workspace
initialization before protocol startup and never approves project-controlled
workloads. A network transport is outside the supported product contract.

## MCP server specification

### SDK and transport

The server uses the official Python SDK pinned to `mcp==2.0.0b2`. The pin is
required because v2 is a prerelease and unpinned resolution may select an
incompatible stable v1 release.

The supported transport is stdio. The server:

- writes protocol messages only to stdout;
- writes diagnostics to stderr or MCP logging;
- initializes workspace and read services through SDK lifespan state;
- returns an explicit MCP result envelope with concise text and one structured
  payload;
- reports progress for long captures and analyses;
- propagates cancellation through the execution broker and containment backend;
- closes DuckDB and Perfetto resources on shutdown.

Handlers translate SDK inputs and outputs but contain no diagnostic logic:

```python
from typing import Annotated

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp_types import CallToolResult, TextContent, ToolAnnotations

server = MCPServer("flameox", lifespan=flameox_lifespan)


@server.tool(
    annotations=ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )
)
async def compare_run_sets(
    request: CompareRunSetsRequest,
    ctx: Context[AppContext],
) -> Annotated[CallToolResult, ToolPayload[ComparisonResult]]:
    services = ctx.request_context.lifespan_context.services
    try:
        result = await services.comparisons.compare(request)
        payload = ToolPayload.ok(result)
        return CallToolResult(
            content=[TextContent(type="text", text=result.short_summary())],
            structured_content=payload.model_dump(mode="json"),
        )
    except DomainError as error:
        payload = ToolPayload.failed(error.to_detail())
        return CallToolResult(
            content=[TextContent(type="text", text=error.concise_message)],
            structured_content=payload.model_dump(mode="json"),
            is_error=True,
        )
```

`ToolPayload[T]` is an object-root schema with `schema_version`, `ok`,
`result | null`, and `error | null`. Domain errors and malformed arguments use
`isError=true` with that structured payload; malformed arguments use the
`INVALID_ARGUMENTS` code and include the affected fields. Missing tools and
invalid JSON-RPC requests remain protocol errors. Returning bare Pydantic models is forbidden because this SDK beta
duplicates them into text and structured content; uncaught domain exceptions
are forbidden because their structured detail is lost.

The concrete v2 SDK context, annotations, content blocks, and lifespan types are
confined to `flameox.mcp`. The server calls synchronous `server.run()` for stdio.
It does not depend on experimental MCP background-task APIs.

### Tool design

Tools are task-shaped and versioned through their result schemas. Every
registration supplies explicit `ToolAnnotations`.

- Query, status, and analysis tools set `read_only_hint=True`,
  `destructive_hint=False`, `idempotent_hint=True`, and
  `open_world_hint=False`.
- Read-only analysis results are deterministic for their pinned inputs and do
  not publish evidence. `record_analysis` and `record_comparison` are explicit
  additive operations that materialize typed provenance.
- `plan_capture` and `plan_experiment` are read-only, idempotent, and
  closed-world with the same annotation values.
- Imports and finding revisions are mutating, additive where applicable,
  non-idempotent, and closed-world:
  `read_only_hint=False`, `destructive_hint=False`,
  `idempotent_hint=False`, `open_world_hint=False`.
- Capture and experiment execution are mutating, non-idempotent, potentially
  destructive, and open-world:
  `read_only_hint=False`, `destructive_hint=True`,
  `idempotent_hint=False`, `open_world_hint=True`, because an uncontained
  workload may modify files or use the network.

`flameox mcp inspect --json` returns the exact installed schemas, annotations,
and the server `instructions` string returned by MCP initialize. The CLI
inspection is additive at schema version 1; in-process and real stdio clients
must observe the same instructions.
The supported tools are grouped as follows:

| Family | Tools |
| --- | --- |
| Workspace | `initialize_workspace`, `workspace_status`, `list_capabilities`, `validate_workspace` |
| Capture and import | `plan_capture`, `execute_capture_plan`, `import_artifact`, `extract_pyperf`, `extract_python_startup`, `extract_pytest`, `extract_coverage`, `extract_memray`, `extract_perfetto`, `extract_observations` |
| Detached capture | `start_detached_capture`, `get_detached_capture`, `cancel_detached_capture` |
| Discovery | `list_declared_workflows`, `get_declared_workflow`, `list_runs`, `list_findings` |
| Investigations | `create_investigation`, `list_investigations`, `get_investigation`, `record_hypothesis`, `get_hypothesis` |
| Experiments | `plan_experiment`, `run_experiment`, `get_experiment`, `freeze_run_set` |
| Runs and artifacts | `list_runs`, `get_run`, `list_artifacts`, `get_artifact`, `get_native_viewer_plan` |
| Evidence products | `summarize_evidence`, `register_artifact_pipeline`, `compare_artifact_pipelines` |
| Reductions | `plan_reduction`, `execute_reduction`, `get_reduction` |
| Analysis | `analyze_hotspots`, `analyze_scaling`, `analyze_pytorch`, `analyze_memory`, `analyze_execution`, `analyze_failures`, `compare_run_sets`, `record_analysis`, `record_comparison` |
| Drill-down | `get_evidence`, `query_measurements`, `get_frame_callers`, `get_frame_callees`, `get_stack_examples`, `get_trace_window` |
| Findings | `record_finding`, `list_findings`, `get_finding` |

#### `initialize_workspace`

Additive and idempotent. Initializes only the MCP server's fixed project root.
If the server already owns an initialized workspace, the call returns its current
status without replacing the detached-capture manager. It cannot select an
external path or approve workloads. Hosts may instead start the server using
`flameox mcp serve --init`.

#### `workspace_status`

Read-only. Returns workspace identity, validation state, catalog state, storage
usage, active captures, and warnings.

#### `list_capabilities`

Read-only. Returns adapter capabilities, installed versions, required
permissions, unavailable features, and remediation. The default call performs
only passive inspection. Active executable probes are separately requested,
bounded, and executed through the subprocess broker; merely listing
capabilities does not run a project-controlled binary found on `PATH`. The
`active_cached` mode uses a previously completed probe, while
`active_refresh` explicitly reruns it.

#### `list_declared_workflows` and `get_declared_workflow`

Workflow discovery is the authoritative source for approved workload names,
declared parameters, requirements, and adapter choices. A workflow detail
returns each requirement's kind, required or optional status, and passive or
active probe requirement. It also returns at most 64 deterministic adapter
options with capability status, planning disposition, required preflight mode,
permission status, supported modes and formats, features, limitations, and
remediation. Options that need a permission check are reported as
`active_probe_required`; discovery remains passive until the caller requests
`list_capabilities(mode="active_refresh")`.

#### `plan_capture`

Read-only. Default inputs are an approved `workload_name`, declared scalar
parameter overrides, adapter, and requested features. Returns the exact
resolved plan, expected artifacts, overhead, permissions, limits, containment
state, warnings, request digest, and a short-lived `plan_id`. The ordinary
schema does not advertise `argv` or `cwd`. MCP does not expose ad-hoc capture
planning.

Plan IDs are 256-bit opaque random values held only in MCP-process memory. They
are bound to workspace ID, workload approval hash, resolved executable and
identity, arguments, working directory, child-environment policy and overrides,
adapter/version/capabilities, requested features, validator, source state,
limits, containment decision, and policy generation. Expiry uses monotonic
time. Plans are atomically single-use, consumed before process creation,
bounded in count, invalid after restart, and rejected if any bound input changes.
A request digest is audit evidence, not an authorization boundary.

#### `execute_capture_plan`

Mutating and command-executing. Executes an unexpired `plan_id`, streams
progress, observes cancellation, and returns a completed or failed run record.
Every bound identity, capability, approval, and policy input is rechecked
immediately before execution. It never accepts a shell string or replacement
arguments.

#### `plan_experiment` and `run_experiment`

Planning is read-only over an approved named workload and experiment
definition. Execution runs the predeclared variants and blocks, registers every
attempted trial, validates outputs, and returns the experiment plus initial
analysis references. The execution receipt links to the immutable experiment
protocol, its bounded trial collection, and the first failing trial when one
exists; it does not duplicate every trial or oracle receipt.

Trial collections are paged at 1,000 entries. The `list_experiment_trials`
read-only tool accepts a smaller page size and the returned cursor; the trial
collection resource exposes the first bounded page and its continuation cursor.
When a trial identifier is reused by multiple historical experiments, callers
must provide the experiment ID to resolve it; an unscoped lookup reports the
ambiguity instead of selecting a newer row.

#### `create_investigation` and `record_hypothesis`

Mutating, additive structured operations. Hypothesis revision requires
`expected_revision`, an explicit prediction, and a discriminating condition.
Corresponding bounded list/get tools allow agents to resume an investigation
without relying on prior conversation context.

#### `import_artifact`

Mutating. Imports a file under allowed roots, computes its identity, validates
the selected or detected kind, creates a new import run, and optionally extracts
evidence.

#### `list_runs`

Read-only, filtered, sorted, and paginated. It cannot return unbounded
manifests.

#### `get_run`

Read-only. Returns one run manifest summary, artifact references, validation,
string limitations, typed `limitation_details`, and runtime-resource
availability. Resource observations remain bounded and are available through
`analyze_memory`; host storage paths are not exposed.

#### `get_artifact`

Read-only metadata. Returns artifact identity, kind, size, sensitivity,
integrity, supported analyses, and paginated registrations. An optional
`run_id` selects one contextual registration. It does not return binary
content.

#### `analyze_hotspots`

Read-only. Runs the appropriate profile or trace query and returns bounded
source-linked hotspots.

#### `analyze_scaling`

Strictly read-only over an existing experiment or frozen run set. It never
executes missing trials. Work requiring collection uses `plan_experiment` and
`run_experiment`.

#### `compare_run_sets`

Read-only. Returns compatibility, metric differences, frame/operator changes,
validation, attempted/failure counts, estimand, and limitations. Pairwise run
IDs are accepted as one-element run-set shorthand.

#### `record_analysis` and `record_comparison`

Mutating and additive. These execute the same curated deterministic recipes as
the read-only tools, then persist an `AnalysisRecord`, result digest, exact
input corpus commit, typed evidence references, coverage, and limitations.
They do not accept arbitrary SQL or caller-supplied result bodies.

#### `analyze_pytorch`

Read-only over an existing trace. Returns operator and accelerator summaries.

#### `analyze_memory`

Read-only over an existing memory artifact or run.

#### `analyze_execution`

Read-only over coverage, trace annotations, configuration observations, and SDK
observations. Returns bounded source-path evidence and, optionally, differences
between two runs.

#### `record_finding`

Mutating. Stores a structured claim with typed evidence references. Updating
requires `expected_revision`. It validates reference existence, assessment
requirements, and evidence-level consistency.

#### `list_findings` and `get_finding`

Read-only, filtered, bounded, and paginated finding retrieval.

#### `get_evidence` and bounded drill-down tools

Read-only operations retrieve a typed evidence reference, curated measurement
query, hotspot callers/callees/representative stacks, or a bounded trace
time-window/event neighborhood. They return stable cursors, total/returned
counts, coverage, and limitations. They do not accept raw SQL.

#### `validate_workspace`

Read-only by default. Checks manifests, artifact hashes on request, Parquet
schemas, references, and catalog freshness. Repairs require the explicit CLI
repair command; MCP never repairs.

### MCP resources

Resources provide stable, bounded representations:

```text
flameox://runs/{run_id}
flameox://artifacts/{artifact_id}
flameox://findings/{finding_id}
flameox://investigations/{investigation_id}
flameox://hypotheses/{hypothesis_id}
flameox://experiments/{experiment_id}
flameox://experiments/{experiment_id}/trials
flameox://experiments/{experiment_id}/trials/{trial_id}
flameox://run-sets/{run_set_id}
```

Resources return JSON or text summaries. Large native artifacts are represented
by metadata and local handles, not injected into model context. Template
resources declare `mime_type="application/json"`, percent-encode identifiers,
and resolve services through a server-local lifespan closure. In the exact
`2.0.0b2` wheel, template-handler `Context` is reconstructed by Pydantic and
loses its private request state, so `ctx.request_context` raises at runtime.
Resource handlers therefore omit `Context`; the MCP adapter stores the active
lifespan value in a closure for the duration of `server.run()`. Tool handlers
continue using injected `ctx.request_context.lifespan_context`. A contract test
must fail if a future SDK change invalidates either path, and the workaround is
removed on upgrade when template context is proven functional.

Mutable workspace and capability views remain tools rather than static
resources because this SDK beta does not inject lifespan context into static
resource handlers. Tool results should include MCP `ResourceLink` blocks for
addressable runs, artifacts, findings, analyses, and comparisons when useful.

### No MCP prompts

Investigation recipes belong in executable domain logic and tool descriptions,
not MCP prompt templates. Prompts can be added only when they express a stable
human-facing workflow that cannot be represented by structured tools.

### Progress

Long operations report named phases:

```text
planning
warming_up
capturing
validating_artifact
extracting
publishing
analyzing
completed
```

Progress is one monotonic stream for the entire request; it never resets at a
phase boundary. A default operation uses fixed phase work units, for example
`0/8 planning` through `8/8 completed`. Measurable sub-work may occupy a fixed
interval. Unknown-duration work reports phase transitions and uses MCP logging
for elapsed time rather than inventing percentages. Reporting is best-effort:
`ctx.report_progress()` is a no-op when the client supplied no progress token.

### Cancellation

An incoming MCP cancellation cancels the handler's AnyIO scope. Cleanup must
therefore be explicitly shielded and bounded:

```python
try:
    return await execution.execute_capture_plan(plan)
except anyio.get_cancelled_exc_class():
    with anyio.CancelScope(shield=True):
        with anyio.fail_after(cleanup_timeout):
            await execution.cancel_containment(plan.run_id)
            await runs.publish_cancelled(plan.run_id)
    raise
```

Cleanup sends a graceful signal, waits a bounded interval, terminates the
entire containment unit or process tree, retains complete validated artifacts,
quarantines incomplete artifacts, publishes terminal lifecycle revisions, and
releases staging resources and locks. It then re-raises cancellation. Startup
recovery handles process death before shielded cleanup completes. Cancellation
must not leave an active lease indefinitely.

### Structured errors

Every tool returns the same object-root transport envelope. For an expected
domain failure, `ok=false`, `result=null`, `error` contains:

```json
{
  "code": "CAPABILITY_UNAVAILABLE",
  "message": "Native stack capture is unavailable on this platform.",
  "retryable": false,
  "details": {},
  "remediation": [
    "Install py-spy or select Python-only stack capture."
  ],
  "run_id": null
}
```

Structured error codes include:

- `WORKSPACE_NOT_FOUND`;
- `WORKSPACE_INVALID`;
- `CAPABILITY_UNAVAILABLE`;
- `INVALID_CAPTURE_PLAN`;
- `EXECUTION_REFUSED`;
- `PROCESS_FAILED`;
- `PROCESS_TIMEOUT`;
- `PROCESS_CANCELLED`;
- `ARTIFACT_TOO_LARGE`;
- `ARTIFACT_INTEGRITY_FAILED`;
- `ARTIFACT_PARSE_FAILED`;
- `EVIDENCE_SCHEMA_MISMATCH`;
- `COMPARISON_INVALID`;
- `QUERY_BUDGET_EXCEEDED`;
- `STORAGE_QUOTA_EXCEEDED`;
- `WRITE_LOCK_TIMEOUT`;
- `SENSITIVE_ARTIFACT_REFUSED`;
- `REVISION_CONFLICT`;
- `STALE_CURSOR`;
- `INTERNAL_ERROR`.

Tracebacks are logged locally but not returned by default.
