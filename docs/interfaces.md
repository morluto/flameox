# CLI and MCP boundaries

## Local client setup

`npx flameox@latest setup` is the supported interactive path for connecting local MCP
clients. The npm package and Python distribution share an exact release
version. npm supplies a thin launcher and the upstream `jsonc-parser` editor;
the Python application service owns discovery, planning, runtime installation,
verification, locking, activation, rollback, and structured results. The same
setup service is available from `flameox setup` for standard JSON and TOML
clients.

Keep `@latest` in the interactive setup command. An unqualified `npx flameox
setup` invocation may reuse an older cached bootstrap and offer stale runtime
choices. The `upgrade` command resolves `flameox@latest` before changing the
managed runtime, even when its own bootstrap was started from an older cache.

`npx flameox upgrade` is the non-interactive npm shorthand for updating the
detected clients to the latest bootstrap package's matching Python runtime. It
keeps the same managed-runtime verification and atomic launcher activation as
setup.

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
the fixed arguments `mcp serve --project-root .`. Setup never passes `--init` or
an external workspace; hosts that need one can add `--workspace PATH` to an
explicit server invocation.
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

### MCP Registry distribution

The official MCP Registry name is `io.github.morluto/flameox`. Its checked-in
`server.json` points to the version-matched `flameox` package on PyPI with the
`uvx` runtime hint, stdio transport, and fixed package arguments
`mcp serve --project-root .`. PyPI is the canonical registry package because it
contains the maintained MCP implementation. The npm package is a setup
bootstrap and must not be advertised as the stdio server.

The package README carries the registry ownership marker. Release planning
updates the Python, npm, and registry metadata versions together. A release
build validates `server.json`; after the exact PyPI artifact is visible, the
tag workflow authenticates `mcp-publisher` with GitHub OIDC and publishes the
same metadata. Registry publication must succeed before the GitHub release is
created.

Registry launch does not imply workspace initialization. The client supplies
its working directory as `.`, and the server retains the normal explicit
`initialize_workspace` boundary.

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
flameox capabilities [--active] [--refresh]
flameox config show
flameox validate
```

`init` creates only local files and never installs collectors. `capabilities`
reports installed tools, versions, supported modes, permissions, and exact
managed setup actions. MCP agents use `start_capability_setup` with an explicit
idempotency key for FlameOx-managed providers, then poll
`get_capability_setup` or cancel with `cancel_capability_setup`. The older
`prepare_adapter` action handles an installed third-party entry point; none of
these actions runs a workload.

### Capture commands

```text
flameox capture plan <adapter> --workload <name> [--parameters '<json>'] [--adapter-options '<json>']
flameox capture run <adapter> --workload <name> [--parameters '<json>'] [--adapter-options '<json>']
flameox import <path> [--kind KIND]
flameox import-nvbench <path>
flameox import-kernel-build <manifest>
```

`capture plan` is side-effect free. `capture run` prints the plan before
execution unless `--json` is used, in which case the plan is part of the result.
Both commands require a named workload and reject `argv`, `cwd`, and trailing
`-- <argv...>` arguments. Every import creates a new import run.
Provider bundle imports select only files declared by the primary native
document; the generic import remains single-file.

### Inference commands

```text
flameox inference list
flameox inference configure-server <name> <managed|existing_local> <model> [options]
flameox inference configure-scenario <name> <server> <aiperf|vllm_bench> [options]
flameox inference plan <scenario> [--timeout SECONDS]
flameox inference run <scenario> [--timeout SECONDS] [--expected-plan-id DIGEST]
flameox inference profile-plan <server> --profiler <torch_profiler|nsight_systems>
flameox inference profile-run <server> <scenario> --profiler <torch_profiler|nsight_systems> \
  --measurement-run-id <unprofiled-run-id> [--expected-plan-id DIGEST]
flameox inference requests <run-id> [--limit N] [--cursor CURSOR]
flameox extract inference-trace <run-id>
flameox extract inference-result <run-id> --provider <aiperf|vllm_bench>
```

The MCP surface provides matching structured configuration, list, plan, run, extraction, and
bounded request-pagination tools. See [Inference replay and profiling](inference.md) for the
evidence and safety boundaries.

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
flameox fault plan <name> --investigation <investigation-id> [--hypothesis <hypothesis-id>] [--parameters '<json>']
flameox fault run [<name> --investigation <investigation-id> | --plan-id <plan-id>] [--parameters '<json>']
flameox fault show <result-id>
```

`flameox.toml` is the project-owned workload declaration. Its strict schema is
`[workloads.<name>]` with an argument array `argv`, project-relative `cwd`,
positive `timeout_seconds`, bounded `parameters` choices, an environment map,
optional `oracle`, `requirements`, `writable_paths`, and `identity`. An exact
plain scalar token such as `{size}` renders from a declared parameter; other
braces in an argument, including JSON and Python literals, remain literal.
Unknown exact tokens such as `{missing}` are rejected. Shell strings, shell
expansion, and trailing `-- <argv...>` forms are unsupported.

MCP's `configure_workload` is the direct agent path for this schema. It accepts
the same typed fields, supports explicit `create` and `replace` operations, and
uses the current configuration digest for replacement. It validates the whole
project before atomically updating `flameox.toml`. The resulting workload is
immediately active and reports `configuration_source = "agent"`; configuration
never executes the workload. A manually authored, valid workload is also
immediately active. There is no separate workload approval or human-check step.

### Analysis commands

```text
flameox analyze hotspots <run-or-artifact>
flameox analyze scaling <experiment-or-run-set>
flameox analyze compare <baseline-run-set> <candidate-run-set>
flameox analyze pytorch <run-or-artifact>
flameox analyze accelerator-launches <run-or-artifact> [--compare-to RUN] [--phase PHASE]
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
flameox trace operation-window <artifact-id> --start NS --end NS
flameox trace transitions <artifact-id> [--trace-id TRACE]
flameox trace gaps <artifact-id>
flameox extract otlp <run-id> [--artifact-id ARTIFACT]
flameox pipelines register <request.json>
flameox pipelines compare <baseline-pipeline-id> <candidate-pipeline-id>
flameox open <artifact-id>
```

Drill-down commands are bounded and use reviewed query families. They report
total and returned counts, truncation, coverage, and stable keyset cursors.
There is no arbitrary SQL or free-form PerfettoSQL command.

`open` prints the appropriate installed viewer command by default.
`flameox open --launch` executes it explicitly. It never launches a browser when
`--json` is active.

Fault experiments are declared under the top-level `fault_experiments` table in
`flameox.toml`. Planning validates the named workload, endpoint parameter,
loopback upstream, typed toxic scenarios, repetition policy, and pinned
Toxiproxy identity. `fault run` measures a no-toxic baseline through the same
proxy before applying each treatment. It records the exact configuration,
resolved ports, tool receipt, logs, process snapshots, containment decisions,
timing, trial, and oracle references. Only the declared endpoint parameter is
rendered into the workload; remote upstreams and arbitrary endpoint injection
are rejected.

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
flameox mcp serve [--init] [--workspace PATH]
flameox mcp inspect [--workspace PATH]
```

`serve` uses stdio exclusively. `--init` performs the additive workspace
initialization before protocol startup and does not configure project workloads.
A network transport is outside the supported product contract.

## MCP server specification

### SDK and transport

The server uses the official Python SDK pinned to stable `mcp==2.0.0` with the
matching `mcp-types==2.0.0` protocol models. Both packages remain exact-pinned
because their wire types are released and consumed as a matched pair.

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
        return _success(result, result.short_summary())
    except DomainError as error:
        return _failure(error)
```

`ToolPayload[T]` is an object-root schema with `schema_version`, `ok`,
`result | null`, and `error | null`. Flameox constructs this payload through
closed Pydantic models before creating a `CallToolResult`; the v2 SDK then
validates the structured content against the annotated `ToolPayload[T]` at the
tool boundary. Domain failures and handler-owned cross-field input failures use
`isError=true` with this typed payload. SDK signature failures (missing fields,
wrong JSON scalar types, or violated field constraints) remain the SDK's native
`isError=true` result with readable text and no `structuredContent`. Unknown
tool names likewise use the SDK's native tool error; malformed JSON-RPC remains
a protocol error. Uncaught domain exceptions are forbidden because their typed
detail would be lost.

The SDK-generated argument model ignores unknown top-level keys in 2.0.0 and
does not expose a registration option that changes that policy. Flameox does
not wrap or subclass the server to imitate stricter SDK behavior. Nested
Flameox request models are closed with `extra="forbid"`, and wire-facing numeric
and boolean parameters use Pydantic strict scalar types so booleans and numeric
strings cannot be silently coerced. Contract tests pin the native unknown-key
behavior so the interface does not claim a stronger contract than the SDK
provides.

Compatibility tests exercise both SDK protocol paths: the legacy initialize
handshake negotiates `2025-11-25`, while modern direct adoption uses
`2026-07-28`. The object-root result schema is valid in both eras even though
the modern protocol permits broader JSON Schema roots.

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
- `configure_workload` is mutating, non-executing, idempotent for an identical
  definition, and closed-world:
  `read_only_hint=False`, `destructive_hint=False`,
  `idempotent_hint=True`, `open_world_hint=False`.
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
| Workspace | `initialize_workspace`, `workspace_status`, `workload_configuration_status`, `configure_workload`, `list_capabilities`, `start_capability_setup`, `get_capability_setup`, `cancel_capability_setup`, `prepare_adapter`, `prepare_workload_dependencies`, `validate_workspace` |
| Capture and import | `plan_capture`, `execute_capture_plan`, `import_artifact`, `import_nvbench`, `import_kernel_build`, `extract_benchmark_samples`, `extract_pyperf`, `extract_python_startup`, `extract_pytest`, `extract_coverage`, `extract_memray`, `extract_perfetto`, `extract_nsight_systems`, `extract_kernel_validation`, `extract_compute_sanitizer`, `extract_nvbench`, `extract_nsight_compute`, `extract_observations` |
| Detached capture | `start_detached_capture`, `get_detached_capture`, `cancel_detached_capture` |
| Discovery | `list_declared_workflows`, `get_declared_workflow`, `list_runs`, `list_findings` |
| Investigations | `create_investigation`, `list_investigations`, `get_investigation`, `record_hypothesis`, `get_hypothesis` |
| Experiments | `plan_experiment`, `run_experiment`, `get_experiment`, `freeze_run_set` |
| Fault experiments | `plan_fault_experiment`, `run_fault_experiment`, `get_fault_experiment` |
| Runs and artifacts | `list_runs`, `get_run`, `list_artifacts`, `get_artifact`, `get_native_viewer_plan` |
| Evidence products | `summarize_evidence`, `register_artifact_pipeline`, `compare_artifact_pipelines` |
| Reductions | `plan_reduction`, `execute_reduction`, `get_reduction` |
| Analysis | `analyze_hotspots`, `analyze_scaling`, `analyze_pytorch`, `analyze_accelerator_launches`, `analyze_memory`, `analyze_execution`, `analyze_failures`, `compare_run_sets`, `record_analysis`, `record_comparison` |
| Lifecycle evidence | `extract_otlp_trace`, `get_operation_window`, `get_operation_transitions`, `find_repeated_operation_sequences`, `get_lifecycle_gaps`, `get_process_snapshot` |
| Drill-down | `get_evidence`, `query_measurements`, `get_frame_callers`, `get_frame_callees`, `get_stack_examples`, `get_trace_window` |
| Findings | `record_finding`, `list_findings`, `get_finding` |

`freeze_run_set` accepts exactly one non-empty membership form: ordered
`run_ids`, or ordered member records. An included member cannot carry an
exclusion reason; an excluded member requires one. The transport parses these
forms into distinct request and member variants before calling the run-set
service.

Reduction planning accepts only the Flameox-native `native_ddmin` engine and a
declared partitioner. `binary_chunks` requires `chunk_size`; all other
partitioners reject that field. Persisted plans and worker requests retain the
same distinction rather than representing chunk size as optional. The
predicate receives the candidate through `FLAMEOX_REDUCTION_CANDIDATE`;
external reducer workloads, wrapper executables, and socket protocols are not
accepted.

Artifact-pipeline stage declarations are discriminated by `status`.
`available` and `cached` stages require `registration_id`; `skipped`,
`unavailable`, and `failed` stages cannot carry a registration identity.
Materialized pipeline stages preserve the same distinction: registered stages
carry artifact identity, length, and sensitivity, while unregistered stages
cannot. Stage comparisons parse into `added`, `missing`, or paired cases, so a
one-sided comparison cannot accidentally claim ordinals or artifacts for the
absent side.
Structural summaries remain a bounded all-or-none extractor name, version,
and summary triple.

The normal first-run sequence is:

```text
initialize_workspace
  → workload_configuration_status
  → configure_workload (when missing)
  → list_declared_workflows (no arguments lists workloads; use kind='experiment' for experiments)
  → get_declared_workflow
  → list_capabilities(adapter='<selected adapter>')
  → start_capability_setup (when the scoped setup_adapters is non-empty)
  → prepare_adapter (when the scoped setup_third_party_adapters is non-empty)
  → prepare_workload_dependencies (when the workload declares missing Python distributions)
  → list_capabilities(adapter='<selected adapter>')
  → plan_capture (preflight_mode='auto')
  → execute_capture_plan
```

Configuration and execution are separate operations. A configuration result
never starts the declared command, and a capture plan never accepts replacement
`argv` or `cwd` values.

`plan_capture` accepts `preflight_mode='auto'` and `capture_mode='auto'` by default. It runs the declared
workload directly and records that no enforced descendant containment was used.
Use `capture_mode='managed'` only when the project policy requires that stronger
guarantee; use `capture_mode='trusted_local'` to request the same direct local
execution explicitly. Planning still does not execute the workload;
`execute_capture_plan` is the explicit execution step.

#### `initialize_workspace`

Additive and idempotent. Initializes only the MCP server's fixed project root,
or the explicit workspace root selected when the server started.
If the server already owns an initialized workspace, the call returns its current
status without replacing the detached-capture manager. It cannot switch to a
different path or configure workloads. Hosts may instead start the server using
`flameox mcp serve --init --workspace PATH`.

#### `workload_configuration_status`

Read-only and bounded. Reports whether the fixed project's `flameox.toml` is
`missing`, `valid`, or `invalid`, returns the relative path, valid configuration
digest, existing workload names, bounded diagnostics, and a machine-readable
`next_tool`. Missing configuration points to `configure_workload`. Invalid
configuration also points to `configure_workload`: a create operation may
repair it when the complete resulting project configuration parses and validates.
The candidate is checked before the atomic replacement; if it is still invalid,
the original file remains untouched and the error carries typed manual-recovery
context plus `workload_configuration_status` as its verification tool.

#### `configure_workload`

Mutating but non-executing. Accepts a typed `name`, `argv`, `cwd`,
`timeout_seconds`, `parameters`, `environment`, `requirements`,
`writable_paths`, `identity`, and `oracle` definition. `operation="create"`
adds a workload or is idempotent for the same definition. `operation="replace"`
requires the current `configuration_id` from
`workload_configuration_status`; a stale digest returns a revision conflict.
The complete resulting `ProjectConfig` is validated, unrelated workloads,
experiments, and comments are preserved, and `flameox.toml` is atomically
updated under the workspace lock. The canonical workload is immediately active.
This tool never runs
the command; call `list_declared_workflows` next, then
`get_declared_workflow`, `list_capabilities`, `start_capability_setup` when
needed, `get_capability_setup`, `list_capabilities` again, `plan_capture`, and finally
`execute_capture_plan`.

#### `workspace_status`

Read-only. Returns workspace identity, validation state, catalog state, storage
usage, active captures, and warnings.

#### `list_capabilities`

Read-only. Returns adapter capabilities, installed versions, required
permissions, unavailable features, remediation, and typed managed setup
actions. Pass the adapter selected for the capture or analysis workflow to
scope mutation guidance to that goal. In a scoped result, `setup_adapters` is
the bounded list of missing providers FlameOx can install and
`setup_third_party_adapters` identifies installed entry points that need exact
identity approval through `prepare_adapter`. Call the relevant setup tool and
then call `list_capabilities(adapter=...)` again.

Omit `adapter` for a complete inventory. The inventory keeps
`available_setup_adapters` and `available_setup_third_party_adapters` as
informational lists, but returns no `next_tool` or prescriptive setup list;
each capability's own setup field remains the source of the exact action. This
prevents broad discovery from turning into an instruction to install unrelated
providers.
Active executable probes are separately requested, bounded, and executed
through the subprocess broker; merely listing capabilities does not run a
project-controlled binary found on `PATH`. The `active_cached` mode uses a
previously completed probe, while `active_refresh` explicitly reruns it.
`plan_capture` uses `preflight_mode='auto'` by default and performs the bounded
active probes needed by the selected plan.

Each capability also reports `provisioning` (`bundled`, `managed_runtime`,
`host`, `third_party_approval`, or `unsupported`) and `setup_verification`.
The latter distinguishes setup that is not applicable, pending, passively
observed, or actively verified. Setup results include a bounded verification
receipt naming the adapters checked and those still unavailable, so an agent can
decide whether to refresh discovery before planning.

#### `start_capability_setup`

Mutating but non-executing. This canonical setup entry point starts a durable
operation and returns its operation ID instead of holding an MCP request open.
It requires an explicit idempotency key. Use `get_capability_setup` to reconnect
and poll, and `cancel_capability_setup` to request cleanup. While setup is
`starting` or `running`, its status includes a
bounded `poll_after_ms` and a `recovery` action with the exact
`get_capability_setup(operation_id=...)` call. Follow that delay and action;
terminal states omit polling guidance and retain their terminal recovery. The
operation accepts only adapter names reported by
`list_capabilities` as managed setup actions: `coverage`, `memray`, `perfetto`,
`py-spy`, `pytest`, and `torch.profiler`. It installs the published FlameOx
extra into the active managed Python runtime and stages the pinned user-space
Trace Processor under the active workspace's `tools/` directory for `perfetto`.
Setup progress uses `validating_request`, `installing_packages`,
`staging_trace_processor`, `verifying`, and `completed`; a staging failure keeps
the staging phase and bounded cause in its durable status. Its receipt includes
the workspace identity, exact request digest, named phase, bounded progress,
item outcomes, cancellation/cleanup state, and next recovery action. It never
runs the declared workload, mutates source, installs arbitrary packages,
changes host permissions, or provisions privileged collectors.

`prepare_workload_dependencies` follows the same side-effect rule and remains a
short setup entry point; its installer runs behind a worker boundary,
and its preflight result names the exact missing distributions and next action.

#### `prepare_adapter`

Mutating but non-executing. Accepts the exact `adapter` and installed
`distribution` pair from `list_capabilities`, hashes the installed distribution
identity, and writes an agent-created approval under the workspace lock. A
version or package-content change invalidates that approval. It does not install
the package, import plugin code, or run a workload; call `list_capabilities`
again before planning.

#### `prepare_workload_dependencies`

Mutating but non-executing. Accepts a declared workload name and installs only
its `requirements.python_distributions` into the active managed Python runtime.
Requirements may use package-index version specifiers but not direct URLs or
local paths. The result includes an active preflight, identifies installed and
remaining distributions, and points to `plan_capture` when ready. Missing
executables, permissions, and privileged host capabilities remain explicit
limitations with their bounded fallback adapters.

#### `list_declared_workflows` and `get_declared_workflow`

Workflow discovery is the authoritative source for current workload names,
declared parameters, requirements, and adapter choices. A workflow detail
returns each requirement's kind, required or optional status, and passive or
active probe requirement. It also returns at most 64 deterministic adapter
options with capability status, planning disposition, required preflight mode,
permission status, supported modes and formats, features, limitations, and
remediation. Options that need a permission check are reported as
`active_probe_required`; discovery remains passive until the caller requests
`list_capabilities(mode="active_refresh")`.

`list_declared_workflows` accepts an argument-free call. It defaults to
`kind="workload"`; pass `kind="experiment"` when the experiment namespace is
needed.

#### `plan_capture`

Read-only. Default inputs are a current `workload_name`, declared scalar
parameter overrides, adapter, and requested features. Returns the exact
resolved plan, expected artifacts, overhead, permissions, limits, containment
state, warnings, request digest, and a short-lived `plan_id`. The ordinary
schema does not advertise `argv` or `cwd`. MCP does not expose ad-hoc capture
planning.

Plan IDs are 256-bit opaque random values held only in MCP-process memory. They
are bound to workspace ID, workload definition hash, resolved executable and
identity, arguments, working directory, child-environment policy and overrides,
adapter/version/capabilities, requested features, validator, source state,
limits, containment decision, and policy generation. Expiry uses monotonic
time. Plans are atomically single-use, consumed before process creation,
bounded in count, invalid after restart, and rejected if any bound input changes.
A request digest is audit evidence, not an authorization boundary.

#### `execute_capture_plan`

Mutating and command-executing. Executes an unexpired `plan_id`, streams
progress, observes cancellation, and returns a completed or failed run record.
Every bound identity, capability, workload definition, and policy input is rechecked
immediately before execution. It never accepts a shell string or replacement
arguments.
Python profiler adapters keep the declared workload interpreter and working
directory, including for `python -m` workloads, while invoking FlameOx's
standalone launcher directly. They do not require FlameOx to be importable from
the workload virtualenv and do not enable `PYTHONPATH` overrides. The
`torch.profiler` whole-entrypoint adapter also accepts declared `python -c`
workloads; it executes the exact inline program through a FlameOx-owned
synthetic filename and records that bound argv in the plan. Undeclared commands
and arbitrary replacement argv remain unsupported.

Workload templates recognize only exact plain scalar tokens such as `{size}`.
Other braces in an argv item, including JSON and Python literals, are passed
through unchanged. An exact `{unknown}` token is still rejected unless that
parameter is declared. Existing doubled-brace escapes continue to collapse to a
single literal brace.

#### `plan_experiment` and `run_experiment`

Planning is read-only over a current named workload and experiment
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

Trial schema version 2 represents an optional numeric parameter as one tagged
`parameter_value`: `{"kind": "integer", "value": 4}` or
`{"kind": "floating", "value": 4.5}`. The tag preserves exact integer
semantics and prevents a trial from carrying both representations.

#### `plan_fault_experiment`, `run_fault_experiment`, and `get_fault_experiment`

Fault planning binds a declared `fault_experiments` workflow and returns an
immutable plan. It does not accept a proxy URL, remote upstream, toxic payload,
or arbitrary workload arguments. Execution starts the pinned Toxiproxy server
through the broker-owned managed-sidecar lease, creates a loopback proxy, runs
the baseline with no toxic, then runs each declared typed treatment through the
normal workload, trial, measurement, and oracle paths. Workload and sidecar
containment decisions are reported separately. Results retain configuration,
ports, version and digest, bounded logs, process observations, cleanup outcome,
timing, and oracle evidence. `get_fault_experiment` reads the immutable result;
it does not rerun the experiment.

#### `create_investigation` and `record_hypothesis`

Mutating, additive structured operations. Hypothesis revision requires
`expected_revision`, an explicit prediction, and a discriminating condition.
Corresponding bounded list/get tools allow agents to resume an investigation
without relying on prior conversation context.

#### `import_artifact`

Mutating. Imports a file under allowed roots, computes its identity, validates
the selected or detected kind, creates a new import run, and optionally extracts
evidence. Use `kind='execution_trace'` for Chrome and Torch profiler traces.
The default `producer='auto'` recognizes common Torch profiler markers and
preserves `producer='torch.profiler'` for analysis routing. If a trace is
ambiguous, pass that producer explicitly. An analysis error that requires Torch
evidence points back to this import step with the required kind and producer.
`source_root='project'` accepts project-local paths, including absolute paths
inside the fixed checkout. `source_root='temp'` accepts absolute or relative
paths beneath the system temporary directory, which covers profiler output
written outside the checkout without allowing arbitrary filesystem reads.

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

Comparison schema version 2 uses the same tagged numeric value for
`baseline_value`, `candidate_value`, and `absolute_change`. Each field is one
integer value, one floating value, or absent; the protocol does not expose
independent integer and floating slots.

#### `record_analysis` and `record_comparison`

Mutating and additive. These execute the same curated deterministic recipes as
the read-only tools, then persist an `AnalysisRecord`, result digest, exact
input corpus commit, typed evidence references, coverage, and limitations.
They do not accept arbitrary SQL or caller-supplied result bodies.

`record_analysis` is a recipe-discriminated request. `hotspots`, `memory`,
`pytorch`, `execution`, and `accelerator_launches` require `input_id`;
`execution` and `accelerator_launches` may also accept `comparison_input_id`,
and only `accelerator_launches` accepts `phase`. `failures` operates on the
pinned corpus population, while `scaling` requires only `experiment_id`.
Fields belonging to another recipe are rejected at the transport boundary.

#### `analyze_pytorch`

Read-only over a Perfetto-extracted trace. Returns operator and accelerator summaries.
For imported Torch Chrome traces, run `extract_perfetto` first. If the trace has not been
normalized, the tool returns the exact `run_id` and recovery action instead of an empty report.

#### `analyze_accelerator_launches`

Read-only over normalized `trace.event` evidence from Perfetto or the maintained Nsight
Systems extractor. It reports direct and graph runtime launches, accelerator kernels,
bounded per-stream idle-gap summaries with device/context/stream identity, and matched
host-to-device correlation coverage. Missing runtime or accelerator tracks produce a
typed partial result. Comparisons are descriptive and do not claim equivalent computation.

#### `analyze_memory`

Read-only over an existing memory artifact or run. Memory-analysis schema
version 2 and `query_measurements` schema version 2 expose each scalar as one
tagged `value`, using the same integer/floating envelope as trials and
comparisons. The public result cannot contain both numeric representations.

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
flameox://pipelines/{pipeline_id}
flameox://schemas/kernel-validation/v1
flameox://schemas/kernel-build/v1
```

Resources return JSON or text summaries. Large native artifacts are represented
by metadata and opaque resource references, never host storage paths and never
injected into model context. Template resources declare
`mime_type="application/json"`, percent-encode identifiers, and use the v2 SDK's
native template-resource `Context` injection to resolve
`ctx.request_context.lifespan_context`. Tools and resources therefore share the
same request-scoped application state without a parallel registry or transport
facade. Resource templates deliberately annotate the SDK's bare `Context`:
parameterizing this Pydantic model causes the SDK's `validate_call` wrapper to
reconstruct it without private request state. The application lifespan type is
recovered with a static protocol cast after injection. Contract tests exercise
resource reads through the real SDK request path so lifecycle changes cannot
silently detach resources from the active workspace.

Mutable workspace and capability views remain tools rather than static
resources because they require current state and explicit recovery semantics.
Tool results should include MCP `ResourceLink` blocks for
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

Detached operations persist the same phase transitions and retain only a bounded
progress history. Percentage values are monotonic when work units are known;
unknown-duration phases report a phase with `completed` and `total` omitted.
Operations using the shared lifecycle record their exact request digest,
workspace identity, idempotency digest, item-level outcomes, cancellation and
cleanup state, and terminal receipt before they are reported complete.
The original idempotency key always reconnects to the same operation, including
after a restart. A retryable failure or cancellation therefore advertises a
`retry_new_operation` recovery with a fresh key; reusing the original key never
starts duplicate side effects.

### Lifecycle matrix

| Operation family | Side effect | Expected duration | Recovery |
| --- | --- | --- | --- |
| Workspace/configuration reads | none | bounded | reread the pinned workspace on stale state |
| Capability setup | package install or tool staging | long | poll status; reconnect with the same key, or use the receipt's fresh key for a new retry |
| Workload dependency setup | declared package install | bounded to long | retry the exact workload name and inspect preflight |
| Import and extraction | copy, external parser, or publication | bounded to long | identify the affected run and required extractor |
| Capture and experiments | external process and/or multiple trials | long | use the durable operation/run identity and poll |
| Analysis and measurement queries | read-only catalog access | bounded | inspect the evidence status and next tool |
| Full validation and reductions | large reads or generated evidence | long | poll the terminal receipt for the declared scope |

An empty array is not evidence of a complete negative result. Analysis and
query results carry an `evidence` object with status `available`, `empty`,
`unavailable`, `partial`, or `unknown`, a stable reason, and exact next-tool
arguments when recovery is possible.

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
