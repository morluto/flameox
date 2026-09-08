# Python MCP SDK v2 review

Research date: 2026-09-08. This is an integration review, not a dependency upgrade.
Flameox pins and currently runs `mcp==2.0.0` and `mcp-types==2.0.0`; its transport
already uses `MCPServer`, snake-case protocol model fields and typed lifespan.
The [current stable package](https://pypi.org/project/mcp/) is 2.2.0, released
2026-09-07. The rolling `/v2/` documentation can therefore describe additions
absent from the installed 2.0.0 package. API claims below were checked against
installed source where relevant.

## Long-running work, progress and cancellation

The [migration guide](https://py.sdk.modelcontextprotocol.io/v2/migration/)
replaces the former experimental Tasks API with inline tools and progress
reporting. The current Tasks extension remains unimplemented according to the
[official roadmap](https://github.com/modelcontextprotocol/python-sdk/blob/main/ROADMAP.md).
Do not introduce a custom detached job system merely to imitate removed APIs.

Use `Context.report_progress` with increasing absolute values. Omit a total when
it is unknown; do not invent percentages. The client opts in per call, and a
server report is a no-op without that opt-in. Over a real transport, callbacks
can race the final result. See the
[progress guide](https://py.sdk.modelcontextprotocol.io/v2/handlers/progress/).

Installed 2.0.0 accepts a per-call `read_timeout_seconds`, falling back to the
client/session setting; an unset setting supplies no dispatcher request timeout.
Progress does not reset the deadline. A real stdio diagnostic with an async tool
reported six progress updates, then timed out after 0.122 seconds for a selected
0.12-second deadline. The client raised `MCPError` code `-32001`; the server
handler finalized after cancellation, and a subsequent call in the same session
succeeded. This is SDK behavior, not a guarantee about every MCP host.

Consequently, separate client request deadlines, optional workload budgets,
decoder budgets and shutdown grace periods. Progress improves visibility; it
does not make work durable or grant additional execution time. Keep cleanup
owned by the broker and do not swallow cancellation.

## Confirmed event-loop blockage in Flameox (initial behavior)

`analysis_handler` in `src/flameox/mcp/server.py` declares an async handler but
calls synchronous `AnalysisRuntime.analyze` without yielding. The SDK's automatic
worker-thread dispatch for synchronous handlers therefore does not apply.

A real stdio diagnostic on the current checkout measured `list_tools` at 0.037
seconds while idle and 4.586 seconds during reanalysis of the preserved Slime
Memray profile. The analysis itself succeeded. Source and timing evidence agree:
blocking analysis delays unrelated protocol requests.

A follow-up actual stdio cancellation diagnostic used `read_timeout_seconds=0.5`
(a float in SDK 2.0.0). The client timed out at 0.515 seconds, but the Memray
decoder was still observed at 3.824 seconds and the immediately following catalog
request took another 3.515 seconds. The diagnostic was itself captured through
`capture_process_output` and preserved as evidence
`3579d7937c4adcefe5b89659667f9ff973daa7aedc6bd7b2526951815c0c8aad`.
This establishes delayed cancellation handling, not a permanently orphaned
worker: both the diagnostic and server subsequently settled normally.

The blocking chain is `analysis_handler` → `AnalysisRuntime.analyze` →
`MemrayProvider.analyze` → `run_typed_sync_session` → `SubprocessBroker.run_sync`.
When called on an event loop, the last method starts a thread and synchronously
joins it. The async worker harness already has `run_typed_session`, whose broker
call and staged-output lifetime can remain request-owned; a replacement worker
framework is unnecessary.

The same synchronous analysis is called after capture. Evidence preservation,
query and resource handlers also call synchronous I/O directly; their latency
has not yet been measured. A transport-only thread change is insufficient:
`analyze` snapshots and restores `_protected_sources`, computes scratch cleanup
from a shared key-set difference, and mutates the shared LRU caches. Overlapping
analyses could therefore restore another request's protections incorrectly or
clean up another request's new artifacts. Keep these ownership transitions
coordinated when introducing asynchronous decoder work; preserve active capture
reservations, failed-attempt artifacts and shutdown ordering. This source-derived
race risk is not claimed as a reproduced concurrency failure.

Priority: move blocking work off the transport event loop while preserving runtime
state ownership and cancellation cleanup. Do not simply add thread offloading
without checking shared analysis caches, repository state, scratch reservations
and worker teardown. Required tests include concurrent catalog requests, capture
cancellation during analysis, shutdown and preserved replay. The follow-up
integration below implements the coordinated request boundary.

The broker prerequisite is now implemented: synchronous adapters invoked through
an AnyIO worker rejoin their originating request scope with `from_thread.run`.
This propagates scoped cancellation and preserves the broker's output/cleanup
receipt across AnyIO's exception translation. Peak-RSS cancellation also now
retains its receipt instead of re-raising a receipt-free cancellation exception.
Four new process regressions cover both cancellation backends, owning-loop
callbacks and callback failure without a second subprocess invocation. Both
cancellation cases failed on the pre-change broker by reaching the three-second
process deadline instead of cancelling. The full broker suite passes (43 tests).

At that intermediate stage the MCP handler still called analysis synchronously,
and runtime state coordination had not changed. An independent exact-diff review
also confirmed that raw `asyncio.Task.cancel()` on the outer AnyIO thread wait can
return before child cleanup; the integration must preserve request scope ownership
and must not abandon the worker.

Startup cancellation is also covered now. Ordinary broker execution checks for
an already-cancelled scope before launch, then shields transport-handle acquisition
from repeated scope cancellation while retaining the existing startup deadline.
Once acquired, the process goes through broker-owned cleanup. Pre-acquisition
cancellation returns a receipt with unreported termination rather than inventing
an exit status. Three phase-specific process tests cover cancellation before
launch, during protocol setup and during the transport handshake; the first two
failed on the preceding implementation. All 49 execution tests pass, including
the stalled-spawn deadline regression, with no introduced findings in the
independent startup review.

The real Slime diagnostic was rerun with the artificial 0.3-second delay removed.
It observed the decoder, immediately cancelled, then confirmed the retained
cleanup receipt, reaped decoder and zero cached analyses in 0.164 seconds.
Evidence `28ecb35113ebbfb4669fdd33cf14d5308e2fc1f948b0174e75097aea9007e88a`
preserves that diagnostic. This is phase-specific cancellation evidence, not a
before/after performance estimate or proof that the MCP handler is fixed.

## Coordinated MCP integration

MCP analysis, preservation, queries and evidence-resource reads now enter
`AnalysisRuntime.run_in_request`. Its task group owns and joins a worker thread;
a per-runtime lock protects shared cache, input protection and scratch state.
Capture admission and finalization use the same boundary, while capture
subprocesses remain concurrent. Direct task cancellation cancels the owning group
and waits for cleanup rather than abandoning state. Domain exceptions keep their
original typed shape instead of being wrapped in an exception group.

Independent review identified and helped resolve a publication race: provider
preparation now acquires the same state lock only when publishing a verified
collector binding. Installation stays outside the lock. A replacement-during-
admission regression failed without this publication lock and passes with it.

Seven new request-boundary tests cover values/errors, catalog responsiveness,
binding replacement and both scope/task cancellation during standalone analysis
and capture finalization, including child reaping and scratch cleanup. The
existing state and execution suites also pass. Non-cooperative synchronous
reader code is still not forcibly interruptible; the request waits for that code
to settle before releasing shared state.

The real stdio Slime Memray reproduction now returns its catalog request in
0.082 seconds after the 0.530-second client timeout, with the decoder reaped by
that check. The original catalog request waited another 3.515 seconds. Evidence
`24d4b2da813d08823088fe13d4d7cf8dddafc3a599dffcc3052c940da1567d5d`
preserves the integrated diagnostic. A fresh Slime memory capture and replay of
the original preserved profile both succeeded, with explicit bounded coverage.
This establishes the exercised cancellation path, not every provider/platform.

## Schemas, failures and artifacts

Use typed tool inputs and declared output schemas. Flameox already returns
`CallToolResult` with `structured_content` plus a concise text explanation, and
resource links when evidence is preserved. Validate real returned payloads against
their catalog schemas; SDK validation is not a substitute for domain invariants.
See [structured output](https://py.sdk.modelcontextprotocol.io/v2/servers/structured-output/).

Recoverable execution failures belong in tool-error results (`is_error=True`),
with bounded diagnostics and preserved evidence. Protocol errors have a different
audience and lifecycle. Do not return a success-shaped string containing an error,
or leak arbitrary exception text. Flameox's explicit error envelope is compatible
with this separation. See
[error handling](https://py.sdk.modelcontextprotocol.io/v2/servers/handling-errors/).

Resources are explicit reads, not automatic byte streams into a host's context.
Keep large native artifacts local, provide bounded evidence projections and
resource references, and test the host's actual handoff. The SDK does not justify
using a response-size limit as a workload-output limit. See
[resources](https://py.sdk.modelcontextprotocol.io/v2/servers/resources/).

## Lifespan, transport and tests

Keep one typed runtime in the server lifespan and clean it up there. The SDK
supports this directly through
[lifespan context](https://py.sdk.modelcontextprotocol.io/v2/handlers/lifespan/).
Flameox's local stdio product does not need an HTTP transport simply to run longer
captures. Transport shutdown is not a replacement for workload descendant cleanup.

Use the SDK's in-memory `Client(server)` for fast schema/error tests, plus actual
stdio tests for progress ordering, cancellation, EOF, process cleanup and concurrent
requests. Direct in-memory tests cannot establish wire-level behavior. The
[testing guide](https://py.sdk.modelcontextprotocol.io/v2/get-started/testing/)
provides the SDK client pattern.

The [2.2.0 release](https://github.com/modelcontextprotocol/python-sdk/releases/tag/v2.2.0)
adds HTTP session bounds and tightens redirects and output-schema reference
resolution, among other changes. Evaluate an upgrade with the exact transport
tests; do not copy rolling-doc APIs into the 2.0.0 checkout or change the lockfile
as part of a research-only request.

## Recommended order

1. Retain the event-loop responsiveness and cancellation/state-ownership regressions.
2. Implement the disk-backed output and separate-budget design in
   [workload resource policy](workload-resource-policy.md).
3. Add progress during meaningful capture/analysis stages, without fake percentages.
4. Test a current v2 SDK upgrade independently, including catalog schemas and stdio.

No SDK dependency was upgraded. The subsequent runtime integration above changed
transport dispatch without changing the tool catalog or evidence envelope.
