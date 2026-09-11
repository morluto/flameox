# Runtime safety

Flameox runs trusted local targets, but it keeps execution and evidence bounds
explicit. Typed argv avoids shell parsing; it does not make an untrusted program
safe.

## Explicit authority

The MCP runtime has no workspace or project root. Capture requires an existing absolute `cwd` and
analysis requires explicit absolute inputs; neither falls back to server startup state. Inputs may
include an expected SHA-256 checked before decoding. Flameox never searches parent directories or
treats local marker directories as discovery state.

Environment overrides are bounded by count and length. The subprocess broker retains its
dangerous-variable and credential-name checks, exact executable binding, selected output ceiling,
explicit workload timeout,
resource observation, process-group cleanup, and descendant cleanup behavior. The target remains a
trusted local process with the operating-system permissions of Flameox; `cwd` is context, not a
sandbox.

## Request-owned work

Capture and analysis run inside the live MCP request. Progress uses the request
context. Cancellation propagates to the broker, which terminates the process
group and settles bounded output readers before unwinding. No operation can be
polled, resumed, or recovered after restart.

After exceptional execution, the broker also closes the asyncio subprocess
transport while the loop remains active. On the tested CPython 3.12 runtime,
`Process` retains that transport privately; leaving interrupted pipes for its
destructor can attempt callbacks on a closed loop. This is an isolated
implementation dependency, not a public `Process.close()` contract. The upstream
[transport contract](https://docs.python.org/3.12/library/asyncio-protocol.html#asyncio.SubprocessTransport.close)
closes pipes and kills a still-running subprocess. The broker already owns
termination and reader settlement before this close. Calling `communicate()` to
collect an unbounded remainder would conflict with bounded diagnostics and could
wait for inherited pipe writers; it is not used as the recovery path.

The broker shields asynchronous finalization from AnyIO cancellation scopes; callers do not
detach or shield broker work themselves. Worker sessions retain their job directory until their
child has settled, including when request encoding, a heartbeat, or the consuming callback fails.

Ordinary subprocess startup checks for request-scope cancellation before launch.
Transport acquisition is shielded from repeated scope cancellation so a created
process reaches broker-owned cleanup; an explicitly configured deadline remains active.
Cancellation before acquisition returns a cleanup receipt with an unreported
termination, not a fabricated exit code.

No-deadline execution passes `None` to the standard
[`asyncio.timeout_at`](https://docs.python.org/3.12/library/asyncio-task.html#asyncio.timeout_at)
context rather than substituting a distant deadline. The observed-process backend
likewise skips deadline comparisons when no budget was selected. Cleanup grace
periods and reader-settlement bounds remain finite; they are not workload budgets.

Writable-root and staging-size baselines are measured before subprocess launch,
not when the asynchronous observer first runs. Otherwise an early workload write
could be mistaken for pre-existing data. Growth enforcement remains sampled;
the baseline is not an atomic filesystem snapshot or a strict disk quota.
Process completion wakes the observer for a final persistent-output check, even
if the process exited before the first periodic sample. Final disk checks do not
invalidate an RSS peak already observed while the process was alive; a process
that exited without any RSS sample still reports that metric as unavailable.

Synchronous adapters running in an AnyIO worker thread return broker execution to the
originating request's event loop and cancellation scope. They do not create a second event loop
for that subprocess. Scope cancellation settles the child before the adapter unwinds and retains
the `ProcessCancelledError` output and cleanup receipt, including peak-RSS execution.
The outer thread wait must remain request-owned: `abandon_on_cancel=True` or raw
`asyncio.Task.cancel()` is not a substitute for cancelling the owning AnyIO scope and joining
cleanup. Arbitrary synchronous reader code is not made interruptible by this bridge.
MCP analysis, preservation, query and resource reads use `AnalysisRuntime.run_in_request`.
It serializes shared-state phases in a worker thread owned by an AnyIO task group;
the group joins the worker before releasing state, including direct caller task
cancellation. Capture admission, finalization, scratch accounting and cleanup use
the same boundary. Capture subprocesses and provider installation do not hold the
state lock; provider preparation acquires it only to publish a verified binding.
Independent capture workloads can therefore overlap while cache mutations remain
coordinated. Synchronous Python callers still own serialization of direct runtime
method calls; those methods are not an independently thread-safe API.

Session scratch has byte and file ceilings. A capture is rejected before its
declared output budget could exhaust remaining capacity. Least-recently-used session analyses and
conversion outputs and materialized evidence are evicted to make room; their `analysis_id` handles
then report `EXPIRED_SESSION_ANALYSIS`. Successful preservation releases capture scratch after the immutable
bundle is published. All remaining scratch disappears at shutdown.

Capture admission reserves bytes and files until the request unwinds. Other captures and evidence
materializations count that reservation even before its output exists; written bytes consume the
reservation rather than being counted twice. Active capture roots cannot be evicted. One capture
scope releases the reservation and removes unretained scratch on every exit, including cancellation
and progress-callback failure.

Evidence requests admit all selected source sizes and file counts before materializing any bundle.
Active analysis inputs stay pinned during subsequent input acquisition and conversion. Evicting a
handle never removes a scratch source still referenced by another cached analysis or active request,
including a request for one member of a cached bundle. Reused materializations are checked before
an analysis-cache hit can return. Failed requests discard newly acquired, unretained scratch artifacts.
Failed analyses use the same bounded handle cache as successful analyses.

Provider projections have a separate least-recently-used cache bounded by entry count and serialized
bytes. Continuation pages reuse the same immutable bounded projection, while every request still
binds the projection to the Flameox implementation and any external decoder executable digest, and
resolves and hashes its sources before and after analysis. Eviction changes latency only.

Capture performs compatibility, invocation binding, executable, aggregate scratch, and provenance
admission before allocating request scratch or executing workload and oracle processes. These
checks are part of the capture request rather than a separate plan or preflight lifecycle.

## Input and output bounds

Capture console retention is independent of its response page. Default diagnostics
drain both streams while retaining bounded in-memory prefixes and counting observed
omissions; console verbosity alone does not terminate that mode. Full-output
collection uses request-owned disk files and the existing combined output ceiling.
Native artifact growth and decoder bounds still apply. Workload time/RSS budgets
are optional fields on `target.budget`; absent values do not inherit decoder
limits. The same explicit budget applies separately to each capture and oracle.
Cancellation and cleanup remain mandatory.

- analysis accepts 1-32 sources and at most 1,000 rows per call;
- result JSON is capped at 256 KiB by default;
- continuations bind request arguments, limits, formats, and input digests;
- capture argv, merged environment, explicitly selected timeout, full output, and durable
  provenance are bounded;
- experiment cases and blocks have explicit maxima.

Invalid digests fail before decoding. Decode and format failures never become
empty successful evidence.

Capture execution results identify the cancellation cause, configured threshold, available
observation, unit, and a bounded recovery hint when the broker terminates a process for timeout,
output, memory, writable-growth, or storage-reserve policy.

## Repository integrity

Preservation re-hashes every native source to catch mutation after analysis.
Publication validates staged files before atomic rename and validates any
concurrent destination before reuse. Readers validate repository versions,
symlink-free content-addressed topology, the complete manifest shape, manifest
identity, data paths, data digests, artifact metadata, and payload digests.

Readers see no bundle or one complete bundle. Corruption is never repaired in
place or hidden by a catalog rebuild. Abandoned staging is removed only when the
recorded owner process is provably absent.

Repository validation treats persisted JSON as boundary input even when its digest is correct.
Nested request and execution structures are parsed before query, materialization, or MCP projection;
downstream code does not discover malformed state through `KeyError` or `AttributeError`.

## Privacy

All work stays local. Manifests preserve explicit paths, provider identity,
effective requests, digests, and execution provenance, so callers must consider
whether those values are sensitive before preservation. Flameox does not upload
artifacts, launch native viewers, or expose payload bytes through MCP resources.
The ordinary MCP evidence resource is a structurally allowlisted projection: full argv,
environment values, working directories, source paths, and scratch paths remain available only
through explicit local canonical-manifest inspection.

The same rule applies to failures. MCP messages use stable, path-free summaries for unexpected I/O
and dependency errors and do not include raw exception strings. Local exception chaining retains the
cause for debugging without making it part of the agent-visible contract.
