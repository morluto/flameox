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
dangerous-variable and credential-name checks, exact executable binding, output ceiling, timeout,
resource observation, process-group cleanup, and descendant cleanup behavior. The target remains a
trusted local process with the operating-system permissions of Flameox; `cwd` is context, not a
sandbox.

## Request-owned work

Capture and analysis run inside the live MCP request. Progress uses the request
context. Cancellation propagates to the broker, which terminates the process
group and settles bounded output readers before unwinding. No operation can be
polled, resumed, or recovered after restart.

The broker shields asynchronous finalization from AnyIO cancellation scopes; callers do not
detach or shield broker work themselves. Worker sessions retain their job directory until their
child has settled, including when request encoding, a heartbeat, or the consuming callback fails.

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

Capture performs compatibility, invocation binding, executable, aggregate scratch, and provenance
admission before allocating request scratch or executing workload and oracle processes. These
checks are part of the capture request rather than a separate plan or preflight lifecycle.

## Input and output bounds

- analysis accepts 1-32 sources and at most 1,000 rows per call;
- result JSON is capped at 256 KiB by default;
- continuations bind request arguments, limits, formats, and input digests;
- capture argv, merged environment, timeout, combined output, and durable
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
