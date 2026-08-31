# Runtime safety

Flameox runs trusted local targets, but it keeps execution and evidence bounds
explicit. Typed argv avoids shell parsing; it does not make an untrusted program
safe.

## Fixed authority

The MCP project root is resolved once at startup. Capture cwd must exist inside
that root after symlink resolution. The runtime never searches parent
directories and never treats `.flameox` or `.diagnostics` as discovery markers.
Explicit analysis inputs are resolved exactly and may include an expected
SHA-256 checked before decoding.

Environment overrides are bounded by count and length. The subprocess broker
retains its dangerous-variable and credential-name checks, exact executable
binding, output ceiling, timeout, resource observation, process-group cleanup,
and descendant cleanup behavior. Effective containment is reported truthfully.

## Request-owned work

Capture and analysis run inside the live MCP request. Progress uses the request
context. Cancellation propagates to the broker, which terminates the process
group and settles bounded output readers before unwinding. No operation can be
polled, resumed, or recovered after restart.

Session scratch has byte and file ceilings. A capture is rejected before its
declared output budget could exhaust remaining capacity. Unpreserved files are
never silently evicted and disappear at shutdown.

## Input and output bounds

- discovery sniffs at most 16 sources and reads only bounded headers;
- inspection accepts 1-16 capability IDs;
- analysis accepts 1-32 sources and at most 1,000 rows per call;
- result JSON is capped at 256 KiB by default;
- continuations bind request arguments, limits, formats, and input digests;
- capture argv, environment, timeout, and combined output are bounded;
- experiment cases and blocks have explicit maxima.

Invalid digests fail before decoding. Decode and format failures never become
empty successful evidence.

## Repository integrity

Preservation re-hashes every native source to catch mutation after analysis.
Publication validates staged files before atomic rename and validates any
concurrent destination before reuse. Readers validate repository versions,
symlink-free content-addressed topology, the complete manifest shape, manifest
identity, data paths, data digests, artifact metadata, and payload digests.

Readers see no bundle or one complete bundle. Corruption is never repaired in
place or hidden by a catalog rebuild. Abandoned staging is removed only when the
recorded owner process is provably absent.

## Privacy

All work stays local. Manifests preserve explicit paths, provider identity,
effective requests, digests, and execution provenance, so callers must consider
whether those values are sensitive before preservation. Flameox does not upload
artifacts, launch native viewers, or expose payload bytes through MCP resources.
