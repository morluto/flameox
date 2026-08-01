# Runtime safety

This document defines the invariants that keep local capture, publication,
analysis, recovery, and retention honest under concurrency, cancellation,
malformed inputs, and agent-controlled requests. These are product contracts,
not implementation suggestions.

Storage formats and snapshot identity are defined in
[Storage and evidence contracts](storage-and-evidence.md). Human and agent
operations are defined in [CLI and MCP boundaries](interfaces.md).

## Concurrency and atomicity

### Operations

Operations fall into three classes:

- `read`: pinned manifests, commit inventories, Parquet, artifact metadata, and
  analytical queries;
- `capture`: long-running external collection into a unique staging directory;
- `commit`: short mutation publishing manifests, artifacts, generations, and a
  new corpus commit;
- `retention`: compaction, trash movement, purge, and repair that may make old
  paths unavailable.

Multiple reads and captures may run concurrently. Commits are serialized.

### Workspace write lock

Portalocker controls `.diagnostics/write.lock` with an exclusive advisory lock.
Both CLI and MCP honor it. Portalocker is used instead of a custom lock because
it exposes cross-platform shared and exclusive lock modes and bounded waits.

The lock covers:

- allocating or transitioning persistent run state;
- publishing an artifact into the content store;
- publishing immutable generation manifests and corpus commits;
- updating or replacing `catalog.duckdb`;
- retention mutations.

It does not cover workload execution, profiling, extraction into staging, or
read-only analysis.

`.diagnostics/catalog.lock` is separate:

- analytical readers hold a shared catalog lock only while a DuckDB connection
  uses `catalog.duckdb`;
- catalog rebuild holds the exclusive catalog lock;
- ordinary evidence publication does not rebuild the catalog; new analyses pin
  the new commit and resolve its explicit inventory;
- workspace commits never wait for long-running analytical readers unless they
  also request catalog replacement.

`.diagnostics/retention.lock` prevents garbage collection or repair from
removing a file used by a pinned reader. Analyses that may open artifacts or
Parquet hold it shared for the operation; GC, purge, and repair hold it
exclusive.

The global lock order is workspace write lock, then retention lock, then
catalog lock when multiple exclusive locks are needed. Readers acquire
retention before shared catalog. Code never acquires the workspace write lock
while holding a read connection. An analysis releases its connection and
retention lock before recording a finding or performing another commit.

### Commit protocol

1. Capture in `.diagnostics/staging/<operation-id>/`.
2. Close producer descriptors; validate, size, and hash payloads.
3. Acquire the write lock, publish content-addressed objects and run-artifact
   registrations, append lifecycle revisions, and release the lock.
4. Extract from immutable objects into a staged generation using checked-in
   Arrow schemas.
5. Validate row budgets, referential integrity, Parquet hashes, and row counts.
6. Write and validate the immutable generation manifest containing exact
   inputs and output files.
7. Acquire the write lock.
8. Pin and recheck current `corpus/HEAD`, run revisions, and supersession
   preconditions.
9. Move all generation files and its manifest to final immutable paths.
10. Write an immutable corpus commit whose inventory is the previous commit
    plus the new/superseding generation.
11. Flush required files and directories under the filesystem durability
    policy.
12. Atomically replace `corpus/HEAD` as the final visibility step.
13. Publish current run projections and release the lock.

Publishing raw artifacts before extraction means an extractor crash cannot lose
a valid, expensive capture. Extraction can be retried without executing the
workload again.

If the process crashes before `HEAD` advances, any new immutable files are
orphans and invisible. Recovery may resume or quarantine staging and reports
unreachable final files without guessing that they are committed. If `HEAD`
advances, the referenced generation is complete by construction.
Content-addressed publication, generation placement, and commit creation are
idempotent.

This protocol assumes a supported local filesystem with advisory locks and
same-volume atomic replace. Initialization validates those assumptions.
Durability requires flushing files before the directory entry or `HEAD`
replacement. Platforms without equivalent semantics are reported as degraded;
their precise crash guarantees are tested and documented rather than assumed.

### Reader behavior

Readers:

- never read from staging;
- pin `corpus/HEAD` once and resolve only files in that commit's immutable
  inventory;
- use read-only DuckDB connections;
- keep the shared retention lock for all file-backed work in the analysis;
- remain unaffected by newer commits appearing between queries;
- include the corpus commit ID and snapshot timestamp in results;
- retry once when a catalog replacement invalidates a connection.

Within the MCP server, a process-local asynchronous read/write coordination
primitive prevents catalog replacement while active readers hold connections.
Cross-process catalog coordination uses the shared/exclusive catalog lock.

## Security and privacy

### Local does not mean harmless

The MCP host may be controlled by an agent. flameox must not turn a diagnostic
tool into unrestricted command, filesystem, or secret access.

The default threat model protects against malformed inputs, accidental
confused-deputy behavior, stale plans, symlink races, resource exhaustion, and
untrusted artifact contents. A named workload is still arbitrary same-user
code. Without an active containment backend it can read the user's accessible
files, mutate the project or `.diagnostics`, access inherited credentials, and
use the network. flameox reports that boundary honestly rather than calling a
named workload safe.

### Command execution

- Accept only argument arrays.
- Never invoke a shell.
- Resolve executables through a documented policy.
- By default, allow MCP execution only for named workloads declared through the
  validated `flameox.toml` configuration path.
- `configure_workload` may create or replace a typed named workload under the
  workspace write lock, atomically updating only `flameox.toml`. It validates the
  complete project and never executes the configured command.
- Do not expose ad-hoc argument-array planning through MCP.
- Bind MCP execution to the single-use plan invariants in [the MCP execution contract](interfaces.md#mcp-server-specification).
- Restrict working directories to configured roots.
- reject NUL bytes and invalid path encodings;
- apply time and artifact-size limits;
- preserve a previewable capture plan;
- treat privileged collectors as disabled by default;
- use Bubblewrap plus cgroup/systemd-scope containment on Linux when policy
  requires it, hiding `.diagnostics`, limiting writable roots, constructing a
  minimal environment, applying CPU/memory/process limits, and controlling
  network access;
- refuse MCP execution when required containment is unavailable; otherwise
  report the process-group fallback as degraded;
- never execute commands embedded in imported artifacts.

### Environment collection

Child environment and recorded environment use separate allowlists. The broker
constructs a minimal child environment rather than forwarding the host
environment. Dangerous loader and interpreter controls such as `LD_PRELOAD`,
`LD_LIBRARY_PATH`, `DYLD_*`, `PYTHONPATH`, debugger init variables, and
credential variables are excluded unless a human allows a specific named
workload to receive them. Recorded metadata uses a second, normally narrower
allowlist. Names indicating tokens, passwords, secrets, keys, credentials, or
cookies are excluded even from broad patterns unless explicitly allowed by
local policy.

### Filesystem access

- Canonicalize and validate every requested root.
- Reject traversal outside configured roots in MCP.
- For imports, open beneath an approved directory without following symlinks,
  require a regular file with an acceptable link count, reject
  devices/FIFOs/sockets and mutable hard-link sources, stream from that same
  descriptor into staging under a size budget, compare `fstat` identity and
  size before and after, then hash the staged destination.
- Do not retain mutable hard links or source references.
- use restrictive permissions for `.diagnostics`;
- mark sensitive artifacts explicitly;
- treat extracted strings, source names, log text, and artifact metadata as
  untrusted data that may contain terminal escapes or prompt injection;
- return no stdout/stderr excerpt over MCP by default; expose only metadata and
  an explicitly bounded local reference.

Workspace, staging, import, file-count, row-count, string-length, nesting,
stdout/stderr, temporary-disk, memory, CPU, and process quotas are enforced
before or while consuming input. Publication checks remaining free space.

### DuckDB

Raw SQL is absent from both flameox interfaces. Internal connections use the
allowlisted-path, locked-configuration, extension, attachment, secret, memory,
thread, and temporary-file restrictions in [the DuckDB contract](storage-and-evidence.md#duckdb-catalog). Parameterized query APIs
select from known views only.

### Core dumps and process memory

Core dumps may contain credentials, source data, user content, and encryption
keys. They require explicit local configuration and are never exposed as MCP
binary resources. Default analyses return only bounded structural metadata.
Future debugger extraction runs in a bounded worker with GDB `-nx`, autoload
and debuginfod disabled, or the LLDB equivalent with user initialization
disabled.

### Network behavior

The flameox control process performs no network calls during capture or analysis.
Capability remediation may print installation documentation but does not fetch
it. Symbol-server or debuginfod access is disabled unless explicitly enabled
in local configuration and invoked through the CLI. Child workloads may use
the network unless an active containment backend denies it; this is displayed
in every capture plan and result.

### Retention and recovery safety

GC and repair use the retention lock described in [the concurrency contract](#concurrency-and-atomicity) and never remove
files beneath active readers. Applying GC first moves eligible material to a
workspace trash area and writes a recovery manifest. A separate explicit purge
removes expired trash. MCP exposes neither operation. Shared content remains
retained while any reachable run registration or pinned corpus commit requires
it.

## Catalog and artifact integrity

### Validation levels

`flameox validate` supports:

- `quick`: manifests, referenced paths, schemas, and sizes;
- `standard`: quick plus Parquet reads and catalog consistency;
- `full`: standard plus rehashing every raw artifact.

MCP uses quick validation at startup and exposes standard validation on request.
Full validation is a CLI operation because it may read many gigabytes.
Validation never mutates evidence. Repair is a separate explicit CLI operation.

### Quarantine

Unparseable or partially written files move to `quarantine` with:

- original staged path;
- reason;
- expected and actual format;
- originating run;
- recovery suggestion.

Quarantine is recoverable and never automatically deleted.

### Garbage collection

No automatic retention policy exists. `gc --dry-run` identifies:

- abandoned staging directories;
- unreferenced content-addressed artifacts;
- generations unreachable from `HEAD`, retained analysis/run-set snapshots, or
  the recovery window;
- superseded rebuildable catalog caches.

`gc --apply` requires explicit CLI invocation and moves eligible objects to
recoverable trash with a manifest. `gc --purge` is a separate explicit,
destructive CLI action after the recovery window. MCP does not expose either.
Corpus commits referenced by an analysis, comparison, finding, or run set are
retention roots and cannot be pruned while that record remains active.

## Observability of flameox itself

flameox writes structured local logs containing:

- operation ID;
- run ID when applicable;
- phase;
- adapter;
- elapsed time;
- bounded error details;
- lock wait duration;
- query name and duration;
- rows and bytes returned.

Logs must not contain raw environment dumps, core contents, or unrestricted
stdout/stderr.

`flameox status` reports:

- workspace and catalog validity;
- storage by artifact kind;
- stale or quarantined runs;
- active capture count;
- last catalog rebuild;
- extractor versions;
- capability warnings.
