# Runtime safety

Flameox executes declared local workloads and consumes untrusted artifacts. The
fact that it is local does not make either operation harmless. These invariants
apply to CLI and MCP equally.

## Execution authority

Every launch starts with a `ResolvedExecutable` produced under an explicit cwd,
effective environment, and trust policy. It contains the absolute invocation
path, canonical target, origin, authorization decision, and file identity.
Planning stores that binding; execution revalidates it. Only
`command_binding.py` performs executable `PATH` search.

An `ExecutionRequest` also contains:

- an argument array, never a shell command;
- an approved cwd and working roots;
- a minimal environment allowlist and validated overrides;
- timeout, graceful-cleanup, and output budgets;
- optional memory and storage resource policy;
- optional systemd-scope identity.

The subprocess broker is the only process-creation boundary. It owns stream
draining, callbacks, observation, deadline accounting, termination, descendant
cleanup, and the final `ProcessResult`. A timeout includes process creation and
startup callbacks. Cleanup may finish after the deadline so returning a timeout
does not leak a child.

Plans are server-owned SQLite intent. Execution accepts an opaque token and
cannot replace argv, roots, network target, outputs, budgets, or identity with
caller-carried fields. Capabilities expire and are consumed atomically.

## Containment

Trusted-local execution is useful but not containment. Results label it
`uncontained`. A project may require managed execution; on supported Linux hosts
that uses Bubblewrap plus systemd/cgroup facilities to restrict visible files,
writable roots, network, resources, and descendants. Planning refuses if the
required guarantee cannot be established.

Containment status describes the actual workload and validation-oracle launch.
Managed sidecars have their own containment record; Flameox does not imply that
a Toxiproxy or inference-server lease shares the workload namespace.

Privileged collectors are disabled unless explicitly selected and available.
Imported artifacts never authorize embedded commands.

## Environment and process privacy

The child environment and recorded environment have separate allowlists.
Loader, interpreter, debugger-init, and credential controls are rejected even
when broadly allowlisted. Names resembling tokens, passwords, secrets, keys,
credentials, or cookies are excluded from recorded metadata.

Process observations use `(pid, create_time)` identity and publish only bounded
parentage, state, RSS/CPU/thread/FD values, phase, cleanup outcome, and per-field
failures. They omit command lines, environments, cwd, executable paths, open
files, and network addresses. Observation is budgeted and cannot delay required
cleanup.

## Structured lifetime and cancellation

An application operation owns its work and cancellation watcher in one scoped
task group where concurrent control tasks are required. Cancellation propagates
to the subprocess broker, DuckDB connection, publisher, or sidecar that owns the
resource. Cleanup is shielded only for the bounded phase needed to reach a
truthful terminal state, after which the original cancellation is re-raised.

The broker's internal asyncio tasks and synchronous worker bridges are part of
the execution substrate. They must be cancelled or joined before their owner
returns. No task may retain a workspace lock or mutate state after the caller
has received a terminal response.

Long-running agent operations use durable operation or detached-capture records.
An idempotency key reconnects only to the same intent. Reusing it for different
intent is a conflict, not a new run.

## Filesystem boundary

Boundary-sensitive reads use `BoundedFileSystem` beneath explicit trusted roots.
On POSIX it traverses components relative to opened directory descriptors with
no-follow flags. On Windows it rejects reparse points and verifies the final
opened path. The descriptor actually consumed supplies type, size, link count,
and identity.

Imports and worker responses must:

- remain beneath an approved root;
- reject traversal, symlink/reparse components, devices, FIFOs, sockets, and
  unexpected hard links;
- enforce byte, row, file-count, nesting, and string budgets before decoding;
- hash the staged immutable bytes rather than a previously checked pathname;
- treat source names, strings, logs, and metadata as untrusted display content.

The current primitive provides descriptor-bound regular-file reads. Mutation
paths that do not yet consume the same descriptor-relative primitive retain a
documented proof gap and must not claim equivalent race resistance.

## Artifact workers

Native readers for Perfetto, OTLP, Nsight, Compute Sanitizer, and reductions use
one isolated worker harness. The parent creates a unique staging root, writes a
bounded request, launches a known module through the broker, validates a bounded
success/error envelope, then validates any file handoff beneath that staging
root. Child handlers use the shared protocol rather than their own argparse and
response writers.

The harness is isolation from the application process, not a sandbox by itself.
Its execution policy, environment, roots, and resource ceilings remain explicit.

## Network boundary

Ordinary capture, import, extraction, and analysis make no control-process
network requests. Declared workloads may use the network unless containment
denies it.

Network access by Flameox is limited to explicit operations:

- npm setup and runtime upgrade;
- approved capability/provider acquisition;
- managed user-space tool acquisition;
- explicitly enabled symbol or debuginfod access.

These operations are visible, bounded, and separate from workload execution.
Host packages, privileged tools, drivers, and permissions are never installed
implicitly.

Fault experiments are loopback-only. The broker owns the pinned Toxiproxy
process, rejects remote upstreams and arbitrary toxic types, captures bounded
diagnostics, deletes tracked proxies, and terminates the lease. This is not a
general proxy API.

## Concurrency and publication

Capture and extraction use unique staging roots and do not hold the workspace
publication lock during long work. SQLite transactions own control-state
atomicity. Corpus publication briefly serializes generation registration and
the atomic `HEAD` advance. Readers pin a snapshot before lookup and hold the
retention protection required by that snapshot.

DuckDB is read-only analytical state during queries. Each concurrent query uses
its own snapshot-local connection. Cancellation interrupts and joins that
connection. Public interfaces cannot submit SQL, attach databases, load
extensions, create secrets, or select arbitrary files.

## Validation, quarantine, and retention

`flameox validate` has three levels:

- `quick` checks manifests, references, schemas, and sizes;
- `standard` also reads Parquet and checks catalog consistency;
- `full` rehashes native artifacts and Parquet.

Validation never mutates evidence. Invalid or incomplete staged material moves
to quarantine with its source, reason, expected format, originating operation,
and recovery action. `flameox recover --quarantine ID` is explicit.

Garbage collection is a dry run by default. `gc --apply` moves unreachable
objects to recoverable trash and writes a manifest. Restore and permanent purge
both require that manifest; purge additionally requires the recovery window to
expire. MCP exposes no destructive retention operation. Active runs, open
snapshots, findings, analyses, comparisons, run sets, and reachable corpus
commits remain retention roots.

## Local diagnostics

Operational logs use bounded typed fields such as operation ID, run ID, phase,
adapter, elapsed time, lock wait, query name, row count, and a sanitized error
summary. They do not contain unrestricted stdout/stderr, raw environment dumps,
artifact contents, or core memory. Native artifacts and explicitly preserved
bounded child output may still contain sensitive workload data and carry their
own sensitivity classification.
