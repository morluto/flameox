# Architecture

Flameox is a local evidence layer for coding agents. It coordinates maintained
measurement tools, preserves native artifacts and provenance, extracts bounded
evidence, and supports reproducible comparisons. It does not replace the tools
that measure runtime behavior or give an agent arbitrary shell, SQL, file, or
network access.

This document owns module and process boundaries. Storage authority is defined
in [Storage and evidence](storage-and-evidence.md); operational guarantees are
defined in [Runtime safety](runtime-safety.md).

## Authority map

```text
flameox.toml
  named workloads, experiments, inference declarations
        │
        ▼
application planning ──► ResolvedExecutable + opaque plan capability
        │                              │
        ▼                              ▼
subprocess broker                SQLite control plane
        │                   plans · operations · runs · revisions
        ▼
native artifacts ──► immutable Parquet generations ──► corpus commit
                                                        │
                                                        ▼
                                              pinned SnapshotHandle
                                                        │
                                                        ▼
                                          DuckDB / task-shaped analysis
```

There are six authoritative boundaries:

1. `ExecutableResolver` resolves a command under an explicit cwd, environment,
   and trust policy. Its `ResolvedExecutable` is consumed by planning,
   execution, identity, and provenance. The broker revalidates it and never
   repeats command discovery.
2. `AuthorizedPlanStore` retains complete execution intent in SQLite. Public
   execution surfaces accept an opaque, expiring, single-use token—not mutable
   plan fields supplied by the caller.
3. `ControlPlane` transactionally owns plans, operations, run and record
   revisions, idempotency, and relationships.
4. The subprocess broker owns child creation, environment construction,
   deadlines, output and resource budgets, observation, and cleanup. The
   artifact-worker harness owns the one staged worker request/response protocol.
5. Native producer formats remain authoritative. Flameox adds schemas only for
   evidence semantics not owned by the producer.
6. Every analysis acquires one `SnapshotHandle` before lookup and resolves all
   evidence through that handle.

Large artifacts and analytical Parquet stay outside SQLite. DuckDB is disposable
analysis state built from one committed corpus inventory.

## Package boundaries

The repository uses a Python `src/` layout:

```text
src/flameox/
├── domain/          immutable contracts, identities, and errors
├── application/     transport-independent use cases
├── storage/         SQLite control state and durable object registration
├── adapters/        maintained producer integrations and extraction
├── analysis/        snapshot-bound recipes and comparisons
├── workers/         isolated child handlers using the shared protocol
├── mcp/             MCP transport and resource projections
├── command_binding.py
├── execution.py     subprocess and managed-sidecar boundary
├── filesystem.py    descriptor-bound trusted-root reads
├── catalog.py       corpus commits, snapshots, and DuckDB projection
└── cli.py           Typer transport
```

Dependencies point inward:

```text
CLI ─┐
     ├─► application ─► domain
MCP ─┘       │
             ├─► storage
             ├─► adapters
             └─► analysis
```

`domain` does not import transports or infrastructure. Storage does not import
CLI, MCP, or adapters. Import-linter enforces these boundaries. CLI and MCP are
thin transports over the same services; transport code does not own business
rules.

## Evidence and reasoning

Flameox preserves three claim levels:

- **observed** — emitted or directly measured by a producer;
- **derived** — deterministically calculated from observed evidence;
- **inferred** — an interpretation that may require another experiment.

A normalized row never replaces its native artifact. Every row carries run,
artifact, generation, and extractor provenance. Missing stacks, symbols,
shapes, samples, or validation remain explicit; adapters do not silently fall
back and call weaker evidence equivalent.

Experimental structure is evidence. Workload identity, treatment assignment,
blocks, repetitions, warm-ups, failed attempts, exclusions, validation, and
stopping rules remain visible. Profiles guide discovery; representative,
predeclared experiments support confirmation.

## Process model

The CLI is short-lived. The stdio MCP server lives with its host client. Both
invoke the same application services and use the same workspace.

All external processes use argument arrays and `shell=False`. A request carries
an absolute executable binding, allowed working roots, a minimal environment,
an absolute deadline, output limits, and optional resource policy. The broker
owns descendants and returns typed termination evidence. Long-lived inference
servers and Toxiproxy are available only through managed leases with bounded
readiness and cleanup.

Artifact-facing native readers execute through `IsolatedWorkerHarness`. The
parent writes a bounded request beneath a unique staging root, launches a known
Python worker through the broker, validates its bounded response, consumes
declared files beneath the same trusted root, and removes the staging root.
Workers do not define another launcher or transport.

Application task lifetime is explicit. Scoped AnyIO task groups own paired work
and cancellation watchers where structured cancellation is required. The
execution substrate retains lower-level asyncio tasks and threads for stream
draining, process observation, and synchronous adapter bridges; those tasks are
joined before the owning operation returns.

## Execution policies

`trusted_local` runs a declared workload directly and records that enforced
descendant containment is absent. Managed execution is opt-in or required by
project policy. On supported Linux hosts it combines Bubblewrap and a
systemd/cgroup scope for filesystem, network, resource, and descendant controls.
Planning refuses when a required guarantee cannot be supplied.

Trust in an executable is separate from containment of the process:

- project-bound commands must resolve beneath approved project roots;
- managed tools must resolve to their recorded managed installation;
- declared host tools may be selected through the request's effective `PATH`
  and are identity-bound;
- an already approved exact path is not searched again.

Canonical targets are checked before authorization, so an in-root symlink does
not authorize an out-of-root executable.

## Storage and snapshot model

The SQLite control plane is a new workspace contract. Initialization creates the
complete schema atomically. A workspace with an incompatible control schema is
rejected; Flameox does not guess how to migrate pre-redesign control files.

Evidence publication is append-only. A new immutable generation and corpus
commit are written before `HEAD` advances. Analysis pins the current commit once
and creates snapshot-local DuckDB views from its inventory. A snapshot cannot
combine an old corpus ID with newer mutable control rows.

Capture and extraction run outside the workspace publication lock. Only short
registration and commit phases serialize. Deleting `catalog.duckdb` never
deletes evidence.

## Maintained infrastructure

Flameox builds on public producer and storage contracts, including Perfetto
Trace Processor, PyArrow/Parquet, DuckDB, pytest-reportlog, pytest-xdist,
pyperf, SciPy, HTTPX, native profiler readers, and provider models. A custom
format or parser belongs only when upstream cannot express Flameox-specific
evidence, safety, or reproducibility semantics.

Runtime dependencies and optional extras are declared in `pyproject.toml` and
pinned in `uv.lock`; those files, not this prose, are the package inventory.
System and privileged tools are detected but never silently installed. Explicit
capability setup may install only allowlisted user-space providers into the
managed runtime and records the result.

## Platform policy

Storage, analysis, CLI, and MCP behavior are platform-neutral. Capture support
is claimed per adapter and feature. Linux is the primary native-capture platform
because its profiler, accelerator, and containment facilities cover the widest
set. macOS and Windows support is reported only where the upstream producer and
Flameox integration have native validation; package installation alone is not
evidence of support.
