# Storage and evidence

Flameox separates transactional control state, immutable evidence, and
rebuildable analysis state. Treating those as interchangeable is a data-integrity
bug.

```text
control-plane.sqlite3       native artifacts + Parquet + corpus commits
transactional authority                  immutable authority
           │                                      │
           └──────────────────┬───────────────────┘
                              ▼
                     catalog.duckdb
                   rebuildable projection
```

[Architecture](architecture.md) defines module ownership. [Runtime
safety](runtime-safety.md) defines locking, recovery, retention, and filesystem
guarantees.

## Workspace

The default workspace is `<project>/.diagnostics`. An explicit `--workspace`
must identify the workspace itself; `--project-root` discovers one beneath a
project. Flameox never searches above the declared project root.

The durable layout is conceptually:

```text
.diagnostics/
├── workspace.json
├── control-plane.sqlite3
├── artifacts/objects/       content-addressed native bytes
├── evidence/generations/    immutable Parquet and manifests
├── corpus/commits/          immutable inventories
├── corpus/HEAD              atomic current-commit pointer
├── catalog.duckdb           disposable analytical projection
├── staging/                 bounded in-progress operations
├── quarantine/              recoverable invalid/incomplete inputs
├── trash/                   recoverable GC moves
└── tools/                   verified managed user-space tools
```

`workspace.json` binds the workspace to its project root and format. It is not a
second database. Configuration lives in project `flameox.toml`; secrets belong
in the environment, not either file.

The redesigned SQLite control plane does not migrate legacy control-file
workspaces. Schema `0` initializes; the current exact schema opens; any other
schema version is rejected with a new-workspace instruction. Native artifacts
can still be imported into a new workspace through supported import operations.

## Control-plane authority

`control-plane.sqlite3` owns:

- authorized plan intent and single-use capability lifecycle;
- operation current state and immutable revisions;
- idempotency bindings;
- run current state and immutable revisions;
- investigations, hypotheses, experiments, findings, pipelines, and other typed
  control records;
- typed relationships between those records.

SQLite uniqueness, revision predicates, and transactions—not directory scans or
file rename choreography—enforce control-plane invariants. Pydantic validates a
payload before commit and after decode. Significant transitions append a
revision; current rows are indexed projections.

The database uses foreign-key enforcement, a bounded busy timeout, and WAL on
the supported local-filesystem contract. Transactions never remain open while a
profiler, network request, extractor, or large Parquet write runs.

### Authorized plans

The caller-visible `plan_id` is deterministic audit identity. The `plan_token`
is an opaque random capability. They are not interchangeable.

Issuing a plan stores its complete execution intent and expiry server-side.
The token is the lookup key and is not duplicated inside the stored intent.
Inspection reconstructs a preview; consumption atomically makes the capability
unavailable. Execution receives only the token and optional expected digest.

Every value that can change execution or side effects belongs in stored intent:
executable bindings, argv, cwd, environment contract, roots, containment,
budgets, outputs, provider and workload identity, network target, expected
artifacts, and oracle authority.

### Revisions and relationships

A revision append supplies the expected current revision and the exact next
revision. SQLite commits the new immutable revision, updates the current row,
and replaces revision-bound relationships in one transaction. A mismatch is a
typed revision conflict, never last-writer-wins.

Relationships are written transactionally with the source revision. Application
services validate typed targets before commit; the generic relationships table
does not enforce target foreign keys, so target integrity remains an
application-level contract. External immutable artifact and corpus references
are likewise checked before they enter published evidence.

## Runs and provenance

A run is one attempted execution or import. Its manifest records:

- run kind, lifecycle state, timestamps, and typed failure;
- canonical workload definition and instance identity;
- executable, source, environment, tool, adapter, and package identities;
- process termination, resource observation, cleanup, and containment;
- native artifacts and normalized evidence generations;
- oracle and experimental relationships;
- explicit limitations and unavailable fields.

Running, failed, cancelled, timed-out, and completed attempts are all evidence.
Failure finalization does not discard partial artifacts that pass integrity and
privacy checks. A later control revision may add publication receipts or
recovery state, but it does not rewrite the historical execution outcome.

Executable provenance consumes the `ResolvedExecutable` selected during
planning. It records requested token, absolute invocation path, canonical
target, resolution origin, trust decision, stable metadata, and content digest.
No provenance service repeats executable discovery.

Environment identity records an allowlisted, redacted projection rather than a
host dump. Source identity uses repository state and bounded diff/untracked
content digests where configured. Workload identity covers the complete
validated command and parameters, including its executable binding and oracle.

## Native artifacts

The native producer artifact is authoritative. Imports stage bytes beneath an
approved root, read from one descriptor under a size budget, verify identity and
type, hash the staged bytes, then register a content-addressed object. Equal
bytes are stored once while each run retains its own provenance relationship.

Artifact metadata includes kind, media type, producer and version, size, digest,
sensitivity, run relationship, and extraction state. An artifact may be
structurally valid yet unsupported by the installed extractor; that state is
not equivalent to an empty extraction.

Sensitive native artifacts are not exposed as MCP binary resources. Agent-facing
resources return bounded metadata and typed evidence references.

## Evidence publication

Normalized evidence uses explicit Arrow schemas and immutable Parquet files.
Every row can be traced to its run, source artifact, generation, schema version,
and extractor version.

Publication follows one commit protocol:

1. pin the input corpus commit;
2. write and close staging files;
3. validate schema, budgets, references, row counts, and hashes;
4. move immutable files into a new generation;
5. write the generation manifest;
6. write a corpus commit whose inventory names every reachable generation;
7. atomically advance `corpus/HEAD`;
8. record the control-plane publication receipt.

Readers see either the previous commit or the new complete commit. Parquet files
are never edited in place. Compaction publishes replacement files and a new
commit before superseded generations become eligible for garbage collection.

Core normalized families are runs, artifacts, environments, source states,
measurements, frames, frame measurements, observations, analyses, comparisons,
and findings. Adapter-specific tables are allowed when a shared table would
erase producer semantics. Exact fields and versions live in
`src/flameox/evidence/schemas.py`; this document deliberately does not mirror
that registry.

## Snapshot isolation

A corpus commit is an immutable inventory, not merely an ID. `Catalog.pin()`
returns a `SnapshotHandle` that binds the commit and its manifest set. Analysis
services acquire one handle at construction or request entry and use it for
every lookup.

Snapshot-local views resolve run manifests, artifacts, Parquet, comparisons,
and evidence relationships from that inventory. Mutable control-plane rows and
later corpus commits are invisible. Holding a snapshot also protects its files
from garbage collection until the reader releases it.

Passing a snapshot ID while consulting unrelated current stores is forbidden;
that is not snapshot isolation.

## DuckDB catalog

`catalog.duckdb` contains schema metadata and views over the committed Parquet
inventory. It does not own runs, revisions, plans, relationships, or evidence.

`flameox catalog rebuild`:

1. validates the selected corpus commit and manifests;
2. validates every referenced path, digest, schema, and row count;
3. creates a new catalog with allowlisted snapshot-local views;
4. validates the catalog;
5. atomically replaces `catalog.duckdb`.

Existing readers continue on their open snapshot. Corrupt DuckDB state is
rebuildable; corrupt artifacts, SQLite revisions, manifests, or Parquet are not.

Public interfaces expose curated parameterized queries, never raw SQL. Internal
connections deny arbitrary extensions, attachments, secrets, and unapproved
paths; enforce memory, row, string, and time budgets; and use one connection per
concurrent query. Cancellation interrupts the owning connection and joins its
worker before returning.

## Compatibility and evolution

Compatibility is scoped, not global:

- control-plane schema support is exact and rejects incompatible workspaces;
- immutable domain and evidence payloads carry integer schema versions;
- readers support only their declared version window;
- producer adapters qualify explicit native versions and required fields;
- new extraction creates a new generation and never rewrites native bytes;
- comparisons declare which identities must match, may differ, or are unknown.

Unknown producer versions, fields, units, or identities remain unavailable or
incompatible. They are never accepted because a permissive parser happened to
decode them.
