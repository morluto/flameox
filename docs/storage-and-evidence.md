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

Schema `0` initializes; the current exact schema opens; any other schema version
is rejected with a new-workspace instruction. Native artifacts can be imported
through supported import operations.

## Control-plane authority

`control-plane.sqlite3` owns:

- authorized plan intent and single-use capability lifecycle;
- operation current state and immutable revisions;
- idempotency bindings;
- run current state and immutable revisions;
- projection intents that bind exact domain revisions to replayable corpus work;
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

The plan is an expiring authorization receipt, not a second durable owner of run
semantics. Its authorization digest is derived from the finalized typed intent;
there is no parallel request-shaped authorization payload or generic identity
bag. Executable bindings remain typed, and an approved third-party adapter
package has one explicit package-identity field.

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

The run manifest is the durable semantic authority for what was attempted. It
records effective values after defaults, adapter policy, and workload binding,
including the adapter mode, capture bounds, filters, and target-process scope
when those values define the property being measured. Imports and internal runs
represent unavailable semantics explicitly rather than inferring them from
artifact contents.

Generic import preserves declared artifact metadata but does not promote a
producer string into trusted run semantics. A registered import profile may do
so only after validating the immutable staged bytes against its bounded native
format contract. Qualified imports retain `origin="import"` while naming the
validated adapter separately; unavailable capture scope remains explicit. A
failed qualification preserves the immutable artifact on a failed import run
instead of silently falling back to provider-specific semantics.

Each effective adapter option has one persisted owner. Property-defining values
live in the typed capture scope; remaining adapter configuration lives beside
that scope. The complete option mapping used by adapter code is derived from
those disjoint fields rather than stored a second time. Run semantics are
immutable across lifecycle revisions and have a content digest that normalized
generation manifests bind alongside their input run identifiers.

This contract starts evidence schema 2.0. Flameox does not reinterpret 1.x run
rows or manifests through fallback columns; create a new workspace and re-import
native evidence when adopting this clean break.

These run-scoped semantics are not artifacts. Content identity answers which
bytes were produced; it cannot answer why they were produced or which property
the execution attempted to measure. Two runs may therefore reference the same
content-addressed artifact while retaining different modes, bounds, filters, or
target scopes.

Post-run correctness evidence follows the same ownership rule. Registering a
`flameox.kernel-validation.v2` document appends its immutable artifact
registration to the exact succeeded execution run and updates that run's typed
validation status. The operation requires the reviewed run revision and derives
producer, workload, environment, source, and execution identity from authoritative
state; callers cannot submit copied identity fields or create a surrogate import
run. Generic imports remain appropriate for genuinely external validation evidence.
If the producing run owns an artifact pipeline, validation creates a new immutable
pipeline generation that appends the registered evidence; it does not mutate the
provider artifact or duplicate pipeline state in the run manifest. The run manifest
remains authoritative for execution and validation semantics, while the pipeline
records ordered evidence lineage. Capture results return only bounded pipeline
references, and list/show projections provide discovery without copying pipeline
records into transport-specific storage.

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

Artifacts preserve produced evidence, not all Flameox state. They are the right
boundary for provider-native, large, binary, independently reprocessable, or
otherwise durable inputs such as profiles, traces, reports, and preserved logs.
Derived normalized evidence remains in immutable evidence generations rather
than being forced into artifact payloads or run manifests. Flameox does not wrap
or rewrite native bytes to embed run semantics.

Small size alone does not make evidence transient. A small provider report may
still be stored as an artifact when integrity, provenance, or later extraction
matters. Conversely, status, effective capture scope, limitations, pagination,
and truncation metadata belong to typed durable records and their bounded
projections rather than separate artifact payloads.

Artifact metadata includes kind, media type, producer and version, size, digest,
sensitivity, run relationship, and extraction state. An artifact may be
structurally valid yet unsupported by the installed extractor; that state is
not equivalent to an empty extraction.

Sensitive native artifacts are not exposed as MCP binary resources. Agent-facing
artifact resources return bounded metadata and typed evidence references. The
`process_output` and `validation_output` kinds additionally support an explicit
UTF-8 preview operation: callers supply byte and line bounds plus an offset, and
the result reports returned and total bytes, truncation, and the next offset.
The artifact service resolves and verifies the content-addressed object, applies
the maximum sensitivity across all registrations, refuses sensitive or
unsupported content, and never returns its host path. Failed run projections
carry this preview operation as recovery guidance while the immutable artifact
remains the diagnostic authority.

## Evidence publication

Normalized evidence uses explicit Arrow schemas and immutable Parquet files.
Every row can be traced to its run and run-semantic identity, source artifact,
generation, evidence schema, and extractor version. A null input semantic
identity is explicit when a generation targets an evidence run that has not yet
been materialized; it is not inferred from artifact bytes.

Publication follows one commit protocol:

1. pin the input corpus commit;
2. write and close staging files;
3. validate schema, budgets, references, row counts, and hashes;
4. move immutable files into a new generation;
5. write the generation manifest;
6. write a corpus commit whose inventory names every reachable generation;
7. atomically advance `corpus/HEAD`;
8. record the control-plane publication receipt.

Required domain projections use a domain-first transactional outbox. The same
SQLite transaction that creates or appends a run revision inserts an immutable
`projection_intents` row containing the workspace, domain kind and ID, exact
revision and digest, projection schema, expected table set, operation digest,
and a bounded replay recipe. It does not copy normalized result rows into the
control plane.

After that transaction commits, the projector builds Parquet without holding a
SQLite transaction or workspace lock. Publication is idempotent by projection
intent ID. Advancing `corpus/HEAD` and marking the intent published are separate
durable steps, so a crash can leave one of three explicit states:

- `pending`: the exact domain revision is durable and publication can be replayed;
- `published`: a generation and corpus commit are linked to the intent;
- `failed`: a bounded failure is recorded and the replay recipe remains available.

Recovery first reconciles run projections. It validates the immutable source
revision and digest, recognizes an already-published operation by its digest,
and either finalizes the receipt or republishes from authoritative state. A
feature-specific projector may retain a pending recipe—for example, an OTLP
extraction replayable from its immutable native artifact—until that bounded
projector is invoked. Missing application routing is not treated as evidence
corruption.

The core run projection contains the run row, all artifact registrations, and
the available environment and source identities. Adapter measurements,
extractor tables, and other producer evidence are independent immutable
generations rather than additional copies of mutable run state. Run rows record
`run_revision` and `run_manifest_digest`; agent-facing run reads also report the
authoritative revision, projected revision, intent state, and whether the
projection is current. Thus a pending or stale projection is never silently
presented as current domain state.

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
Compaction changes the authoritative corpus inventory. It does not require a
catalog rebuild because each read constructs transient views from its pinned
corpus commit. Catalog metadata records only when the disposable shell was
built; it does not duplicate corpus HEAD or maintain a freshness sentinel.

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
