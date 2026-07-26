# Architectural decisions

These choices constrain Flamo's design and implementation. Revisit a settled
choice only with concrete implementation evidence and an explicit update to
this page. Open questions stay here until implementation evidence supports a
decision.

## Settled decisions

The following choices are settled for Flamo:

- Flamo is permanently local.
- Python is the implementation language.
- The MCP SDK is pinned to `mcp==2.0.0b2`.
- stdio is the supported MCP transport.
- DuckDB is the local cross-run analytical engine.
- `catalog.duckdb` is a persistent, rebuildable cache over authoritative
  Parquet and native artifacts; deleting it does not delete evidence.
- Parquet is the normalized evidence format.
- native artifacts remain authoritative.
- corpus commits and generation manifests define atomic analytical snapshots.
- Perfetto Trace Processor handles detailed temporal trace analysis.
- there is no PostgreSQL or SQLite database.
- the DuckDB catalog is rebuildable.
- writes are serialized; captures and reads may run concurrently.
- runs, trials, experiments, and investigations are distinct domain entities.
- content identity is separate from contextual artifact registration.
- confirmatory comparisons operate on frozen run sets and preserve failed
  attempts.
- named workloads require CLI approval and are not described as sandboxed
  without active containment.
- MCP does not expose unrestricted command execution, raw SQL, deletion, or
  sensitive artifact content.
- the CLI and MCP share one domain and application layer; MCP adds a transport
  envelope.
- third-party adapter loading is disabled until the exact
  distribution identity is explicitly approved through the CLI.
- existing profilers and viewers are reused rather than reimplemented.

## Open decisions

These choices should be made only with implementation evidence:

- whether pprof or Perfetto should be preferred when a collector genuinely
  supports both and the required queries are equivalent;
- whether retaining source snapshots is worth the sensitivity cost; dirty
  source identity itself is never optional;
- which additional normalized stack representation is needed for
  population-level cross-profile analysis;
- which macOS and Windows adapters can meet the same evidence contract;
- whether OpenTelemetry Profiles is stable enough to become an accepted import
  or export contract;
- whether detached background captures are necessary. The default remains
  synchronous progress and cancellation;
- which platform-specific Trace Processor provisioning method gives the best
  auditable installation experience. Runtime download remains disallowed.
