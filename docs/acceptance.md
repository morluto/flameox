# Acceptance and verification

The acceptance criteria are the completion contract for changes that affect
flameox's product behavior. The evidence map below records the implementation
reviewed on 2026-08-01. A passing ordinary test suite is not enough when the
relevant criterion requires crash, concurrency, containment, protocol, golden,
or scale proof.

## Release acceptance criteria

A release is acceptable when:

1. deleting `catalog.duckdb` and running `flameox catalog rebuild` preserves all
   queryable evidence at the same corpus commit;
2. two identical payloads from different runs occupy one artifact object while
   retaining distinct registration provenance and the maximum sensitivity;
3. crash injection at every publication boundary never makes a partial
   generation visible through `corpus/HEAD`;
4. concurrent captures can execute while commits remain serialized;
5. CLI JSON validates as the domain result and MCP's `result` field contains
   that same model inside a stable success/error envelope;
6. no MCP tool accepts a shell command or unrestricted SQL;
7. cancellation during every awaited phase performs bounded shielded cleanup,
   interrupts DuckDB where applicable, terminates contained descendants, and
   leaves terminal lifecycle revisions;
8. comparison results operate on frozen run sets, preserve all attempted
   trials, identify identity/environment mismatches, and refuse invalid proof;
9. every persisted analysis and finding has typed references to existing
   immutable evidence and the exact pinned corpus commit, while ephemeral
   read-only results carry their pinned commit without acquiring the write
   lock;
10. the three golden investigations can be reproduced from a fresh workspace;
11. profiler and extractor limitations appear in results instead of logs alone;
12. native artifacts can be opened with existing ecosystem viewers;
13. a full workspace integrity check detects altered artifact bytes;
14. ordinary read-only analyses do not acquire the workspace write lock;
15. the flameox control process makes no network requests during operations and
   capture results truthfully report whether child network access was
   contained;
16. every confirmatory optimization result records a primary metric, estimand,
   practical threshold, confidence method, independent unit, paired count,
   source identity, and passing cross-treatment oracle;
17. the real stdio server emits only protocol messages on stdout and passes
   initialize, list, call, resource, progress, structured-error, and
   cancellation contract tests;
18. an agent can move from a bounded hotspot or anomaly result to its callers,
   representative stack, measurement hierarchy, analysis provenance, run,
   artifact registration, and native viewer command without raw SQL;
19. a 100,000-run synthetic corpus meets recorded startup, query, file-count,
   and rebuild budgets after compaction;
20. stale cursors, modified workload definitions, replayed plans, partial source
   identity, incomplete blocks, and failed validation are surfaced explicitly
   rather than silently weakened.
21. failure analysis distinguishes absent, filtered-empty, successful, and
   failed populations, while structured oracle receipts preserve raw bytes and
   produce conservative trial outcomes with bounded lazy drill-down.
22. invalid native collector outputs are quarantined with run and adapter
    provenance, partial timeout artifacts are explicitly marked, and profile
    analysis distinguishes unavailable evidence from zero hotspots;
23. permission-sensitive adapters exercise active readiness through the broker,
    bind that report into capture plans, and recheck it before execution;
24. terminal captures publish bounded runtime-resource summaries, writable-root
    observations, and explicit unavailable metrics without leaking host paths;
25. MCP inspection exposes the first-run configuration and capture workflow,
    bounded workload schemas, recovery actions, and declared workflow
    requirements and adapter options.
26. direct agent configuration is idempotent, digest-checked for replacement,
    preserves unrelated project state, records configuration provenance, and
    rejects removed ad-hoc execution settings without a human execution gate.

## Implementation conformance

The reviewed implementation contains the core local architecture: shared domain
and application services for CLI and MCP, content-addressed native artifacts,
immutable record projections, generation manifests and atomic corpus commits,
Parquet evidence, rebuildable DuckDB snapshots, current capture plans,
experiments, comparisons, bounded evidence queries, structured findings,
native-viewer planning, and the three golden investigation fixtures.

Conformance status uses three terms:

- **Proven** means a representative behavior test exercises the public or
  storage/runtime seam required by the criterion.
- **Partial** means useful implementation and evidence exist, but part of the
  required behavior is absent or lacks representative proof.
- **Open defect** means the current implementation can violate the criterion,
  even if related tests pass.

| Criterion | Status | Representative evidence |
| --- | --- | --- |
| 1 | Proven | Catalog deletion and rebuild preserve queryable evidence at the pinned commit (`tests/evidence/test_publication_and_catalog.py`). |
| 2 | Proven | Artifact content deduplicates while registration provenance and maximum sensitivity remain distinct (`tests/storage/test_artifacts_and_runs.py`, `tests/application/test_artifact_service.py`). |
| 3 | Proven | Crash injection covers staged Parquet, staged manifest, moved evidence, moved manifest, written commit, pre-HEAD, and post-HEAD boundaries (`tests/evidence/test_publication_and_catalog.py`). |
| 4 | Proven | Concurrent captures execute together while corpus publications serialize without losing either run (`tests/application/test_capture_lifecycle.py`). |
| 5 | Proven | CLI JSON and MCP `result` are checked against the same `WorkspaceStatus` domain model (`tests/mcp/test_contracts.py`). |
| 6 | Proven | MCP exposes neither shell execution nor unrestricted SQL, and subprocesses use argument arrays (`tests/mcp/test_contracts.py`, `tests/execution/test_broker.py`). |
| 7 | Proven | Cancellation is exercised at capture phases 1–7, during experiment work, artifact registration, atomic publication, DuckDB materialization and comparison, and subprocess cleanup; the systemd test proves escaped-descendant cleanup when a user manager is available (`tests/application/test_capture_lifecycle.py`, `tests/application/test_experiments.py`, `tests/application/test_analysis_records.py`, `tests/application/test_comparison_compatibility.py`, `tests/execution/test_broker.py`). |
| 8 | Proven | Frozen run sets preserve attempts and reject identity, environment, block, and oracle incompatibilities (`tests/application/test_comparison_compatibility.py`, `tests/application/test_experiments.py`). |
| 9 | Proven | Comparisons and materialized analyses use one explicitly pinned `Snapshot`; persisted analyses and findings validate typed immutable references and record the exact source commit (`tests/application/test_comparison_compatibility.py`, `tests/application/test_analysis_records.py`, `tests/application/test_artifact_service.py`). |
| 10 | Proven | All three fresh-workspace golden investigations produce investigation, hypothesis, analysis, comparison, validation, and finding evidence; reverse scan executes 32K–128K scaling and proves a normally exiting broken treatment is rejected by its oracle (`tests/golden/`). |
| 11 | Proven | Results expose limitations and compatibility uncertainty; fixtures cover Python startup/import costs, pytest/xdist fixture and failure evidence, pyperf, Perfetto, Memray, coverage, synthetic torch evidence, empty/malformed/truncated inputs, recursive stacks, multi-process/thread traces, and newer producer versions (`tests/adapters/`, `tests/application/test_python_evidence_capture.py`, `tests/application/test_capture_lifecycle.py`). |
| 12 | Proven | Every supported native artifact kind dispatches to an ecosystem viewer, and explicit launch runs through the bounded broker (`tests/application/test_viewers.py`, `tests/adapters/test_perfetto_parsing.py`). |
| 13 | Proven | Full integrity verification detects altered artifact bytes (`tests/application/test_integrity.py`). |
| 14 | Proven | Read-only recipes fail the test if they acquire the workspace write lock, comparisons leave HEAD unchanged, and both paths return their pinned commit (`tests/analysis/test_recipe_invariants.py`, `tests/analysis/test_execution_recipes.py`, `tests/application/test_comparison_compatibility.py`). |
| 15 | Proven | A control-process socket trap covers ordinary operations; capture plans report uncontained/degraded/active state, active Bubblewrap hides `.diagnostics`, and active systemd scopes apply CPU, memory, and process limits (`tests/security/test_offline.py`, `tests/application/test_capture_lifecycle.py`). |
| 16 | Proven | Confirmatory comparisons persist the metric, estimand, practical threshold, confidence method and level, independent unit, paired count, source and protocol compatibility, and cross-treatment validation; incompatible or incomplete evidence is invalid or exploratory (`tests/analysis/test_comparison.py`, `tests/application/test_comparison_compatibility.py`, `tests/application/test_experiments.py`). |
| 17 | Proven | The real stdio server covers initialize, list, call, resource, named progress, structured errors, cancellation, and repeated fresh handshakes with the flameox runtime version; every tool has bounded object schemas and explicit annotations (`tests/mcp/test_stdio_transport.py`, `tests/application/test_analysis_records.py`, `tests/application/test_comparison_compatibility.py`). |
| 18 | Proven | The Perfetto integration test follows hotspot → callers/stack → analysis provenance → run → artifact registration → viewer command without raw SQL (`tests/adapters/test_perfetto_provider.py`). |
| 19 | Proven | The gated 10/1K/100K matrix records publication, compaction file count, rebuild, startup plus cohort-comparison query, and serialization budgets; hashing has a separate throughput budget (`tests/performance/test_catalog_scale.py`). |
| 20 | Proven | Public behavior tests cover stale cursors, workload configuration mutation, replayed plans, partial source identity, incomplete blocks, failed validation, lock contention, quota errors, malformed artifacts, and structured CLI/MCP failures (`tests/application/`, `tests/storage/`, `tests/adapters/`, `tests/mcp/test_contracts.py`, `tests/test_cli_smoke.py`). |
| 21 | Proven | Failure-population invariants, strict receipt parsing and lifecycle projection, typed trial persistence, and the fresh-workspace semantic matrix are exercised by `tests/analysis/test_recipe_invariants.py`, `tests/domain/test_models.py`, `tests/application/test_capture_lifecycle.py`, `tests/application/test_experiments.py`, and `tests/golden/test_semantic_matrix.py`. The downstream fixture contract is checked separately by `tests/test_external_receipt_fixture.py`. |
| 22 | Proven | Native-output success, empty, missing, failed, and timeout-partial cases exercise publication gates, quarantine metadata, lifecycle statuses, and unavailable hotspot evidence (`tests/application/test_capture_native_outputs.py`, `tests/application/test_capture_lifecycle.py`). |
| 23 | Proven | Deterministic granted, denied, degraded, cached, refreshed, managed-provider setup, idempotent setup, fallback routing, fixed-command, staging-cleanup, and execution-recheck cases cover capability readiness (`tests/application/test_capabilities.py`, `tests/application/test_capture_planning.py`, `tests/application/test_capture_native_outputs.py`). |
| 24 | Proven | Normal, short-process, storage-policy, and project-relative writable-root observations cover dedicated runtime-resource publication and unavailable metrics (`tests/execution/test_broker.py`, `tests/application/test_capture_containment.py`, `tests/application/test_capture_lifecycle.py`). |
| 25 | Proven | Workload status, typed invalid-configuration recovery context, direct configuration, explicit capability provisioning/setup verification, workflow requirements/options, inspect parity, and the real stdio configure → discover → plan → execute flow are covered by `tests/application/test_workloads.py`, `tests/mcp/test_workflows.py`, `tests/mcp/test_contracts.py`, and `tests/mcp/test_stdio_transport.py`. |
| 26 | Proven | Idempotence, replacement revision conflicts, state/comment preservation, invalid-config non-overwrite, direct configuration provenance, named-only planning, and six deterministic offline agent traces are covered by `tests/application/test_workloads.py`, `tests/storage/test_workspace.py`, `tests/mcp/test_capture_workflows.py`, and `tests/mcp/test_agent_workflows.py`. |

The snapshot, retention, containment, recovery, adapter, analysis,
observability, and protocol defects recorded by the prior conformance review
are closed in this implementation tree. Platform-dependent live collector and
systemd checks remain conditional on those public tools being installed and
permitted; deterministic fixtures cover their normalized contracts when they
are unavailable.

MCP remains a thin transport over proven application services. A criterion is
not promoted to **Proven** merely because a lower-level helper exists or the
ordinary test suite passes; the evidence must exercise the behavior and seam
named by the criterion.

## Testing strategy

### Unit tests

- canonical identity and hashing;
- Pydantic contract validation;
- independent lifecycle transitions and run lease recovery;
- balanced block generation and trial accounting;
- path and command safety;
- comparison compatibility;
- evidence-level rules;
- query parameter construction;
- statistical estimands, bootstrap reproducibility, equivalence decisions, and
  incomplete-pair behavior;
- schema evolution.

### Adapter fixture tests

Each adapter includes small, legally redistributable fixture artifacts. Tests
validate extraction independently of collector availability. Fixtures cover:

- complete and partial symbols;
- malformed and truncated artifacts;
- empty profiles;
- multiple processes and threads;
- recursive and inlined frames;
- unknown producer versions.

Python startup fixtures additionally preserve raw `-X importtime` lines, cache
semantics, package grouping, repeated wall samples, and optional peak RSS.
Pytest fixtures cover serial and two-worker xdist reports, repeated
worker-scoped fixture setup, worker lifecycle, bounded fixture-sidecar recovery
after a forced worker crash, first observed and controller-reported failure
latency, malformed event streams, and a timed-out partial run with explicit
unexecuted tests. Python startup integration also records whether peak RSS came
from POSIX `wait4` resource usage or the portable polling fallback.

### Integration tests

- initialize and rebuild a workspace;
- capture a deterministic local workload;
- cancel and time out a containment unit or degraded process tree;
- deduplicate identical artifact bytes across runs;
- preserve distinct sensitivity and provenance registrations for identical
  bytes;
- run a randomized complete-block experiment and compare frozen run sets;
- delete and rebuild `catalog.duckdb`;
- recover every interrupted commit boundary;
- prove an unpublished generation remains invisible;
- run CLI JSON and MCP calls against the same expected domain result;
- start the real stdio MCP server and perform initialize, list, call, resource
  read, progress, and cancellation operations;
- run deterministic offline agent traces for missing, additive, stale,
  invalid, ad-hoc, and normal configure → discover → plan → execute workflows;
- assert that server stdout contains only valid JSON-RPC messages.

Contract snapshots cover every MCP input schema, output envelope schema,
annotations object, structured success, and structured `isError` result.

### Concurrency tests

- concurrent read-only analyses;
- concurrent captures committing in either order;
- CLI commit while MCP reads;
- two external CLI commits contending for the lock;
- catalog rebuild while a read is active;
- reader pinned to an old commit while new evidence publishes and GC waits;
- cancellation during spawn, capture, validation, extraction, write-lock
  acquisition, DuckDB query, and corpus publication;
- crash injection after every commit-protocol step.

### Security tests

- shell metacharacters remain literal arguments;
- path traversal and symlink escapes fail;
- import symlink-swap, FIFO, device, hard-link, growth, and truncation races
  fail without blocking;
- raw SQL cannot enter MCP;
- external DuckDB access is unavailable through internal queries;
- unexpected DuckDB attachments, extensions, secrets, and catalog objects fail
  validation;
- secrets are redacted from environment and logs;
- sensitive artifacts cannot be read as resources;
- artifact-derived terminal escapes and instruction-shaped text remain quoted
  untrusted data;
- output and artifact budgets are enforced;
- imported files cannot trigger execution;
- changed workload definition hashes and replayed or expired plan IDs fail;
- managed capture refuses MCP execution when its required containment is
  unavailable; the default trusted-local agent path remains executable and
  records the limitation;
- escaped descendants are terminated under the Linux containment backend.

### Performance tests

flameox's own benchmarks cover:

- catalog startup with 10, 1,000, and 100,000 runs;
- file counts and compaction behavior at those scales;
- common cohort and comparison queries;
- artifact hashing throughput;
- Parquet extraction and publication;
- bounded MCP serialization;
- catalog rebuild.

Performance tests must distinguish collector overhead from flameox orchestration
overhead.

### Golden investigations

The acceptance corpus includes three end-to-end investigations:

1. a Python reverse scan whose time grows linearly with sequence length;
2. a configuration interaction where execution observations reveal a disabled
   algorithmic safeguard even when the CPU profile is unremarkable;
3. a retained-memory or repeated-allocation regression.

Each golden investigation must produce a hypothesis, evidence references, a
prediction, discriminating validation, analysis record, and before/after
run-set result. The reverse-scan case measures 32K–128K sequence lengths,
preserves the pyperf worker hierarchy, and demonstrates that the semantic
oracle fails when the implementation is deliberately perturbed. The
configuration-interaction case requires semantic observations rather than
inferring correctness from CPU time.

## Issue 16 and 18–36 acceptance evidence

The 2026-07 contract expansion is checked by observable tests rather than by
the presence of schemas alone:

| Issues | Proven behavior | Primary evidence |
| --- | --- | --- |
| #30, #34 | Closed MCP inputs, explicit modes, discriminated outcomes, and structured recovery actions | `tests/mcp/test_contracts.py` |
| #32 | Compact lifecycle receipts and structural result bounds | `tests/mcp/test_contracts.py`, `tests/application/test_summaries.py` |
| #31, #33, #35 | Declared-workflow discovery, task routing, filtered snapshot cursors, and cohort selection | `tests/application/test_discovery.py`, `tests/mcp/test_contracts.py` |
| #21, #22, #23 | Requirement preflight, contained writable roots, storage reserve enforcement, and descendant resource observation | `tests/application/test_capture_planning.py`, `tests/application/test_capture_containment.py`, `tests/application/test_capture_lifecycle.py`, `tests/execution/test_broker.py` |
| #18, #29, #36 | Module/native-library identity, accelerator/topology identity, and remote worker/lease provenance | `tests/application/test_environment_identity.py`, `tests/application/test_capture_environment.py`, `tests/application/test_comparison_compatibility.py` |
| #19 | Idempotent detached start, reconnect, terminal status, cancellation, timeout, and output-limit recovery | `tests/application/test_detached.py`, `tests/mcp/test_capture_workflows.py` |
| #24 | Approved third-party adapter probe, plan, execution, validation, extraction, cancellation, identity recheck, and artifact quotas | `tests/application/test_third_party_adapters.py` |
| #25, #26 | Bounded multi-factor matrices, exclusions, stable randomization, categorical outcomes, unmatched cells, and unattempted trials | `tests/application/test_experiments.py` |
| #20 | Canonical evidence summaries, proof roles, compatibility limitations, redaction, structural truncation, and safe Markdown rendering | `tests/application/test_summaries.py` |
| #27 | Ordered pipeline lineage, content-addressed reuse, structural comparison, skipped/incompatible stages, first observed divergence, and immutable catalog rows | `tests/application/test_pipelines.py` |
| #28 | Approved reducer/predicate binding, coordinator-owned bounded predicate attempts, contradictory outcomes, final revalidation, malicious candidate rejection, terminal cancellation, immutable artifacts, and cleanup | `tests/application/test_reductions.py` |
| #16 | Visible npm/runtime/wizard handoff and a real PTY reaching the first Python prompt | `npm/test/jsonc-edit.test.cjs`, `tests/test_cli_setup.py`, `tests/cli/test_setup_process.py`, `tests/cli/test_npx_upgrade.py` |
| #63–65 | Structured workload status/configuration, named-only capture planning, no ad-hoc MCP command setting, and deterministic agent workflow traces | `tests/application/test_workloads.py`, `tests/mcp/test_agent_workflows.py`, `tests/mcp/test_capture_workflows.py`, `tests/storage/test_workspace.py` |

Pipeline structural summaries are supplied through the versioned declaration
contract in the first implementation. A future adapter convenience API may
invoke format-specific extractors directly, but core comparison does not infer
format semantics or expose native content.

Reducer predicate brokering currently uses a Unix-domain socket. Linux is the
first-class capture platform, and the same subprocess resource observer,
output limits, timeouts, process-group cancellation, artifact import checks,
and storage reserve policy are reused. A platform-neutral broker and parity
with active Bubblewrap/systemd containment remain required before reduction is
claimed as supported on macOS or Windows. Reducer results state limitations;
they do not claim global minimality or candidate quality.
