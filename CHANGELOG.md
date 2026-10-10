# Changelog

All notable changes to flameox are documented in this file.

## [0.2.9] - 2026-10-10

### Bug Fixes

- **setup:** Honor active client profiles and review interactive changes
- **runtime:** Avoid following workload links during scratch accounting
- **inference:** Include prefix reuse hashes in workload identity
- **cpu:** Reject unresolved pstats callers
- **runtime:** Bind cached projections to external reader identity
- **readers:** Bind native parser reads to regular files
- **runtime:** Reject source-kind changes during decoding
- **execution:** Tolerate disappearing scratch files during accounting
- **evidence:** Retain source kind and reject incomplete directory inputs
- **setup:** Verify final launcher environments with bounded installers
- **traces:** Preserve native OTLP encoding and source-table provenance
- **providers:** Validate unresolved weights and report omitted stack frames
- **comparison:** Retain wide identities and check infinite kernel metrics
- **parsers:** Preserve literal JSON structure and native SARIF coordinates
- **inference:** Retain failed records and enforce native input bounds
- **pytest:** Preserve bounded test identities and worker outcomes
- **mcp:** Retain live recovery and redact native decoder diagnostics
- **preview:** Preserve native fields and canonicalize bounded values
- **runtime:** Publish verified projections and tolerate stale cache inputs
- **capture:** Retain admitted identities and completed experiment evidence
- **setup:** Bound configuration reads and verify preparation responses
- **repository:** Validate layouts before initializing evidence storage
- **pstats:** Reject invalid native measurements before projection
- **runtime:** Detach continuation handoffs from cached provenance
- **files:** Bind native reads to regular-file descriptors
- **identity:** Preserve wide native integers without tag collisions
- **triton:** Average finite timings without sum overflow
- **repository:** Reject special metadata files without blocking
- **setup:** Reject null MCP sections consistently
- **benchmarks:** Exclude empty series from mean comparisons
- **kernel:** Validate rows before comparing consistency
- **pytest:** Use one outcome classifier for metrics and rows
- **setup:** Reject ambiguous duplicate JSON keys
- **mcp:** Advertise and enforce selector item bounds
- **nsight:** Classify malformed Parquet tables as decoding failures
- **cpu:** Preserve sample counts and reject weight overflow
- **filesystem:** Reject special files without blocking
- **setup:** Preserve scalar comments and reject ambiguous configuration
- **evidence:** Rescue preserved bundles from unusable store metadata
- **inference:** Preserve exact AIPerf values and share finite comparisons
- **nvbench:** Enforce sample bounds before integer conversion
- **runtime:** Contain recursive artifact and metadata decoding failures
- **evidence:** Retain completed captures when publication fails
- **providers:** Validate native boundaries and preserve exact counts
- **runtime:** Reject recursive continuation tokens safely
- **trace:** Preserve exact epoch nanosecond windows
- **runtime:** Enforce capture admission and scratch ownership
- **providers:** Bound and validate native profile inputs
- **benchmarks:** Preserve semantic series identity

### Chores

- **deps:** Update AIPerf and vulnerable transitive packages

### Documentation

- **adapters:** State collector admission limits accurately
- Remove historical release and performance records
- **testing:** Record audit regression coverage
- Record audited evidence contracts and native regression proofs
- **testing:** Record native-read and handoff regression proofs
- **testing:** Record native edge-case regressions
- **testing:** Record retained comparison and failure proofs
- **testing:** Record retained execution and boundary coverage

### Features

- **npm:** Expose MCP update through the uvx bootstrap
- **cli:** Verify releases before updating MCP client pins
- **mcp:** Return bounded evidence inline and remove resource receipts

### Performance

- **memray:** Filter bounded frames before fetching rows
- **pytest:** Index workers with teardown failures
- **v8:** Reuse frame identities within extraction
- **repository:** Reuse validated inventory traversal
- **runtime:** Reuse model validators and validated results
- **mcp:** Reuse schemas and inspect contracts directly

### Refactoring

- **setup:** Remove legacy launcher compatibility
- **api:** Use task names directly across runtime and transports
- **execution:** Remove unused contracts and normalize path failures
- **memray:** Bound extraction to the requested frame population
- **benchmarks:** Share bounded series comparison
- **setup:** Reuse external provider requirements
- **contracts:** Share source-kind normalization
- **v8:** Share CPU and heap worker orchestration
- **workers:** Keep one request-owned session path
- **execution:** Use one bounded subprocess lifecycle
- **setup:** Reuse unchanged client configuration without rendering
- **setup:** Import environment contracts from their owner

### Testing

- **setup:** Measure bounded diagnostics independently of terminal styling
- **inference:** Prove line limits with valid native control records
- **performance:** Retain scale workloads and measured evidence
- **transports:** Isolate evidence stores and normalize terminal styling
# Changelog

Release notes are published on the [GitHub releases page](https://github.com/morluto/flameox/releases).
