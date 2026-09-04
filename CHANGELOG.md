# Changelog

All notable changes to flameox are documented in this file.

## [0.2.5] - 2026-09-04

### Bug Fixes

- **evidence:** Enforce trustworthy projection contracts ([#455](https://github.com/morluto/flameox/pull/455))
- **mcp:** Clarify runtime evidence contracts
- **setup:** Migrate project-bound launchers
- **setup:** Require explicit clients without a terminal

### Documentation

- Codify projection correctness invariants

### Features

- **evidence:** Complete typed investigation workflows ([#440](https://github.com/morluto/flameox/pull/440))
- **cli:** Expose experiments and setup version advice

### Refactoring

- **workers:** Remove retired nsight systems rejection worker

### Testing

- **setup:** Remove unused tmp-path fixtures
## [0.2.4] - 2026-09-02

### Bug Fixes

- **mcp:** Include cwd in capture examples
- **release:** Retry MCP registry propagation

### Documentation

- Explain workspace-free evidence runtime

### Features

- **setup:** Configure global MCP clients

### Refactoring

- **runtime:** Simplify bounded capture lifecycle
- **runtime:** Remove workspace binding

### Styling

- Format capture provider module

### Testing

- Remove stale redesign assertions
- Remove unused execution helper
## [0.2.3] - 2026-09-02

### Bug Fixes

- **setup:** Pin generated MCP launchers

### Documentation

- **mcp:** Clarify provider and comparison workflows
- **setup:** Clarify uvx preparation lifecycle (#421) ([#421](https://github.com/morluto/flameox/pull/421))

### Features

- **mcp:** Return provider reconnection action
- **mcp:** Expose typed capability tools (#422) ([#422](https://github.com/morluto/flameox/pull/422))
- **mcp:** Restore explicit provider preparation (#419) ([#419](https://github.com/morluto/flameox/pull/419))
- **runtime:** Unify bounded capture contracts (#417) ([#417](https://github.com/morluto/flameox/pull/417))

### Refactoring

- **setup:** Prepare the returned uvx environment (#420) ([#420](https://github.com/morluto/flameox/pull/420))
- **runtime:** Validate captures atomically (#418) ([#418](https://github.com/morluto/flameox/pull/418))
## [0.2.2] - 2026-09-01

### Bug Fixes

- **runtime:** Close canonical evidence edge cases
- **capture:** Bind providers to owning runtimes
- **setup:** Retain managed provider extras
- **capture:** Preserve native evidence across analysis failures
- **runtime:** Make result continuations truthful
- **cpu:** Retain resolved py-spy samples
- **cpu:** Actively probe perf event permissions
- **memory:** Bound Memray aggregation resources
- **runtime:** Reject blocked environments at validation

### Documentation

- Align runtime ownership and evidence contracts
- Define provider runtime ownership

### Refactoring

- **runtime:** Remove unsupported provider fallbacks

### Styling

- Normalize integrated runtime changes

### Testing

- **cpu:** Isolate host py-spy fallback
## [0.2.1] - 2026-09-02

### Bug Fixes

- **packaging:** Restore DuckDB timezone support in isolated installations

## [0.2.0] - 2026-08-31

### Breaking changes

- Replace workspaces, SQLite control-plane state, persisted DuckDB catalogs,
  action graphs, plan tokens, generic records, detached operations, recovery,
  and GC with a process-lifespan capability runtime.
- Remove initialization, named workloads, `.diagnostics` discovery, migration,
  compatibility aliases, durable provider setup operations/receipts, and
  resumable jobs.
- Reduce MCP to six tools and one immutable evidence resource template. Existing
  native artifacts remain analyzable only through explicit paths and formats.

### Features

- Add typed direct capture with bounded argv, cwd, environment, request limits,
  progress, cancellation, and single/experiment modes. Experiment mode emits
  paired wall-time effects, uncertainty coverage, and practical-threshold
  decisions.
- Retain direct pytest reliability capture through a request-bound bounded event
  plugin, without restoring workspace or reportlog infrastructure.
- Add strict path/evidence source unions, digest-bound continuations, bounded
  streaming preview readers, and a process-local analysis cache.
- Retain prompt-free offline readers for AIPerf, vLLM and SGLang aggregate
  benchmark exports, and Mooncake request traces without managed server leases.
- Add lazy `.flameox` preservation with content-addressed native artifacts,
  canonical evidence manifests, atomic publication, deterministic queries, and
  repository-local Git exclusion.
- Add CLI mirrors for capability discovery/inspection, analysis, capture,
  evidence query/show, and MCP serving/inspection.
- Keep `flameox setup` as a stateless MCP bootstrap and explicit installer for
  selected Python provider extras; system/vendor tools receive external guidance.

### Safety

- Make unpreserved analysis and capture leave no durable Flameox state.
- Reject input mutation before preservation and validate every artifact,
  manifest, and evidence data digest on reuse or read.
- Reject symlinked repository paths and parse the complete versioned manifest
  shape before repository queries or evidence reads trust its fields.
- Report missing providers as discovery state with CLI setup and external
  remediation; discovery and analysis never install them implicitly.

## [0.1.15] - 2026-08-30

### Breaking changes

- Existing workspaces are not migrated. Keep the old `.diagnostics` directory,
  create a new workspace, and import any native artifacts you still need before
  deleting it.

### Bug Fixes

- **capabilities:** Expose GPU adapter workflows
- **faults:** Preserve pre-capture diagnostics
- **evidence:** Qualify process snapshot coverage
- **viewers:** Resolve managed provider runtimes
- **capabilities:** Expose all managed setup adapters
- **pipelines:** Report first observed divergence
- **memray:** Bound and isolate extraction
- **memray:** Separate allocation operations from records
- **memray:** Bind frames to captured source context
- **memray:** Bind extraction to producer-qualified readers
- **types:** Support optional aiperf installations
- **operations:** Bound cancellation cleanup waits
- **providers:** Bind runtime package provenance
- **collectors:** Bind workload capture code
- **nsight:** Disable unattended symbol downloads
- **sanitizer:** Bind bounded capture evidence
- **extraction:** Recover from missing evidence inputs
- **perfetto:** Route extraction to managed setup
- **torch:** Preserve workload launcher semantics
- **nvbench:** Qualify benchmark workloads
- **capture:** Preserve terminal process output
- **analysis:** Bind comparisons to exact evidence
- **faults:** Generate valid proxy names
- **core:** Enforce authority at execution boundaries
- **evidence:** Project scoped sanitizer semantics

### Documentation

- Define the hybrid run evidence boundary
- Clarify evidence authority boundaries
- **contributing:** Improve issue and PR evidence prompts
- Improve GitHub issue and pull request templates
- Remove redundant README disclaimer
- Clarify flameox boundaries
- Add simplified Chinese README

### Features

- Hard-break run semantics and evidence ownership
- **memory:** Add source and lifetime analysis
- **reductions:** Register reduced artifacts
- **memray:** Publish bounded stack navigation
- **analysis:** Make execution evidence retrievable
- **extraction:** Add durable Memray lifecycle
- **setup:** Resume managed asset downloads
- **capture:** Execute reviewed plans from CLI
- **nsight:** Capture declared workloads
- **trace:** Unify bounded evidence windows
- **imports:** Qualify preserved profiler evidence
- **status:** Expose server and workspace versions
- **pipelines:** Expose managed evidence lineage
- **validation:** Link evidence to producing runs
- **artifacts:** Expose bounded output previews
- **evidence:** Make run semantics authoritative
- **setup:** Install Flameox agent guidance

### Performance

- **memray:** Cache normalized frame identities

### Refactoring

- **coverage:** Remove obsolete optional import
- **storage:** Hard-break control-plane format
- **catalog:** Remove stale-state bookkeeping

### Styling

- Apply Ruff formatting

### Testing

- **memory:** Narrow optional allocation count
- **memory:** Narrow retained measurement contract
- **runtime:** Cover sidecar transport shutdown
- **memray:** Bind capture to workload interpreter
- **capabilities:** Prove managed py-spy discovery
- **catalog:** Prove concurrent snapshot isolation
- **pipelines:** Prove public capture handoff

## [0.1.14] - 2026-08-13

### Bug Fixes

- **xctrace:** Admit managed external staging
- **runtime:** Fail closed on abandoned resource state
- **reductions:** Retry failed durable operations
- **providers:** Preserve and contain runtime paths
- **capture:** Preserve timeout evidence reporting
- **types:** Tolerate absent optional OTLP modules
- **ci:** Align deterministic lanes with collected dependencies
- Structural issue batch — bounded reads, oracle coherence, protocol comparison, vLLM percentiles (#293) ([#293](https://github.com/morluto/flameox/pull/293))
- **adapters:** Harden GPU evidence boundaries

### Chores

- Reconcile issues resolved by #295 (#296) ([#296](https://github.com/morluto/flameox/pull/296))
- **ci:** Replace custom test scheduling with pytest lanes

### Documentation

- Align guides with redesigned architecture
- **adapters:** Align rocprofv3 test provenance

### Features

- **adapters:** Add GPU profiling and trace evidence (#139) ([#139](https://github.com/morluto/flameox/pull/139))
- **adapters:** Add GPU benchmark and compiler evidence (#138) ([#138](https://github.com/morluto/flameox/pull/138))
- **adapters:** Add GPU correctness evidence (#137) ([#137](https://github.com/morluto/flameox/pull/137))

### Refactoring

- Center runtime evidence on explicit authorities (#295) ([#295](https://github.com/morluto/flameox/pull/295))
- **core:** Center runtime evidence on explicit authorities
- Establish authoritative execution and control boundaries (#294) ([#294](https://github.com/morluto/flameox/pull/294))
- Make invalid contract states unrepresentable (#152) ([#152](https://github.com/morluto/flameox/pull/152))

### Testing

- **gpu:** Focus evidence lanes on behavioral coverage (#204) ([#204](https://github.com/morluto/flameox/pull/204))

## [0.1.13] - 2026-08-09

### Bug Fixes

- **runtime:** Harden cancellation and recovery (#127) ([#127](https://github.com/morluto/flameox/pull/127))

### Documentation

- Align public descriptions and examples (#130) ([#130](https://github.com/morluto/flameox/pull/130))
- Require changelog-ready pull request titles (#129) ([#129](https://github.com/morluto/flameox/pull/129))
- **readme:** Mention SGLang inference replay
- Clarify supported workflows

### Refactoring

- Make invalid evidence and MCP states unrepresentable (#128) ([#128](https://github.com/morluto/flameox/pull/128))
- **mcp:** Extract resource registrations (#127) ([#127](https://github.com/morluto/flameox/pull/127))

## [0.1.12] - 2026-08-08

### Bug Fixes

- **inference:** Preserve provider schedule contracts (#123) ([#123](https://github.com/morluto/flameox/pull/123))
- **inference:** Harden replay review boundaries

### Continuous Integration

- Register inference test collection

### Documentation

- Clarify runtime evidence workflow
- **adapters:** Clarify maintained tool integration

### Features

- **experiments:** Compare runtime resource metrics (#125) ([#125](https://github.com/morluto/flameox/pull/125))
- **inference:** Add replay and profiling workflows

### Styling

- **inference:** Format executable resolution

### Testing

- Update collection baseline
- Update inference collection receipt

## [0.1.11] - 2026-08-07

### Bug Fixes

- **toxiproxy:** Validate staged release behavior
- **recovery:** Preserve indeterminate process leases
- **runtime:** Harden worker and recovery lifecycles
- **review:** Close evidence and lifecycle correctness gaps
- **capture:** Keep failure finalization bounded
- **faults:** Close transport and reducer review gaps
- **evidence:** Preserve lifecycle and process observations
- **npm:** Resolve latest bootstrap for upgrades
- **capture:** Prevent exception loss on cancellation, bound diagnostics, evict consumed plans
- **storage,capture:** Close TOCTOU races, path traversal, and /proc/stat parsing bugs
- **storage,recovery:** Replace assert with explicit raises, add symlink checks

### Features

- **capabilities:** Stage managed toxiproxy
- **evidence:** Add runtime evidence and transport fault experiments

### Testing

- Replace false-green fixtures with behavioral proof
- Refresh collection preservation receipt
- **reducer:** Constrain UTF-8 property inputs
- **evidence:** Cover lifecycle and process provenance

## [0.1.10] - 2026-08-05

### Bug Fixes

- **capture:** Preserve inline workload argument semantics
- Validate workload and workspace boundaries
- **setup:** Preserve installer environment and nested Windows paths
- **security:** Update cryptography past known CVE
- **storage:** Preserve artifact imports on Windows
- **execution:** Bound brokered workloads and preserve recovery context
- **execution:** Avoid reaping observed children twice
- **cli:** Retain DuckDB timezone runtime dependency
- **storage:** Harden artifact staging and provenance
- Preserve failure evidence across capture boundaries
- **evidence:** Preserve adapter and setup failure diagnostics

### Continuous Integration

- **test:** Route performance changes through the planner
- Make dependency audit classify optional tooling
- **test:** Formalize affected plans and required checks
- **test:** Make lanes ownership-driven and coverage-aware

### Features

- **mcp:** Make setup lifecycle guidance goal-scoped
- **capture:** Support inline Python and literal workload arguments
- **mcp:** Support explicit external workspaces
- **preflight:** Classify unavailable perf and CUDA toolchains

### Refactoring

- **execution:** Centralize observed child execution
- **analysis:** Decompose recipe service and lazy-load barrels
- **execution:** Centralize bounded subprocess setup

### Testing

- Refresh collection baseline
- Refresh collection preservation baseline
- **release:** Verify published npm bootstrap freshness

## [0.1.9] - 2026-08-01

### Bug Fixes

- **test:** Expect failed execution for missing torch steps
- **capture:** Preserve launcher paths and setup cancellation
- **lifecycle:** Close cross-process recovery gaps
- **capture:** Support module workloads in separate environments
- **analysis:** Preserve evidence availability states
- **application:** Close lifecycle review gaps
- **ci:** Refresh formatting and collection baseline
- **capture:** Preserve detached idempotency after restart
- **mcp:** Make profiler setup recovery actionable

### Chores

- **mcp:** Adopt stable 2.0 packages

### Documentation

- Add contributing guide
- Clarify lifecycle retry recovery
- Align lifecycle and discovery guidance

### Features

- Publish flameox MCP registry support
- **accelerator:** Add structured launch evidence workflows
- **mcp:** Standardize lifecycle and evidence contracts

## [0.1.8] - 2026-08-01

### Bug Fixes

- **ci:** Avoid empty release workflow cache
- **setup:** Explain stale npx bootstrap recovery

## [0.1.7] - 2026-08-01

### Bug Fixes

- **storage:** Preserve and safely migrate legacy config
- **storage:** Migrate removed execution setting
- **setup:** Prevent stale bootstrap downgrades

### Documentation

- Document managed agent workflows

### Features

- **mcp:** Complete typed agent workflow contracts
- **import:** Classify profiler traces and bound temp roots
- **capture:** Make agent preflight and containment explicit
- **capabilities:** Add managed provider preparation

### Styling

- Apply Ruff formatting

### Testing

- Refresh collection baseline
- **cli:** Stabilize npx collection receipt
- **cli:** Cover stale npx bootstrap upgrades

## [0.1.6] - 2026-08-01

### Bug Fixes

- **npm:** Refresh pinned runtime metadata before setup

### Features

- **mcp:** Add direct named workload workflow

### Testing

- **cli:** Assert rendered help semantics
- **cli:** Make help assertions color independent

## [0.1.5] - 2026-08-01

### Bug Fixes

- **release:** Authenticate git-cliff metadata requests

### Features

- **npm:** Add end-to-end upgrade command

### Testing

- **npm:** Use packed fixture for upgrade e2e

## [0.1.4] - 2026-08-01

### Bug Fixes

- **ci:** Refresh formatting and collection baseline
- **mcp:** Include server instructions in inspect output
- **capture:** Preserve typed run limitations
- **capabilities:** Probe perf sampling permissions
- **capture:** Reject invalid native profiles

### Features

- **mcp:** Expose workflow requirements and adapter options
- **evidence:** Publish runtime resource summaries

## [0.1.3] - 2026-07-31

### Bug Fixes

- **experiments:** Preserve historical trial lookup semantics
- **capture:** Reconcile cancellation revision races
- **capture:** Preserve terminal state during cancellation
- **storage:** Type workspace error details
- **mcp:** Make workspace initialization idempotent
- **storage:** Normalize workspace initialization failures

### Continuous Integration

- Preserve optional collection and upload diagnostics
- Route explicit test lanes
- Add build performance tracking with uv caching and summary job

### Documentation

- **experiments:** Add semantic matrix and receipt fixtures
- **mcp:** Clarify project initialization flow

### Features

- **experiments:** Preserve structured oracle receipts
- **analysis:** Expose failure population semantics

### Styling

- Apply ruff formatting

### Testing

- Add ownership and lane runner
- Decompose suite by semantic owner

## [0.1.2] - 2026-07-30

### Bug Fixes

- **ci:** Make release branch retries safe
- **ci:** Unblock release PR creation
- **application:** Settle atomic writes after cancellation
- **mcp:** Normalize wire schemas and argument errors
- **ci:** Satisfy format, dead-code, and manifest checks
- **setup:** Show npm runtime handoff
- **setup:** Preselect detected MCP clients
- **test:** Update proc stat tests to mock os.open/os.read API
- Address high-severity audit findings (H1-H4)
- Address medium and low severity audit findings (M1-M7, L1-L2)
- Correct /proc/[pid]/stat parsing and guard unblocked sample keys (#11) ([#11](https://github.com/morluto/flameox/pull/11))
- **storage:** Contain artifact and generation paths (#9) ([#9](https://github.com/morluto/flameox/pull/9))

### Chores

- **ci:** Satisfy format and dead-code checks
- Restore static-check compliance
- Remove gitleaks from CI
- Remove gitleaks from CI (#10) ([#10](https://github.com/morluto/flameox/pull/10))

### Continuous Integration

- Automate trusted PyPI and npm releases (#8) ([#8](https://github.com/morluto/flameox/pull/8))

### Documentation

- Sharpen README product language
- Use flameox mascot as README hero
- Tighten README prose

### Features

- **adapters:** Capture Python runtime evidence
- **mcp:** Improve evidence discovery and recovery
- **mcp:** Expose bounded lifecycle discovery and outcomes
- **core:** Add evidence capture and experiment lifecycle
- **setup:** Verify configured client launchers

### Styling

- **mcp:** Apply ruff formatting

### Testing

- **npm:** Follow package version in bootstrap test
- **mcp:** Make capture lifecycle test host-independent

## [0.1.1] - 2026-07-26

### Features

- Publish the first `flameox` Python package, npm setup bootstrap, and lowercase CLI/import.
