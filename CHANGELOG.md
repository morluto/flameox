# Changelog

All notable changes to flameox are documented in this file.
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
