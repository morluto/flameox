# Changelog

All notable changes to flameox are documented in this file.
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
