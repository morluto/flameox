# Testing flameox

Tests are owned by behavior. [tests/ownership.toml](../tests/ownership.toml) assigns every test
file one semantic owner and primary lane; `tools/test.py` is the shared local/CI runner. Markers
describe requirements such as processes, providers, and performance, but do not create a second
owner.

The suite has five kinds of proof:

- property and conformance tests for central invariants such as command binding, plan
  capabilities, SQLite revisions, bounded file access, and snapshot pinning;
- behavioral tests through application, CLI, and MCP seams;
- representative SQLite, process, and provider integration tests;
- a small golden set of end-to-end evidence investigations;
- explicit optional, live-provider, platform, and performance lanes.

Provider families live in separate modules and share conformance contracts. A provider module
must not become a general inference, extraction, privacy, or publication test bucket.

## Commands

Install development dependencies, inspect ownership, then run the narrowest relevant lane:

```console
uv sync --extra dev
uv run python tools/test.py list
uv run python tools/test.py ownership
uv run python tools/test.py collection
uv run python tools/test.py storage
uv run python tools/test.py application
uv run python tools/test.py analysis
uv run python tools/test.py mcp
uv run python tools/test.py process
uv run python tools/test.py adapters
uv run python tools/test.py cli
uv run python tools/test.py golden
```

`uv run pytest -q` follows pytest's default marker exclusions; it is not an owned CI lane.
`tools/test.py` prints the exact pytest command and writes logs, JUnit, and a command receipt under
`.test-results/`.

Before moving, splitting, or deleting tests, run `ownership` and `collection`. Collection is
checked against [tests/collection-baseline.toml](../tests/collection-baseline.toml); update the
baseline only when the changed node IDs are intentional and behavior remains accounted for.

## Optional providers and performance

Optional providers fail as classified skips when their package, executable, permission, or host
facility is unavailable. A skip is not validation evidence. Install only the required extra and
run its named `optional-*` lane, or use:

```console
uv run python tools/test.py providers
uv run python tools/test.py capabilities
uv run python tools/test.py optional
uv run python tools/test.py performance
uv run python tools/test.py full
```

Performance checks are measurements with declared workloads and budgets. The scheduled and manual
lanes enable the larger acceptance cases; ordinary regression results do not substitute for them.

## CI contract

Pull requests use the affected-path planner. Source, dependency, test-topology, workflow, or
unknown changes conservatively select the required lanes. Missing git history also selects the
full matrix. Standard lanes publish coverage fragments for one aggregate gate.

`merge_group`, scheduled, and manual runs select the deterministic full matrix. The single
`required` job validates the planner receipt and every selected job result, so branch protection
does not depend on a changing matrix of check names. Live/provider/platform lanes stay explicit
and never masquerade as deterministic evidence.

Keep mutable workspaces and process handles function-scoped. Add shared fixtures only for a real
semantic boundary. Process tests must prove cleanup and terminal state; snapshot tests must pin a
handle before analysis; storage tests must assert SQLite transactions rather than filesystem
projections. Do not make flaky tests pass by adding retries or broad timeouts.
