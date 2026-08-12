# Testing

Tests are owned by behavior, not by implementation file. `tests/ownership.toml`
assigns each test module one semantic owner and primary lane. `tools/test.py` is
the shared local and CI runner; markers describe process, provider, platform, or
performance requirements without creating another ownership system.

The suite contains:

- property and conformance tests for central invariants;
- behavior through application, CLI, and MCP seams;
- representative SQLite, process, and provider integration;
- a small set of end-to-end evidence investigations;
- explicit live-provider, platform, and performance lanes.

Provider families live in separate modules and share conformance behavior. A
provider test file must not become an omnibus home for unrelated extraction,
privacy, correlation, and publication tests.

## Commands

```console
uv sync --extra dev
uv run python tools/test.py list
uv run python tools/test.py ownership
uv run python tools/test.py collection
uv run python tools/test.py core
uv run python tools/test.py storage
uv run python tools/test.py application
uv run python tools/test.py analysis
uv run python tools/test.py mcp
uv run python tools/test.py process
uv run python tools/test.py adapters
uv run python tools/test.py cli
uv run python tools/test.py security
uv run python tools/test.py golden
```

Each lane prints its exact pytest command and writes JUnit, log, and command
receipts under `.test-results/`. `pytest -q` uses the configured default marker
selection; it is useful locally but is not a named ownership lane.

Run `ownership` and `collection` before and after moving, splitting, or deleting
tests. Update `tests/collection-baseline.toml` only when node-ID changes are
intentional and every removed behavior is accounted for.

## Optional and live evidence

```console
uv run python tools/test.py providers
uv run python tools/test.py capabilities
uv run python tools/test.py optional
uv run python tools/test.py performance
uv run python tools/test.py full
```

An unavailable package, executable, permission, driver, or host facility is a
classified skip. A skip is not provider evidence. Run the named `optional-*`
lane on a qualified host for claims about that producer or platform.

Performance tests are measurements with declared workloads and budgets. Enable
the larger acceptance cases only in the scheduled/manual lane or an equivalent
explicit local environment.

## CI

Pull requests use the ownership-driven affected planner for early feedback and
required lane selection. Source, dependency, test-topology, workflow, unknown,
or missing-history changes conservatively select broader validation. The planner
is CI routing metadata, not a product contract.

Merge queues, scheduled runs, and manual full runs execute the deterministic
full matrix. The stable `required` job validates the planner receipt and every
selected result so branch protection does not depend on dynamic matrix names.
Provider and platform jobs remain explicit and cannot masquerade as deterministic
coverage when their prerequisite was absent.

## Test design

- Keep mutable workspaces, processes, and handles function-scoped.
- Share a fixture only when it represents a real semantic boundary.
- Assert observable behavior or stable artifacts, not private helper text.
- Process tests prove cleanup and terminal state.
- Storage tests prove transactions, conflicts, and immutable revision history.
- Analysis tests pin one snapshot before lookup.
- Adapter tests preserve the native format and prove explicit compatibility.
- Security tests include hostile paths, bounds, identities, and cancellation.
- Do not hide nondeterminism with retries or broad timeouts.

A passing suite is not evidence for behavior it never exercises. State provider,
platform, race, crash, or performance proof gaps explicitly.
