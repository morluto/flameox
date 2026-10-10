# Testing

## Choose validation for the change

Prefer installed CLI or real MCP stdio workflows with native evidence, then integration tests
across the affected boundary, then golden examples for format projections and serialization.
Extend an existing workflow when it can prove the regression. Keep narrower tests for distinct
contracts those workflows cannot reliably reach; use fault injection for otherwise inaccessible
failure boundaries. Assert observable outcomes rather than private cache state or internal calls.
When testing a termination cause, keep unrelated limits permissive so they cannot win the race.

For transport changes, check advertised schemas against accepted inputs and emitted results,
including relevant failures, continuation, and recovery. Derive the expected tool catalog from
the operation registry. Documentation-only edits need links and affected claims checked against
their implementation; they do not require collector or performance runs.

## Commands and selections

Install development tools with `uv sync --extra dev`, adding provider extras as needed.
The default selection includes process tests and excludes `optional` and `performance`:

```console
uv run pytest -q
```

Useful selections:

```console
uv run pytest tests/test_runtime.py -q
uv run pytest -m e2e -q
uv run pytest -o addopts='' -m golden -q
uv run pytest -o addopts='' -m requires_memray -q
uv run pytest -o addopts='' -m requires_torch -q
uv run pytest -o addopts='' -m performance --durations=0
```

Provider selections require the matching extras and host tools. `-o addopts=''` removes the default
marker exclusions. Mark tests that spawn or communicate with subprocesses `process`; use `e2e`
for installed CLI or MCP workflows with real processes and native evidence, and `golden` for fixed
artifacts with explicit expected projections. Marker definitions live in
[pyproject.toml](../pyproject.toml).

[CI](../.github/workflows/ci.yml) splits the default suite into deterministic and process-boundary
jobs and reports combined branch coverage without a percentage gate. Optional Memray, PyTorch,
and performance jobs run on schedules and manual dispatch. Style, type, dependency, architecture,
and npm checks are defined there; contributor commands are in
[CONTRIBUTING.md](../CONTRIBUTING.md#development).

## Evidence and limits

The maintained workflows cover these boundaries:

| Surface | Evidence exercised | Limit of that proof |
| --- | --- | --- |
| CLI and MCP | Schemas, inline results, validation, continuation, preservation, rescue, restart, replay | Does not measure agents' tool-selection quality |
| Setup and update | Client formats and profile overrides, terminal consent, pinned launchers, environment forwarding, bounded failures, atomic writes | Installer shims do not prove live dependency resolution or native clients loading registrations |
| Native analysis and capture | Node CPU capture and saved-input replay, Python profiles, inference and benchmark exports, pytest, coverage, streaming traces | Supplied artifacts do not establish every producer release or complete vendor capture lifecycle |
| Processes and storage | Cancellation, descendant cleanup, output and scratch bounds, FIFO substitution, corruption, source-kind changes, immutable publication | Sampled resource observations are not strict OS quotas |
| Comparisons | Native identities, exclusions, oracle outcomes, incompatible and partial coverage | Profiles alone do not establish causality or improvement |
| Scale | Publication and paginated queries across 1,000 real manifests after restart | Timings depend on corpus, workload, environment, semantic oracle, and metric |

Node replay uses a workload marker to prove it does not execute the target again. PyTorch trace
replay needs PyTorch and a local Perfetto reader. Simulated vendor-exporter failures establish
cleanup and retry behavior, not installed xctrace or Nsight compatibility. Setup/update workflows
do not establish project or administrator policy precedence, Windows behavior, or private-index
resolution. Real uvx preparation and native-client loading require separate matching host checks.

Other proof gaps include resource-baseline races and unavailable metrics, cancellation during
process startup, V8 traversal-ceiling overflow, perf demangled/unknown frames, and a managed
dependency reconnect branch. Vendor hardware, permissions, and other platforms require matching
host workflows. A skipped provider test is not provider verification.
