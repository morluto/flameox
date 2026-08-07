# Testing flameox

flameox's tests are organized by the behavior they prove, not by the amount of
code in a module. The checked-in [ownership manifest](../tests/ownership.toml)
assigns every test file one semantic owner and one primary lane. Pytest applies
the declared markers during collection, and `tools/test.py` uses the same
manifest for local commands and CI. Each named lane runs its manifest-owned
paths; aggregate lanes (`full`, `optional`, `performance`, and the
`optional-*` provider lanes) are the only commands that intentionally select
the whole test tree. Process, provider, and performance markers describe
requirements; they do not silently move a test into a second lane.

## Fast local workflow

Install the development dependencies once:

```console
uv sync --extra dev
```

See the available lanes and their exact scopes:

```console
uv run python tools/test.py list
uv run python tools/test.py providers
uv run python tools/test.py capabilities  # check managed setup metadata vs extras
```

Useful focused commands are:

```console
uv run python tools/test.py core          # foundational core-owned tests
uv run python tools/test.py storage       # storage, publication, recovery, integrity
uv run python tools/test.py application   # application-owned service composition
uv run python tools/test.py analysis      # recipes and statistical comparisons
uv run python tools/test.py mcp           # in-process MCP contracts
uv run python tools/test.py process       # core subprocess, stdio, and cancellation paths
uv run python tools/test.py adapters      # deterministic adapter parsing
uv run python tools/test.py cli           # CLI and setup behavior
uv run python tools/test.py security      # offline control-process checks
uv run python tools/test.py golden        # representative investigation fixtures
```

The direct `uv run pytest -q` command follows the repository-wide pytest
default in `pyproject.toml`; it excludes process, performance, and optional
provider tests but is not the same thing as the manifest-owned `core` lane.
In particular, direct pytest can include non-optional golden tests. Use
`tools/test.py` when you need an explicit lane, and use `golden` or `process`
when you intentionally want those slower boundaries.
The runner prints the exact pytest command, disk and temporary-directory
telemetry, and writes a JUnit report, log, and command receipt under
`.test-results/`.

Before moving or consolidating tests, run the ownership and collection checks:

```console
uv run python tools/test.py ownership
uv run python tools/test.py collection
uv run python tools/test.py affected --base origin/main
```

The collection check compares normalized node IDs and parametrized case counts
against `tests/collection-baseline.toml`. A move is complete only when the
receipt passes or the baseline is deliberately updated with a replacement
mapping and reviewable behavioral rationale.

## Optional providers and performance

Optional tests are explicit and never fail at import time merely because a
provider is absent. A missing package or executable appears as a classified
skip. Provider lanes can be run individually when the environment is ready:

```console
uv sync --extra dev --extra execution  # coverage.py
uv sync --extra dev --extra memory     # Memray
uv sync --extra dev --extra trace      # Perfetto Python API
uv sync --extra dev --extra cpu        # py-spy package
uv sync --extra dev --extra torch     # PyTorch
uv run python tools/test.py optional-coverage
uv run python tools/test.py optional-memray
uv run python tools/test.py optional-perfetto
uv run python tools/test.py optional-pyspy
uv run python tools/test.py optional-torch
uv run python tools/test.py optional-host  # Bubblewrap, systemd, Cargo, perf
uv run python tools/test.py optional       # all optional cases
uv run python tools/test.py full           # every retained case, including optional/performance
```

The real Toxiproxy loopback lane is opt-in because it starts a pinned external
server binary. Set `FLAMEOX_TOXIPROXY_SERVER` to the verified managed
`toxiproxy-server` path and run:

```console
FLAMEOX_TOXIPROXY_SERVER=/path/to/toxiproxy-server \
  uv run pytest -o addopts='' tests/adapters/test_toxiproxy_integration.py -q \
  -m 'optional and requires_toxiproxy'
```

This lane uses a local HTTP upstream and covers baseline passthrough, latency,
peer reset, timeout/stall, bandwidth, and truncation. It does not contact a
remote upstream. The default suite intentionally deselects these tests.

The shared provider registry checks packages, executables, and the effective
Trace Processor binary. Some capabilities still depend on permission or a
running user manager; those tests report a classified skip when the host cannot
provide them. A skip is not provider evidence; report the unavailable provider
when describing validation results.

Performance acceptance is a separate lane because its timing budgets are
measurements, not ordinary regression assertions:

```console
uv run python tools/test.py performance
```

The command runs the quick scale cases by default. Set
`FLAMEOX_RUN_PERFORMANCE=1` to include the 100,000-run acceptance case; the
scheduled and manually dispatched CI lane enables that flag. Run it on a quiet
machine with the declared Python version and record the generated JUnit and log
files.

## CI selection

CI uses `tools/test.py affected` to select primary lanes from changed test
paths. Changes to source, tooling, ownership metadata, dependency locks,
workflow configuration, or an unknown path conservatively select every
required test and optional-provider lane. A non-optional test change runs the
coverage-owned standard lanes (`core`, `storage`, `application`, `analysis`,
`mcp`, `adapters`, `cli`, and `golden`) once, plus any explicit process/security lane,
while an optional or performance-only change selects its provider or explicit
lane. Documentation changes do not schedule code tests. The planner falls back
to the full matrix when the git base revision is unavailable, so shallow
checkouts and unusual event payloads cannot silently reduce coverage. Standard
lanes emit coverage fragments that the separate coverage gate combines, so the
existing threshold is not applied to a partial suite.

## Lane ownership

| Lane | Owns | Typical evidence |
| --- | --- | --- |
| `core` | foundational domain and release metadata | invariants and compatibility contracts |
| `storage` | workspace, artifacts, publication, recovery, integrity | durability and rebuild behavior |
| `application` | orchestration and compatibility services | service composition and lifecycle |
| `analysis` | recipes and comparisons | estimands, limitations, and pinned inputs |
| `mcp` | in-process protocol contracts | schemas, envelopes, and routing |
| `process` | brokers, stdio, capture, and cancellation | process ownership and cleanup |
| `adapters` | native-format parsing and setup | preserved artifacts and normalized extraction |
| `cli` | CLI, setup, and user-facing JSON | command behavior and recovery guidance |
| `golden` | representative investigation fixtures | end-to-end semantic behavior |
| `optional` | provider-specific integrations | capability-dependent evidence |
| `performance` | scale and throughput acceptance | reproducible budgets and methodology |

Keep mutable workspaces, clients, and process handles function-scoped. Shared
support belongs in a narrowly named module or fixture only when it expresses a
real ownership boundary; there is no broad autouse fixture layer.

## Troubleshooting

`[Errno 122] Disk quota exceeded` is classified as a resource or infrastructure
failure, not as a product assertion failure. Re-run the affected lane with its
`.test-results` receipt, check the reported workspace and temporary-directory
free space, and inspect the JUnit log for the first failing operation. The
runner never retries silently, so the first failure remains visible.

Process and containment failures should be isolated with:

```console
uv run python tools/test.py process
uv run pytest tests/execution/test_broker.py -q -p no:randomly
```

Check for available `systemd-run`, `bwrap`, `cargo`, `perf`, and the local Trace
Processor binary before treating a skip or permission error as product evidence.
Do not increase timeouts or rerun counts to make a resource failure disappear;
capture the lane, host limits, provider inventory, and artifact receipt instead.

## Evidence vocabulary

Unit tests prove one semantic owner in a deterministic process. Integration
tests prove composition across Flameox services. Process tests add a subprocess
or transport boundary and must verify cleanup and terminal state. Optional
tests prove behavior only when their declared provider is available.
Performance tests measure a declared budget and workload; they do not prove
causality or semantic correctness. Golden tests exercise representative
investigations and should be reported separately from ordinary regression
coverage.
