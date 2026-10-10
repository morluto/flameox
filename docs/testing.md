# Testing

Prefer evidence in this order:

1. End-to-end workflows through the installed CLI or real MCP stdio transport,
   using real subprocesses and native evidence.
2. Integration workflows across runtime, provider, worker, process, filesystem,
   or repository boundaries.
3. Golden examples with explicit expected projections for native formats and
   stable serialization.

Keep a narrower test only when these workflows cannot establish a behavioral
contract. Avoid tests that mirror methods, count internal calls, restate library
validators, or manufacture successful collector and decoder results. Fault
injection is useful when a real workflow cannot reliably reach a failure
boundary. Assert the observable outcome and keep the test tied to the contract.
When asserting a termination cause, keep unrelated limits permissive so they cannot win the race.

Keep assertions tied to the behavior under investigation. Prefer a public preservation or replay
call over inspecting a private cache, and check the expected result rather than merely accepting
any valid result. Fold regression checks into an existing workflow when it already reaches the
same boundary; avoid repeating summary fields across equivalent fixtures.

There is no unit-test marker. The default pytest selection excludes optional
provider and performance tests, while process tests remain included. CI divides
the default suite into deterministic and process-boundary jobs and reports their
combined branch coverage without a percentage gate. The default local suite is:

```console
uv run pytest -q
```

Useful explicit selections are:

```console
uv run pytest -m e2e -q
uv run pytest -o addopts='' -m golden -q
uv run pytest -o addopts='' -m performance --durations=0
```

The real MCP stdio workflow checks the catalog against the operation registry,
including advertised schemas, examples, and result validation. Derive expected
tools from that registry rather than hard-coding a count or a removed interface.

The installed CLI update workflow uses a uvx shim to run the real release metadata and tool catalog
commands. It verifies preserved provider sets, successful pin changes, repeat-update no-ops,
and unchanged configurations after resolver, version, extra, catalog, timeout, and output-limit
failures. Distinct client index and cache environments reach their preparation processes.
Configuration workflows cover every supported client, explicit rollback, disabled state,
TOML/JSONC comments, previews, newer-release protection, and invalid registrations. The shim does
not prove dependency resolution against a live index; real uvx preparation is a separate manual
check. CI does not establish update behavior on Windows or against private package indexes.
Installed CLI workflows also verify selected client home/config overrides through setup previews,
repeat setup, and update discovery. POSIX terminal workflows check cancellation before dependency
preparation and reject interactive setup with redirected stderr. Inaccessible profile directories
produce CLI diagnostics without tracebacks. These checks do not establish registration loading by
every native client or precedence against project and administrator settings.

A separate real stdio workflow starts with reduced server ceilings and verifies inherited defaults,
pre-execution limit rejection, and exact preservation/replay of lower request limits. Schema
regressions belong in the existing transport workflow, including numeric coercion and query bounds.
The stdio workflow also checks inline JSON against structured results, evidence inspection by ID,
and replay through returned source selectors. Restart, rescue, and provenance redaction workflows
use `inspect_evidence`; MCP resource capabilities are absent. Fault injection verifies that malformed
runtime results produce a typed contract failure before JSON serialization.

## Evidence and limits

The scale workflow publishes 1,000 real manifests and queries every page after restart. Report
performance against the same corpus, workload, environment, semantic oracle, and declared metric;
there is no hardware-independent timing target.

Native workflows cover Node CPU capture and saved-profile reanalysis, cProfile and Python profile
caller metrics, V8 profiles, inference and benchmark exports, pytest, coverage, and streaming trace
formats. The Node workload marker proves replay does not execute the workload again. Optional
PyTorch trace replay requires PyTorch and a local Perfetto reader. Simulated vendor exporter
failures prove cleanup and retry contracts, not installed xctrace or Nsight compatibility.

Process workflows exercise cancellation, descendant settlement, sampled RSS, scratch file churn,
symlink loops, output limits, and typed failures after admitted files are replaced by FIFOs.
Repository workflows cover atomic publication, corruption, immutable identities, source-kind
replacement after decoding and before preservation, rescue, and exact continuation after restart.
Comparison workflows preserve native identities, exclusions, oracle outcomes, and incompatible or
partial coverage; profiles alone do not establish causality or improvement.

Golden projections establish behavior for their supplied artifacts. They do not establish
compatibility with every producer release or complete provider capture lifecycles. Remaining proof
gaps include resource-baseline races and unavailable metrics, cancellation during process startup,
V8 traversal ceiling overflow, perf demangled/unknown frames, a managed dependency reconnect
branch, and live package installation. Vendor hardware, permissions, private indexes, Windows,
and other platforms require matching host workflows. A skipped provider test is not provider
verification.
