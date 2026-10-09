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
combined coverage. The default local suite is:

```console
uv run pytest -q
```

Useful explicit selections are:

```console
uv run pytest -m e2e -q
uv run pytest -o addopts='' -m golden -q
uv run pytest -o addopts='' -m performance --durations=0
```

The real MCP stdio test checks the complete current catalog of 50 direct tools
(26 analysis, 20 capture, and 4 lifecycle), including schemas and examples. The
older `inspect_capabilities`, `analyze`, and `capture_and_analyze` surface has
been replaced. Keep transport tests aligned with the current catalog rather
than encoding an old tool count or wrapper contract.

A separate real stdio workflow starts with reduced server ceilings and verifies inherited defaults,
pre-execution limit rejection, and exact preservation/replay of lower request limits. Schema
regressions belong in the existing transport workflow, including numeric coercion and query bounds.

## Evidence and limits

The retained scale check publishes 1,000 real immutable manifests, closes the
runtime, and queries every page through a new runtime. Performance claims should
report the corpus, command, host-relevant limits, and result; compare the same
workload and environment. This check has no hardware-independent timing target.

Golden projections establish behavior for their supplied artifact and expected
fields. They do not establish compatibility with every upstream producer
release or a full provider capture lifecycle. A passing suite also does not
prove every publication boundary, provider, package-install route, or platform.
Name the missing evidence when it matters to a change.

Known proof gaps include:

- Cache bounds and eviction internals lack public-workflow proof.
- Resource-baseline race handling, bounded file scanning, and unavailable-metric
  branches lack retained direct proof.
- Cancellation during subprocess startup, V8 hard traversal ceiling overflow,
  and perf demangled/unknown-frame conversion lack retained direct proof.
- A managed dependency reconnect branch and live package installation remain
  unproved.
- Vendor tools, optional hardware, permissions, and other platforms are not
  exercised by the ordinary CI suite. Node CPU and heap stdio captures were
  manually proved, but there is no retained live Node capture test.

Provider tests should identify their actual artifact or host requirement. A
skip because a provider or host capability is unavailable is not provider
evidence. Keep observed, derived, and inferred claims distinct, including in
tests and performance reports.
