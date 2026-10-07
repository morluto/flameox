# REA tool UX audit

Audit date: 2026-10-07. The audit exercises Flameox against the built checkout at
`/Users/will/dev/active/rea`, then implements the failures found in discovery,
native V8 evidence, durable handoffs, and process cleanup. After the scope was
clarified, live exercises were restricted to REA Node CPU/heap capture,
benchmarks, semantic experiment contracts, native traces, SARIF, and evidence
handoffs. No Python fixture, PyTorch, GPU, or inference workload was added.

## Recorded variants

- Flameox source baseline: `cab2f6b09585f67d286a20dae112857d7fc3ea02`, version
  `0.2.7`. Changes are on `codex/rea-tool-ux-audit`; the wire evidence
  was collected before those changes were committed.
- Fresh stdio servers expose seven tools and 26 capabilities. The already-connected
  host exposed six tools without `inspect_capabilities`; that host is a separate
  older variant and needs a reconnect to consume the current catalog.
- Host: macOS 26.3, arm64. Final REA runs use the exact executable
  `/Users/will/.nvm/versions/node/v22.22.2/bin/node`, within REA's declared engines.
  Earlier exploratory runs used PATH's Node 25.1.0, outside that engine range.
- Initial REA baseline: `faaee693d5bb4bb957387ebd0f4338513c4c914a`. Concurrent work
  changed sources, rebuilt JavaScript, and advanced HEAD during this session.
  The earlier complete transport audit used
  `98f8e29177bf69e44a2bd303226bb6345ce0ad6a`. The final relevant-workflow baseline
  is `f1fe51f37f40b4c12df6d4935cf67e35e0ba7f14`. Its before/after snapshots
  confirm stable hashes of the built JavaScript,
  CLI entrypoint, package metadata, and selected fixture during the final audit.
  Earlier profiles belong to their recorded build variants; elapsed times are
  not compared across those builds.
- The installed `design-mcp-discoverability` skill was updated to upstream
  `morluto/codex-global-settings` commit `0d008ac0131b554acdad401298293683252413e4`
  and its five files verified against Git blob identities.

These are scripted protocol and workflow checks. They do not measure a model's
tool-selection rate, first-use abandonment, or performance improvement.

## Selection and workflow

| User intent | Preferred tool | Nearest alternative and boundary | Next step |
| --- | --- | --- | --- |
| Choose an evidence question and valid options | `inspect_capabilities` | CLI inspection; listing every transport schema costs more context | Get one capability, then analyze or capture |
| Satisfy a provider's dependency requirement | `prepare_providers` | CLI setup or upstream installation; host-only tools receive guidance | Follow activation/reconnect guidance and retry |
| Inspect or compare existing native exports | `analyze` | Native profiler UI or bounded upstream query; Flameox supplies evidence identity and handoffs | Follow the returned page or preserve |
| Collect new runtime evidence | `capture_and_analyze` | Direct profiler invocation; Flameox owns bounds, output, provenance, and failures | Inspect existing pages without repeating the workload |
| Keep a session result | `preserve_evidence` | Copying files loses the immutable manifest and selectors | Read its resource or reopen its evidence source |
| Move a live result into a recovery store | `rescue_evidence` | Manual store copying lacks the validated restart handoff | Restart with the returned environment and resource URI |
| Find results after a session ends | `query_evidence` | Filesystem search lacks typed immutable metadata | Read the exact returned URI or analyze its selectors |

The seven tools keep distinct effect boundaries. Analysis and execution stay
separate; discovery provides detail on demand. Server instructions now route
artifact comparisons, randomized experiments, SARIF imports, pagination,
session expiry, preservation, and resource reads explicitly.

The final wire catalog is 87,940 bytes versus 87,306 bytes at baseline, using
the same compact JSON serialization: a 634-byte increase with no new tools.
The largest contributions remain `capture_and_analyze` (26,022 bytes) and
`analyze` (15,723 bytes), including their typed result schemas. Exact capability
options remain behind discovery instead of expanding `tools/list`.

## Findings implemented

| Issue | Before | Resulting behavior |
| --- | --- | --- |
| [#517](https://github.com/morluto/flameox/issues/517) | V8 hotspot tables contained hashed frame IDs without the function or file | CPU and heap rows include the native function, file, line, column, and script identity beside measurements |
| [#518](https://github.com/morluto/flameox/issues/518) | Heap sample counts counted call-tree nodes | Counts join native allocation sample records to their nodes, then aggregate matching frames |
| [#519](https://github.com/morluto/flameox/issues/519) | CLI could inspect transport schemas but lacked exact capability options; window/scaling examples missed required fields | `mcp inspect --capability ID` shares MCP detail; every discovered example validates against its selected model |
| [#520](https://github.com/morluto/flameox/issues/520) | Rescue after preservation tried deleted capture scratch paths | Rescue verifies immutable native sources and rematerializes bundles under scratch admission before publication |
| [#521](https://github.com/morluto/flameox/issues/521) | A native heap sample referencing an absent tree node rejected the whole profile | Known rows remain usable; unresolved counts and estimated bytes are explicit and coverage is incomplete |
| [#522](https://github.com/morluto/flameox/issues/522) | macOS denied signals to already-dead groups broke peak RSS timeout/cancellation receipts | EPERM is accepted only after verifying that every remaining group member is dead; live/unreadable failures propagate |
| [#523](https://github.com/morluto/flameox/issues/523) | Cleanup could wait forever on a paused output pipe after the reader stopped at its budget | Root-exit waiting is independent of pipe completion; reader settlement and transport closure retain their existing owner |
| [#524](https://github.com/morluto/flameox/issues/524) | CPU rows used optional hit counts, dropping 81 exported samples in a real REA profile | Self counts and subtree aggregation use the bounded exported sample sequence; all 4,409 samples are accounted for on reanalysis |

V8 coordinates remain native and zero-based; `-1` means unavailable, and source
maps are not resolved. Recursive CPU inclusive counts sum call-tree occurrences;
they are not an exclusive percentage of total samples. The worker identity advances
to `v2` for changed CPU and heap count semantics. Existing evidence is immutable; reanalysis records the new
extractor identity. CPU samples referencing absent nodes still fail validation;
heap handling does not relax malformed sizes, types, duplicate IDs, or node bounds.

Rescue validates primary-store payloads before writing a destination, does not
reimpose analysis reader limits on publication, and restores temporary scratch
protections. Regression cases include removed original directories, failed
captures without native sources, captures exceeding reader byte limits, retries,
restart, and corrupt/missing preserved payloads.

## Direct REA evidence

The final relevant-workflow run performs 49 MCP calls, validates each result against its
actual advertised output schema, reads two evidence resources through the MCP
resource API, and exercises all CLI command families:

- Capture the real `analyze-javascript-application` command against REA's
  checked-in `tests/conformance/readiness/javascript-cli` fixture using Node CPU,
  Node heap, direct output, and two pyperf baseline runs.
- Preserve, repeat preservation, paginate without recapture, reopen immutable
  sources, rescue after scratch release, and consume the rescue URI from a new
  server process after the original server exits.
- Query an absent store and a populated store; validate incompatible-provider
  and expired-analysis errors. The earlier transport audit also obtained
  host-only dependency guidance; unrelated provider preparation is omitted
  from the narrowed live harness.
- Run a randomized three-block baseline/repeat-control experiment: six REA
  executions pass a declared oracle comparing the source digest, statistics,
  and findings summary. A separate rejection control preserves both successful
  workloads while marking the rejected oracle and comparison incomplete.
  This checks experiment accounting and output equivalence on the selected
  fields, without claiming a speedup or full program semantic equivalence.
- Inspect compact, capability-specific, and full CLI discovery; plan setup with
  `--dry-run`; locate/query/show evidence; preview REA package metadata; capture
  CPU evidence and execute its next-page command in a separate CLI process.

The final CPU run contains 2,019 nodes and 506 samples. Rows identify functions
and source paths, and self counts follow exported sample IDs rather than optional
hit metadata. The heap run contains 1,259 nodes and 224 samples, including one
unresolved sample representing 526,240 estimated bytes. It returns bounded
known rows with incomplete coverage. An earlier preserved Node 25 heap
with two absent references also reopens successfully with explicit uncertainty.

Native Node trace exports are analyzed through the pinned Perfetto processor
(v56.1) as `trace.summary`, `trace.call_graph`, and `trace.window`. Summary evidence
is a bounded 1,001-slice extraction and reports incomplete coverage. The window
query observes 61,730 matching slices and paginates; those are different bounded
questions, not interchangeable totals.

Oxlint 1.87.0 exports native SARIF from REA's `src` using
`--allow all --warn perf --format sarif`. Flameox normalizes 214 diagnostics with
no invalid or excluded records. These are static candidates, not measured
runtime bottlenecks. No REA source changes were made by this audit.

Whole-`dist` analysis exceeded its declared 60-second workload budget, including
a repeat on supported Node 22. That failed attempt was preserved; it is not
counted as a completed profile.
Node CPU profiles do not support `cpu.callers`; the native artifact probe returns
that limitation rather than inventing caller evidence.

### REA findings discovered during profiling

[REA #904](https://github.com/morluto/rea/issues/904) tracks a concrete optimization
candidate from a completed `dist/server` CPU capture: 59 JavaScript files,
143,782 source bytes, 5,795 profile nodes, and 4,409 exported samples. Native
sample IDs attribute 1,053 self samples to crypto `update` and 924 to
`emitCanonical`, approximately 44.8% together. Current source sends each canonical
punctuation/key/value fragment separately to the crypto hash. Bounded buffering
is a hypothesis for reducing those crossings; it requires exact digest oracles
and isolated representative experiments before claiming improvement.

The [follow-up on REA #623](https://github.com/morluto/rea/issues/623#issuecomment-6034279182)
records a supported-Node whole-tree attempt with zero stdout/stderr before the
external deadline. It does not assert a terminal RangeError: current hashing
already streams, and the run was interrupted before a native result. The larger
attempt and smaller control were concurrent, so their elapsed times are not an
isolated scaling comparison. The REA issue and comment focus on REA behavior and
direct Node/REA reproduction commands.

[REA #912](https://github.com/morluto/rea/issues/912) records a separate CLI
contract bug: `--format json --filter-output summary` exits 0 with empty stdout
and stderr, whereas `normalized_result.summary` returns valid JSON. The issue
includes the direct REA reproduction and distinguishes unknown top-level
selectors from missing nested fields.

The completed native CPU artifact has SHA-256
`00aaaa832373aeca4f1924bfc2ae756b4d7c20deda0ef9eea76780d334b24189` and is preserved
under `.diagnostics/rea-optimization/server-store/`. Its evidence ID is
`2cfafb670ac0d6a669b3c911bbaa37bf1b872f8ffc4e708121d1c388968ce2a9`.
`server-cpu-recounted.json` contains the complete 929-row reanalysis, with self
counts summing to 4,409. The timed-out attempt is retained in `whole-store/` and
`whole-cpu.json` rather than being presented as a successful profile.

## Coverage and proof gaps

| Capabilities | Evidence in this audit |
| --- | --- |
| `cpu.hotspots`, `memory.hotspots`, `artifact.preview` | Real REA capture, bounded rows, preservation and/or restart |
| `benchmark.summary`, `benchmark.compare` | Real REA pyperf inputs and compatible two-source comparison |
| `trace.summary`, `trace.call_graph`, `trace.window` | Native REA Node trace export and real Perfetto queries |
| `static.performance_candidates` | Native REA Oxlint SARIF export |
| `cpu.callers` | Actual unsupported-format response for the REA V8 artifact |
| `benchmark.scaling`, `memory.retained`, `coverage.summary`, `failures.summary`, `pytest.fixtures` | Discovery/schema examples and repository regression fixtures; no corresponding REA native workload |
| `trace.pytorch`, `trace.operations`, `trace.lifecycle`, `inference.summary`, `inference.compare` | Discovery/schema examples and repository regression fixtures; no corresponding REA service/trace workload |
| `gpu.launches`, `gpu.kernel_metrics`, `triton.autotune`, `sanitizer.failures`, `kernel.validation`, `kernel.compare` | Discovery/schema examples and repository regression fixtures; no GPU hardware or native vendor capture on this host |

All 26 capability details were fetched from a live server. Required options,
source cardinality, and provider example contracts have regression coverage.
Passing these checks does not establish live readiness for every provider.
Linux/Windows cleanup and randomized representative optimization experiments
remain unproved here. Vendor/GPU capture, Python-specific workloads, inference
services, and scaling studies without native REA inputs are outside the narrowed
live scope. Semantic pass/rejection controls establish the experiment contract
only. The two repeated pyperf baselines establish compatibility, not an
optimization or causal conclusion.

## Reproduction and artifacts

Run from a built Flameox checkout with the committed lock:

```console
uv sync --locked --extra dev --extra memory --extra trace --extra cpu
uv run python -m tools.rea_ux_audit \
  --rea /absolute/path/to/rea \
  --node /absolute/path/to/supported/node \
  --output /absolute/path/to/new-audit-directory \
  --trace /absolute/path/to/native-node-trace.json \
  --sarif /absolute/path/to/native-oxlint.sarif \
  --experiments
```

Set `FLAMEOX_TRACE_PROCESSOR` to an exact native processor path when needed.
Trace and SARIF inputs are optional. The output directory must be new. Setup is
a dry run; captures execute REA and write into the audit's own evidence stores.
The harness records source/build identity, timestamps, hashes, protocol payloads,
per-call timings, CLI results, and summaries.

Local final artifacts are under `.diagnostics/rea-ux-audit-relevant/`: `summary.json`,
`initialize.json`, `catalog.json`, `calls.json`, resource responses, native evidence
stores, and individual tool/CLI results. These files are ignored rather than
committed. Source and fixture snapshots appear in `rea-before.json` and
`rea-after.json`. Native trace/SARIF inputs remain under
`.diagnostics/rea-native-traces-v22/` and `.diagnostics/rea-native-traces/`.

Validation includes the default deterministic suite, the complete required
process suite, Memray provider checks, marked performance cases, Ruff formatting
and lint, strict mypy, and both import boundaries. The default suite alone would
have missed both cleanup failures; process checks are part of completion.

Commands actually run:

```console
uv run pytest -q
uv run pytest -o addopts='' -m 'not optional and not performance and process' -q
uv run pytest -o addopts='' -m requires_memray -q
uv run pytest -o addopts='' -m performance -q
uv run pytest -o addopts='' -m optional -q -ra
uv run ruff check src tests tools
uv run ruff format --check src tests tools
uv run mypy src tests tools
uv run lint-imports
git diff --check
```

Final results: 416 deterministic tests passed; 216 process tests passed and two
skipped. All three Memray checks and all three performance checks passed. Ruff,
strict mypy, formatting, and both import boundaries passed. The final relevant
live audit passed all 49 tool calls and two resource reads, including advertised output
schema validation and CLI handoffs.

Earlier repository checks skipped optional PyTorch because it is unavailable;
no live PyTorch exercise was added. Systemd checks skip on this macOS host.
See the coverage table for the remaining live-provider and experimental proof
gaps. The failed exploratory `--jitless` candidate remains preserved under
`.diagnostics/rea-ux-audit-extended/`; its three failed workloads were excluded
from comparison, and no result is presented as an optimization.
