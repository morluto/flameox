# Slime tool investigation

Status: ongoing. This is a coverage record, not a claim that every tool or all
optimization opportunities have been exhausted.

## Scope and method

The initial Flameox revision is `e3e909e0663c18db11e5b9a247c02a4713040249`
(0.2.5); the Slime revision is `3778dbf6d1a533ab478ecf5ddaa11449a47752b2`.
The host is Linux with an NVIDIA RTX 3060. Slime source is unchanged.
All current runs use a fresh MCP stdio process launched from this checkout.
The already-connected MCP server reports the same package version but exposes
older behavior, so its results are not evidence about the current diff.

The local investigation directory is `.diagnostics/slime-audit/` in the Flameox
checkout. It contains the complete catalog, append-only `calls.jsonl`, individual
request/result records, runnable workload scripts, and an explicitly selected
format-2 evidence store. It is ignored by Git because native artifacts and full
local provenance should remain local. The old Slime workspace configuration,
catalogs and runs were moved to desktop Trash at the user's request.

Use real Slime implementations and existing tests as semantic oracles. Profiling
locates candidates; it does not establish causal improvement. Keep initialization,
warmup and measurement separate when benchmarking operators. Command benchmarks
include process startup. The host has other active workloads, so timings are
exploratory and have not established a production effect.

Research basis:

- [pyperf execution model](https://github.com/psf/pyperf/blob/main/doc/run_benchmark.rst):
  calibration, independent workers, warmups and repeated measurements.
- [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html): scheduled
  capture, warmup and the extra overhead from shapes and stacks.
- [DuckDB insertion guidance](https://duckdb.org/docs/stable/data/insert): avoid
  individual inserts for bulk ingestion.
- [Arrow ingestion](https://duckdb.org/docs/stable/guides/python/import_arrow):
  insert Arrow batches through the maintained Python interface.

## Confirmed Flameox defects and improvements

### Memray analysis exhausts memory on a small native profile

[Issue #470](https://github.com/morluto/flameox/issues/470) records the defect.
The workload runs Slime's `build_dp_schedule` 20 times with 2,048 deterministic
rollout lengths, dynamic packing, four DP ranks and balancing. Every output is
checked against Slime's existing scheduling invariants.

The native profile is 173,883 bytes with 37,307 allocation records. Collection
succeeds, but both hotspot and retained-memory analysis exceed the default 1 GiB
worker budget after approximately 48 seconds. Preservation correctly retains
the capture and its separate analysis failure.

The extractor issues individual SQL upserts for frame contributions, edges and
stacks. Bounded Arrow ingestion removes this pattern. Replaying the exact
preserved profiles after the final cleanup succeeds in 4.32 and 4.02 seconds,
respectively, under the same memory limit. These are successful replay latencies,
not speedup ratios against completed baseline analyses.

The same tables used 32-bit byte counters. A regression with an allocation of
2,147,483,648 bytes fails on the base with an INT64-to-INT32 conversion error.
The fix uses signed 64-bit byte values and unsigned 64-bit sample counts,
matching the existing published table contracts.

The manual extractor revision was removed: its only consumer is the temporary
worker request/response identity check. A stable implementation identifier is
sufficient. Artifact schema versions remain meaningful independent contracts.
Redundant pending-stack state, an unused contribution counter and a no-op budget
check were also removed.

Independent review compared the final implementation with the base on randomized
recursive stacks across two metrics and all truncation bounds. Aggregates,
rankings, coverage and drop counters matched. Eleven focused tests passed.

Preserved original failures:

- Hotspots: `81d4f2a40afaa8ea76a5f22977bc8187abf0d542e1073a833e0ba54e2233efdf`
- Retained: `33ec1576aa3d8472f79d6bb92ae2546c555da7927ce30a1cfa45cc04813a09ca`

### Perfetto graph metric and redundant projection

[Issue #471](https://github.com/morluto/flameox/issues/471) records a real trace
whose graph contains 111 grouped edges but reports `slice_count=0`. The provider
now reports the worker's actual grouped population as `edge_count`. A regression
fails on the base and passes after the fix. The unused Python graph projection
was removed; the isolated trace worker already owns production graph queries.

### Caller analysis becomes a capture tool

The scheduler investigation needed callers of a sampled hotspot, but `cpu.callers`
accepted only deterministic pstats. It now also accepts py-spy Speedscope, exposing
`capture_cpu_callers` through the shared capability registry. The catalog grows
from 47 to 48 tools. Replaying the original scheduler profile finds two caller
edges for `first_fit_pack`; a fresh capture observes 124 edges and returns the
first 100 with a continuation. Each adjacent edge is counted once per sample,
including recursive self-edges, and profile identities remain separate. Weights
normalize time units; sample counts are not invocation counts.

The [Speedscope format](https://github.com/jlfwong/speedscope/blob/main/src/lib/file-format-spec.ts)
defines the native weighted sampled stacks. Four behavioral regressions cover
weights, recursion, profile separation, directional filters and empty selections.
The actual MCP capture and explicit preservation also succeeded.
Resuming its capture token against preserved evidence in a new MCP process returns
the remaining 24 edges and marks coverage complete.

### Workload-independent pytest plugin

The absolute runner still imported `flameox.pytest_capture` through the workload
interpreter's older Flameox package. It failed before Slime's discounted-return
tests ran. The runner now exposes only its adjacent plugin in a request-owned
temporary module directory; it neither imports nor replaces workload-side
Flameox. Zero-worker and two-worker xdist regressions fail before the change and
pass afterward. The real retry records seven fixtures, 244 invocations and zero
incomplete invocations. Evidence was added to canonical
[issue #369](https://github.com/morluto/flameox/issues/369#issuecomment-5576913889).

### SARIF exports retain source scoping

Ruff 0.16.0 produced one PERF401 finding in Slime, but Flameox discarded it because
the report was exported outside the source tree. `static.performance_candidates`
now accepts an explicit absolute `source_root`; replaying identical bytes yields
one candidate and zero invalid results. Containment, URI and traversal checks
remain in place, and source contents are not read. Three regressions exercise
normalization and rejected roots. The
[SARIF location contract](https://docs.oasis-open.org/sarif/sarif/v2.1.0/os/sarif-v2.1.0-os.html)
allows absolute source URIs independently of the report's location. Evidence was
added to canonical
[issue #366](https://github.com/morluto/flameox/issues/366#issuecomment-5576914051).

### Explicit startup limits without a custom launcher

The default 1 GiB budget cannot start this Torch CUDA workload. `analyze`,
`capture` and `mcp serve` now accept `--limits` using the existing RequestLimits
JSON contract. An 8 GiB MCP launch successfully runs the real GAE oracle and
PyTorch capture. Defaults and hard maxima remain unchanged, and MCP requests
cannot increase startup bounds. Invalid limits are rejected before server launch.
Independent review passed for each public-contract change described above.

## Slime optimization candidates

### Native Triton autotuning follow-up

Installed Triton 3.7.1 exposes native `cache_results=True` output, not the listener
hook implied by the existing import format. The maintained
[autotuner implementation](https://github.com/triton-lang/triton/blob/v3.7.1/python/triton/runtime/autotuner.py)
stores `key` and `configs_timings` and selects configurations lexicographically.
The existing `analyze_triton_autotune` tool now accepts these native
`*.autotune.json` files without a custom event producer.

The real Slime route-scatter kernel was exercised on 32×256 BF16 hidden states
with two routes per token. Three block/warp configurations agree exactly with an
independent tensor-indexing oracle before autotuning. The native cache contains
all three candidates; Flameox's derived winner agrees with Triton's selected
256-element block and eight warps. This does not establish a production speedup.
The experiment wrapper and unchanged native cache remain in the local evidence
directory; Slime source is unchanged.

Timing values preserve producer order rather than being averaged as repetitions.
Native caches do not identify the function, device, later cache hits or tuning
duration; the result explicitly discloses these gaps. Positive-infinity sentinels
are preserved. Independent review identified integer-to-float precision loss in
the initial implementation; the correction retains exact integer rankings and
adds regressions above 2⁵³ and at 10⁴⁰⁰.

`capture_triton_autotune` now runs the declared workload using Triton's own cache
controls and a fresh request-owned cache. The actual Slime workload completes in
8.8 seconds, preserving 28 native artifacts including one autotune cache with
three candidates. Replaying the preserved bundle in a fresh MCP process returns
the same three candidates. Independent review found no defects in the capture
and bundle-analysis changes. This deliberately cold-cache experiment includes
compilation and tuning; it is not representative warm-cache execution.

### Sanitizer producer-side evidence loss

A deliberate out-of-bounds mapping passed to Slime's route-scatter kernel under
Compute Sanitizer produces 133 reported errors, but its default print limit saves
only 100 XML records. Flameox's capture now disables that producer cap; the same
fault injection saves all 133 records. Imported-report limitations clarify that
counts cover saved XML, not necessarily all errors observed by the producer.
[Issue #472](https://github.com/morluto/flameox/issues/472) records the reproduction
and local fix. Independent review and both regression cases pass. The invalid
mapping was introduced by the diagnostic wrapper, not observed in Slime usage.

The user's limits concern is tracked separately in the proposed
[workload resource policy](workload-resource-policy.md). That design separates
response bounds, disk-backed native output, workload budgets and decoder
protections. It is not yet implemented; the sanitizer fix does not change
Flameox's execution budgets.

`first_fit_pack` is the leading sampled scheduler function: the initial py-spy
capture attributed 0.28 seconds to one line and 0.05 seconds to its neighboring
line, among 51 total samples. This is an exploration lead. Test its scaling and
candidate algorithms while preserving first-fit ordering and the full scheduling
invariants before proposing a change. Five measurements per input size at 64,
256, 1,024 and 4,096 elements yield an exploratory exponent of 1.9223 (R² 0.9974).
Independent repeated runs have compatible benchmark identities and compare
successfully. This A/A comparison is not a candidate speedup claim. No Slime
optimization has been implemented.

The existing discounted-returns tests pass: 45 tests exercised the real numerical
implementations on CPU. A CUDA GAE workload compares `chunked_gae` with
`vanilla_gae`; its initial capture exceeded the 1 GiB default process-tree budget
during startup. That is a truthful limit outcome, not proof of a numerical defect.
An explicitly configured 8 GiB session passes the serial numerical oracle for
FP32 tensors of shape 4×257. Two independent real validation reports also pass
kernel-validation summary and comparison. These do not cover other shapes or
dtypes. Nsight Compute's one-kernel capture includes setup rather than an isolated
GAE hotspot, so its metrics do not establish GAE's limiting factor.

## Tool coverage so far

| Tools | Evidence and current status |
| --- | --- |
| `query_evidence` | Empty explicit store succeeds; connected older server's store returns corruption. |
| `capture_process_output` | Nine real scheduling tests pass; missing `aiohttp` causes an accurately retained F1 collection failure. CUDA startup limit retained. |
| `capture_pytest_fixtures`, `analyze_pytest_fixtures` | Nonempty real fixture capture succeeds after the environment fix; seven fixtures, 244 invocations. |
| `capture_failures_summary` | Real F1 collection error yields one errored collector and exit status 2. |
| `capture_coverage_summary` | Scheduling suite yields 445 observed rows, 100 returned, with a continuation and incomplete coverage. |
| `capture_cpu_hotspots` | py-spy capture succeeds; native empty-stack samples are disclosed as unresolved. |
| `capture_memory_hotspots`, `capture_memory_retained` | Captures succeed; base analysis exceeds memory; original bytes and failure preserved. |
| `analyze_memory_hotspots`, `analyze_memory_retained` | Exact preserved profiles successfully reanalyzed after batching. |
| `capture_benchmark_summary` | pyperf command capture succeeds with two workers, six measured values and two warmups. |
| CPU caller tools | pstats analysis, preserved py-spy replay, and new sampled capture succeed. |
| Trace summary, window, graph, PyTorch, operations and lifecycle analysis/capture | Real PyTorch and Nsight Systems traces succeed with installed official Perfetto processor. Unsupported provider/capability combinations were also checked. |
| GPU launch and kernel-metric analysis/capture | Nsight Systems observes 5,969 launches; Nsight Compute yields 487 metrics and 23 observations. |
| Benchmark summary, scaling and comparison | Real command, operator and first-fit scaling samples; repeated-run comparison succeeds. |
| Kernel validation and comparison | Actual CUDA GAE errors against the serial oracle, two independent runs. |
| Static performance candidates | Real Ruff SARIF finding recovered through explicit source-root support. |
| Sanitizer capture and analysis | GAE times out after 300 seconds. Smaller route-scatter workload completes in 13 seconds with zero errors. Deliberate invalid-mapping capture exercises 133 real error records and exposes/fixes producer print-limit loss. Other sanitizer tools and error families remain unproved. |
| `prepare_providers`, `preserve_evidence`, `preview_artifact` | Host-only recovery guidance, explicit preservation, and bounded Slime README preview succeed. |
| Evidence resource read | Original failed memory capture remains readable after MCP restart. |

The append-only call log currently records 47 of 49 tool names invoked. Invocation
is not equivalent to full behavioral coverage. Still unexercised on real inputs:
`analyze_inference_summary` and `analyze_inference_compare`.
No representative inference exports were found.
The SDK follow-up reproduced delayed cancellation of analysis: a 0.5-second
client deadline returned at 0.515 seconds while the Memray decoder remained
observable at 3.824 seconds. The next catalog request waited another 3.515 seconds.
Evidence `3579d7937c4adcefe5b89659667f9ff973daa7aedc6bd7b2526951815c0c8aad`
preserves the diagnostic stdout and execution provenance. See the
[SDK review](mcp-sdk-v2-review.md) for the blocking chain and ownership constraints.
The runtime fix is pending; a client timeout is not evidence of decoder cleanup.

The broker prerequisite now propagates AnyIO request-scope cancellation through
synchronous provider calls and retains the cancellation output/cleanup receipt
for both ordinary and peak-RSS execution. Four new behavioral regressions pass;
the complete execution test directory passes (46 tests), with an independent
exact-diff review finding no introduced defects. A single-request diagnostic on
the preserved Slime Memray input cancelled after observing the decoder and
allowing 0.3 seconds of startup/decoding. It confirmed decoder reaping, a retained
cleanup receipt and zero cached analyses. Its evidence is
`44ed05ec6e2d8cdae4e1b2f38748fc53398f899434b1dc291d6b0c383cc67fc7`.
This validates the bridge, not concurrent runtime ownership or MCP integration.
An earlier diagnostic cancelled during subprocess startup and received no process
receipt. The startup follow-up now checks cancellation before launch, acquires the
transport handle under a deadline-bounded scope shield, and retains the cleanup
receipt. Three phase-specific tests cover pre-launch, protocol setup and transport
handshake cancellation; the execution suite now passes 49 tests. Independent
review found no introduced defects. Repeating the real diagnostic without its
artificial startup delay confirms a reaped decoder, complete cleanup and zero
cached analyses in 0.164 seconds; evidence
`28ecb35113ebbfb4669fdd33cf14d5308e2fc1f948b0174e75097aea9007e88a`.
Failed diagnostic attempts remain preserved rather than being counted as
successful validation. These checks still do not establish MCP integration or
concurrent runtime state ownership.

Final execution validation exposed a writable-growth baseline race: the child
could write its output before the asynchronous observer sampled initial sizes,
causing the byte-limit test to time out instead of reporting the byte violation.
A deterministic real-child regression reproduced this ordering. Baselines now
precede subprocess launch; the regression observes exactly 1,024 new bytes while
excluding 200 pre-existing bytes. Independent review found no introduced defects.
This fixes baseline timing, not the separate limitations of sampled enforcement.
Tracked in [issue 473](https://github.com/morluto/flameox/issues/473). Final validation
after the fix: 50 execution tests passed; the default suite passed 319 tests with
137 deselected. Changed execution files pass lint, formatting and strict typing.

A fast-exit follow-up found that a child exiting before observation could still
bypass writable-growth enforcement. Another ordering allowed the observer's sleep
to delay completion until the request deadline after the child had already exited.
Final observation now runs even for already-exited children, and process completion
wakes the observer. Deterministic real-child regressions cover both cases.
Independent review caught and helped resolve an introduced availability-label
regression: the final disk check now retains previously measured RSS without
looking up an exited PID. Never-sampled RSS remains explicitly unavailable.

The reviewed change was exercised through real MCP capture of Slime scheduling:
20 schedules with 2,048 samples each passed the existing Slime invariants.
Evidence `2b4f735ba8d840c4a4f8bbb6f013a2e6ada14bfdd28d1104179040a2b18a3102`
preserves its stdout and execution provenance. This is a successful behavioral
replay, not a controlled performance-improvement estimate.
Final checks for the exit-observer changes: 53 execution tests passed; 319 default
tests passed with 140 deselected. Lint, formatting, strict typing and diff checks
pass for the changed execution files. Slime's tracked checkout remains clean.

### Coordinated MCP requests

The event-loop defect is now fixed for the exercised MCP analysis path. Analysis,
preservation, queries and resource reads run through a task-group-owned worker
boundary with serialized shared-state access. Capture admission/finalization and
scratch bookkeeping use the same lock; workload processes remain concurrent.
Provider preparation only takes the lock to publish a verified binding. An
independent review identified that publication race, and its regression fails
without the publication lock and passes with it.

Real stdio cancellation of the preserved Slime Memray analysis now returns the
catalog request in 0.082 seconds after the 0.530-second client timeout, with the
decoder already reaped. Previously the catalog request waited another 3.515
seconds. Evidence:
`24d4b2da813d08823088fe13d4d7cf8dddafc3a599dffcc3052c940da1567d5d`.
A fresh capture succeeded with 718 observed rows and 100 returned (bounded,
incomplete coverage), preserved as
`e4665ad6fef516fbd2f7cb4d5e1148c724c88ecece8efe52feadf504be80a6c6`.
Reanalysis and explicit preservation of the original profile also succeeded.

Two concurrent Slime memory captures completed successfully in one MCP session.
The process observer confirmed two simultaneous workloads; each result was
preserved independently. Diagnostic evidence:
`dd19de9596634f75a9be5147539e43f1bb53f85effcc86673b22d5cf646d7575`.

Seven new request-boundary tests pass, covering domain errors, catalog
responsiveness, provider publication, and scope/task cancellation during analysis
and capture finalization with child and scratch cleanup. The default suite passes
322 tests with 144 deselected. The broader state/execution suite passed 178 tests
with one skip before the final two cancellation cases were added; all seven new
cases were then run together. Ruff, formatting, strict mypy (124 files) and both
import contracts pass. Non-cooperative synchronous readers still have to settle
before state can be released; this does not prove prompt cancellation for every
provider or platform. All changes remain local and uncommitted.

Remaining work includes the workload-resource redesign, broader sanitizer validation,
confirmatory experiment design, and broader
shape/provider/platform coverage. Fabricated inputs do not count as real-workload
coverage. All code changes remain local and uncommitted.

## Zero-exit policy failures

Tracked in [issue #474](https://github.com/morluto/flameox/issues/474).

A short producer can exit with code zero before the broker detects its output
limit violation. Capture and semantic-oracle status previously used only that
exit code, incorrectly classifying the execution as successful despite a
`LIMIT_EXCEEDED` failure record. Both now require a zero exit code and no broker
failure. Failed captures skip the semantic oracle; failed oracles invalidate the
experiment execution even when their process exited zero.

Two real-subprocess regression cases deterministically exercise this schedule,
including preservation and reopening. Both fail with the two status fixes
removed and pass with them restored. Independent scoped review found a mistaken
test manifest key, which was corrected; it found no production issue.

A fresh MCP capture ran Slime's actual scheduling workload, then deliberately
wrote 2,048 extra console bytes under a 1,024-byte output budget. Its exit code
was zero, its capture status was failed, and MCP returned `EXECUTION_FAILURE`.
Preserved evidence:
`199b196a941e52c3ed3112951304d9619a63fae9626b857a42010f138f6b54f5`.
This tests Flameox policy attribution, not a Slime scheduling defect.

## Disk-backed broker output foundation

An internal sink mode now preserves stdout/stderr directly on disk with separate
64 KiB in-memory previews. It keeps the combined output limit unchanged and
rejects child-peak-RSS observation before launch. Ordinary capture tools do not
select this mode yet; this is groundwork for their output lifecycle, not a
completed resource-policy redesign.

Independent review caught shared-budget ordering, overwrite, and writer-lifetime
defects in the initial draft. The corrected implementation reserves bytes before
awaiting writes, uses exclusive file creation, and joins worker writes before
closing files. The consolidated follow-up found those defects resolved. Eleven
focused tests cover simultaneous 17 MiB streams, exact bytes, small previews,
combined limits, regular/hardlinked existing files, cancellation with blocked or
inherited writers, and typed write/close failures.

A real Slime scheduling workload followed by deliberate large console output
completed with 17,825,838 stdout bytes and 17,825,792 stderr bytes, with a 65,536-byte
preview for each. The outer MCP capture preserved its execution receipt as
`c3daba184c4e43c9a2c55f4f226d5c8c56184c4dda13fd085ee9f96153d65023`.
The native files remain in ignored investigation storage. A first replay used an
incorrect tool name; retrying the catalog's `preview_artifact` exposed a genuine
gap: the long text line cannot fit a result page, returning `LIMIT_EXCEEDED`
without an analysis handle to preserve. No successful native-log preservation is
claimed.

After review: the execution/request/status suite passed 72 tests; the default
suite passed 322 with 156 deselected. Ruff, formatting, strict mypy (125 files)
and both import contracts passed. Capture integration, oversized-line drill-down,
independent workload budgets and shared storage admission remain open.

## Recoverable oversized-text fragments

The oversized-log gap above is now handled by
`preview_artifact` with `options: {"text_fragment_chars": 1024}`. Default
line-based offsets remain unchanged. Fragment mode uses bounded standard-library
text reads, preserves LF/CR characters in the projection, reports line and
fragment indices, and explicitly discloses UTF-8 replacement decoding. Source
digests and options bind continuation tokens. The native artifact is not edited.

The original two Slime console files now preview successfully: the first page
returned 100 fragments with incomplete coverage in 0.241 seconds. Explicit
preservation succeeded in 0.356 seconds with both native artifacts (17,825,838
and 17,825,792 bytes):
`73aff1bcfb3aaaf465200b91586b84cfd0c8916913c0f585694ae6eddc658d89`.
A fresh stdio server reopened the resource and validated both original digests.
Resuming the original continuation through its ordered `analysis_sources`
returned the next 100 fragments in 0.186 seconds. A direct final-window query
returned the last fragment and reported 34,817 observed rows in 0.168 seconds.
These are single-run observations, not comparative performance claims.

Twenty-two focused tests pass, including Unicode/CRLF/invalid-byte semantics,
empty files, large lines, preservation and restart, stale options/content,
non-text rejection, the MCP path, and minimal-fragment result-envelope recovery.
Independent review found and resolved a misleading hint at the minimum fragment
size. Fragment mode is available through the existing tool, not an extra catalog
entry. Whole-line mode still rejects an oversized result row; it now names the
fragment recovery option. Capture sink integration and independent workload
budgets remain unfinished.

The default suite passes 344 tests with 156 deselected; Ruff, formatting, strict
mypy (128 files) and both import contracts pass. After verifying original and
preserved hashes, the two task-owned scratch log copies were removed. Their exact
bytes remain recoverable through the preserved evidence above.

## Console retention default and recovery

Capture now defaults to bounded in-memory console diagnostics, retaining at most
4,096 bytes per stream (less under tight provenance budgets) and reporting exact
observed/retained/omitted byte counts. Process-output evidence, semantic-oracle
inputs, and explicit `target.console_output: "full"` use disk-backed full output.
An oracle's own output remains diagnostic unless full retention is explicitly
requested. Preservation does not change the retention choice.

Two real Slime scheduling runs used Memray through `capture_memory_hotspots`,
with deliberate console noise after the workload's scheduling invariants passed:

| Mode | Observed stdout / stderr bytes | Retained console | Preserved artifacts |
| --- | --- | --- | --- |
| Default diagnostics | 17,826,366 / 17,825,944 | 4,096 bytes per stream in memory | One Memray profile |
| Explicit full | 82,494 / 82,072 | Exact native streams on disk | Profile, stdout, stderr |

The default run completed in 5.668 seconds and preserved evidence
`b9bd20d18b860506309284e2bd836a9c9e73fd8ccd520ef0337c8a221a7824bd`.
It reported 17,822,270 stdout bytes and 17,821,848 stderr bytes omitted; those
discarded bytes are not available for later recovery. The full-output run
completed in 5.409 seconds and preserved
`7e3c06761424869edafd908117b563f7fbe2bae0d7b6e2954b294f2d2c0dae99`.
These are single-run execution observations, not a performance comparison: the
deliberate console volumes differ.

A fresh stdio server reopened both evidence resources. The default resource
exposes diagnostic counts without console text. The explicit-full stdout replay
returned all 88 text fragments with complete coverage in 0.045 seconds. Both
captures returned ten allocation rows with incomplete table coverage; neither
claims that the ten-row page exhausts the profile.

Tests exposed two recovery defects during integration. Diagnostics-only failed
captures had no native inputs, which the repository previously rejected; they
now preserve an explicit failed analysis and execution diagnostics without
placeholder log artifacts. Cancellation could also leave an asyncio pipe
transport alive until after its event loop closed; broker cleanup now closes the
transport while the loop remains active. The focused cancellation tests pass
with unraisable-exception warnings promoted to errors.

A real Slime scheduling run followed by a deliberate exit 7, without the requested
observations artifact, returned MCP `isError=true` and `EXECUTION_FAILURE` in
0.682 seconds. Evidence
`8b9338380a1e87fb54060fe88985c2712a803aafaa75fd110a3a7ba1a361eab4`
preserves the failed attempt with zero native artifacts and 46 retained stdout
bytes. A fresh server reopened it in 0.004 seconds and exposed the workload exit,
missing artifact role, failure status, and diagnostic counts without console text.

Independent Luna review and its consolidated follow-up found no remaining
actionable retention defects in the scoped diff. Validation includes 19 focused
retention/output/status tests, 77 broker/request tests, and 33 CLI tests. Native
coverage tests exercise 17 MiB on each stream, exact full-log preservation,
oracle retention, and missing-artifact recovery with and without immediate
preservation. These are not evidence of complete provider or platform coverage.

The final default suite passed 345 tests with 175 deselected. Ruff, formatting,
strict mypy (131 files), both import contracts, and `git diff --check` passed.

Workload time/RSS separation, independent native-artifact storage bounds, shared
storage admission, and the broader investigation coverage gaps remain open.

## Independent workload time and RSS budgets

`target.budget` now owns optional per-invocation workload time and sampled RSS
limits. Both default to null; capture no longer inherits the decoder's 300-second
or 1 GiB defaults. The CLI exposes `--workload-budget`. Explicit budgets apply
separately to each capture and semantic oracle, while cancellation, cleanup,
decoder protection, console retention, and storage bounds remain active.

The real CUDA GAE workload completed with its serial oracle passing for fp32
batch 4, 257 tokens even when decoder limits were set to 0.01 seconds and 16 MiB.
The tool completed in 7.551 seconds; recorded workload wall time was 7.364 seconds.
Its preserved evidence is
`db02b63b1422a394b003a61f997e3e5ec04b68450003b493f5ad7d231e8b8398`.
This checks budget independence for the actual Slime workload, not an optimization
claim or proof across other shapes and dtypes.

Two deliberate budget failures preserved their exact attribution:

- A 512 MiB workload RSS cap stopped the process after observing 621,674,496
  resident bytes. Evidence:
  `14b35c606f64b785fcfc0184b849404cf45dd54071c642bbaf0817c2708b989b`.
- A 0.2-second workload deadline stopped the process with 0.232 seconds recorded
  including settlement. Evidence:
  `0319dc344361ee9cc53335219f4783de5637ad9882bf63e9601b20b894f14828`.

These failed during startup and do not claim completed GAE execution. Fresh
servers reopened all three resources. New resources expose the recorded numeric
budget; an older capture without that field still omits it rather than claiming
historical execution was unbudgeted.

A separate real Slime scheduling/Memray capture succeeded while its analysis
worker hit an explicitly selected 0.01-second decoder deadline. The result
reported `analysis_failure` with `EXECUTION_FAILURE`, incomplete analysis coverage,
and a successful capture outcome; the original profile was preserved as
`323e934b98e67bb2a7b31978693e3e466c62664d0e97ecda706a27c59f9dc028`.
Thus removing an implicit workload deadline does not disable decoder protection.
A fresh server reanalyzed that preserved profile with normal decoder limits in
3.800 seconds, without rerunning Slime. It returned ten of 729 allocation rows
with explicit incomplete page coverage and preserved the recovered analysis as
`eb88635c4c8669cb53bdbb20773d32a736b0bda4a6e5845db6e4d1fbda5c4e64`.

Independent Luna review and its follow-up found no actionable defect in this
scoped budget change. With process exclusions cleared, 94 broker, request,
retention/failure, and budget tests passed. The default suite passed 349 tests
with 186 deselected; the six dedicated workload-budget tests also passed after
tightening the decoder-timeout assertion. Ruff, formatting, strict mypy (133
files), both import contracts, and `git diff --check` passed.

Native-artifact byte limits remain coupled to console/worker output limits, and
shared storage admission remains unfinished. This increment does not establish
complete tool/provider/platform coverage or close the broader investigation.

## Validation of the current changes

After the required independent reviews:

- Default suite after Triton capture and sanitizer fixes: 319 passed, 129 deselected.
- Triton/kernel focused suite: 14 passed, including bundle and exact-integer regressions.
- Sanitizer print-limit regressions: 2 passed.
- CPU, source-evidence and CLI tests with exclusions cleared: 47 passed.
- Explicit pytest environment, fixture ownership and stdio checks: 5 passed.
- Ruff, formatting, strict mypy, both import contracts and `git diff --check` pass.

These checks supplement the real workloads above. They do not establish coverage
of every provider, all hard resource ceilings, every operating system, or all
numerical shapes and dtypes.
