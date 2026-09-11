# Workload resources and evidence bounds

Status: console-retention selection is implemented and exercised with real Slime
captures. Optional workload time/RSS budgets are now separate from decoder limits;
shared storage admission and native-artifact budget separation remain incomplete.
The default is bounded, memory-backed console diagnostics, not
disk-backed full console output. This follows the Slime investigation and
the request to reconsider byte and time limits rather than keep increasing them.

## Retention default

Native artifacts and console diagnostics have different purposes:

- Keep native profiles, traces, and other artifacts when the investigation needs
  them for analysis, comparison, or later verification. Do not create an artifact
  merely because an operation ran.
- Normally retain bounded console diagnostics in memory. Report what was retained
  and omitted; never present an excerpt as the complete stream.
- Retain full console output when it is the evidence being requested (for example,
  process-output capture), a semantic oracle requires it, or the caller explicitly
  requests full console retention.
- Use disk backing for that full-output case. It is not a universal prerequisite
  for capture or analysis.

Evidence preservation controls durability of the evidence selected for retention;
it does not silently upgrade console diagnostics to full-log collection. If bytes
were omitted, later preservation cannot recover them. Native artifacts may remain
ephemeral when the caller does not request preservation. A bounded console excerpt
may be included in preserved evidence without retaining the original full stream.

Omission counts must describe observed bytes actually omitted. Cancellation or an
interrupted reader may leave the eventual stream length unknown; report that
incompleteness instead of inventing a total. A full-output storage failure must
likewise retain an explicitly incomplete prefix where possible.

The optional broker sink writes exact bounded stdout/stderr prefixes to exclusive
request-owned files and returns at most 64 KiB of each stream in memory. Its
metadata distinguishes disk bytes, preview bytes, stream completeness and I/O
failure. Sink writes run in joined worker tasks; cancellation cannot close a file
under an active write. Both the combined output cap and existing deadlines remain
active for full-output collection. Capture selects this sink for process-output
evidence, oracle inputs, or `target.console_output: "full"`. Default diagnostics
drain both pipes without applying the full-output byte cap to discarded bytes.
They retain at most 4,096 bytes per stream, lowered by the provenance budget.
Combining the sink with child-peak-RSS observation is explicitly rejected before
launch. Existing observed execution without sinks is unchanged.

An actual Slime scheduling run followed by deliberate large console output
completed with 17,825,838 stdout bytes and 17,825,792 stderr bytes, retaining
65,536 bytes per stream in memory. The execution receipt is preserved, but normal
analysis/preservation of the native logs exposed a separate oversized-text-row
gap. The subsequent `artifact.preview` fragment option now recovers and preserves
those native logs through bounded pages. Neither milestone proves the planned
resource-policy redesign complete. Subsequent real captures exercise both
retention choices and restart recovery; see the console-retention section in the
[Slime investigation](slime-investigation.md#console-retention-default-and-recovery).

## Evidence for changing the boundary

`RequestLimits` previously combined page size, decoder input limits, workload
timeout, process-tree memory, process output, artifact growth and provenance.
Workload time/RSS now belong to `target.budget`, with neither imposed by default.
`AnalysisRuntime._capture_resource_policy` still uses `max_output_bytes` as a writable-growth
ceiling while `ExecutionRequest` uses that same value for stdout plus stderr.
Before the sink draft, the broker retained console output in bytearrays, then
capture wrote those bytes to files after the process settled. Capture admission reserves a multiple of
that limit against a fixed 1 GiB session scratch ceiling.

The consequences are observable:

- A real Torch/Slime CUDA workload could not start under the former default 1 GiB RSS
  budget, but succeeded with an explicitly selected 8 GiB budget.
- A GAE sanitizer run reached the former default 300-second timeout, while a smaller
  route-scatter investigation completes. This is an experimental choice, not
  evidence that all legitimate investigations fit five minutes.
- Compute Sanitizer's own default print limit saved only 100 of 133 reported
  errors. Flameox now disables that producer cap. It should not replace that
  silent loss with a blanket workload termination caused by console verbosity.
- Native profiling artifacts and the small JSON response have different sizes
  and lifetimes. The artifact does not have to fit in the agent's response.

## Proposed ownership

| Concern | Owner | Desired behavior |
| --- | --- | --- |
| Rows and response bytes | Evidence projection | Bounded pages, explicit truncation, resumable queries. Never a workload kill criterion. |
| Console diagnostics (default) | Request-owned bounded memory buffer | Retain an excerpt with observed/retained/omitted byte counts and completeness. No full-output disk sink. |
| Full console output (when needed or requested) | Request-owned disk writer | Stream exact bytes to native files with bounded buffering; identify partial files on failure. |
| Native profiler outputs | Capture/evidence storage | Keep upstream artifacts on disk; preserve by content identity. Do not apply a console-response size limit. |
| Workload runtime and RSS | Explicit execution budget | Optional per-workload controls. No universal five-minute or one-GiB requirement. |
| Decoder runtime, RSS and output | Flameox worker policy | Independent protective limits for parsing and extraction. A failed decoder must not lose a successful capture. |
| Disk exhaustion and concurrent captures | Storage admission | Shared accounting and a free-space reserve. Explicit storage failure, with partial evidence retained where possible. |
| Cancellation and child cleanup | Subprocess broker | Always active, including when no workload deadline is configured. |

The existing CLI startup options are a workaround, not this design's end state.
Adding separate numbers without changing these owners would leave the same
coupling in place.

## Required behavior

Ordinary trusted local workload execution should not acquire a timeout or memory
cap from a parser setting. A caller can explicitly budget a costly experiment;
any deployment-owned ceilings should be separately named and visible. A missing
deadline means no Flameox workload deadline, not detached execution or immunity
from client cancellation. Hosts may impose their own request timeouts.

Console output must not accumulate without bound in Python memory. Default
diagnostic collection should keep a bounded excerpt while accounting for omitted
observed bytes. Full-output collection should stream stdout and stderr to separate
request-owned files and return their identities after settling the writers.
Small worker protocols retain their bounded in-memory transport. When full logs
are required, capture, semantic oracles and failure preservation must consume the
disk-backed artifacts without reading them all back into RAM.

No silent log dropping: a returned excerpt identifies its omitted bytes and
whether a full native stream was retained. Do not offer a native-artifact handoff
when the omitted bytes were discarded. If full-output storage cannot retain more,
stop explicitly with a storage failure and preserve the retained prefix and
termination reason. Never label that prefix as a complete native stream.

The fixed scratch ceiling cannot remain a hidden replacement for removed artifact
limits. Request admission, active reservations, eviction and preservation staging
need one storage owner that accounts for actual disk use and concurrent growth.
The design must account for temporary duplication during immutable publication.
Low-disk protection is best-effort on a filesystem shared with unrelated writers;
do not advertise it as an operating-system quota or a security sandbox.

## Implementation and proof obligations

The retention choice, diagnostics/counting contract, and workload time/RSS
separation are implemented. Next separate native-artifact storage accounting from
console and decoder output bounds. Do not disable
the current output ceiling before the corresponding collector can drain safely
with bounded memory or stream to bounded storage. Do not remove cancellation,
descriptor safety, native integrity checks, or decoder protections.

Required tests and live evidence:

- Default diagnostic collection does not create a full-output disk sink, keeps
  memory bounded under noisy output, and reports exact observed omission counts.
- Full retention is selected for process-output evidence, oracle inputs, and an
  explicit caller request; preservation alone does not silently select it.
- Large simultaneous stdout/stderr streams remain byte-exact on disk with
  bounded memory; inherited pipe writers cannot prevent cancellation cleanup.
- Noisy successful Slime workloads complete when only the response page is
  small. Default diagnostics report omissions without creating full logs; when
  full retention is selected, those logs remain retrievable after preservation
  and restart.
- Explicit timeout and memory budgets still terminate with accurate attribution;
  omitted budgets do not silently inherit decoder defaults.
- Cancellation with no deadline settles the process group, descendants and
  output writers on supported platforms.
- Concurrent captures cannot evict active inputs or double-count reservations;
  storage exhaustion returns partial evidence rather than successful truncation.
- A native artifact larger than the former console cap remains analyzable through
  small pages. Decoder failure retains the original artifact and capture outcome.
- Real GPU and sanitizer workloads exercise both successful and failing paths;
  a passing mock transport test is insufficient.

This is a public-contract, concurrency and evidence-integrity change. It requires
an independent exact-diff review before final validation. The investigation is
not complete merely because the design is documented.
