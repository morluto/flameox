# Direct MCP tool design

## Decision

Flameox exposes 26 task-named analysis tools, 20 matching capture tools where a compatible
provider exists, and four lifecycle tools. Each analysis or capture name represents one stable
runtime capability. Its MCP schema comes from the shared runtime contract: capability fields are
top-level inputs, and capture provider fields are typed fields alongside `provider.kind`. MCP has no
opaque request wrapper, options map, or capability selector. The complete mapping is in
[the interface catalog](interfaces.md).

This is a deliberate breaking change from `analyze`, `capture_and_analyze`, and
`inspect_capabilities`; no aliases remain. Runtime capability IDs, persisted evidence schemas, and
CLI `analyze` and `capture` commands remain stable. `flameox mcp inspect` provides compact catalog
discovery, and `--capability` prints direct examples and schemas for CLI inspection. MCP clients can
select from the named tools without a preliminary discovery tool call.

## Evidence workflow

The direct topology follows the evidence task:

```text
exact artifact paths ──> named analysis tool ──> bounded evidence
                                      │                 │
                                      │             next_page names
                                      │             that analysis tool
                                      ▼
typed target + collector ──> named capture tool ──> captured native artifacts
                                                        │
                                                immediate analysis
                                                        │
                                               preserve_evidence
                                                        │
                                                immutable evidence
```

An agent chooses the evidence question from the tool name, passes explicit ordered sources or an
explicit argv target and cwd, then follows the returned evidence and coverage. Comparisons consume
existing baseline and candidate artifacts through comparison analysis tools. Experiments are
capture inputs only for multi-source capabilities: they declare cases, randomization, repetitions,
metric, estimand, threshold, and optional semantic oracle. An analysis or capture `next_page` names an
analysis tool with complete arguments. It continues reading existing artifacts and never repeats a
capture. Preservation refreshes any continuation that previously referred to session scratch so it
uses immutable evidence sources.

## Design lessons

The local REA MCP implementation was inspected at source HEAD `0c6852b3` with a clean worktree. Its
live stdio server was package `rea@6.1.0`, Node `24.18.0`, and MCP SDK `2.3.1`; it advertised 139
tools with a compact serialized catalog of 1,856,991 bytes on 2026-10-09. The relevant source was
`docs/tool-design.md`, `src/contracts/toolContracts.ts`,
`src/server/toolRegistrationOptions.ts`, and `src/server/createServer.ts`. The installed build had
`build_commit: null`, so source-to-package parity was not established.

The design lesson I drew from this inspection is to make common agent questions directly callable,
with schemas generated from canonical typed contracts, while keeping shared behavior and validation
at one runtime boundary. Flameox applies this to its existing provider-neutral evidence contracts:
named tools select the capability, and the provider remains a typed implementation choice inside
capture. A shared catalog and detailed task schemas let MCP clients select a task without
requiring a separate discovery call, while retaining CLI schema inspection.

Flameox does not reproduce REA's mutable binary session model or duplicate full evidence as text.
Its authoritative outputs remain typed evidence with observed, derived, and inferred distinctions;
native bytes, provenance, coverage, limitations, and failed attempts remain inspectable and
optionally immutable. Text summaries stay concise, while structured content carries the bounded
result.

## Compatibility and contracts

- MCP tool names and input schema shapes are breaking changes. Existing MCP callers must migrate
  to the corresponding named task tool and flatten their old `request.options` fields.
- Capability IDs, native format adapters, evidence identities and schemas, CLI commands, and the
  `AnalysisRuntime` capability registry are unchanged by this transport redesign.
- Analysis remains read-only and idempotent. Capture remains an explicit local execution with
  effect annotations, typed argv/cwd/provider inputs, and no shell-string gateway.
- MCP inputs are projections of shared capability and provider models. They do not create a second
  source of validation truth; unknown fields and incompatible source/provider combinations fail
  before execution.
- `next_page` is an executable handoff to a named analysis tool. Capture pagination cannot rerun a
  workload. Preservation remains explicit and publishes immutable manifests and content-addressed
  native artifacts.

## Validation and proof gaps

The clean pre-change Flameox baseline at HEAD `706e205` exposed seven tools and 87,300 bytes
of compact JSON. The initial replacement at `d946123` exposes 50 tools and 911,561 bytes, an increase of 824,261
bytes. These sizes serialize `{"tools": [...]}` with each tool's
`model_dump(mode="json", by_alias=True, exclude_none=True)` and
`json.dumps(..., separators=(",", ":"))`. Direct tool selection trades the smaller gateway catalog
for complete schemas up front. Output schemas remain advertised and validated; none were removed
to make the catalog smaller. Serialized bytes do not establish model context cost or usability.

| Contract | Verification | Remaining proof gap |
| --- | --- | --- |
| Named tools and typed inputs | Fresh real stdio E2E validates every advertised schema and input example, checks malformed requests against the advertised schema, and rejects invalid source counts, formats, metrics, collector settings, and legacy wrappers. | No empirical tool-selection study or verification across all MCP hosts. |
| Evidence source and error boundaries | Real transport accepts omitted source kinds and reports actionable field paths without internal union labels; missing immutable evidence stays a typed failure. | JSON Schema does not express every cross-field or runtime admission constraint. |
| Capture, pagination and preservation | Real subprocess success and exit-7 failure, exact next-page calls, immutable source refresh, and replay; marker proves pagination does not rerun capture. Existing boundary workflows also cover rescue and restart. | This proves the exercised workflows, not every adapter. |
| Native collector integration | Real Node 24.18.0 CPU and heap captures through fresh stdio, bounded nonempty native rows, named read pagination and preservation. | No all-platform profiler, driver, permission or interpreter matrix. |
| Quality checks | `uv run pytest -q`: 267 passed, one skipped (`systemd-run` unavailable), five optional/performance cases deselected. Ruff, formatting, strict mypy, import boundaries, dead-code and dependency checks pass; the 1,000-manifest query/restart scale test and three npm subprocess tests pass. Coverage is reported without a percentage gate. | Passing checks do not prove causality, semantic equivalence, or performance improvement. |

The cleanup reduces test definitions from 369 to 215 and Python test/support code from 13,457 to
9,270 lines relative to HEAD `706e205`. It removes gateway discovery scaffolding, helper and
private-state tests, mocked scheduling and sink mechanics, and duplicated validation, capture,
pagination, and setup scenarios. Real process, corruption, containment, cancellation, partial
capture, native-format and source-identity workflows remain. Invalid UTF-8 and Unicode previews are
checked through preservation and replay rather than direct helper tests. Failure diagnostic
redaction and failed mixed provider preparation retain integration proof. The remaining test
policy and proof gaps are in [the testing guide](testing.md).

Executable revalidation is checked through actual broker admission, and no-deadline completion
and cancellation assertions share existing process workflows. Private cache assertions and
duplicate summaries are removed. Retained regression checks verify the declared Triton winner
and the combined stdout/stderr limit, rather than weaker membership or single-stream checks.

Memray's worker now returns only the two consumed frame tables and scalar metrics. Unused
measurement, call-edge and stack tables, null metadata columns, progress-file writes without a
reader, and their private protocol fields are removed. Both memory capabilities match a fixed
native-profile baseline exactly for public metrics, attribution rows, coverage, truncation and
limitations at row limits 1 and 100. The four metric aggregations still participate in bounded
frame selection, and the harness still enforces timeout and RSS limits.

## Live hardening follow-up

Fresh stdio probes on 2026-10-09 found and fixed these contract issues:

- Omitted page sizes previously supplied 100 even when startup policy allowed only two rows.
  Requests now inherit startup limits and can lower typed decoder, input, output, memory and
  provenance limits. Continuations retain the effective limits across preservation and restart.
- Pydantic coercion previously accepted numeric strings and booleans outside the advertised
  integer schema. The boundary now validates the original arguments against the generated JSON
  Schema before execution, while accepting integral JSON numbers such as `2.0`.
- Evidence selectors now advertise their non-null mutual exclusion. Query creation bounds require
  timezone-aware strings and ordered dates; admission failures include an actionable field path.
- Missing optional AIPerf dependencies now report unavailable capability through the worker
  protocol instead of looking like corrupt evidence. Output-limit recovery distinguishes a request
  increase within the startup ceiling from a server restart to raise that ceiling.

The Python SDK and type package are pinned to `2.3.0`. Fresh legacy stdio sessions and the SDK's
high-level client both worked; the latter negotiated protocol `2026-07-28`. The current 50-tool
catalog measures 999,785 compact bytes by the method above. The increase carries typed request
limits and corrected schema constraints; output contracts remain advertised in full.

Live probes exercised direct execution success, failure, timeout and cancellation; coverage,
cProfile, Memray, Node CPU and heap captures; native-format analysis and pagination; source changes;
immutable corruption; preservation, rescue and restart; and representative invalid schemas.
An actual Xcode Instruments Time Profiler capture also paged through its native trace metadata.
Independent follow-up probes found no further actionable issues in those exercised workflows.
Unavailable vendor profilers, GPU hardware and other operating systems remain proof gaps.

The default suite passes with 269 tests, one host skip (`systemd-run` unavailable), and five
optional/performance cases deselected. Ruff, formatting, strict mypy, import boundaries, dead-code
and dependency checks also pass. New schema regressions extend the existing real transport
workflow; a single new stdio workflow covers startup limits and preservation/replay.

The follow-up removes an unused Torch option compatibility wrapper, the pass-through MCP server
factory and package re-exports, and redundant subclass exception catches. The retained server class,
runtime and provider boundaries continue to own their existing behavior.
