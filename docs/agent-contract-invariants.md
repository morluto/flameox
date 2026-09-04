# Agent contract invariants

An agent investigation crosses discovery, preparation, capture, analysis, preservation, and
retrieval. Each stage must preserve the meaning and usability of the preceding stage's result.
Use these invariants when designing changes and reviewing their behavioral evidence.

These are design and review requirements, not a statement that every current path satisfies them.
The linked issues record gaps found against revision
`c4965a23e49daa1804fdd14fdfe23cb9a48d77ec`; consult their current state for implementation progress.
Documentation changes alone do not resolve those gaps.

## Separate outcomes from their presentation

Determine execution status from the complete admitted execution record before applying response
limits. Retain a bounded aggregate outcome even when diagnostic rows are omitted. Reducing a byte
budget, changing diagnostic order, or adding unrelated table rows must not turn failure into
success. The transport projects the runtime's outcome; it must not reconstruct that outcome from
the visible sample of records. See [#465](https://github.com/morluto/flameox/issues/465).

Identify the process whose status was observed. A collector's nonzero exit does not establish the
workload's exit status, and a readable profile does not establish successful workload completion.
Report unknown status where it was not observed. Keep useful partial artifacts alongside failure
diagnostics. See [#457](https://github.com/morluto/flameox/issues/457).

## Make producer outputs usable by consumers

Exercise the next operation using the actual serialized response of the previous operation.
Required selectors, units, source identity, and pagination arguments must be available without
guessing private naming conventions or reading implementation code.

Redaction must preserve a usable selection mechanism. When a canonical artifact role contains
sensitive information, provide a safe selector rather than exposing private paths or removing the
only way to select an artifact. See [#464](https://github.com/morluto/flameox/issues/464).

A continuation needs a defined source lifetime. Preservation, cache eviction, and scratch cleanup
must account for outstanding references. Making evidence durable must either retain a working
continuation or return an explicit replacement and its accepted source. Do not advertise a token
whose only backing files have been deleted. Retain digest and request binding when designing that
handoff; do not keep scratch indefinitely. See [#467](https://github.com/morluto/flameox/issues/467).

## Treat schemas and recovery instructions as executable contracts

Field descriptions must match actual units and semantics. Row offsets and byte offsets are
different contracts. Equivalent valid input representations must produce equivalent observations;
formatting whitespace must not turn nonempty evidence into an empty complete result. Use maintained
parsers for the declared grammar and preserve malformed-input failures. See
[#462](https://github.com/morluto/flameox/issues/462) and
[#463](https://github.com/morluto/flameox/issues/463).

A recovery instruction must change the condition that caused the failure. State whether the action
belongs to Flameox, the workload environment, or the host. Distinguish installed, prepared, usable,
unsupported, and unknown states. Text summaries and structured results must agree about outstanding
requirements. Preservation provides durability; it cannot recover observations never collected or
excluded by a terminal provider limit. See [#458](https://github.com/morluto/flameox/issues/458),
[#461](https://github.com/morluto/flameox/issues/461), and
[#454](https://github.com/morluto/flameox/issues/454).

## Assign dependencies to their execution environment

Current setup prepares a version-pinned server environment from the complete requested provider
set and returns a launcher for reconnection. Independent external-engine preparation is a proposal
in [#459](https://github.com/morluto/flameox/issues/459), not an available setup mode.

Evaluate provider requirements by execution role:

| Role | Examples | Design obligation |
| --- | --- | --- |
| Standalone collector or analysis engine | py-spy, Perfetto Trace Processor | Bind the exact executable and compatibility requirements; evaluate whether server replacement is necessary. |
| Instrumentation inside the workload | Memray, coverage.py, PyTorch profiler | Check the declared workload interpreter/runtime; server installation alone does not prove capture readiness. |
| Host or platform prerequisite | perf, Nsight, ROCProfiler, xctrace | Report platform, tool, and permission requirements without silently changing host configuration. |
| Specialized benchmark workload | NVBench, Triton-based benchmarks | Validate the target's own execution contract and environment. |

A provider can have separate capture-time and analysis-time dependencies. Avoid assigning both to
one environment merely because they share a package name. Preparation should be idempotent for an
already-satisfied request. Reconnection should reflect an actual server replacement requirement,
with session-evidence consequences made explicit; see [#460](https://github.com/morluto/flameox/issues/460).
Recovery must respect complete-set selection semantics rather than recommending an incomplete
replacement list; see [#405](https://github.com/morluto/flameox/issues/405).

## Distinguish descriptions from statistical conclusions

A point estimate inside a practical margin describes the observed estimate. It does not by itself
establish equivalence. Likewise, a negative estimated difference does not establish improvement
when the uncertainty spans meaningful regression. Name descriptive classifications explicitly.
An inferential decision needs a stated method, assumptions, uncertainty rule, and inconclusive
outcome. Current experiment labels use the point estimate; [#466](https://github.com/morluto/flameox/issues/466)
tracks that distinction. Do not interpret those labels as confidence-qualified decisions.

Keep paired observations, failed or oracle-invalid pairs, sample size, and coverage visible.
Choosing a confidence interval or equivalence procedure requires evidence for the chosen estimand;
adding a confidence field alone does not validate the decision rule.

## References

- [CLI design guidelines](https://clig.dev/): composability, discovery, and actionable recovery.
- [MCP tool schema](https://modelcontextprotocol.io/specification/2025-06-18/schema): structured
  outcomes, tool errors, and resource links.
- [RFC 8259](https://www.rfc-editor.org/rfc/rfc8259): JSON grammar and insignificant whitespace.
- [NIST on confidence intervals and tests](https://www.itl.nist.gov/div898/handbook/prc/section1/prc15.htm)
  and [equivalence-testing guidance](https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/): uncertainty
  and the distinction between observed similarity and statistical equivalence.

These references support the principles. They do not prescribe Flameox's dependency defaults,
authorize installations, or replace provider-specific compatibility and boundedness evidence.
