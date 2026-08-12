# Investigations and analysis

Flameox records the path from a runtime symptom to a conclusion that another
person or agent can inspect and try to disprove.

```text
question → hypothesis → evidence request → capture/import
         → bounded analysis → discriminating experiment
         → supported, refuted, or inconclusive finding
```

Runs and evidence are defined in [Storage and evidence](storage-and-evidence.md).
Commands and MCP workflow are defined in [CLI and MCP](interfaces.md).

## Investigation records

An investigation has a question and optional parent. A hypothesis states a
falsifiable explanation, its expected observations, alternatives, and current
assessment. An experiment binds a predeclared protocol. A finding cites typed
evidence relationships and records limitations and next experiments.

These records are revisioned in SQLite and published into immutable evidence.
Updating an assessment appends a revision; it does not rewrite history.

Evidence relations have meaning:

- `supports` and `refutes` require evidence whose validity permits that claim;
- `context` supplies relevant observed background without deciding the claim;
- `limits` records a reason the evidence cannot support a stronger conclusion.

An invalid or inconclusive comparison cannot be relabeled as support merely by
attaching it to a finding.

## Workloads and experiments

`flameox.toml` is the authority for named workloads, oracles, experiments,
inference servers, scenarios, and fault experiments. A workload contains an
argument array, cwd, environment contract, parameter domain, timeout, optional
dependencies, and optional oracle. Valid manual edits and `configure_workload`
produce the same canonical definition.

An experiment declares:

- workload and bounded factor domains;
- explicit or Cartesian treatment combinations;
- randomized complete blocks and block count;
- primary metric, unit semantics, and polarity;
- estimand, practical threshold, and confidence level;
- deterministic random seed;
- validation-oracle requirement;
- maximum trials and resource bounds.

Planning expands and validates the complete matrix, resolves executable and
provider authority, randomizes treatment order within blocks, and stores the
complete plan behind an opaque capability. Execution consumes that capability;
it does not accept replacement factor values or commands.

Every attempted trial remains visible. Cancellation, timeout, environment
failure, workload failure, oracle failure, missing evidence, and explicit
exclusion are different states. Automatic paired comparison runs only when the
declared block and identity contract is complete.

## Oracles

Process exit success is not semantic correctness. An oracle produces a bounded
typed receipt tied to the workload instance and run.

Common strengths are:

- a contract check for one execution;
- reference agreement for a declared input;
- cross-treatment equivalence where the receipt genuinely binds both sides.

Planning binds oracle argv, semantic executable identity, outer launch binding,
cwd, environment, containment, network status, timeout, output paths, and
reference artifacts. Execution revalidates and consumes those exact fields.

A command observing one treatment cannot prove equivalence between treatments.
Missing or inapplicable oracle evidence keeps a confirmatory claim exploratory or
inconclusive.

## Analysis contract

Every analysis pins one `SnapshotHandle` before its first lookup. All runs,
artifacts, measurements, relationships, and evidence references resolve through
that handle. Later imports or record revisions are invisible.

Analysis output identifies:

- recipe and schema version;
- pinned corpus commit;
- complete input identities;
- observed and derived values;
- coverage and population denominator;
- exclusions and incompatibilities;
- limitations and unavailable fields;
- whether the result is exploratory, valid, invalid, or inconclusive.

Read-only recipes do not create claims. Recording an analysis preserves its
request, snapshot, inputs, result, and evidence references as a durable record.

## Supported recipes

### Hotspots

Aggregates compatible frame samples by semantic frame identity and returns
bounded source-linked totals, callers, callees, example stacks, and coverage.
Unextracted profiles differ from extracted profiles with no samples.

### Scaling

Summarizes declared experiment coordinates and per-trial samples before fitting
growth behavior. It requires one compatible measurement series and unit and
rejects environment changes masquerading as input-size effects.

### Run-set comparison

Compares frozen baseline and candidate cohorts using the declared paired
estimand. It validates membership, protocol, workload/source/environment/tool
identity, oracle status, metric dimension, unit, polarity, block coverage, and
sample eligibility before calculating an effect and interval.

### PyTorch and accelerator launches

Summarizes operator populations, direct and graph launches, kernels,
synchronization, correlations, and idle gaps from normalized traces. Added and
removed regions remain visible; analysis does not silently intersect them away.

### Memory growth

Keeps allocation, retained/high-water, RSS, and other memory concepts separate.
Series identity and chronological phase order are preserved before computing
growth.

### Execution path

Returns bounded coverage contexts and structured semantic observations. It does
not infer that an executed path is correct or that an unobserved path is dead.

### Failure population

Analyzes complete eligible terminal-run populations with an explicit time
window and denominator. It separates workload, environment, validation,
resource, cancellation, and infrastructure failures rather than clustering
everything by message text.

## Comparison validity

Comparison identity is dimension-specific. Required equality may include:

- workload definition and parameter coordinate;
- source revision and dirty-state digest;
- environment, interpreter, package, tool, and executable identity;
- hardware and accelerator identity;
- collector and extraction generation;
- protocol, profiler, schedule, model, tokenizer, and cache identity;
- validation strength and result;
- sampling interval and resource-observation semantics.

Each dimension is equal, different, unavailable, or not applicable. Missing is
not equal. A force/display option may show an exploratory delta but cannot turn
an invalid comparison into confirmatory evidence.

Run sets freeze explicit members against one corpus snapshot. Excluded members
remain in the denominator with reasons. Comparison never substitutes the latest
matching runs or silently analyzes only the intersection of paired blocks.

## Statistical policy

The protocol chooses the estimand before data collection. Supported paired
performance comparisons use preserved samples and complete declared blocks;
they do not pair values by incidental pyperf worker or value ordinal.

For positive ratio metrics, Flameox analyzes paired log ratios and reports a
back-transformed effect. Nonpositive values are ineligible rather than silently
dropped. Confidence intervals retain exact degeneracy and reject non-finite
results. Practical significance uses the declared threshold in addition to the
interval; statistical exclusion of zero alone is not a performance verdict.

Reliability and categorical outcomes use their own evaluable contracts. Exact
agreement, expected rejection, typed mismatch, unavailable backend, and process
failure remain distinct. Scalar factor identity is typed, so `1`, `1.0`, and
`"1"` do not collapse into the same treatment label.

## Evidence quality

A useful result answers four questions:

1. What population and exact snapshot did this analyze?
2. Which values were observed, which were derived, and which claim is inferred?
3. What evidence is missing, incompatible, excluded, or truncated?
4. What next experiment would distinguish the remaining explanations?

Profile-guided optimization without a representative benchmark and semantic
oracle remains exploratory. A passing suite does not prove the documented
behavior complete. When the evidence cannot decide the hypothesis, the correct
finding is inconclusive—not a softened claim of success.
