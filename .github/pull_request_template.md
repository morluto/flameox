<!--
PR title: type(optional-scope): imperative outcome
Example: fix(storage): preserve comparison evidence

The squash-merge commit and generated changelog use the PR title.
-->

## Summary

<!--
What user, agent, or maintainer problem does this solve? Link the issue with
"Fixes #123" when the PR closes one. If there is no issue, explain why.
Keep the description self-contained: a reviewer should understand the prior
behavior, the expected contract, and the new behavior without guessing.
-->

## Problem and expected behavior

<!-- For fixes, give the smallest triggering condition and the violated
invariant. For features, describe the use case and observable outcome. -->

## Change

<!-- Explain the chosen approach and why it fits flameox's architecture.
Mention intentional non-goals so the scope is clear. -->

## Suggested review order

<!-- Optional for larger or cross-cutting diffs. Name the semantic entry point,
then the supporting contracts, tests, and docs. Remove this section when the
diff is small enough to review directly. -->

## Contract and boundary impact

<!--
Complete the lines that apply. Use "none" or "not applicable" explicitly.
For behavior changes, name the semantic owner and the earliest changed stage.
-->

- Semantic owner and changed stage:
- Public CLI or MCP contract:
- Storage, artifact, provenance, or schema contract:
- Adapter, provider, platform, or workload compatibility:
- Cancellation, concurrency, security, or containment impact:
- Native artifact, failed-attempt, and observed/derived/inferred claim handling:

## Evidence and regression coverage

<!--
What tests or other evidence prove the behavior? Say whether a reproduction
was executed, source-derived, or only proposed. For fixes, explain how the
regression test would fail on the affected base revision when practical.
-->

- Tests added or updated:
- Base reproduction or other evidence:
- User-visible CLI or MCP output (if applicable):
- Remaining proof gaps:

For performance or resource-budget claims, include the workload, baseline and
candidate, metric and units, platform, warmup and repetition policy, measurement
method, variability, and the exact revisions compared.

For changes that affect evidence or conclusions:

- [ ] Observed, derived, and inferred claims remain distinguishable.
- [ ] Inputs, versions, provenance, and relevant corpus or artifact identity remain bound.
- [ ] Any compatibility, limitation, incompleteness, or uncertainty is exposed to callers.

## Validation

<!-- List only commands that actually ran, with the observed result. Remove
commands that were not run. Include focused checks as well as broader checks
when they materially support the change. State the exact commit or unchanged
working tree the results apply to; do not reuse evidence invalidated by a later
edit. -->

- `command` — result
- Validation tree or commit:

## Compatibility and safety

<!-- Call out breaking changes, migration needs, platform/provider sensitivity,
storage or protocol effects, security/privacy/containment implications, and
whether native artifacts, provenance, failed attempts, and experiment
structure remain preserved. Write "None" only after checking. -->

- Breaking changes or migration steps:
- Supported platform/provider changes:
- Security, privacy, or containment review:
- Performance or resource-budget impact:

## Review checklist

- [ ] The PR has one focused outcome and the title follows `type(scope): outcome`.
- [ ] Related issue is linked, or the reason for not linking one is stated above.
- [ ] Tests cover the changed observable behavior and meaningful failure path.
- [ ] For a fix, the regression proof fails on the affected base revision for the intended reason, or the proof gap is stated.
- [ ] Documentation or the owning contract is updated when behavior changed.
- [ ] User-visible CLI or MCP changes include a representative example or output.
- [ ] I checked the final diff for secrets, unrelated cleanup, and unsupported claims.
