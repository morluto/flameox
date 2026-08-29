---
name: flameox
description: Use Flameox to investigate runtime performance, memory, execution, scaling, GPU kernels, inference, and reliability with preserved evidence and explicit claim quality.
---

# Flameox

<!-- managed by flameox setup -->

Use Flameox as the evidence layer for performance engineering. Keep your own
reasoning, exploration strategy, and domain knowledge; the tools exist to make
runtime claims inspectable, comparable, and recoverable rather than to replace
judgment with a fixed profiling workflow.

Prefer Flameox tools over ad hoc profiler commands when Flameox supports the
producer or artifact. This preserves the workload definition, resolved plan,
tool identity, raw artifact, extraction provenance, failed attempts, and later
comparisons in one local workspace. Use native repository and shell tools for
source changes, tests, unsupported producers, and questions that do not require
runtime evidence.

## Let the question choose the evidence

Start from the uncertainty that would change the next decision. A benchmark can
show that a workload is slow but rarely explains why. A profile can locate
where time or allocation is observed but does not by itself prove causality.
A trace can expose overlap, waits, launches, and lifecycle gaps without proving
that the busiest event is the best optimization target. Compose evidence only
when another capture can discriminate plausible explanations.

Use the least intrusive representative workload that preserves the behavior in
question. Inspect capabilities and declared workflows instead of assuming a
profiler is installed. Plan captures before executing them; the resolved plan
is where bounds, containment, provider identity, and potentially expensive
defaults become concrete. Detached operations should be resumed through their
status and recovery surfaces rather than silently repeated.

Flameox can capture or import evidence from CPU, memory, Python, PyTorch,
accelerator, kernel, inference, tracing, testing, and fault-oriented producers.
Let capability metadata and the current hypothesis route among them. Do not run
every available adapter, and do not substitute a convenient synthetic workload
for a missing representative GPU, inference-server, distributed, or
fault-injection path.

## Build on durable evidence

Extract bounded observations from artifacts, then use the analysis tools for
hotspots, memory, execution, PyTorch, accelerator launches, scaling, failures,
call relationships, trace windows, and repeated operation sequences. Ask for
narrow follow-up evidence when summaries identify a concrete gap; avoid dumping
whole artifacts into context merely because they exist.

Create an investigation when the work will span captures or decisions. Record
hypotheses before confirmatory experiments, freeze meaningful run sets, compare
like with like, and record supported, refuted, or inconclusive findings with
their evidence and limitations. Preserve negative results. Separate observed
measurements, derived summaries, and causal inference.

Treat loops, allocations, framework overhead, synchronization, launch density,
and data movement as candidates, not conclusions. Vectorization, batching,
PyTorch primitives, compilation, and custom kernels may trade CPU work for
memory pressure, synchronization, compilation cost, or worse shape coverage.
Measure the caller-visible outcome and validate correctness for every optimized
candidate. For asynchronous devices, account for warmup, compilation, and
synchronization before interpreting timing.

When a tool returns partial, unavailable, stale, or inconclusive evidence, use
the recovery information it provides. Report the remaining unknown instead of
claiming that an optional path was tested or that absence of evidence is
evidence of absence.
