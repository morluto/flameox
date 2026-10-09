# Semantic outcome example

This standard-library-only program emits a JSON record for one set of command-line inputs. It is
useful for exercising direct capture and artifact preview, but a single run does not compare
implementations or establish a semantic result. The program's `expected_rejection` case also
reports its expected marker itself; it is not an independent correctness oracle.

Run one sample from this directory and retain full console output:

```console
uv run flameox capture --provider direct --cwd "$PWD" \
  --console-output full --preserve -- \
  python semantic_workload.py reference portable float32 contiguous 4 stateless ordinary
```

Use [the investigation guide](../../docs/investigations.md) for the requirements on a meaningful
comparison or confirmatory experiment. In particular, validate candidate behavior with an
independent semantic oracle and preserve representative samples. The resulting evidence records
the command, execution provenance, captured output, coverage, and limitations; keep hypotheses
and interpretation in the agent's notes.
