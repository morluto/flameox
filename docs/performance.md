# Performance investigation: 2026-10-11

This investigation compared commit `3fd3961030ac2e81fc13daac9629b62b658e196c`
with the optimizations described below, using the same locked environment on
macOS 26.3 arm64, Python 3.12.13, Pydantic 2.13.4, MCP 2.3.0, DuckDB 1.5.5,
PyArrow 25.0.0, and Memray 1.19.3. Results describe these workloads on this
host; they are not timing budgets or general provider speedup claims.

## Observed results

Each entry reports the median of five uninstrumented wall-clock samples.
Warm measurements include output materialization after one warmup. Cold CLI
and repository measurements alternate baseline and candidate subprocesses.
The warm measurements ran as separate baseline and candidate groups and
therefore provide weaker control over host drift. No other performance
workload ran concurrently with the reported measurements.

| Workload | Baseline | Candidate | Semantic oracle |
| --- | ---: | ---: | --- |
| Warm MCP catalog: list and serialize 51 tools | 159.9 ms | 17.7 ms | Identical serialized catalog digest |
| V8 CPU extraction: 20,001 nodes, 11 distinct normalized frames | 250.7 ms | 98.1 ms | Identical complete extraction digest |
| Pytest projection: 4,000 completed session fixtures and 4,000 other-worker teardown failures | 593.8 ms | 41.7 ms | Identical complete projection digest |
| Cold `flameox mcp inspect` | 955.2 ms | 775.3 ms | Byte-identical stdout |
| Cold single-tool inspection | 965.1 ms | 817.0 ms | Byte-identical stdout |
| Cold `flameox mcp inspect --full` | 989.3 ms | 835.1 ms | Byte-identical stdout |
| Publish, reopen, and query 1,000 manifests through the runtime | 6.174 s | 5.791 s | Existing end-to-end assertions on every page and evidence ID |

The repository workflow is noisy: baseline samples ranged from 5.880 to
6.800 seconds; candidate samples ranged from 5.159 to 6.354 seconds. Its
approximately 6% median reduction is an observation, not a statistically
established improvement.

## Changes and evidence

- MCP generates serialization schemas once per output model, then copies them
  for each public projection. Input schemas and annotations are also detached
  from caller mutation. CLI inspection reads the same registry directly and
  projects only the requested tools; compact inspection avoids generating
  output schemas that it would discard.
- Runtime validation uses each capability model's existing validator.
  Freshly validated results become cache-owned without another internal copy;
  outward results remain independent copies.
- V8 extraction reuses frame identities within one extraction. A CPU profile
  reduced canonical digest calls from 20,001 to 11. Coordinates are validated
  before cache lookup, and the key includes every varying identity and
  symbolization field. Heap extraction uses the same request-local cache.
- Pytest fixture analysis indexes workers with teardown failures once,
  replacing a repeated scan of all failures for each non-function fixture.
  Failures on another worker still leave completed fixtures complete.
- Repository queries reuse the validated inventory walk. Symlink validation
  retains component-by-component checks while avoiding repeated `Path`
  construction. Native artifact and manifest integrity checks remain active.
- Memray selects referenced frames in DuckDB before fetching Python rows.
  A native capture with 10,000 allocator frames and bounds of ten frames and
  ten aggregate rows produced an identical complete worker-result digest,
  including output-file digests. No controlled timing claim is made for this
  change. A retained native worker test checks bounded rows, referenced frame
  membership, and truthful dropped-contribution coverage.

The V8 before/after CPU profiles were preserved through Flameox as evidence
IDs `942ff80031e18b10faa3144b8a8524eea45a142b5a99d4a689082d6615f8620f`
and `1daca18941090022ae082e07e7915962a13ac9493f0f023953307484dc5c9039`.
These are local evidence, not repository-distributed benchmark artifacts.

## External guidance and rejected shortcuts

DeepWiki was consulted for Pydantic, DuckDB, Apache Arrow, and the MCP Python
SDK. Suggestions were checked against installed implementations and native
examples before changing behavior.

[Pydantic's performance guidance](https://docs.pydantic.dev/latest/concepts/performance/)
supports reusing validators rather than repeatedly constructing adapters.
Schema reuse still requires independent mutable public projections. DuckDB
filtering removes unnecessary Python materialization; it does not replace
the extraction bounds or evidence validation.

A proposed removal of the Nsight Arrow-to-JSON round trip was rejected.
Arrow map values default to association lists containing tuples; the existing
JSON conversion turns those tuples into arrays, whereas direct canonicalization
would stringify them. A scalar-only benchmark missed this semantic difference.
See the [Arrow conversion contract](https://arrow.apache.org/docs/python/generated/pyarrow.Table.html#pyarrow.Table.to_pylist).
The installed low-level MCP server also does not guarantee the strict rejection
contract used here, so Flameox retains schema admission and result validation.

## Retained verification

Run the public scale workloads with:

```console
uv run pytest -o addopts='' tests/performance/test_provider_scale.py tests/performance/test_runtime_scale.py -q
uv run python -m cProfile -o /tmp/flameox-scale.pstats -m pytest -o addopts='' tests/performance/test_runtime_scale.py -q
```

The V8 scale test imports native JSON through the isolated worker and checks
shared-frame aggregation plus cache isolation. The pytest scale test checks
worker-specific failure accounting through the public runtime. These tests
retain workload and semantic proofs, not hardware-dependent time assertions.
The exploratory timing harness and before/after profiles are local artifacts;
the grouped warm timings are historical observations rather than a committed
benchmark runner. Real MCP stdio and installed CLI workflows separately prove
catalog contracts, inline results, invalid input rejection, and capture replay.
