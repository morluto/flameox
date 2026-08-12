# Inference replay and profiling

Flameox coordinates maintained inference tools; it does not implement a load generator,
scheduler, profiler, or trace viewer. AIPerf 0.12 owns synthetic load and Mooncake replay,
`vllm bench serve` owns native aggregate benchmarks, vLLM owns profiler capture controls,
and PyTorch Profiler and Nsight Systems retain their native formats.

Servers and scenarios are declared in `flameox.toml`:

```toml
[workloads.vllm]
argv = ["vllm", "serve", "model"]

[inference_servers.local]
provider = "vllm"
mode = "managed"
workload = "vllm"
base_url = "http://127.0.0.1:8000"
model = "model"
tokenizer = "model"
model_revision = "immutable-model-revision"
tokenizer_revision = "immutable-tokenizer-revision"
quantization = "none"

[workloads.semantic-check]
argv = ["python", "scripts/check_inference.py"]

[workloads.semantic-check.oracle]
strength = "contract_check"
argv = ["python", "scripts/check_inference.py", "--emit-receipt"]
receipt_schema = "flameox.oracle-receipt.v1"

[inference_scenarios.replay]
server = "local"
provider = "aiperf"
endpoint_type = "chat"
streaming = true
request_rate = 10.0
burstiness = 1.0
concurrency = 8
warmup_request_count = 2
seed = 7
semantic_oracle_workload = "semantic-check"
```

`existing_local` targets are restricted to loopback HTTP endpoints and are probed only through
`/health` and `/v1/models`. They remain exploratory because Flameox cannot observe all server,
scheduler, and cache settings. Managed servers reuse the broker's sidecar lease and one absolute
deadline across startup, readiness, replay, and cleanup.

Server declarations are parsed by both provider and lifecycle. Managed servers
always carry a workload; existing-local servers cannot carry one. SGLang
servers always carry an absolute `benchmark_python` launcher, while vLLM
servers cannot carry that field. Scenario declarations are likewise parsed by
provider: SGLang bench scenarios require random input and output lengths,
vLLM/SGLang bench scenarios are streaming-only in v1, and trace timing options
belong only to AIPerf. Invalid mixtures are rejected at configuration load,
before replay or profiling code receives them.

Tool discovery is parsed into available and unavailable cases. An available
tool always has a resolved executable and cannot carry compatibility failure
or remediation fields. Replay plans remain flat on the wire but are parsed by
provider, preserving the scenario-specific trace, streaming, and random-input
constraints instead of widening them back into optional plan fields.

Confirmatory inference comparisons require immutable model and tokenizer revisions, exact
provider and managed-vLLM executable/version identities, a managed-server command identity,
complete accelerator identity, and a passing declared contract-check oracle. The oracle runs after a successful benchmark while the
managed server is still alive. It receives `FLAMEOX_INFERENCE_BASE_URL`,
`FLAMEOX_INFERENCE_RESULT_DIR`, and `FLAMEOX_ORACLE_RECEIPT`; its native output and receipt are
preserved, while normalized protocol evidence contains only the receipt's bounded status, reason,
error, and tolerance fields. Existing-local targets cannot satisfy the managed-command identity
requirement and therefore remain exploratory.

Per-run `cross_treatment_equivalence` receipts are rejected because a command observing only one
run cannot prove equivalence between treatments. A future comparison-time oracle must bind its
receipt to both runs before that strength can support confirmation. Plans also bind the current
configuration plus provider and managed-server executable identities; execution refuses a stale
plan after any of those identities changes.

AIPerf commands preserve fixed Mooncake timing when a trace artifact is declared. Native AIPerf
outputs are imported as sensitive immutable artifacts. vLLM commands always request structured
aggregate JSON and never enable `--save-detailed`; normalized evidence therefore contains no
prompt, generated text, or raw provider error text.

AIPerf is invoked with `--export-level records`. Its `profile_export.jsonl` is normalized one
line at a time using conversation/turn identifiers, published token and timing metrics, and only
safe error type/code fields. Provider prompt bodies, generated responses, and error messages stay
solely in the sensitive native artifact. `vllm bench serve` maps chat scenarios to the documented
`openai-chat` backend and `/v1/chat/completions` endpoint; non-streaming vLLM response scenarios
are rejected in v1 because the maintained CLI does not expose that response-mode control.

Each execution creates one canonical inference run. Native provider files remain immutable import
runs, while their normalized request or aggregate rows are published under the canonical run ID
with both source run and artifact provenance. The canonical manifest records the exact protocol
identity used by the existing run-set and comparison services and links the preserved native
registrations so later extraction needs only the canonical run ID. Successfully imported staging
copies are removed; failed and cancelled executions retain the same run ID and preserve any
partial provider artifacts emitted before cleanup.

Mooncake JSONL is parsed line by line with bounded line and row limits. Normalized
`inference_requests` preserve token counts, scheduled timing, and prefix-hash counts. Queue,
prefill, decode, and cache-hit fields remain null unless a maintained structured provider reports
them with request correlation. Prefix reuse is not treated as proof of a cache hit.

Profiling is diagnostic and `profile-run` requires the ID of a successful, protocol-compatible,
unprofiled measurement run. The diagnostic result and manifest limitations preserve that link.
Profiling plans are parsed into vLLM Torch, SGLang Torch, or Nsight Systems variants. Only the
SGLang Torch variant carries a profile ID and fixed SGLang options, while only the Nsight variant
carries an Nsight executable.
The canonical diagnostic manifest also records it as `source_measurement_run_id`, and evidence
publication includes the measurement run as an input. The linked run must have succeeded, retained
provider artifacts, published measurements, and carry the same unprofiled inference protocol.
Torch plans configure vLLM's profiler output directory and use `/start_profile` and `/stop_profile`.
Nsight plans use the CUDA profiler capture range, preserve `.nsys-rep`, and require official
`nsys export --type sqlite` before the existing Nsight SQLite extractor is used. Existing-local
servers cannot be profiled.

After capture, compressed PyTorch `.pt.trace.json.gz` files are registered as execution traces
and passed to the existing Perfetto extractor. Nsight `.nsys-rep` remains an opaque native
artifact; only its official SQLite export is passed to the Nsight extractor. Extraction shares
the capture deadline, and failures leave the native artifacts intact with partial coverage.

Inference protocol comparison reports exact mismatched fields across provider, trace, schedule,
model/tokenizer, server/cache, hardware, profiler, and semantic-oracle facets. Missing identity,
missing semantic oracles, and profiled-versus-unprofiled evidence keep conclusions exploratory.
Optional scheduling fields that do not apply to a run are not themselves missing evidence, but a
one-sided scheduling value is an exact incompatibility.

GPU capture is supported only where the corresponding vLLM, PyTorch/CUDA, or Nsight toolchain is
available. The deterministic tests use fake processes and endpoints; Windows GPU profiling remains
unsupported and unexecuted until it is validated on a supported vLLM environment.

For agent handoff, the opaque `plan_token` authorizes execution; `plan_id` is stable audit evidence
and may also be passed as `expected_plan_id`. Execution refuses when the configuration or
provider/server executable identity changed after review. Request queries return
both a total and typed evidence availability, and reject unknown run IDs instead of returning an
ambiguous empty page.
