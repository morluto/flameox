"""Selection-oriented MCP instructions and tool descriptions."""

SERVER_DESCRIPTION = "Bounded local runtime evidence over explicit artifacts and process targets."

SERVER_INSTRUCTIONS = """Use Flameox to investigate local runtime time, memory, execution, and
reliability with bounded evidence and preserved native artifacts. Start with inspect_capabilities
list mode filtered by artifact format or capture support, then get the selected capability's exact
options and compatible providers. Use analyze for existing artifacts; capture_and_analyze executes
an explicit argv target and cwd. For baseline/candidate comparisons, capture inputs separately and
analyze them with a .compare capability; a capture experiment instead runs declared randomized
cases and paired repetitions. static.performance_candidates consumes SARIF rather than scanning
source files. Session results expire at shutdown or eviction: preserve_evidence makes them durable;
query_evidence finds preserved records. Copy returned resource URIs and next_page arguments
verbatim. Preservation returns a refreshed evidence-backed next_page; use it instead of a prior
scratch-backed page and do not rerun capture to paginate. Partial, retryable, and unavailable
results are typed product states; follow next_action when present. Profiles are exploratory
evidence rather than proof of causality."""

TOOL_DESCRIPTIONS = {
    "inspect_capabilities": (
        "Choose an evidence question before analysis or capture. List capabilities by artifact "
        "format or capture support, then get one capability's exact options, compatible providers, "
        "source cardinality, and valid request examples."
    ),
    "prepare_providers": (
        "Use when a provider result requests dependency preparation. Install the requested managed "
        "dependencies, then follow activation or reconnect guidance before retrying. Host "
        "profilers and workload-interpreter requirements receive guidance rather than installation."
    ),
    "analyze": (
        "Inspect or compare explicit existing native artifacts without executing a workload. "
        "Use inspect_capabilities for accepted formats and exact options; follow next_page for "
        "bounded drill-down, then preserve_evidence if the result must survive the session."
    ),
    "capture_and_analyze": (
        "Collect new evidence by executing an explicit argv target and cwd with a compatible "
        "provider. Analyze native outputs immediately; optionally preserve them or run a declared "
        "paired experiment. Follow next_page to inspect more rows without rerunning the workload."
    ),
    "preserve_evidence": (
        "Keep a live analysis beyond eviction or shutdown. Publish its native artifacts and "
        "provenance as immutable evidence, returning an evidence ID, resource URI, and refreshed "
        "next_page when available."
    ),
    "rescue_evidence": (
        "Use when the active evidence store needs recovery or a live analysis needs a distinct "
        "store. Publish into an explicit new evidence directory and return its restart handoff; "
        "the active store stays selected. Already-preserved captures use their immutable sources."
    ),
    "query_evidence": (
        "Find preserved evidence after a session ends. Search immutable manifest metadata with "
        "typed filters and rows, then read a returned resource URI or use its source selectors "
        "with analyze. Reports inventory coverage and a next_page when more matches exist."
    ),
}
