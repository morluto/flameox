"""Selection-oriented MCP instructions and tool descriptions."""

from flameox.mcp.catalog import ANALYSIS_TOOLS
from flameox.runtime_contracts import Capability

SERVER_DESCRIPTION = "Bounded local runtime evidence over explicit artifacts and process targets."

SERVER_INSTRUCTIONS = """Use Flameox for bounded local runtime evidence with native artifacts and
provenance. Call the named analysis tool that answers the question directly with explicit artifact
paths or preserved evidence sources. Each tool advertises its exact typed arguments; discovery is
optional. capture_* tools execute the supplied argv in an explicit cwd with a compatible typed
collector and immediately analyze its outputs. Comparisons consume existing baseline/candidate
artifacts; capture experiments declare randomized cases, repetitions, a metric and semantic oracle.
inspect_performance_candidates reads SARIF rather than scanning source. Follow next_page.tool with
next_page.arguments verbatim; pagination always reads existing evidence and never recaptures.
Session analyses expire at shutdown or eviction: preserve_evidence publishes immutable evidence;
query_evidence finds it later. Preservation refreshes next_page to immutable sources. Copy resource
URIs verbatim. Partial, retryable and unavailable results are typed product states; follow
next_action. Profiles locate exploratory evidence and do not prove causality or improvement."""

TOOL_DESCRIPTIONS = {
    "prepare_providers": (
        "Use when a provider result requests dependency preparation. Install the requested managed "
        "dependencies, then follow activation or reconnect guidance before retrying. Host "
        "profilers and workload-interpreter requirements receive guidance rather than installation."
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
        "with a named analysis tool. Reports inventory coverage and a next_page when more matches "
        "exist."
    ),
}


def analysis_description(capability: Capability) -> str:
    return (
        f"{capability.summary} Read existing {', '.join(capability.formats)} artifacts; "
        "returns input identities, metrics/rows, coverage and limitations. "
        "Follow next_page for more evidence; preserve_evidence makes it durable."
    )


def capture_description(capability: Capability) -> str:
    return (
        f"Execute an explicit argv target and collect native artifacts to: {capability.summary} "
        "Returns immediate evidence and execution provenance, including failed attempts. "
        f"For existing artifacts use {ANALYSIS_TOOLS[capability.id]}. "
        "Pagination reads captured artifacts without executing again."
    )
