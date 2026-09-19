"""Selection-oriented MCP instructions and tool descriptions."""

SERVER_DESCRIPTION = "Bounded local runtime evidence over explicit artifacts and process targets."

SERVER_INSTRUCTIONS = """Flameox analyzes explicit local artifacts and captures explicit process
targets. inspect_capabilities reports the formats, providers, option schemas, examples, and limits
for a selected capability. analyze never executes a target; capture_and_analyze does.
static.performance_candidates consumes SARIF rather than source files. Session results remain
ephemeral until preserved. Partial, retryable, and unavailable results are typed product states;
their next_action describes an executable recovery when one exists. Profiles are exploratory
evidence rather than proof of causality."""

TOOL_DESCRIPTIONS = {
    "inspect_capabilities": (
        "List capabilities by format or capture support, or return one capability's accepted "
        "formats, source cardinality, compatible providers, exact schemas, and examples."
    ),
    "prepare_providers": (
        "Prepare the requested Flameox-managed provider dependencies and report unresolved host "
        "or workload-interpreter requirements plus any reconnect handoff."
    ),
    "analyze": (
        "Analyze explicit existing native artifacts without executing a process. Capability "
        "discovery supplies accepted formats and the exact options schema."
    ),
    "capture_and_analyze": (
        "Execute one explicit argv target with a compatible provider, retain native capture "
        "diagnostics, and analyze the resulting artifacts."
    ),
    "preserve_evidence": (
        "Publish one session analysis and its native artifacts as immutable evidence in the "
        "active evidence store."
    ),
    "rescue_evidence": (
        "Publish one live session analysis into an agent-selected distinct new evidence directory "
        "and return a restart or reconnect handoff without changing the active store."
    ),
    "query_evidence": (
        "Search immutable evidence manifests with typed rows. Distinguishes an absent "
        "repository, an empty inventory, and a populated inventory with no matching rows."
    ),
}
