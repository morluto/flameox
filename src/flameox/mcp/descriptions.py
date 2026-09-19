"""Selection-oriented MCP instructions and tool descriptions."""

SERVER_DESCRIPTION = "Bounded local runtime evidence over explicit artifacts and process targets."

SERVER_INSTRUCTIONS = """Use inspect_capabilities before analyze or capture_and_analyze when the
capability, artifact format, provider, or option contract is uncertain. analyze reads existing
native artifacts; it never captures a process. capture_and_analyze executes the explicit target and
then analyzes the captured native artifacts. static.performance_candidates consumes an existing
SARIF report and never scans source files. Session results are ephemeral until preserve_evidence is
called. A partial capture result is usable evidence: inspect its typed capture outcome and
analysis_failure, and do not rerun a completed capture merely because analysis failed. If a missing
managed provider returns a retryable action, call prepare_providers with the complete desired set,
preserve any live results, then reconnect with the returned launcher. Profiles are exploratory;
confirm causality with representative paired experiments and a semantic oracle."""

TOOL_DESCRIPTIONS = {
    "inspect_capabilities": (
        "Discover accepted artifact formats, source cardinality, compatible capture providers, "
        "and exact option schemas. Call this before analysis when selection is uncertain."
    ),
    "prepare_providers": (
        "Prepare Flameox-managed provider dependencies and report host requirements. Use only "
        "after a retryable provider response or when preflighting a known provider set."
    ),
    "analyze": (
        "Analyze existing native artifacts without executing a workload. Use inspect_capabilities "
        "for exact formats and options. For static.performance_candidates, pass SARIF; source "
        "files are not scanned."
    ),
    "capture_and_analyze": (
        "Execute one explicit argv target with a compatible provider, preserve native capture "
        "diagnostics, and analyze its artifacts. Use analyze instead when artifacts already exist."
    ),
    "preserve_evidence": (
        "Publish one session analysis and its native artifacts as immutable evidence. Use before "
        "reconnect or shutdown when the result must outlive this server process."
    ),
    "rescue_evidence": (
        "Publish one live session analysis into a distinct empty evidence directory when the "
        "configured store cannot accept it, then return the required restart handoff."
    ),
    "query_evidence": (
        "Search immutable evidence manifests with typed rows. Distinguishes an absent "
        "repository, an empty inventory, and a populated inventory with no matching rows."
    ),
}
