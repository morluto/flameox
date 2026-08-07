from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

# The conditional imports below exist only for static type checkers; runtime
# exports are resolved lazily to keep short-lived transports lightweight.
# ruff: noqa: F405

if TYPE_CHECKING:
    from flameox.application.analysis_records import *  # noqa: F403
    from flameox.application.artifacts import *  # noqa: F403
    from flameox.application.capabilities import *  # noqa: F403
    from flameox.application.capture import *  # noqa: F403
    from flameox.application.compaction import *  # noqa: F403
    from flameox.application.comparisons import *  # noqa: F403
    from flameox.application.dependencies import *  # noqa: F403
    from flameox.application.detached import *  # noqa: F403
    from flameox.application.discovery import *  # noqa: F403
    from flameox.application.drilldown import *  # noqa: F403
    from flameox.application.evidence_lookup import *  # noqa: F403
    from flameox.application.evidence_query import *  # noqa: F403
    from flameox.application.execution_identity import *  # noqa: F403
    from flameox.application.execution_policy import *  # noqa: F403
    from flameox.application.experiments import *  # noqa: F403
    from flameox.application.faults import *  # noqa: F403
    from flameox.application.gc import *  # noqa: F403
    from flameox.application.imports import *  # noqa: F403
    from flameox.application.inference import *  # noqa: F403
    from flameox.application.inference_profiling import *  # noqa: F403
    from flameox.application.inference_providers import *  # noqa: F403
    from flameox.application.integrity import *  # noqa: F403
    from flameox.application.lifecycle import *  # noqa: F403
    from flameox.application.native_reducer import *  # noqa: F403
    from flameox.application.operations import *  # noqa: F403
    from flameox.application.otlp import *  # noqa: F403
    from flameox.application.pipelines import *  # noqa: F403
    from flameox.application.preflight import *  # noqa: F403
    from flameox.application.quarantine import *  # noqa: F403
    from flameox.application.records import *  # noqa: F403
    from flameox.application.recovery import *  # noqa: F403
    from flameox.application.reductions import *  # noqa: F403
    from flameox.application.repair import *  # noqa: F403
    from flameox.application.setup import *  # noqa: F403
    from flameox.application.status import *  # noqa: F403
    from flameox.application.summaries import *  # noqa: F403
    from flameox.application.viewers import *  # noqa: F403
    from flameox.application.workloads import *  # noqa: F403

_MODULES = (
    "analysis_records",
    "artifacts",
    "capabilities",
    "capture",
    "compaction",
    "comparisons",
    "dependencies",
    "detached",
    "discovery",
    "drilldown",
    "evidence_lookup",
    "evidence_query",
    "execution_identity",
    "execution_policy",
    "experiments",
    "faults",
    "gc",
    "imports",
    "inference",
    "inference_profiling",
    "lifecycle",
    "native_reducer",
    "otlp",
    "integrity",
    "inference_providers",
    "operations",
    "pipelines",
    "preflight",
    "quarantine",
    "records",
    "recovery",
    "reductions",
    "repair",
    "setup",
    "status",
    "summaries",
    "viewers",
    "workloads",
)


__all__ = [
    "AIPerfProfileRequest",
    "AdapterOption",
    "AdapterPreparationResult",
    "AnalysisMaterializationService",
    "ArtifactListItem",
    "ArtifactListResult",
    "ArtifactMetadataResult",
    "ArtifactPipeline",
    "ArtifactPipelineService",
    "ArtifactRegistrationSummary",
    "ArtifactService",
    "BandwidthFault",
    "CallEdgeResult",
    "CapabilityList",
    "CapabilityService",
    "CapabilitySetupManager",
    "CapabilitySetupResult",
    "CapturePlanRegistry",
    "CaptureResult",
    "CaptureService",
    "ClientSetupPlan",
    "CompactionResult",
    "CompactionService",
    "CompareRunSetsRequest",
    "ComparisonResult",
    "ComparisonService",
    "ConfigureInferenceScenarioRequest",
    "ConfigureInferenceServerRequest",
    "ConfigureWorkloadRequest",
    "CreateInvestigationRequest",
    "DeclaredWorkflowDetail",
    "DeclaredWorkflowList",
    "DeclaredWorkflowRequirement",
    "DeclaredWorkflowSummary",
    "DetachedCaptureManager",
    "DetachedCaptureRecord",
    "DetachedCaptureStatus",
    "DetachedProgress",
    "DiscoveryCoverage",
    "DrilldownService",
    "EvidenceInput",
    "EvidenceLookupResult",
    "EvidenceLookupService",
    "EvidenceQueryService",
    "EvidenceSummary",
    "EvidenceSummaryBundle",
    "EvidenceSummaryRequest",
    "EvidenceSummaryService",
    "ExecutionIdentityService",
    "ExecutionPolicy",
    "ExistingServerProbe",
    "ExperimentBlock",
    "ExperimentCell",
    "ExperimentConfig",
    "ExperimentPlan",
    "ExperimentPlanRegistry",
    "ExperimentRunResult",
    "ExperimentService",
    "ExperimentTrialCollection",
    "FaultExperimentConfig",
    "FaultExperimentPlan",
    "FaultExperimentResult",
    "FaultExperimentService",
    "FindingListResult",
    "FindingResult",
    "FindingService",
    "FrameDetail",
    "FreezeRunSetMember",
    "FreezeRunSetRequest",
    "GarbageApplyResult",
    "GarbageCollector",
    "GarbageEntry",
    "GarbagePlan",
    "GarbagePurgeResult",
    "GarbageRestoreResult",
    "ImportArtifactRequest",
    "ImportResult",
    "ImportService",
    "InferenceConfigurationList",
    "InferenceConfigurationResult",
    "InferenceProfilingPlan",
    "InferenceProfilingResult",
    "InferenceProfilingService",
    "InferenceReplayPlan",
    "InferenceReplayResult",
    "InferenceReplayService",
    "InferenceRequestItem",
    "InferenceRequestQueryResult",
    "InferenceScenarioConfig",
    "InferenceServerConfig",
    "InferenceToolDiscovery",
    "IntegrityIssue",
    "IntegrityResult",
    "IntegrityService",
    "InvestigationListResult",
    "InvestigationService",
    "LatencyFault",
    "LifecycleEvidenceService",
    "LifecycleItem",
    "LifecycleQueryResult",
    "LimitDataFault",
    "MaterializeAnalysisRequest",
    "MaterializedAnalysisResult",
    "MeasurementItem",
    "MeasurementQueryResult",
    "NativeDdminReducer",
    "NativePredicateClassification",
    "NativeReductionAttempt",
    "NativeReductionLimits",
    "NativeReductionResult",
    "NativeViewerLaunchResult",
    "NativeViewerPlan",
    "NativeViewerService",
    "OperationFailure",
    "OperationItemOutcome",
    "OperationProgress",
    "OperationRecord",
    "OperationRecovery",
    "OperationRunner",
    "OperationState",
    "OperationStatus",
    "OtlpExtractionResult",
    "OtlpTraceService",
    "OutcomeCount",
    "OutcomeExperimentResult",
    "PipelineComparison",
    "PipelineStage",
    "PipelineStageComparison",
    "PipelineStageDeclaration",
    "PlanReductionRequest",
    "PreflightService",
    "ProfileChange",
    "ProjectConfig",
    "ProxyFault",
    "QuarantineManifest",
    "QuarantineRestoreResult",
    "QuarantineService",
    "RecordFindingRequest",
    "RecordHypothesisRequest",
    "RecoveryInspection",
    "RecoveryResult",
    "RecoveryService",
    "ReductionAttemptSummary",
    "ReductionLimits",
    "ReductionPlan",
    "ReductionResult",
    "ReductionService",
    "RegisterPipelineRequest",
    "RepairEntry",
    "RepairPlan",
    "RepairResult",
    "RepairService",
    "ResetPeerFault",
    "ResolvedOracle",
    "ResolvedSetupPlan",
    "RunDiscoveryService",
    "RunFilter",
    "RunListResult",
    "RunSetService",
    "RunSummary",
    "RuntimeAction",
    "Scalar",
    "SetupInspection",
    "SetupOperation",
    "SetupPlan",
    "SetupReport",
    "SetupService",
    "SetupVerification",
    "SlicerFault",
    "SlowCloseFault",
    "StackExample",
    "StackExamplesResult",
    "SummaryArtifact",
    "SummaryClaim",
    "SummaryReference",
    "SummaryRun",
    "TimeoutFault",
    "TrashManifest",
    "VllmBenchServeRequest",
    "VllmProfilerControlClient",
    "WorkloadConfig",
    "WorkloadConfigurationResult",
    "WorkloadConfigurationStatus",
    "WorkloadDependencyService",
    "WorkloadDependencySetupResult",
    "WorkloadIdentityConfig",
    "WorkloadInspection",
    "WorkloadOracleConfig",
    "WorkloadRequirementsConfig",
    "WorkloadService",
    "WorkspaceStatus",
    "discover_inference_tool",
    "probe_existing_vllm_server",
    "render_evidence_summary_markdown",
    "workspace_status",
]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    for module_name in _MODULES:
        module = import_module(f"{__name__}.{module_name}")
        try:
            value = getattr(module, name)
        except AttributeError:
            continue
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
