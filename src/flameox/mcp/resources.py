from __future__ import annotations

import json
from typing import Protocol, cast

from mcp.server import MCPServer
from mcp.server.mcpserver import Context

from flameox.adapters.kernel_build import kernel_build_json_schema
from flameox.adapters.kernel_validation import kernel_validation_json_schema
from flameox.application.artifacts import ArtifactService
from flameox.application.comparisons import RunSetService
from flameox.application.evidence_lookup import EvidenceLookupService
from flameox.application.experiments import ExperimentService
from flameox.application.pipelines import ArtifactPipelineService
from flameox.application.records import (
    FindingService,
    InvestigationService,
)
from flameox.application.run_projection import RunProjectionService
from flameox.application.static_analysis import StaticAnalysisService
from flameox.application.triton_autotune import TritonAutotuneService
from flameox.domain import DomainError, ErrorCode, EvidenceReferenceType
from flameox.storage import Workspace


class WorkspaceContext(Protocol):
    def require_workspace(self) -> Workspace: ...


def _workspace(ctx: Context) -> Workspace:
    state = cast(WorkspaceContext, ctx.request_context.lifespan_context)
    return state.require_workspace()


def register_resources[T: WorkspaceContext](server: MCPServer[T]) -> None:  # noqa: C901
    """Register resource projections independently from the MCP tool transport."""

    def error_payload(error: DomainError) -> str:
        return json.dumps({"ok": False, "error": error.to_detail()})

    @server.resource(
        "flameox://runs/{run_id}",
        mime_type="application/json",
        description="Bounded run manifest projection.",
    )
    async def run_resource(run_id: str, ctx: Context) -> str:
        try:
            workspace = _workspace(ctx)
            return RunProjectionService(workspace).get(run_id).model_dump_json(indent=2)
        except DomainError as error:
            return error_payload(error)

    @server.resource(
        "flameox://artifacts/{artifact_id}",
        mime_type="application/json",
        description="Artifact metadata without binary content.",
    )
    async def artifact_resource(artifact_id: str, ctx: Context) -> str:
        try:
            workspace = _workspace(ctx)
            return ArtifactService(workspace).get(artifact_id).model_dump_json(indent=2)
        except DomainError as error:
            return error_payload(error)

    @server.resource(
        "flameox://static-analysis/{run_id}/candidates",
        mime_type="application/json",
        description="Bounded source-scoped static-analysis candidates with an opaque next cursor.",
    )
    async def static_candidates_resource(run_id: str, ctx: Context) -> str:
        try:
            workspace = _workspace(ctx)
            result = StaticAnalysisService(workspace).candidates(run_id=run_id)
            return result.model_dump_json(indent=2)
        except DomainError as error:
            return error_payload(error)

    @server.resource(
        "flameox://triton-autotune/{run_id}/selections",
        mime_type="application/json",
        description="Bounded Triton autotune selections with an opaque next cursor.",
    )
    async def triton_autotune_selections_resource(run_id: str, ctx: Context) -> str:
        try:
            workspace = _workspace(ctx)
            return (
                TritonAutotuneService(workspace).selections(run_id=run_id).model_dump_json(indent=2)
            )
        except DomainError as error:
            return error_payload(error)

    @server.resource(
        "flameox://schemas/kernel-validation/v2",
        mime_type="application/schema+json",
        description="Published JSON Schema for flameox.kernel-validation.v2.",
    )
    async def kernel_validation_schema_resource() -> str:
        return json.dumps(kernel_validation_json_schema(), indent=2, sort_keys=True)

    @server.resource(
        "flameox://schemas/kernel-build",
        mime_type="application/schema+json",
        description="Published JSON Schema for the kernel-build provenance document.",
    )
    async def kernel_build_schema_resource() -> str:
        return json.dumps(kernel_build_json_schema(), indent=2, sort_keys=True)

    @server.resource(
        "flameox://pipelines/{pipeline_id}",
        mime_type="application/json",
        description="Immutable artifact-pipeline projection.",
    )
    async def pipeline_resource(pipeline_id: str, ctx: Context) -> str:
        try:
            workspace = _workspace(ctx)
            return ArtifactPipelineService(workspace).get(pipeline_id).model_dump_json(indent=2)
        except DomainError as error:
            return error_payload(error)

    @server.resource(
        "flameox://investigations/{investigation_id}",
        mime_type="application/json",
        description="Current investigation projection.",
    )
    async def investigation_resource(
        investigation_id: str,
        ctx: Context,
    ) -> str:
        try:
            workspace = _workspace(ctx)
            return (
                InvestigationService(workspace)
                .investigations.read(investigation_id)
                .model_dump_json(indent=2)
            )
        except DomainError as error:
            return error_payload(error)

    @server.resource(
        "flameox://hypotheses/{hypothesis_id}",
        mime_type="application/json",
        description="Current hypothesis revision.",
    )
    async def hypothesis_resource(hypothesis_id: str, ctx: Context) -> str:
        try:
            workspace = _workspace(ctx)
            return (
                InvestigationService(workspace)
                .hypotheses.read(hypothesis_id)
                .model_dump_json(indent=2)
            )
        except DomainError as error:
            return error_payload(error)

    @server.resource(
        "flameox://findings/{finding_id}",
        mime_type="application/json",
        description="Current finding revision.",
    )
    async def finding_resource(finding_id: str, ctx: Context) -> str:
        try:
            workspace = _workspace(ctx)
            return FindingService(workspace).get(finding_id).model_dump_json(indent=2)
        except DomainError as error:
            return error_payload(error)

    @server.resource(
        "flameox://experiments/{experiment_id}",
        mime_type="application/json",
        description="Bounded experiment outcome reconstructed from durable evidence.",
    )
    async def experiment_resource(experiment_id: str, ctx: Context) -> str:
        try:
            workspace = _workspace(ctx)
            return ExperimentService(workspace).status(experiment_id).model_dump_json(indent=2)
        except DomainError as error:
            return error_payload(error)

    @server.resource(
        "flameox://experiments/{experiment_id}/trials",
        mime_type="application/json",
        description="Bounded immutable trial collection for an experiment.",
    )
    async def experiment_trials_resource(
        experiment_id: str,
        ctx: Context,
    ) -> str:
        try:
            workspace = _workspace(ctx)
            return ExperimentService(workspace).list_trials(experiment_id).model_dump_json(indent=2)
        except DomainError as error:
            return error_payload(error)

    @server.resource(
        "flameox://experiments/{experiment_id}/trials/{trial_id}",
        mime_type="application/json",
        description="One immutable trial and its structured oracle receipt.",
    )
    async def experiment_trial_resource(
        experiment_id: str,
        trial_id: str,
        ctx: Context,
    ) -> str:
        try:
            workspace = _workspace(ctx)
            return (
                ExperimentService(workspace)
                .get_trial(trial_id, experiment_id=experiment_id)
                .model_dump_json(indent=2)
            )
        except DomainError as error:
            return error_payload(error)

    @server.resource(
        "flameox://run-sets/{run_set_id}",
        mime_type="application/json",
        description="Immutable frozen run cohort.",
    )
    async def run_set_resource(run_set_id: str, ctx: Context) -> str:
        try:
            workspace = _workspace(ctx)
            return RunSetService(workspace).store.read(run_set_id).model_dump_json(indent=2)
        except DomainError as error:
            return error_payload(error)

    @server.resource(
        "flameox://evidence/{ref_type}/{ref_id}",
        mime_type="application/json",
        description="Authoritative persisted analysis or comparison evidence.",
    )
    async def evidence_resource(
        ref_type: str,
        ref_id: str,
        ctx: Context,
    ) -> str:
        try:
            try:
                parsed_ref_type = EvidenceReferenceType(ref_type)
            except ValueError:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    f"Unsupported evidence resource type {ref_type!r}.",
                ) from None
            if parsed_ref_type not in {
                EvidenceReferenceType.ANALYSIS,
                EvidenceReferenceType.COMPARISON,
            }:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    f"Unsupported evidence resource type {ref_type!r}.",
                )
            workspace = _workspace(ctx)
            return (
                EvidenceLookupService(workspace)
                .get(parsed_ref_type, ref_id)
                .model_dump_json(indent=2)
            )
        except DomainError as error:
            return error_payload(error)
