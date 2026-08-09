from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal, cast

from mcp.server import MCPServer

from flameox.application import (
    ArtifactService,
    EvidenceLookupService,
    ExperimentService,
    FindingService,
    InvestigationService,
    RunSetService,
)
from flameox.domain import DomainError, ErrorCode
from flameox.storage import RunStore, Workspace


def register_resources(  # noqa: C901
    server: MCPServer[Any],
    workspace: Callable[[], Workspace],
) -> None:
    """Register resource projections independently from the MCP tool transport."""

    def error_payload(error: DomainError) -> str:
        return json.dumps({"ok": False, "error": error.to_detail()})

    @server.resource(
        "flameox://runs/{run_id}",
        mime_type="application/json",
        description="Bounded run manifest projection.",
    )
    async def run_resource(run_id: str) -> str:
        try:
            return RunStore(workspace()).read(run_id).model_dump_json(indent=2)
        except DomainError as error:
            return error_payload(error)

    @server.resource(
        "flameox://artifacts/{artifact_id}",
        mime_type="application/json",
        description="Artifact metadata without binary content.",
    )
    async def artifact_resource(artifact_id: str) -> str:
        try:
            return ArtifactService(workspace()).get(artifact_id).model_dump_json(indent=2)
        except DomainError as error:
            return error_payload(error)

    @server.resource(
        "flameox://investigations/{investigation_id}",
        mime_type="application/json",
        description="Current investigation projection.",
    )
    async def investigation_resource(investigation_id: str) -> str:
        try:
            return (
                InvestigationService(workspace())
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
    async def hypothesis_resource(hypothesis_id: str) -> str:
        try:
            return (
                InvestigationService(workspace())
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
    async def finding_resource(finding_id: str) -> str:
        try:
            return FindingService(workspace()).findings.read(finding_id).model_dump_json(indent=2)
        except DomainError as error:
            return error_payload(error)

    @server.resource(
        "flameox://experiments/{experiment_id}",
        mime_type="application/json",
        description="Immutable experiment protocol.",
    )
    async def experiment_resource(experiment_id: str) -> str:
        try:
            return (
                ExperimentService(workspace())
                .experiments.read(experiment_id)
                .model_dump_json(indent=2)
            )
        except DomainError as error:
            return error_payload(error)

    @server.resource(
        "flameox://experiments/{experiment_id}/trials",
        mime_type="application/json",
        description="Bounded immutable trial collection for an experiment.",
    )
    async def experiment_trials_resource(experiment_id: str) -> str:
        try:
            return (
                ExperimentService(workspace()).list_trials(experiment_id).model_dump_json(indent=2)
            )
        except DomainError as error:
            return error_payload(error)

    @server.resource(
        "flameox://experiments/{experiment_id}/trials/{trial_id}",
        mime_type="application/json",
        description="One immutable trial and its structured oracle receipt.",
    )
    async def experiment_trial_resource(experiment_id: str, trial_id: str) -> str:
        try:
            return (
                ExperimentService(workspace())
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
    async def run_set_resource(run_set_id: str) -> str:
        try:
            return RunSetService(workspace()).store.read(run_set_id).model_dump_json(indent=2)
        except DomainError as error:
            return error_payload(error)

    @server.resource(
        "flameox://evidence/{ref_type}/{ref_id}",
        mime_type="application/json",
        description="Authoritative persisted analysis or comparison evidence.",
    )
    async def evidence_resource(ref_type: str, ref_id: str) -> str:
        try:
            if ref_type not in {"analysis", "comparison"}:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    f"Unsupported evidence resource type {ref_type!r}.",
                )
            return (
                EvidenceLookupService(workspace())
                .get(cast(Literal["analysis", "comparison"], ref_type), ref_id)
                .model_dump_json(indent=2)
            )
        except DomainError as error:
            return error_payload(error)
