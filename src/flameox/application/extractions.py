from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

from flameox.action_graph import ActionId
from flameox.adapters.memray import (
    MemrayExtractionResult,
    MemrayExtractor,
    memray_extraction_limits,
)
from flameox.application.operations import (
    CompletedOperationRecord,
    OperationAdapter,
    OperationRunner,
    OperationStatus,
)
from flameox.application.task_supervisor import TaskSupervisor
from flameox.domain import DomainError, ErrorCode
from flameox.filesystem import BoundedFileSystem
from flameox.storage import GenerationManifest, Workspace
from flameox.workers.memray_contract import MEMRAY_WORKER, MemrayExtractionLimits


class ExtractionManager:
    """Own durable lifecycle for provider extraction that may outlive one request."""

    _OPERATION = OperationAdapter(
        kind="evidence.extract.memray",
        start_action=ActionId.EXTRACT_MEMRAY,
        status_action=ActionId.GET_EXTRACTION,
    )

    def __init__(
        self,
        workspace: Workspace,
        *,
        supervisor: TaskSupervisor | None = None,
        memray_factory: Callable[[Workspace], MemrayExtractor] = MemrayExtractor,
    ) -> None:
        self.workspace = workspace
        self.memray_factory = memray_factory
        self.runner = OperationRunner(workspace, self._OPERATION, supervisor=supervisor)

    async def start_memray(
        self,
        run_id: str,
        idempotency_key: str,
        *,
        temporary_allocation_threshold: int = 1,
    ) -> OperationStatus:
        limits = memray_extraction_limits(self.workspace).validated_copy(
            update={"temporary_allocation_threshold": temporary_allocation_threshold}
        )
        return await self.runner.start(
            {
                "run_id": run_id,
                "extractor_profile": MEMRAY_WORKER.implementation,
                "limits": limits.model_dump(mode="json"),
            },
            idempotency_key,
            self._run_memray,
            items=(run_id,),
        )

    async def status(self, operation_id: str) -> OperationStatus:
        return await self.runner.status(operation_id)

    async def cancel(self, operation_id: str) -> OperationStatus:
        return await self.runner.cancel(operation_id)

    async def shutdown(self) -> None:
        await self.runner.shutdown()

    async def _run_memray(
        self,
        operation_id: str,
        progress: Callable[[str, float | None, float | None, str], Awaitable[None]],
    ) -> dict[str, object]:
        reused = self._completed_receipt(operation_id)
        if reused is not None:
            await progress(
                "reusing_completed_generation",
                1,
                1,
                "Reused the exact completed Memray extraction.",
            )
            return reused

        async def report(phase: str, completed: int, total: int | None) -> None:
            await progress(
                phase,
                None if total is None else completed,
                total,
                (
                    f"Memray extraction {phase.replace('_', ' ')}: {completed} records."
                    if total is None
                    else f"Memray extraction {phase.replace('_', ' ')}: "
                    f"{completed} of {total} records."
                ),
            )

        try:
            extractor = self.memray_factory(self.workspace)
            task = asyncio.current_task()
            assert task is not None

            def cancel() -> None:
                task.cancel()

            self.runner.set_cancel_hook(operation_id, cancel)
            result: MemrayExtractionResult = await extractor.extract(
                self._run_id(operation_id),
                limits=self._limits(operation_id),
                progress=report,
            )
            return {"extraction": result.model_dump(mode="json")}
        finally:
            self.runner.clear_cancel_hook(operation_id)

    def _run_id(self, operation_id: str) -> str:
        record = self.runner.store.read(operation_id)
        run_id = record.request.get("run_id")
        if not isinstance(run_id, str):
            raise DomainError(
                ErrorCode.INTERNAL_ERROR,
                "Extraction operation does not contain a valid run identifier.",
            )
        return run_id

    def _limits(self, operation_id: str) -> MemrayExtractionLimits:
        record = self.runner.store.read(operation_id)
        try:
            return MemrayExtractionLimits.model_validate(record.request.get("limits"))
        except ValueError as error:
            raise DomainError(
                ErrorCode.INTERNAL_ERROR,
                "Extraction operation does not contain valid limits.",
            ) from error

    def _completed_receipt(self, operation_id: str) -> dict[str, object] | None:
        current = self.runner.store.read(operation_id)
        for record in reversed(self.runner.store.list()):
            if (
                isinstance(record, CompletedOperationRecord)
                and record.operation_id != operation_id
                and record.operation == current.operation
                and record.request_digest == current.request_digest
            ):
                receipt = dict(record.terminal_receipt)
                if self._receipt_evidence_exists(receipt):
                    return receipt
        return None

    def _receipt_evidence_exists(self, receipt: dict[str, object]) -> bool:
        try:
            result = MemrayExtractionResult.model_validate(receipt.get("extraction"))
            commit = self.workspace.corpus.read_commit(result.corpus_commit_id)
            relative_manifest = (
                Path("generations") / result.evidence_generation_id / "manifest.json"
            )
            if relative_manifest.as_posix() not in commit.generation_manifests:
                return False
            manifest_path = self.workspace.paths.root / relative_manifest
            manifest = GenerationManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            return manifest.generation_id == result.evidence_generation_id and all(
                self._generation_file_matches(item.path, item.byte_length, item.sha256)
                for item in manifest.files
            )
        except (DomainError, OSError, ValueError):
            return False

    def _generation_file_matches(
        self,
        relative_path: str,
        byte_length: int,
        expected_sha256: str,
    ) -> bool:
        digest = hashlib.sha256()
        with BoundedFileSystem((self.workspace.paths.root,)).open_regular(
            self.workspace.paths.root / relative_path,
            max_bytes=byte_length,
            require_single_link=True,
        ) as descriptor:
            metadata = os.fstat(descriptor)
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
        return metadata.st_size == byte_length and digest.hexdigest() == expected_sha256
