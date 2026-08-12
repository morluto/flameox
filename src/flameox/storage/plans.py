from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from pydantic import BaseModel, TypeAdapter

from flameox.storage.control_plane import ControlPlane, canonical_json
from flameox.storage.workspace import Workspace

type FieldSelection = set[str] | Mapping[str, FieldSelection | bool]


class AuthorizedPlanStore[PlanT: BaseModel]:
    """Typed access to server-owned, opaque, single-use plan capabilities."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        family: str,
        model: type[PlanT] | TypeAdapter[PlanT],
        output_only_fields: FieldSelection | None = None,
    ) -> None:
        self.control_plane = ControlPlane(workspace)
        self.family = family
        self._adapter = TypeAdapter(model) if isinstance(model, type) else model
        self.output_only_fields = output_only_fields or set()

    def issue(
        self,
        token: str,
        intent_digest: str,
        plan: PlanT,
        *,
        expires_at: datetime,
    ) -> None:
        canonical = self._adapter.validate_python(
            plan.model_dump(mode="python", exclude=self.output_only_fields)
        )
        self.control_plane.issue_plan(
            token=token,
            family=self.family,
            intent_digest=intent_digest,
            payload_json=canonical_json(
                canonical.model_dump(mode="json", exclude=self.output_only_fields)
            ),
            expires_at=expires_at,
        )

    def inspect(self, token: str) -> PlanT:
        _, payload = self.control_plane.inspect_plan(token=token, family=self.family)
        return self._adapter.validate_json(payload)

    def consume(self, token: str, *, expected_digest: str | None = None) -> PlanT:
        _, payload = self.control_plane.consume_plan(
            token=token,
            family=self.family,
            expected_digest=expected_digest,
        )
        return self._adapter.validate_json(payload)
