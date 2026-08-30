from __future__ import annotations

import json
from datetime import datetime

from pydantic import BaseModel, TypeAdapter

from flameox.storage.control_plane import ControlPlane, _serialize_control_payload
from flameox.storage.workspace import Workspace


class AuthorizedPlanStore[PlanT: BaseModel]:
    """Typed access to server-owned, opaque, single-use plan capabilities."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        family: str,
        model: type[PlanT] | TypeAdapter[PlanT],
    ) -> None:
        self.control_plane = ControlPlane(workspace)
        self.family = family
        self._adapter = TypeAdapter(model) if isinstance(model, type) else model

    def issue(
        self,
        token: str,
        intent_digest: str,
        plan: PlanT,
        *,
        expires_at: datetime,
    ) -> None:
        canonical = self._adapter.validate_python(
            plan.model_dump(mode="python", exclude_computed_fields=True)
        )
        if getattr(canonical, "plan_token", None) != token:
            raise ValueError("the issued capability must match the plan capability")
        payload = canonical.model_dump(mode="json", exclude_computed_fields=True)
        payload.pop("plan_token")
        self.control_plane.issue_plan(
            token=token,
            family=self.family,
            intent_digest=intent_digest,
            payload_json=_serialize_control_payload(payload),
            expires_at=expires_at,
        )

    def inspect(self, token: str) -> PlanT:
        _, payload = self.control_plane.inspect_plan(token=token, family=self.family)
        return self._parse(payload, token)

    def consume(self, token: str, *, expected_digest: str | None = None) -> PlanT:
        _, payload = self.control_plane.consume_plan(
            token=token,
            family=self.family,
            expected_digest=expected_digest,
        )
        return self._parse(payload, token)

    def _parse(self, payload: str, token: str) -> PlanT:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("authorized plan payload must be an object")
        value["plan_token"] = token
        return self._adapter.validate_python(value)
