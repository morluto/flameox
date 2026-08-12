from __future__ import annotations

import json
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
        self._has_capability = _schema_has_property(
            self._adapter.json_schema(mode="serialization"), "plan_token"
        )

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
        if self._has_capability and canonical.model_dump(mode="python")["plan_token"] != token:
            raise ValueError("the issued capability must match the plan capability")
        payload = canonical.model_dump(mode="json", exclude=self.output_only_fields)
        self.control_plane.issue_plan(
            token=token,
            family=self.family,
            intent_digest=intent_digest,
            payload_json=canonical_json(_without_plan_capability(payload)),
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
        if self._has_capability:
            value["plan_token"] = token
        return self._adapter.validate_python(value)


def _without_plan_capability(value: object) -> object:
    if not isinstance(value, dict):
        return value
    return {key: item for key, item in value.items() if key != "plan_token"}


def _schema_has_property(value: object, name: str) -> bool:
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict) and name in properties:
            return True
        return any(_schema_has_property(item, name) for item in value.values())
    if isinstance(value, list):
        return any(_schema_has_property(item, name) for item in value)
    return False
