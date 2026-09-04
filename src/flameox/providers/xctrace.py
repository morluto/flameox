from __future__ import annotations

from pathlib import Path
from typing import Any

from defusedxml.ElementTree import ParseError, iterparse  # type: ignore[import-untyped]

from flameox.providers.contracts import ProviderAnalysis, ProviderFailure

_MAX_ATTRIBUTE_LENGTH = 2_000


class XctraceProvider:
    """Project bounded metadata from an xctrace table-of-contents export."""

    @staticmethod
    def analyze(path: Path, *, max_rows: int, provider_version: str) -> ProviderAnalysis:
        rows: list[dict[str, Any]] = []
        observed = 0
        try:
            for _event, element in iterparse(path, events=("end",)):
                observed += 1
                if len(rows) < max_rows:
                    rows.append(
                        {
                            "element": str(element.tag)[:500],
                            "attributes": {
                                str(key)[:200]: str(value)[:_MAX_ATTRIBUTE_LENGTH]
                                for key, value in sorted(element.attrib.items())[:32]
                            },
                            "text": (element.text or "").strip()[:_MAX_ATTRIBUTE_LENGTH] or None,
                        }
                    )
                element.clear()
        except (OSError, ParseError, ValueError) as error:
            raise ProviderFailure(
                "DECODE_FAILURE", "xctrace table-of-contents XML is invalid"
            ) from error
        if observed == 0:
            raise ProviderFailure("DECODE_FAILURE", "xctrace table of contents is empty")
        return ProviderAnalysis(
            provider_id="xctrace",
            provider_version=provider_version,
            blocks=[
                {"type": "metrics", "values": {"toc_element_count": observed}},
                {"type": "table", "rows": rows},
            ],
            rows_observed=observed,
            complete=observed <= max_rows,
            limitations=[
                "The table of contents describes the native trace bundle; it does not expose "
                "all recorded samples or establish causal dependence."
            ],
        )
