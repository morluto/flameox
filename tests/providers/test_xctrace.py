from __future__ import annotations

from pathlib import Path

import pytest

from flameox.providers.contracts import ProviderFailure
from flameox.providers.xctrace import XctraceProvider


@pytest.mark.golden
def test_xctrace_toc_projects_a_bounded_native_format_example(tmp_path: Path) -> None:
    toc = tmp_path / "toc.xml"
    toc.write_text('<trace><run name="sample"><process pid="42"/></run></trace>')

    result = XctraceProvider.analyze(toc, max_rows=2, provider_version="fixture")

    assert result.rows_observed == 3
    assert result.complete is False
    assert result.blocks[0]["values"] == {"toc_element_count": 3}
    assert result.blocks[1]["rows"] == [
        {"element": "process", "attributes": {"pid": "42"}, "text": None},
        {"element": "run", "attributes": {"name": "sample"}, "text": None},
    ]


@pytest.mark.golden
def test_xctrace_rejects_malformed_toc(tmp_path: Path) -> None:
    toc = tmp_path / "toc.xml"
    toc.write_text("<trace><run></trace>")

    with pytest.raises(ProviderFailure, match="table-of-contents XML is invalid") as failure:
        XctraceProvider.analyze(toc, max_rows=10, provider_version="fixture")

    assert failure.value.code == "DECODE_FAILURE"
