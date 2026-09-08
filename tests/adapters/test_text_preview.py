from pathlib import Path

import pytest

from flameox.adapters.text_preview import iter_text_fragments


def test_fragments_preserve_unicode_newlines_and_positions(tmp_path: Path) -> None:
    content = "αβ\r\n🙂\nlast"
    path = tmp_path / "output.txt"
    path.write_text(content, encoding="utf-8", newline="")

    rows = list(iter_text_fragments(path, 2))

    assert "".join(row["text"] for row in rows) == content
    assert [(row["line"], row["fragment"], row["line_terminated"]) for row in rows] == [
        (1, 0, False),
        (1, 1, True),
        (2, 0, True),
        (3, 0, False),
        (3, 1, False),
    ]


def test_empty_and_final_unterminated_lines_are_bounded(tmp_path: Path) -> None:
    path = tmp_path / "output.txt"
    path.write_text("\nfinal", encoding="utf-8", newline="")

    rows = list(iter_text_fragments(path, 3))

    assert rows == [
        {"line": 1, "fragment": 0, "text": "\n", "line_terminated": True},
        {"line": 2, "fragment": 0, "text": "fin", "line_terminated": False},
        {"line": 2, "fragment": 1, "text": "al", "line_terminated": False},
    ]


def test_long_line_never_yields_a_fragment_over_limit(tmp_path: Path) -> None:
    content = "x" * 100_003 + "\n"
    path = tmp_path / "output.txt"
    path.write_text(content, encoding="utf-8", newline="")

    rows = list(iter_text_fragments(path, 1024))

    assert all(len(row["text"]) <= 1024 for row in rows)
    assert "".join(row["text"] for row in rows) == content
    assert rows[-1]["line_terminated"] is True


@pytest.mark.parametrize("size", [1, 2, 3, 128])
@pytest.mark.parametrize("payload", [b"", b"a\r\nb\rc\n", b"\xffa\xe6\x97\xa5\xf0\x9f\x99\x82"])
def test_fragment_projection_matches_declared_decoding(
    tmp_path: Path, size: int, payload: bytes
) -> None:
    path = tmp_path / "output.txt"
    path.write_bytes(payload)
    rows = list(iter_text_fragments(path, size))
    assert "".join(row["text"] for row in rows) == payload.decode("utf-8", errors="replace")
    assert all(0 < len(row["text"]) <= size for row in rows)
    assert path.read_bytes() == payload
