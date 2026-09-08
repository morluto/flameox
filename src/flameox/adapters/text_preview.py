"""Bounded UTF-8 text fragments for artifact previews."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_text_fragments(path: Path, text_fragment_chars: int) -> Iterator[dict[str, Any]]:
    """Yield bounded fragments, retaining LF delimiters and replacing invalid UTF-8.

    Lines are one-based and fragment indices within each line are zero-based.
    line_terminated is true only when the fragment ends with LF. A final line
    without LF remains unterminated even when the file has been fully read.
    """
    if text_fragment_chars < 1:
        raise ValueError("text_fragment_chars must be positive")

    line = 1
    fragment = 0
    with path.open(encoding="utf-8", errors="replace", newline="\n") as stream:
        while text := stream.readline(text_fragment_chars):
            terminated = text.endswith("\n")
            yield {
                "line": line,
                "fragment": fragment,
                "text": text,
                "line_terminated": terminated,
            }
            if terminated:
                line += 1
                fragment = 0
            else:
                fragment += 1
