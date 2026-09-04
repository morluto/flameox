from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

import ijson
from ijson import IncompleteJSONError, JSONError

DEFAULT_EXCLUDE_PATHS = (
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "node_modules",
    "build",
    "dist",
    "target",
    "*.egg-info",
)

_RESULT_PREFIX = "runs.item.results.item"
_LOCATION_PREFIX = f"{_RESULT_PREFIX}.locations.item"
_MAX_IDENTIFIER_LENGTH = 1_024
_MAX_MESSAGE_LENGTH = 16_384
_MAX_DEFERRED_RESULTS = 1_000
_MAX_ANALYZERS = 16
_MAX_ANALYZER_FIELD_LENGTH = 256


@dataclass(frozen=True, slots=True)
class SarifAnalyzer:
    name: str
    version: str | None


@dataclass(frozen=True, slots=True)
class SarifCandidate:
    run_index: int
    result_index: int
    rule_id: str | None
    level: str | None
    message: str
    relative_path: str
    start_line: int | None
    start_column: int | None
    end_line: int | None
    end_column: int | None
    provider_fingerprint: str | None
    provider_confidence: float | None


@dataclass(frozen=True, slots=True)
class SarifCoverage:
    result_count: int
    normalized_count: int
    excluded_count: int
    invalid_count: int
    omitted_count: int


@dataclass(frozen=True, slots=True)
class SarifParseResult:
    supported: bool
    complete: bool
    candidates: tuple[SarifCandidate, ...]
    analyzers: tuple[SarifAnalyzer, ...]
    exit_status: int | None
    coverage: SarifCoverage
    limitations: tuple[str, ...]


@dataclass(slots=True)
class _Run:
    name: str | None = None
    version: str | None = None
    pending_results: list[_Result] = field(default_factory=list)
    pending_omitted_count: int = 0
    analyzer_field_truncated: bool = False


@dataclass(slots=True)
class _Result:
    run_index: int
    result_index: int
    rule_id: str | None = None
    level: str | None = None
    message_text: str | None = None
    message_markdown: str | None = None
    uri: str | None = None
    uri_base_id: str | None = None
    start_line: int | None = None
    start_column: int | None = None
    end_line: int | None = None
    end_column: int | None = None
    guid: str | None = None
    fingerprint_name: str | None = None
    fingerprint_value: str | None = None
    confidence: float | None = None
    location_count: int = 0
    current_location_index: int | None = None
    field_errors: set[str] = field(default_factory=set)


@dataclass(slots=True)
class _Document:
    version: str | None = None
    root_started: bool = False
    root_finished: bool = False
    current_run_index: int = -1
    current_run: _Run | None = None
    current_result: _Result | None = None
    result_count: int = 0
    analyzers: list[SarifAnalyzer] = field(default_factory=list)
    analyzers_omitted_count: int = 0
    analyzer_truncation_count: int = 0
    exit_codes: set[int] = field(default_factory=set)
    multiple_exit_statuses: bool = False


@dataclass(slots=True)
class _Normalization:
    source_root: Path
    include_paths: tuple[str, ...]
    exclude_paths: tuple[str, ...]
    default_exclude_paths: tuple[str, ...]
    maximum_candidates: int
    candidates: list[SarifCandidate] = field(default_factory=list)
    excluded_count: int = 0
    invalid_count: int = 0
    omitted_count: int = 0
    deferred_omitted_count: int = 0
    invalid_reasons: dict[str, int] = field(default_factory=dict)

    def add(self, raw: _Result, analyzer: _Run | None) -> None:
        candidate, reason, excluded = _normalize_result(
            raw,
            analyzer=analyzer,
            source_root=self.source_root,
            include_paths=self.include_paths,
            exclude_paths=self.exclude_paths,
            default_exclude_paths=self.default_exclude_paths,
        )
        if candidate is None:
            if excluded:
                self.excluded_count += 1
            else:
                self._invalid(reason or "invalid candidate")
            return
        if len(self.candidates) >= self.maximum_candidates:
            self.omitted_count += 1
            return
        self.candidates.append(candidate)

    def _invalid(self, reason: str) -> None:
        self.invalid_count += 1
        self.invalid_reasons[reason] = self.invalid_reasons.get(reason, 0) + 1

    def incomplete(self) -> None:
        self._invalid("the result did not finish before parsing stopped")

    def unknown_run(self) -> None:
        self._invalid("the result has no SARIF run identity")

    def defer(self, run: _Run, raw: _Result) -> None:
        if len(run.pending_results) >= _MAX_DEFERRED_RESULTS:
            run.pending_omitted_count += 1
            return
        run.pending_results.append(raw)

    def flush(self, run: _Run) -> None:
        for raw in run.pending_results:
            self.add(raw, run)
        run.pending_results.clear()
        self.deferred_omitted_count += run.pending_omitted_count
        run.pending_omitted_count = 0

    def finish_run(self, run: _Run) -> None:
        if run.name is not None:
            self.flush(run)
            return
        for _ in run.pending_results:
            self._invalid("the run has no SARIF tool driver identity")
        if run.pending_omitted_count:
            self.invalid_count += run.pending_omitted_count
            self.invalid_reasons["the run has no SARIF tool driver identity"] = (
                self.invalid_reasons.get("the run has no SARIF tool driver identity", 0)
                + run.pending_omitted_count
            )


def parse_sarif(
    path: Path,
    *,
    source_root: Path,
    include_paths: tuple[str, ...],
    exclude_paths: tuple[str, ...],
    default_exclude_paths: tuple[str, ...],
    maximum_candidates: int,
) -> SarifParseResult:
    """Read only the stable SARIF fields used by Flameox's candidate projection.

    The native report remains the authority for extensions and provider-specific
    detail. This parser deliberately streams result objects and never copies
    arbitrary `properties` values into normalized evidence.
    """
    normalization = _Normalization(
        source_root=source_root,
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        default_exclude_paths=default_exclude_paths,
        maximum_candidates=maximum_candidates,
    )
    document, parse_error = _stream_document(path, normalization)
    return _normalize_document(
        document,
        normalization=normalization,
        parse_error=parse_error,
    )


def _stream_document(path: Path, normalization: _Normalization) -> tuple[_Document, str | None]:
    document = _Document()
    parse_error: str | None = None
    try:
        with path.open("rb") as stream:
            for prefix, event, value in ijson.parse(stream):
                if _consume_document_event(document, normalization, prefix, event, value):
                    continue
                if document.current_result is not None:
                    _consume_result_event(document.current_result, prefix, event, value)
                    if prefix == _RESULT_PREFIX and event == "end_map":
                        run = document.current_run
                        if run is None:
                            normalization.unknown_run()
                        elif run.name is None:
                            normalization.defer(run, document.current_result)
                        else:
                            normalization.add(document.current_result, run)
                        document.current_result = None
    except (IncompleteJSONError, JSONError, OSError, ValueError) as error:
        parse_error = str(error) or type(error).__name__
    _finish_run(document, normalization)
    return document, parse_error


def _consume_document_event(
    document: _Document,
    normalization: _Normalization,
    prefix: str,
    event: str,
    value: object,
) -> bool:
    if prefix == "" and event == "start_map":
        document.root_started = True
        return True
    if prefix == "" and event == "end_map":
        document.root_finished = True
        return True
    if prefix == "version":
        document.version = value if event == "string" and isinstance(value, str) else None
        return True
    if prefix == "runs.item" and event == "start_map":
        document.current_run_index += 1
        document.current_run = _Run()
        return True
    if prefix == "runs.item" and event == "end_map":
        _finish_run(document, normalization)
        return True
    if prefix == "runs.item.tool.driver.name":
        run = document.current_run
        _set_run_text(run, "name", event, value)
        if run is not None and run.name is not None:
            normalization.flush(run)
        return True
    if prefix == "runs.item.tool.driver.version":
        _set_run_text(document.current_run, "version", event, value)
        return True
    if prefix == "runs.item.invocations.item.exitCode":
        exit_code = _integer(value) if event == "number" else None
        if exit_code is not None and exit_code >= 0 and document.current_run is not None:
            if exit_code in document.exit_codes:
                return True
            if len(document.exit_codes) < 2:
                document.exit_codes.add(exit_code)
            else:
                document.multiple_exit_statuses = True
        return True
    if prefix == _RESULT_PREFIX and event == "start_map":
        document.result_count += 1
        document.current_result = _Result(
            run_index=document.current_run_index,
            result_index=document.result_count - 1,
        )
        return True
    return False


def _finish_run(document: _Document, normalization: _Normalization) -> None:
    run = document.current_run
    if run is None:
        return
    normalization.finish_run(run)
    if run.name is not None:
        if len(document.analyzers) < _MAX_ANALYZERS:
            document.analyzers.append(SarifAnalyzer(name=run.name, version=run.version))
        else:
            document.analyzers_omitted_count += 1
        if run.analyzer_field_truncated:
            document.analyzer_truncation_count += 1
    document.current_run = None


def _normalize_document(
    document: _Document,
    *,
    normalization: _Normalization,
    parse_error: str | None,
) -> SarifParseResult:
    limitations: list[str] = []
    if not document.root_started or document.version != "2.1.0":
        limitations.append("Only provider-native SARIF 2.1.0 reports can be normalized.")
        return SarifParseResult(
            supported=False,
            complete=parse_error is None and document.root_finished,
            candidates=(),
            analyzers=(),
            exit_status=None,
            coverage=SarifCoverage(
                result_count=document.result_count,
                normalized_count=0,
                excluded_count=0,
                invalid_count=document.result_count,
                omitted_count=0,
            ),
            limitations=tuple(limitations),
        )
    if parse_error is not None:
        limitations.append(f"SARIF parsing stopped before the document ended: {parse_error}")
    elif not document.root_finished:
        limitations.append("SARIF parsing stopped before the document ended.")
    if document.current_result is not None:
        normalization.incomplete()

    for reason, count in sorted(normalization.invalid_reasons.items()):
        limitations.append(f"{count} SARIF result(s) were not normalized: {reason}.")
    if normalization.omitted_count:
        limitations.append(
            f"{normalization.omitted_count} SARIF result(s) exceeded the configured "
            "normalized-evidence limit."
        )
    if normalization.deferred_omitted_count:
        limitations.append(
            f"{normalization.deferred_omitted_count} SARIF result(s) preceded tool identity and "
            f"exceeded the {_MAX_DEFERRED_RESULTS}-result streaming buffer."
        )
    if document.analyzers_omitted_count:
        limitations.append(
            f"{document.analyzers_omitted_count} SARIF analyzer provenance record(s) were "
            f"omitted after the {_MAX_ANALYZERS}-analyzer bound."
        )
    if document.analyzer_truncation_count:
        limitations.append(
            f"{document.analyzer_truncation_count} SARIF analyzer identity record(s) contained "
            f"fields truncated to {_MAX_ANALYZER_FIELD_LENGTH} characters."
        )
    exit_status = next(iter(document.exit_codes)) if len(document.exit_codes) == 1 else None
    if document.multiple_exit_statuses or len(document.exit_codes) > 1:
        limitations.append("SARIF runs reported multiple analyzer exit statuses.")
    return SarifParseResult(
        supported=True,
        complete=parse_error is None and document.root_finished,
        candidates=tuple(normalization.candidates),
        analyzers=tuple(document.analyzers),
        exit_status=exit_status,
        coverage=SarifCoverage(
            result_count=document.result_count,
            normalized_count=len(normalization.candidates),
            excluded_count=normalization.excluded_count,
            invalid_count=normalization.invalid_count,
            omitted_count=(normalization.omitted_count + normalization.deferred_omitted_count),
        ),
        limitations=tuple(limitations),
    )


def _set_run_text(
    run: _Run | None,
    field_name: str,
    event: str,
    value: object,
) -> None:
    if event != "string" or not isinstance(value, str) or not value:
        return
    if run is not None:
        setattr(run, field_name, value[:_MAX_ANALYZER_FIELD_LENGTH])
        if len(value) > _MAX_ANALYZER_FIELD_LENGTH:
            run.analyzer_field_truncated = True


def _consume_result_event(result: _Result, prefix: str, event: str, value: object) -> None:
    if prefix == _LOCATION_PREFIX:
        if event == "start_map":
            result.current_location_index = result.location_count
            result.location_count += 1
        elif event == "end_map":
            result.current_location_index = None
        return
    if prefix == _RESULT_PREFIX and event == "end_map":
        return
    if prefix == f"{_RESULT_PREFIX}.ruleId":
        result.rule_id = _result_text(result, "rule ID", event, value, _MAX_IDENTIFIER_LENGTH)
    elif prefix == f"{_RESULT_PREFIX}.level":
        result.level = _result_text(result, "level", event, value, _MAX_IDENTIFIER_LENGTH)
    elif prefix == f"{_RESULT_PREFIX}.message.text":
        result.message_text = _result_text(result, "message", event, value, _MAX_MESSAGE_LENGTH)
    elif prefix == f"{_RESULT_PREFIX}.message.markdown":
        result.message_markdown = _result_text(result, "message", event, value, _MAX_MESSAGE_LENGTH)
    elif prefix == f"{_RESULT_PREFIX}.guid":
        result.guid = _result_text(result, "guid", event, value, _MAX_IDENTIFIER_LENGTH)
    elif prefix.startswith(f"{_RESULT_PREFIX}.partialFingerprints.") or prefix.startswith(
        f"{_RESULT_PREFIX}.fingerprints."
    ):
        fingerprint = _result_text(
            result,
            "provider fingerprint",
            event,
            value,
            _MAX_IDENTIFIER_LENGTH,
        )
        if fingerprint is not None:
            name = prefix.rsplit(".", maxsplit=1)[-1]
            if len(name) <= _MAX_IDENTIFIER_LENGTH and (
                result.fingerprint_name is None
                or (name, fingerprint) < (result.fingerprint_name, result.fingerprint_value or "")
            ):
                result.fingerprint_name = name
                result.fingerprint_value = fingerprint
    elif prefix == f"{_RESULT_PREFIX}.properties.confidence":
        confidence = _confidence(value) if event == "number" else None
        if confidence is not None:
            result.confidence = confidence
        elif event != "null":
            result.field_errors.add("provider confidence is not a 0-to-1 number")
    elif result.current_location_index == 0:
        _consume_first_location_event(result, prefix, event, value)


def _consume_first_location_event(result: _Result, prefix: str, event: str, value: object) -> None:
    if prefix == f"{_LOCATION_PREFIX}.physicalLocation.artifactLocation.uri":
        result.uri = _result_text(result, "artifact URI", event, value, _MAX_MESSAGE_LENGTH)
    elif prefix == f"{_LOCATION_PREFIX}.physicalLocation.artifactLocation.uriBaseId":
        result.uri_base_id = _result_text(
            result,
            "artifact URI base",
            event,
            value,
            _MAX_IDENTIFIER_LENGTH,
        )
    elif prefix == f"{_LOCATION_PREFIX}.physicalLocation.region.startLine":
        result.start_line = _position(result, "start line", event, value)
    elif prefix == f"{_LOCATION_PREFIX}.physicalLocation.region.startColumn":
        result.start_column = _position(result, "start column", event, value)
    elif prefix == f"{_LOCATION_PREFIX}.physicalLocation.region.endLine":
        result.end_line = _position(result, "end line", event, value)
    elif prefix == f"{_LOCATION_PREFIX}.physicalLocation.region.endColumn":
        result.end_column = _position(result, "end column", event, value)


def _result_text(
    result: _Result,
    label: str,
    event: str,
    value: object,
    maximum_length: int,
) -> str | None:
    if event != "string" or not isinstance(value, str) or not value:
        result.field_errors.add(f"{label} is not a non-empty string")
        return None
    if len(value) > maximum_length:
        result.field_errors.add(f"{label} exceeds its bounded length")
        return None
    return value


def _position(result: _Result, label: str, event: str, value: object) -> int | None:
    parsed = _integer(value) if event == "number" else None
    if parsed is None or parsed < 1:
        result.field_errors.add(f"{label} is not a positive integer")
        return None
    return parsed


def _integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, Decimal) and value == value.to_integral_value():
        return int(value)
    return None


def _confidence(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float | Decimal):
        return None
    parsed = float(value)
    return parsed if 0 <= parsed <= 1 else None


def _normalize_result(
    raw: _Result,
    *,
    analyzer: _Run | None,
    source_root: Path,
    include_paths: tuple[str, ...],
    exclude_paths: tuple[str, ...],
    default_exclude_paths: tuple[str, ...],
) -> tuple[SarifCandidate | None, str | None, bool]:
    if analyzer is None or analyzer.name is None:
        return None, "the run has no SARIF tool driver identity", False
    if raw.field_errors:
        return None, "; ".join(sorted(raw.field_errors)), False
    message = raw.message_text or raw.message_markdown
    if message is None:
        return None, "the result has no bounded message", False
    relative_path, path_error = _relative_source_path(raw.uri, raw.uri_base_id, source_root)
    if relative_path is None:
        return None, path_error, False
    if _is_excluded(relative_path, include_paths, exclude_paths, default_exclude_paths):
        return None, None, True
    if raw.end_line is not None and raw.start_line is not None and raw.end_line < raw.start_line:
        return None, "end line precedes start line", False
    return (
        SarifCandidate(
            run_index=raw.run_index,
            result_index=raw.result_index,
            rule_id=raw.rule_id,
            level=raw.level,
            message=message,
            relative_path=relative_path,
            start_line=raw.start_line,
            start_column=raw.start_column,
            end_line=raw.end_line,
            end_column=raw.end_column,
            provider_fingerprint=raw.guid or raw.fingerprint_value,
            provider_confidence=raw.confidence,
        ),
        None,
        False,
    )


def _relative_source_path(
    uri: str | None,
    uri_base_id: str | None,
    source_root: Path,
) -> tuple[str | None, str]:
    if uri is None:
        return None, "the result has no physical artifact URI"
    if uri_base_id not in {None, "SRCROOT"}:
        return None, "the result uses an unsupported URI base"
    parsed = urlsplit(uri)
    if parsed.scheme not in {"", "file"} or parsed.netloc not in {"", "localhost"}:
        return None, "the result URI is not a local source path"
    if parsed.query or parsed.fragment:
        return None, "the result URI contains a query or fragment"
    raw_path = unquote(parsed.path if parsed.scheme == "file" else uri)
    if not raw_path or "\x00" in raw_path or "\\" in raw_path:
        return None, "the result URI is not a normalized source path"
    lexical = PurePosixPath(raw_path)
    if ".." in lexical.parts:
        return None, "the result URI traverses outside the source root"
    candidate = Path(raw_path) if lexical.is_absolute() else source_root / lexical
    try:
        resolved = candidate.resolve(strict=False)
        relative = resolved.relative_to(source_root)
    except ValueError:
        return None, "the result URI is outside the declared source root"
    return relative.as_posix(), ""


def _is_excluded(
    path: str,
    include_paths: tuple[str, ...],
    exclude_paths: tuple[str, ...],
    default_exclude_paths: tuple[str, ...],
) -> bool:
    if _matches_any(path, exclude_paths):
        return True
    if include_paths and not _matches_any(path, include_paths):
        return True
    return not include_paths and _matches_any(path, default_exclude_paths)


def _matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    parts = PurePosixPath(path).parts
    for pattern in patterns:
        candidate = PurePosixPath(pattern)
        if any(token in pattern for token in "*?["):
            if fnmatchcase(path, pattern) or any(fnmatchcase(part, pattern) for part in parts):
                return True
        elif path == candidate.as_posix() or candidate in PurePosixPath(path).parents:
            return True
    return False
