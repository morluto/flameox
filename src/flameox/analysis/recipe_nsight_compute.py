from __future__ import annotations

import json
import re

from flameox.action_graph import ActionId, next_action_for_action, tool_action
from flameox.analysis.recipe_context import RecipeContext
from flameox.analysis.recipe_models import (
    NsightComputeAnalysisCoverage,
    NsightComputeAnalysisResult,
    NsightComputeRecaptureGuidance,
    NsightComputeRecaptureSelection,
    NsightComputeRuleProvenance,
    NsightComputeTargetQualification,
    NsightComputeTargetStatus,
)
from flameox.catalog import Snapshot
from flameox.domain import DomainError, digest_model
from flameox.evidence_scope import EvidenceScope, resolve_evidence_scope
from flameox.evidence_status import (
    available_availability,
    empty_availability,
    recoverable_unavailable_evidence,
    unavailable_availability,
)
from flameox.nsight_compute import NsightComputeProviderRuleFact, NsightComputeReportLocation

_ROOFLINE_SECTION = "SpeedOfLight_RooflineChart"


class NsightComputeRecipes(RecipeContext):
    def nsight_compute(
        self,
        input_id: str,
        *,
        limit: int | None = None,
        corpus_commit_id: str | None = None,
    ) -> NsightComputeAnalysisResult:
        corpus_commit_id = self._pinned_commit_id(corpus_commit_id)
        bounded = self._limit(limit)
        with self._open_snapshot(corpus_commit_id) as snapshot:
            scope = resolve_evidence_scope(snapshot, input_id)
            profile_artifact_ids = self._profile_artifact_ids(snapshot, scope)
            if not profile_artifact_ids:
                return self._unavailable_result(
                    snapshot=snapshot,
                    input_id=input_id,
                    reason="no_nsight_compute_profile",
                    limitation=(
                        "No Nsight Compute kernel-profile artifact is registered for this input."
                    ),
                )
            profile_run_ids = self._profile_run_ids(snapshot, profile_artifact_ids)

            facts, malformed_fact_count = self._rule_facts(
                snapshot,
                scope,
                profile_artifact_ids,
            )
            actions, action_total, extraction_truncated, section_ids = self._actions(
                snapshot,
                scope,
                profile_artifact_ids,
            )
            requested_kernel_name = self._requested_kernel_name(
                snapshot,
                profile_run_ids,
            )
            target = self._target_qualification(
                requested_kernel_name=requested_kernel_name,
                actions=actions,
                action_total=action_total,
                limit=bounded,
                action_evidence_truncated=extraction_truncated,
            )
            extraction_present = self._extraction_present(snapshot, scope, profile_artifact_ids)

        if not facts:
            roofline_collected = self._roofline_collected(section_ids)
            if not extraction_present and len(profile_run_ids) == 1:
                evidence = recoverable_unavailable_evidence(
                    "nsight_compute_not_extracted",
                    next_action=tool_action(
                        ActionId.EXTRACT_NSIGHT_COMPUTE,
                        run_id=profile_run_ids[0],
                    ),
                )
                limitation = (
                    "This Nsight Compute report has not produced typed normalized rule facts; "
                    "extract it before analysis."
                )
            else:
                evidence = empty_availability("no_provider_rule_facts")
                limitation = "The extracted report contains no provider rule facts."
            return NsightComputeAnalysisResult(
                corpus_commit_id=snapshot.commit.commit_id,
                input_id=input_id,
                findings=(),
                total=0,
                target=target,
                coverage=NsightComputeAnalysisCoverage(
                    section_count=len(section_ids),
                    global_runtime_reduction_findings=0,
                    local_hardware_efficiency_findings=0,
                    roofline_collected=roofline_collected,
                    normalized_evidence_truncated=extraction_truncated,
                ),
                recapture=self._recapture(
                    target=target,
                    extraction_truncated=extraction_truncated,
                    roofline_collected=roofline_collected,
                ),
                limitations=(
                    limitation,
                    "Native .ncu-rep bytes remain the authoritative provider report.",
                ),
                evidence=evidence,
            )

        ordered = sorted(facts, key=self._priority_key)
        coverage = NsightComputeAnalysisCoverage(
            section_count=len(section_ids),
            global_runtime_reduction_findings=sum(
                item.rule.speedup_estimation is not None
                and item.rule.speedup_estimation.meaning == "global_runtime_reduction"
                for item in ordered
            ),
            local_hardware_efficiency_findings=sum(
                item.rule.speedup_estimation is not None
                and item.rule.speedup_estimation.meaning == "local_hardware_efficiency_increase"
                for item in ordered
            ),
            roofline_collected=self._roofline_collected(section_ids),
            normalized_evidence_truncated=extraction_truncated,
        )
        limitations = [
            "Findings are ordered from typed provider rule facts; FlameOx does not infer a "
            "separate bottleneck.",
            "Native .ncu-rep bytes remain the authoritative provider report.",
        ]
        if malformed_fact_count:
            limitations.append(
                f"Ignored {malformed_fact_count} malformed normalized provider rule facts."
            )
        if extraction_truncated:
            limitations.append(
                "The normalized extraction was bounded; a missing rule is not evidence of "
                "absence in the native report."
            )
        if not coverage.roofline_collected and not extraction_truncated:
            limitations.append(
                "No roofline section was reported by the persisted provider extraction metadata."
            )
        return NsightComputeAnalysisResult(
            corpus_commit_id=snapshot.commit.commit_id,
            input_id=input_id,
            findings=tuple(ordered[:bounded]),
            total=len(ordered),
            target=target,
            coverage=coverage,
            recapture=self._recapture(
                target=target,
                extraction_truncated=extraction_truncated,
                roofline_collected=coverage.roofline_collected,
            ),
            limitations=tuple(limitations),
            evidence=available_availability(),
        )

    @staticmethod
    def _priority_key(
        item: NsightComputeRuleProvenance,
    ) -> tuple[int, float, int, str, str, int, int]:
        speedup = item.rule.speedup_estimation
        meaning_rank = {
            "global_runtime_reduction": 0,
            "local_hardware_efficiency_increase": 1,
            "unknown": 2,
            None: 3,
        }[speedup.meaning if speedup is not None else None]
        estimate = -(speedup.estimated_speedup if speedup is not None else 0.0)
        location = item.rule.location
        return (
            0 if speedup is not None else 1,
            estimate,
            meaning_rank,
            item.rule.section_identifier,
            item.rule.rule_identifier,
            location.range_index,
            location.action_index,
        )

    def _unavailable_result(
        self,
        *,
        snapshot: Snapshot,
        input_id: str,
        reason: str,
        limitation: str,
    ) -> NsightComputeAnalysisResult:
        return NsightComputeAnalysisResult(
            corpus_commit_id=snapshot.commit.commit_id,
            input_id=input_id,
            findings=(),
            total=0,
            target=NsightComputeTargetQualification(
                status=NsightComputeTargetStatus.UNQUALIFIED,
                reason="No Nsight Compute action evidence is available for target qualification.",
                observed_action_total=0,
            ),
            coverage=NsightComputeAnalysisCoverage(
                section_count=0,
                global_runtime_reduction_findings=0,
                local_hardware_efficiency_findings=0,
                roofline_collected=False,
                normalized_evidence_truncated=False,
            ),
            limitations=(limitation,),
            evidence=unavailable_availability(reason),
        )

    @staticmethod
    def _profile_artifact_ids(snapshot: Snapshot, scope: EvidenceScope) -> tuple[str, ...]:
        where, parameters = scope.predicate(
            run_column="ar.run_id",
            artifact_column="ar.artifact_id",
        )
        rows = snapshot.execute(
            "SELECT DISTINCT ar.artifact_id FROM artifact_registrations ar WHERE ("
            + where
            + ") AND ar.kind = 'kernel_profile' AND lower(coalesce(ar.producer, '')) "
            "IN ('nsight.compute', 'ncu', 'flameox.import') "
            "AND (lower(ar.display_name) LIKE '%.ncu-rep' "
            "OR lower(ar.display_name) LIKE '%.ncu-repz') ORDER BY ar.artifact_id",
            parameters,
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    @staticmethod
    def _profile_observation_predicate(
        scope: EvidenceScope,
        artifact_ids: tuple[str, ...],
    ) -> tuple[str, tuple[object, ...]]:
        where, parameters = scope.predicate(run_column="run_id", artifact_column="artifact_id")
        placeholders = ", ".join("?" for _ in artifact_ids)
        return f"({where}) AND artifact_id IN ({placeholders})", (*parameters, *artifact_ids)

    @staticmethod
    def _profile_run_ids(snapshot: Snapshot, artifact_ids: tuple[str, ...]) -> tuple[str, ...]:
        placeholders = ", ".join("?" for _ in artifact_ids)
        rows = snapshot.execute(
            "SELECT DISTINCT run_id FROM artifact_registrations WHERE artifact_id IN ("
            + placeholders
            + ") ORDER BY run_id",
            artifact_ids,
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    @staticmethod
    def _rule_facts(
        snapshot: Snapshot,
        scope: EvidenceScope,
        artifact_ids: tuple[str, ...],
    ) -> tuple[tuple[NsightComputeRuleProvenance, ...], int]:
        where, parameters = NsightComputeRecipes._profile_observation_predicate(
            scope,
            artifact_ids,
        )
        rows = snapshot.execute(
            "SELECT artifact_id, value_json FROM observations WHERE "
            + where
            + " AND kind = 'nsight_compute.rule' ORDER BY artifact_id, observation_id",
            parameters,
        ).fetchall()
        facts: dict[str, NsightComputeRuleProvenance] = {}
        malformed = 0
        for artifact_id, value_json in rows:
            if artifact_id is None:
                malformed += 1
                continue
            try:
                fact = NsightComputeProviderRuleFact.model_validate_json(str(value_json))
            except ValueError:
                malformed += 1
                continue
            provenance = NsightComputeRuleProvenance(artifact_id=str(artifact_id), rule=fact)
            facts[digest_model(provenance.model_dump(mode="json"))] = provenance
        return tuple(facts.values()), malformed

    @staticmethod
    def _actions(
        snapshot: Snapshot,
        scope: EvidenceScope,
        artifact_ids: tuple[str, ...],
    ) -> tuple[tuple[NsightComputeReportLocation, ...], int, bool, tuple[str, ...]]:
        where, parameters = NsightComputeRecipes._profile_observation_predicate(
            scope,
            artifact_ids,
        )
        rows = snapshot.execute(
            "SELECT artifact_id, name, value_json FROM observations WHERE "
            + where
            + " AND kind = 'profile.action' ORDER BY artifact_id, observation_id",
            parameters,
        ).fetchall()
        actions: dict[tuple[str, int, int, str], NsightComputeReportLocation] = {}
        for artifact_id, name, value_json in rows:
            if artifact_id is None:
                continue
            try:
                value = json.loads(str(value_json))
            except (TypeError, ValueError):
                continue
            if not isinstance(value, dict):
                continue
            range_index = value.get("range_index")
            action_index = value.get("action_index")
            if isinstance(range_index, bool) or not isinstance(range_index, int):
                continue
            if isinstance(action_index, bool) or not isinstance(action_index, int):
                continue
            action_name = str(name)
            location = NsightComputeReportLocation(
                range_index=range_index,
                action_index=action_index,
                action_name=action_name,
            )
            actions[(str(artifact_id), range_index, action_index, action_name)] = location

        metadata_rows = snapshot.execute(
            "SELECT artifact_id, value_json FROM observations WHERE "
            + where
            + " AND kind = 'profile.extraction' ORDER BY artifact_id, observation_id",
            parameters,
        ).fetchall()
        reported_counts: dict[str, int] = {}
        section_ids: set[str] = set()
        truncated = not metadata_rows
        for artifact_id, value_json in metadata_rows:
            if artifact_id is None:
                continue
            try:
                value = json.loads(str(value_json))
            except (TypeError, ValueError):
                truncated = True
                continue
            if not isinstance(value, dict):
                truncated = True
                continue
            count = value.get("action_count")
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                reported_counts[str(artifact_id)] = count
            else:
                truncated = True
            raw_section_ids = value.get("section_ids")
            if not isinstance(raw_section_ids, list):
                truncated = True
                continue
            for section_id in raw_section_ids:
                if not isinstance(section_id, str) or not section_id:
                    truncated = True
                    continue
                section_ids.add(section_id)
            if value.get("truncated") is not False:
                truncated = True
        observed = tuple(
            sorted(
                actions.values(),
                key=lambda item: (item.range_index, item.action_index, item.action_name),
            )
        )
        known = sum(reported_counts.values()) if reported_counts else len(observed)
        return observed, max(known, len(observed)), truncated, tuple(sorted(section_ids))

    @staticmethod
    def _roofline_collected(section_ids: tuple[str, ...]) -> bool:
        return any("roofline" in section_id.casefold() for section_id in section_ids)

    @staticmethod
    def _extraction_present(
        snapshot: Snapshot,
        scope: EvidenceScope,
        artifact_ids: tuple[str, ...],
    ) -> bool:
        where, parameters = NsightComputeRecipes._profile_observation_predicate(
            scope,
            artifact_ids,
        )
        row = snapshot.execute(
            "SELECT 1 FROM observations WHERE "
            + where
            + " AND kind = 'profile.extraction' LIMIT 1",
            parameters,
        ).fetchone()
        return row is not None

    @staticmethod
    def _requested_kernel_name(
        snapshot: Snapshot,
        run_ids: tuple[str, ...],
    ) -> str | None:
        if len(run_ids) != 1:
            return None
        try:
            manifest = snapshot.run(run_ids[0])
        except DomainError:
            return None
        value = manifest.semantics.scope.filters.get("kernel_name")
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _target_qualification(
        *,
        requested_kernel_name: str | None,
        actions: tuple[NsightComputeReportLocation, ...],
        action_total: int,
        limit: int,
        action_evidence_truncated: bool,
    ) -> NsightComputeTargetQualification:
        observed_actions = actions[:limit]
        if not actions:
            return NsightComputeTargetQualification(
                requested_kernel_name=requested_kernel_name,
                status=NsightComputeTargetStatus.INDETERMINATE,
                reason="The normalized extraction contains no action identity for target matching.",
                observed_actions=observed_actions,
                observed_action_total=action_total,
            )
        if requested_kernel_name is None:
            return NsightComputeTargetQualification(
                status=NsightComputeTargetStatus.UNQUALIFIED,
                reason=(
                    "No Nsight Compute kernel-name filter was recorded; the captured action "
                    "may not be the intended workload target."
                ),
                observed_actions=observed_actions,
                observed_action_total=action_total,
            )
        matches = NsightComputeRecipes._kernel_matches(requested_kernel_name, actions)
        if matches is None:
            status = NsightComputeTargetStatus.INDETERMINATE
            reason = "The recorded kernel-name expression is not a valid Python-compatible regex."
        elif not matches:
            if action_evidence_truncated:
                status = NsightComputeTargetStatus.INDETERMINATE
                reason = (
                    "The normalized action evidence was truncated, so the absence of a matching "
                    "action cannot establish a target mismatch."
                )
            else:
                status = NsightComputeTargetStatus.MISMATCH
                reason = (
                    "No captured action matches the recorded Nsight Compute kernel-name filter; "
                    "the report target is inconsistent with its capture semantics."
                )
        else:
            status = NsightComputeTargetStatus.MATCHED
            reason = "At least one captured action matches the recorded kernel-name filter."
        return NsightComputeTargetQualification(
            requested_kernel_name=requested_kernel_name,
            status=status,
            reason=reason,
            observed_actions=observed_actions,
            observed_action_total=action_total,
        )

    @staticmethod
    def _kernel_matches(
        requested_kernel_name: str,
        actions: tuple[NsightComputeReportLocation, ...],
    ) -> bool | None:
        if not requested_kernel_name.startswith("regex:"):
            return any(action.action_name == requested_kernel_name for action in actions)
        try:
            pattern = re.compile(requested_kernel_name.removeprefix("regex:"))
        except re.error:
            return None
        return any(pattern.search(action.action_name) is not None for action in actions)

    @staticmethod
    def _recapture(
        *,
        target: NsightComputeTargetQualification,
        extraction_truncated: bool,
        roofline_collected: bool = False,
    ) -> NsightComputeRecaptureGuidance | None:
        kernel_name = target.requested_kernel_name
        if extraction_truncated:
            reason = "The bounded normalized extraction was truncated."
            sections: tuple[str, ...] = ()
        elif target.status is not NsightComputeTargetStatus.MATCHED:
            reason = "The captured kernel is not qualified against a recorded target filter."
            sections = ()
        elif not roofline_collected:
            reason = "The persisted provider extraction metadata reports no roofline section."
            sections = (_ROOFLINE_SECTION,)
        else:
            return None
        return NsightComputeRecaptureGuidance(
            reason=reason,
            selection=NsightComputeRecaptureSelection(
                kernel_name=kernel_name,
                sections=sections,
                replay_mode="kernel",
            ),
            next_action=next_action_for_action(
                ActionId.PLAN_CAPTURE,
                context={"adapter": "nsight.compute"},
                instruction=(
                    "Select the declared workload and its parameters, then plan the displayed "
                    "Nsight Compute recapture selection."
                ),
            ),
        )
