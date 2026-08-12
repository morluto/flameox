"""Composition root for snapshot-pinned analysis recipes."""

from flameox.analysis.recipe_accelerator import AcceleratorRecipes
from flameox.analysis.recipe_execution import ExecutionRecipes
from flameox.analysis.recipe_failures import FailureRecipes
from flameox.analysis.recipe_hotspots import HotspotRecipes
from flameox.analysis.recipe_models import (
    AcceleratorLaunchAnalysisResult,
    AcceleratorLaunchComparison,
    AcceleratorLaunchRegion,
    AcceleratorStreamSummary,
    ExecutionAnalysisResult,
    ExecutionObservation,
    ExecutionObservationChange,
    FailureAnalysisResult,
    FailureChangePoint,
    FailureCluster,
    Hotspot,
    HotspotResult,
    KernelNameCount,
    MeasurementSummary,
    MemoryAnalysisResult,
    MemoryPhaseGrowth,
    OperatorSummary,
    PyTorchAnalysisResult,
    RuntimeResourceObservation,
    RuntimeResourceTotals,
    ScalingAnalysisResult,
    ScalingCorrelatedHotspot,
    ScalingFit,
    ScalingPoint,
    ScalingTrialSummary,
    WritableRootObservation,
)
from flameox.analysis.recipe_pytorch import PyTorchRecipes
from flameox.analysis.recipe_scaling import ScalingRecipes


class RecipeService(
    HotspotRecipes,
    ExecutionRecipes,
    PyTorchRecipes,
    AcceleratorRecipes,
    FailureRecipes,
    ScalingRecipes,
):
    """Snapshot-pinned facade combining the independent analysis recipe families."""


__all__ = [
    "AcceleratorLaunchAnalysisResult",
    "AcceleratorLaunchComparison",
    "AcceleratorLaunchRegion",
    "AcceleratorStreamSummary",
    "ExecutionAnalysisResult",
    "ExecutionObservation",
    "ExecutionObservationChange",
    "FailureAnalysisResult",
    "FailureChangePoint",
    "FailureCluster",
    "Hotspot",
    "HotspotResult",
    "KernelNameCount",
    "MeasurementSummary",
    "MemoryAnalysisResult",
    "MemoryPhaseGrowth",
    "OperatorSummary",
    "PyTorchAnalysisResult",
    "RecipeService",
    "RuntimeResourceObservation",
    "RuntimeResourceTotals",
    "ScalingAnalysisResult",
    "ScalingCorrelatedHotspot",
    "ScalingFit",
    "ScalingPoint",
    "ScalingTrialSummary",
    "WritableRootObservation",
]
