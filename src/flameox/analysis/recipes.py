"""Composition root for snapshot-pinned analysis recipes."""

from flameox.analysis.recipe_accelerator import AcceleratorRecipes
from flameox.analysis.recipe_execution import ExecutionRecipes
from flameox.analysis.recipe_failures import FailureRecipes
from flameox.analysis.recipe_hotspots import HotspotRecipes
from flameox.analysis.recipe_nsight_compute import NsightComputeRecipes
from flameox.analysis.recipe_pytorch import PyTorchRecipes
from flameox.analysis.recipe_scaling import ScalingRecipes


class RecipeService(
    HotspotRecipes,
    ExecutionRecipes,
    PyTorchRecipes,
    NsightComputeRecipes,
    AcceleratorRecipes,
    FailureRecipes,
    ScalingRecipes,
):
    """Snapshot-pinned facade combining the independent analysis recipe families."""
