from __future__ import annotations

import ast
from pathlib import Path

import pytest

from flameox.workers.registry import ARTIFACT_WORKERS, worker_for_module

pytestmark = pytest.mark.unit


def test_worker_registry_has_one_definition_per_operation_and_module() -> None:
    operations = [definition.operation for definition in ARTIFACT_WORKERS]
    modules = [definition.module for definition in ARTIFACT_WORKERS]

    assert len(operations) == len(set(operations))
    assert len(modules) == len(set(modules))
    assert all(worker_for_module(module) is not None for module in modules)


def test_every_registered_worker_uses_the_shared_typed_transport() -> None:
    source_root = Path(__file__).parents[2] / "src"
    for definition in ARTIFACT_WORKERS:
        path = source_root.joinpath(*definition.module.split(".")).with_suffix(".py")
        tree = ast.parse(path.read_text())
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }

        assert "run_typed_worker" in names
        assert "run_typed_worker" in calls
        assert "run_worker" not in names
