from __future__ import annotations

from pathlib import Path

from flameox.config import WorkspaceConfig
from flameox.storage import Workspace


def write_workload(project: Path, *, message: str = "hello") -> None:
    (project / "flameox.toml").write_text(
        f"""
schema_version = 1

[workloads.echo]
argv = ["python", "-c", "print('{{message}}')"]
cwd = "."
timeout_seconds = 5

[workloads.echo.parameters]
message = ["{message}", "candidate"]
"""
    )


def disable_containment(workspace: Workspace) -> None:
    config = workspace.config.model_copy(
        update={
            "execution": workspace.config.execution.model_copy(update={"containment": "disabled"})
        }
    )
    assert isinstance(config, WorkspaceConfig)
    workspace.paths.config.write_text(config.to_toml())
