from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.e2e
@pytest.mark.process
def test_capture_analyzes_real_sdk_observations(tmp_path: Path) -> None:
    executable = str(Path(sys.executable).with_name("flameox"))
    environment = {**os.environ, "FLAMEOX_DATA_DIR": str(tmp_path / "evidence")}
    script = (
        "from flameox.sdk import observe, phase\n"
        "with phase('compile'):\n"
        "    observe('compiled', duration_ns=12)\n"
        "observe('finished')\n"
    )

    result = subprocess.run(
        [
            executable,
            "capture",
            "--provider",
            "observations",
            "--capability",
            "failures.summary",
            "--cwd",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            script,
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    rows = json.loads(result.stdout)["blocks"][1]["rows"]
    assert [(row["name"], row["phase"]) for row in rows] == [
        ("flameox.phase.start", "compile"),
        ("compiled", "compile"),
        ("flameox.phase.end", "compile"),
        ("finished", None),
    ]
    assert rows[1]["values"] == {"duration_ns": 12}

    invalid_events = (
        b'{"name":"","phase":null,"monotonic_ns":1,"values":{}}\n',
        b'{"name":"event","phase":"' + b"p" * 201 + b'","monotonic_ns":1,"values":{}}\n',
        b'{"name":"event","phase":null,"monotonic_ns":true,"values":{}}\n',
        b'{"name":"event","phase":null,"monotonic_ns":1,"values":{"x":NaN}}\n',
        b'{"name":"event","phase":null,"monotonic_ns":1,"values":{"x":"\\ud800"}}\n',
        b'{"name":"event","phase":null,"monotonic_ns":1,"values":{"x":"\\ud800"}}\n',
        b'{"name":"event","phase":null,"monotonic_ns":1,"values":{"x":1e999}}\n',
        b'{"name":"event","phase":null,"monotonic_ns":1,"values":{"x":'
        + b"[" * 9
        + b"0"
        + b"]" * 9
        + b"}}\n",
        b'{"name":"event","phase":null,"monotonic_ns":1,"values":{"x":['
        + b",".join([b"0"] * 257)
        + b"]}}\n",
        b'{"name":"event","phase":null,"monotonic_ns":1,"values":{"x":"'
        + b"x" * (16 * 1024)
        + b'"}}\n',
    )
    for index, invalid_event in enumerate(invalid_events):
        artifact = tmp_path / f"invalid-{index}.jsonl"
        artifact.write_bytes(invalid_event)
        rejected = subprocess.run(
            [executable, "analyze", "failures.summary", str(artifact), "--format", "observations"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert rejected.returncode == 1
        assert rejected.stdout == ""
        assert json.loads(rejected.stderr)["code"] == "DECODE_FAILURE"
