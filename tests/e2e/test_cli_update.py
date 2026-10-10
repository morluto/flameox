from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import tomlkit

from flameox import __version__
from flameox.setup import (
    ClientUpdatePlan,
    SetupClient,
    apply_client_setup,
    plan_client_setup,
    read_client_installations,
)


def plan_client_update(
    clients: list[SetupClient], version: str, *, home: Path
) -> list[ClientUpdatePlan]:
    return [item.plan_update(version) for item in read_client_installations(clients, home=home)]


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.process,
    pytest.mark.skipif(os.name != "posix", reason="Test uvx shim uses a POSIX executable script"),
]


@pytest.mark.parametrize(
    "failure", [None, "resolver", "version", "extras", "catalog", "timeout", "output"]
)
@pytest.mark.parametrize("second_client", [SetupClient.CODEX, SetupClient.OPENCODE])
def test_installed_update_verifies_every_environment_before_switching_client_pins(
    tmp_path: Path, failure: str | None, second_client: SetupClient
) -> None:
    cli = Path(sys.executable).with_name("flameox")
    plans = plan_client_setup([SetupClient.CURSOR], [], home=tmp_path)
    plans += plan_client_setup([second_client], ["memray"], home=tmp_path)
    apply_client_setup(plans)
    apply_client_setup(
        [
            plan.setup
            for plan in plan_client_update(
                [SetupClient.CURSOR, second_client], "0.1.0", home=tmp_path
            )
        ]
    )
    for setup_plan in plans:
        if setup_plan.client is SetupClient.CODEX:
            document = tomlkit.parse(setup_plan.path.read_text())
            entry = document["mcp_servers"]["flameox"]
        else:
            document = json.loads(setup_plan.path.read_text())
            entry = document[setup_plan.client.server_section]["flameox"]
        environment_key = "environment" if setup_plan.client is SetupClient.OPENCODE else "env"
        entry[environment_key] = {
            "UV_DEFAULT_INDEX": f"https://{setup_plan.client.value}.example.invalid/simple",
            "UV_CACHE_DIR": str(tmp_path / f"cache-{setup_plan.client.value}"),
        }
        setup_plan.path.write_text(
            tomlkit.dumps(document)
            if setup_plan.client is SetupClient.CODEX
            else json.dumps(document)
        )
    originals = {plan.path: plan.path.read_text() for plan in plans}
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "prepared.jsonl"
    uvx = bin_dir / "uvx"
    uvx.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "assert args[:2] == ['--no-config', '--no-sources']\n"
        "requirement = args[args.index('--from') + 1]\n"
        f"with Path({str(record)!r}).open('a') as stream:\n"
        "    stream.write(json.dumps({'args': args, 'index': os.getenv('UV_DEFAULT_INDEX'), "
        "'cache': os.getenv('UV_CACHE_DIR')}) + '\\n')\n"
        f"failure = {failure!r}\n"
        "if '[memory]' in requirement:\n"
        "    if failure == 'resolver':\n"
        "        print('resolver diagnostic', file=sys.stderr)\n"
        "        sys.exit(23)\n"
        "    if failure == 'timeout':\n"
        "        time.sleep(30)\n"
        "    if failure == 'output':\n"
        "        sys.stdout.buffer.write(b'x' * (2 * 1024 * 1024))\n"
        "        sys.exit(0)\n"
        "    if failure in ('version', 'extras') and args[-2] == '-c':\n"
        f"        version = '0.0.0' if failure == 'version' else {__version__!r}\n"
        "        print(json.dumps({'version': version, 'extras': []}))\n"
        "        sys.exit(0)\n"
        "    if failure == 'catalog' and args[-1] == 'inspect':\n"
        "        print('{\"tools\": []}')\n"
        "        sys.exit(0)\n"
        "offset = args.index('--from') + 2\n"
        "forwarded = args[offset + 1:]\n"
        "if args[offset] == 'python':\n"
        f"    os.execv({sys.executable!r}, [{sys.executable!r}, *forwarded])\n"
        f"os.execv({str(cli)!r}, [{str(cli)!r}, *forwarded])\n"
    )
    uvx.chmod(0o700)
    environment = {
        **os.environ,
        "HOME": str(tmp_path),
        "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
        "FLAMEOX_DATA_DIR": str(tmp_path / "evidence"),
        "NO_COLOR": "1",
    }
    command = [str(cli), "update", "--version", __version__, "--json"]
    if failure == "timeout":
        command += ["--timeout-seconds", "2"]
    completed = subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=20, check=False
    )
    if failure:
        assert completed.returncode == 2, completed.stdout + completed.stderr
        assert {path: path.read_text() for path in originals} == originals
        assert "Traceback" not in completed.stderr
        if failure == "resolver":
            assert "resolver diagnostic" in completed.stderr
    else:
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout)
        assert payload["restart_required"] is True
        assert {client["status"] for client in payload["clients"]} == {"updated"}
        assert all(f"=={__version__}" in path.read_text() for path in originals)
        prepared = [json.loads(line) for line in record.read_text().splitlines()]
        assert len(prepared) == 4
        assert {item["index"] for item in prepared} == {
            "https://cursor.example.invalid/simple",
            f"https://{second_client.value}.example.invalid/simple",
        }
        assert {item["cache"] for item in prepared} == {
            str(tmp_path / "cache-cursor"),
            str(tmp_path / f"cache-{second_client.value}"),
        }
        current = subprocess.run(
            command, env=environment, capture_output=True, text=True, timeout=10, check=False
        )
        assert current.returncode == 0, current.stderr
        assert json.loads(current.stdout)["restart_required"] is False
        assert len(record.read_text().splitlines()) == 4
    assert not SetupClient.CLAUDE.config_path(tmp_path).exists()
    assert not (tmp_path / "evidence").exists()
