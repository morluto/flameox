from __future__ import annotations

import os
import re
from collections.abc import Mapping

_DANGEROUS_ENVIRONMENT = {
    "BASH_ENV",
    "CDPATH",
    "ENV",
    "GIT_ASKPASS",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PYTHONHOME",
    "PYTHONPATH",
    "SSH_ASKPASS",
}
_DANGEROUS_ENVIRONMENT_PREFIXES = ("DYLD_", "GDB_", "GIT_CONFIG_", "LD_", "LLDB_")
_DANGEROUS_ENVIRONMENT_NAMES = {
    "JAVA_TOOL_OPTIONS",
    "NODE_EXTRA_CA_CERTS",
    "NODE_OPTIONS",
    "PERL5LIB",
    "PERL5OPT",
    "RUBYLIB",
    "RUBYOPT",
    "GDBINIT",
    "GDBHISTFILE",
    "PYTHONSTARTUP",
}
_CREDENTIAL_ENVIRONMENT = re.compile(
    r"(?:^|_)(?:TOKEN|PASSWORD|PASSWD|SECRET|KEY|CREDENTIALS?|COOKIES?)(?:_|$)"
)
_SAFE_CONTROL_OVERRIDES = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


def is_dangerous_environment_name(name: str) -> bool:
    normalized = name.upper()
    return (
        normalized in _DANGEROUS_ENVIRONMENT
        or normalized in _DANGEROUS_ENVIRONMENT_NAMES
        or normalized.startswith(_DANGEROUS_ENVIRONMENT_PREFIXES)
        or _CREDENTIAL_ENVIRONMENT.search(normalized) is not None
    )


def is_safe_control_override(name: str, value: str) -> bool:
    return _SAFE_CONTROL_OVERRIDES.get(name.upper()) == value


def blocked_environment_override(overrides: Mapping[str, str]) -> str | None:
    return next(
        (
            name
            for name, value in overrides.items()
            if is_dangerous_environment_name(name) and not is_safe_control_override(name, value)
        ),
        None,
    )
