from __future__ import annotations

import importlib.util

PACKAGE_PROVIDERS = {
    "requires_memray": "memray",
    "requires_torch": "torch",
}
PROVIDER_MARKERS = frozenset(PACKAGE_PROVIDERS)


def provider_available(marker: str) -> bool:
    package = PACKAGE_PROVIDERS.get(marker)
    if package is not None:
        return importlib.util.find_spec(package) is not None
    raise ValueError(f"Unknown provider marker: {marker}")
