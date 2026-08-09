from __future__ import annotations

import json
from pathlib import Path

from flameox.adapters.kernel_build import kernel_build_json_schema
from flameox.adapters.kernel_validation import kernel_validation_json_schema

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = PROJECT_ROOT / "src" / "flameox" / "schemas"


def main() -> None:
    SCHEMA_ROOT.mkdir(parents=True, exist_ok=True)
    schemas = {
        "kernel-build-v1.schema.json": kernel_build_json_schema(),
        "kernel-validation-v1.schema.json": kernel_validation_json_schema(),
    }
    for filename, schema in schemas.items():
        path = SCHEMA_ROOT / filename
        path.write_text(
            json.dumps(schema, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
