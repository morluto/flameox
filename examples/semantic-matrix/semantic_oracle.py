from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def receipt(**values: Any) -> None:
    payload = {"schema_version": "flameox.oracle-receipt.v1", **values}
    Path(os.environ["FLAMEOX_ORACLE_RECEIPT"]).write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True)
    )


def main() -> None:
    treatment, backend, _dtype, _layout, length, state_mode, case = sys.argv[1:]
    case_id = f"{treatment}-{backend}-{case}"
    if backend == "unavailable":
        receipt(
            status="unsupported",
            reason="unsupported_capability",
            case_id=case_id,
            limitations=["The example deliberately marks this backend unavailable."],
        )
    elif case == "expected_rejection":
        receipt(status="pass", reason="expected_rejection", case_id=case_id)
    elif case == "mismatch" and treatment == "candidate":
        expected = float(sum(range(int(length))) + (state_mode == "initial_state"))
        observed = expected + 0.25
        receipt(
            status="fail",
            reason="cross_treatment_mismatch",
            case_id=case_id,
            output_field="forward",
            coordinate=[0],
            expected={"kind": "scalar", "value": expected},
            observed={"kind": "scalar", "value": observed},
            absolute_error=abs(observed - expected),
            relative_error=abs(observed - expected) / abs(expected),
            tolerance={"absolute": 1e-6, "relative": 1e-6, "equal_nan": False},
        )
    else:
        receipt(status="pass", reason="contract_match", case_id=case_id)


if __name__ == "__main__":
    main()
