from __future__ import annotations

import json
import sys


def evaluate(sequence_length: int, *, initial_state: bool) -> dict[str, float]:
    state = 1.0 if initial_state else 0.0
    forward = sum(float(index) for index in range(sequence_length)) + state
    return {
        "forward": forward,
        "backward": float(sequence_length),
        "initial_state_gradient": 1.0 if initial_state else 0.0,
    }


def main() -> None:
    treatment, backend, dtype, layout, length, state_mode, case = sys.argv[1:]
    if case == "expected_rejection":
        try:
            evaluate(-1, initial_state=state_mode == "initial_state")
            raise ValueError("sequence_length must be non-negative")
        except ValueError:
            print(json.dumps({"expected_rejection": True}))
            return
    result = evaluate(int(length), initial_state=state_mode == "initial_state")
    print(
        json.dumps(
            {
                "treatment": treatment,
                "backend": backend,
                "dtype": dtype,
                "layout": layout,
                "outputs": result,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
