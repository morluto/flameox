from __future__ import annotations


def exercise() -> str:
    value = sum(range(4))
    if value > 0:
        return "hit"
    return "miss"


if __name__ == "__main__":
    print(exercise())
