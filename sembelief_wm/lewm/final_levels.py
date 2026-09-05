"""Create the predeclared, immutable final Sokoban holdout level set."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ..collectors.real import _fixed_sokoban_solution_length
from ..data.adapters.sokoban import SokobanAdapter


FORMAT = "lewm_final_holdout_levels_v1"


def _identity(level: dict[str, Any]) -> str:
    encoded = json.dumps(
        {
            "room_state": level["room_state"],
            "room_fixed": level["room_fixed"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_excluded(path: Path | None) -> set[str]:
    if path is None:
        return set()
    with path.open() as handle:
        payload = json.load(handle)
    return {_identity(level) for level in payload["levels"]}


def _validate(
    payload: dict[str, Any], *, count: int, seed_base: int,
    excluded: set[str],
) -> None:
    if payload.get("format") != FORMAT:
        raise RuntimeError("final holdout file has the wrong format")
    if payload.get("count") != count or payload.get("seed_base") != seed_base:
        raise RuntimeError("final holdout manifest does not match requested protocol")
    levels = payload.get("levels")
    if not isinstance(levels, list) or len(levels) != count:
        raise RuntimeError("final holdout level count mismatch")
    identities = [_identity(level) for level in levels]
    if len(set(identities)) != count:
        raise RuntimeError("final holdout contains duplicate layouts")
    overlap = set(identities) & excluded
    if overlap:
        raise RuntimeError(
            f"final holdout overlaps {len(overlap)} previously used layouts"
        )
    digest = hashlib.sha256("".join(identities).encode()).hexdigest()
    if payload.get("levels_sha256") != digest:
        raise RuntimeError("final holdout layout digest mismatch")


def create(
    output: Path, *, count: int, seed_base: int,
    excluded_path: Path | None,
) -> dict[str, Any]:
    excluded = _load_excluded(excluded_path)
    if output.exists():
        with output.open() as handle:
            payload = json.load(handle)
        _validate(
            payload, count=count, seed_base=seed_base, excluded=excluded
        )
        print(
            f"Reusing locked final holdout: {output} "
            f"sha256={payload['levels_sha256']}",
            flush=True,
        )
        return payload

    adapter = SokobanAdapter(require_real=True)
    environment = adapter.make_env(seed=seed_base)
    levels: list[dict[str, Any]] = []
    identities: set[str] = set()
    seed = seed_base
    while len(levels) < count:
        environment.reset(seed=seed)
        raw = environment._env
        if raw is None:
            raise RuntimeError("final holdout generation requires gym_sokoban")
        level = {
            "room_state": raw.room_state.tolist(),
            "room_fixed": raw.room_fixed.tolist(),
            "generation_seed": seed,
        }
        identity = _identity(level)
        solution_length = _fixed_sokoban_solution_length(level)
        if (
            identity not in excluded
            and identity not in identities
            and solution_length is not None
            and solution_length > 0
        ):
            level["solution_length"] = solution_length
            levels.append(level)
            identities.add(identity)
        seed += 1
    close = getattr(environment, "close", None)
    if callable(close):
        close()
    payload = {
        "format": FORMAT,
        "count": count,
        "seed_base": seed_base,
        "last_seed_examined": seed - 1,
        "excluded_source": (
            str(excluded_path.resolve()) if excluded_path is not None else None
        ),
        "excluded_layouts": len(excluded),
        "levels_sha256": hashlib.sha256(
            "".join(_identity(level) for level in levels).encode()
        ).hexdigest(),
        "levels": levels,
    }
    _validate(payload, count=count, seed_base=seed_base, excluded=excluded)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(
        f"Created locked final holdout: {output} "
        f"sha256={payload['levels_sha256']}",
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--seed-base", type=int, default=610_000_000)
    parser.add_argument("--exclude", type=Path)
    args = parser.parse_args()
    if args.count <= 0 or args.seed_base < 0:
        parser.error("count must be positive and seed-base non-negative")
    create(
        args.output, count=args.count, seed_base=args.seed_base,
        excluded_path=args.exclude,
    )


if __name__ == "__main__":
    main()
