#!/usr/bin/env python3
"""Stage 2 diagnostic SeedSequence collision probe (spec section 3.2)."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = 20261001
NEW_NS = 30
OLD_SEEDS = tuple(range(20)) + tuple(range(1000, 1020))
NEW_SEEDS = tuple(range(2000, 2020))
OLD_BOOT = tuple(range(15)) + tuple(range(100, 120))
NEW_BOOT = tuple(range(9))


def old_identities() -> List[Tuple[str, Tuple[int, ...]]]:
    out: List[Tuple[str, Tuple[int, ...]]] = []
    for seed in OLD_SEEDS:
        for component in range(1, 7):
            out.append((f"legacy/object/seed={seed}/component={component}", (ROOT, seed, component)))
        out.append((f"legacy/object/seed={seed}/component=21", (ROOT, seed, 21)))
        for checkpoint in range(1, 5):
            out.append((f"legacy/tie/seed={seed}/checkpoint={checkpoint}", (ROOT, seed, 22, checkpoint)))
    for k in OLD_BOOT:
        out.append((f"legacy/bootstrap/k={k}", (ROOT, 7, k)))
    return out


def new_identities() -> List[Tuple[str, Tuple[int, ...]]]:
    out: List[Tuple[str, Tuple[int, ...]]] = []
    per_seed = (
        ("weights_w1_w2_o1", (1,)),
        ("weights_w2_o0", (21,)),
        ("A_train", (2,)),
        ("A_test", (5,)),
        ("A_tie", (22, 1)),
        ("lag1_train", (31,)),
        ("lag1_test", (32,)),
        ("lag1_tie", (22, 2)),
        ("background_drive", (40,)),
    )
    for seed in NEW_SEEDS:
        for name, suffix in per_seed:
            out.append((f"new/{name}/seed={seed}", (ROOT, seed, NEW_NS, *suffix)))
        for q in (1, 2):
            out.append((f"new/D1_block_bootstrap/seed={seed}/q={q}", (ROOT, seed, NEW_NS, 60, q)))
    for k in NEW_BOOT:
        out.append((f"new/across_seed_bootstrap/k={k}", (ROOT, NEW_NS, 7, k)))
    return out


def state(entropy: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(x) for x in np.random.SeedSequence(list(entropy)).generate_state(8))


def collision_groups(records: Sequence[Tuple[str, Tuple[int, ...]]], key_index: int) -> List[Dict[str, object]]:
    groups: Dict[Tuple[int, ...], List[Tuple[str, Tuple[int, ...]]]] = defaultdict(list)
    for name, entropy in records:
        key = entropy if key_index == 0 else state(entropy)
        groups[key].append((name, entropy))
    collisions = []
    for key, members in groups.items():
        if len(members) > 1:
            collisions.append(
                {
                    "key": list(key),
                    "members": [{"name": name, "entropy": list(entropy)} for name, entropy in members],
                }
            )
    return sorted(collisions, key=lambda x: json.dumps(x, sort_keys=True))


def cross_collisions(
    old: Sequence[Tuple[str, Tuple[int, ...]]],
    new: Sequence[Tuple[str, Tuple[int, ...]]],
    key_index: int,
) -> List[Dict[str, object]]:
    old_by: Dict[Tuple[int, ...], List[Tuple[str, Tuple[int, ...]]]] = defaultdict(list)
    for name, entropy in old:
        old_by[entropy if key_index == 0 else state(entropy)].append((name, entropy))
    out = []
    for name, entropy in new:
        key = entropy if key_index == 0 else state(entropy)
        if key in old_by:
            out.append(
                {
                    "key": list(key),
                    "new": {"name": name, "entropy": list(entropy)},
                    "legacy": [{"name": n, "entropy": list(e)} for n, e in old_by[key]],
                }
            )
    return sorted(out, key=lambda x: json.dumps(x, sort_keys=True))


def probe() -> Dict[str, object]:
    old = old_identities()
    new = new_identities()
    old_tuple = collision_groups(old, 0)
    old_state = collision_groups(old, 1)
    new_tuple = collision_groups(new, 0)
    new_state = collision_groups(new, 1)
    cross_tuple = cross_collisions(old, new, 0)
    cross_state = cross_collisions(old, new, 1)
    canonical = [
        {"scope": "legacy", "name": n, "entropy": list(e), "state": list(state(e))} for n, e in old
    ] + [{"scope": "new", "name": n, "entropy": list(e), "state": list(state(e))} for n, e in new]
    identity_sha = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result: Dict[str, object] = {
        "schema_version": 1,
        "numpy_version": np.__version__,
        "generate_state_words": 8,
        "counts": {
            "legacy_records": len(old),
            "legacy_unique_entropy_tuples": len({e for _, e in old}),
            "legacy_unique_states": len({state(e) for _, e in old}),
            "new_records": len(new),
            "new_unique_entropy_tuples": len({e for _, e in new}),
            "new_unique_states": len({state(e) for _, e in new}),
        },
        "known_legacy_only_duplicate_count": len(old_tuple),
        "legacy_entropy_collision_groups": old_tuple,
        "legacy_state_collision_groups": old_state,
        "new_entropy_collision_groups": new_tuple,
        "new_state_collision_groups": new_state,
        "new_vs_legacy_entropy_collisions": cross_tuple,
        "new_vs_legacy_state_collisions": cross_state,
        "identity_state_sha256": identity_sha,
    }
    passed = not (new_tuple or new_state or cross_tuple or cross_state)
    result["decision"] = "PASS" if passed else "INVALID"
    return result


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results_stage2_diagnostic/ss_probe.json")
    args = parser.parse_args()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = probe()
    result["probe_code_path"] = str(Path(__file__).resolve().relative_to(Path.cwd()))
    result["probe_code_sha256"] = file_sha(Path(__file__).resolve())
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    path.write_text(text)
    output_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = path.parent / "SHA256SUMS"
    manifest.write_text(
        f"{result['probe_code_sha256']}  {result['probe_code_path']}\n{output_sha}  {path.name}\n"
    )
    print(json.dumps({"decision": result["decision"], "counts": result["counts"], "output_sha256": output_sha}))


if __name__ == "__main__":
    main()
