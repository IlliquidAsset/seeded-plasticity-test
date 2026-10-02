#!/usr/bin/env python3
"""D1R2 SeedSequence collision probe (frozen spec section 3.3).

Identity enumeration only. This module never constructs a network, stream, or
D1R2 decoder and cannot produce a D1R2 scientific datum.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage2_diagnostic_d1r_ss_probe as d1r_probe  # noqa: E402
ROOT = 20261001
D1R2_NS = 32
D1R2_SEEDS = tuple(range(2200, 2220))
FIXTURE_SEED = 4242
PER_SEED = (
    ("weights_w1_w2_o1", (1,)), ("weights_w2_o0", (21,)),
    ("A_train", (2,)), ("A_test", (5,)),
    ("D1R2_block_bootstrap_A", (60, 1)),
    ("D1R2_block_bootstrap_network_positive_control", (60, 2)),
)
FROZEN_EXPECTED = {
    "counts": {"legacy_records": 824, "legacy_unique_entropy_tuples": 818,
               "legacy_unique_states": 818, "new_records": 120,
               "new_unique_entropy_tuples": 120, "new_unique_states": 120},
    "legacy_only_duplicate_groups": 6,
    "identity_state_sha256": "4d8dd8b477aa2615bc14f66ed11223d0995c10df0985919577bacc9648b6848e",
}

def legacy_identities() -> List[Tuple[str, Tuple[int, ...]]]:
    return list(d1r_probe.legacy_identities()) + [
        (name.replace("d1r/", "ns31/", 1), entropy)
        for name, entropy in d1r_probe.d1r_identities()
    ]

def d1r2_identities() -> List[Tuple[str, Tuple[int, ...]]]:
    return [(f"d1r2/{name}/seed={seed}", (ROOT, seed, D1R2_NS, *suffix))
            for seed in D1R2_SEEDS for name, suffix in PER_SEED]

def probe(old: Sequence[Tuple[str, Tuple[int, ...]]] | None = None,
          new: Sequence[Tuple[str, Tuple[int, ...]]] | None = None) -> Dict[str, object]:
    """Section 3.3 probe. ``old``/``new`` overrides exist only for the
    synthetic identity-set tests (11d, 11f); the committed probe uses neither."""
    old = list(legacy_identities() if old is None else old)
    new = list(d1r2_identities() if new is None else new)
    base, state = d1r_probe.base, d1r_probe.base.state
    result: Dict[str, object] = {
        "schema_version": 1, "probe": "D1R2",
        "spec": "docs/STAGE2_DIAGNOSTIC_D1R2_SPEC.md section 3.3",
        "numpy_version": np.__version__, "generate_state_words": 8,
        "seeds_new": [D1R2_SEEDS[0], D1R2_SEEDS[-1]],
        "seed_ranges_disjoint": not (set(D1R2_SEEDS) & (set(base.OLD_SEEDS) | set(base.NEW_SEEDS) | set(d1r_probe.D1R_SEEDS) | {FIXTURE_SEED})),
        "counts": {"legacy_records": len(old), "legacy_unique_entropy_tuples": len({e for _, e in old}),
                   "legacy_unique_states": len({state(e) for _, e in old}), "new_records": len(new),
                   "new_unique_entropy_tuples": len({e for _, e in new}), "new_unique_states": len({state(e) for _, e in new})},
        "legacy_only_entropy_duplicate_groups": base.collision_groups(old, 0),
        "legacy_only_state_duplicate_groups": base.collision_groups(old, 1),
        "new_entropy_collision_groups": base.collision_groups(new, 0),
        "new_state_collision_groups": base.collision_groups(new, 1),
        "new_vs_legacy_entropy_collisions": base.cross_collisions(old, new, 0),
        "new_vs_legacy_state_collisions": base.cross_collisions(old, new, 1),
    }
    result["legacy_only_duplicate_count"] = len(result["legacy_only_entropy_duplicate_groups"])  # type: ignore[arg-type]
    canonical = [{"scope": scope, "name": name, "entropy": list(entropy), "state": list(state(entropy))}
                 for scope, records in (("legacy", old), ("new", new)) for name, entropy in records]
    result["identity_state_sha256"] = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    keys = ("new_entropy_collision_groups", "new_state_collision_groups", "new_vs_legacy_entropy_collisions", "new_vs_legacy_state_collisions")
    clean = not any(result[k] for k in keys) and result["seed_ranges_disjoint"] is True
    result["reproduces_spec_section_3_3"] = bool(
        result["counts"] == FROZEN_EXPECTED["counts"] and
        result["legacy_only_duplicate_count"] == FROZEN_EXPECTED["legacy_only_duplicate_groups"] and
        len(result["legacy_only_state_duplicate_groups"]) == FROZEN_EXPECTED["legacy_only_duplicate_groups"] and  # type: ignore[arg-type]
        result["identity_state_sha256"] == FROZEN_EXPECTED["identity_state_sha256"])
    result["decision"] = "PASS" if clean and result["reproduces_spec_section_3_3"] else "INVALID"
    return result

def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_result(path: Path, result: Dict[str, object] | None = None) -> Dict[str, object]:
    result = dict(probe() if result is None else result)
    path.parent.mkdir(parents=True, exist_ok=True)
    repo = HERE.parent
    for key, module_file in (("probe", __file__), ("d1r_probe", d1r_probe.__file__), ("base_probe", d1r_probe.base.__file__)):
        module_path = Path(module_file).resolve()
        result[f"{key}_code_path"] = str(module_path.relative_to(repo))
        result[f"{key}_code_sha256"] = file_sha(module_path)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    result["output_sha256"] = file_sha(path)
    return result

def write_retained(path: Path, result: Dict[str, object]) -> Dict[str, object]:
    """Retain a PASS probe output and its manifest (section 3.3)."""
    result = write_result(path, result)
    lines = [f"{result[k + '_code_sha256']}  {result[k + '_code_path']}" for k in ("probe", "d1r_probe", "base_probe")]
    lines.append(f"{result['output_sha256']}  {path.name}")
    (path.parent / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results_stage2_diagnostic_d1r2/ss_probe.json")
    args = parser.parse_args()
    path = Path(args.out)
    result = probe()
    if result["decision"] != "PASS":
        # Shakedown execution (section 8 step 1, section 11 item 12): an
        # implementation failure. Nothing is written; no package, no row.
        print(json.dumps({k: result[k] for k in ("decision", "counts", "identity_state_sha256", "reproduces_spec_section_3_3")}), file=sys.stderr)
        raise SystemExit(5)
    result = write_retained(path, result)
    keys = ("new_entropy_collision_groups", "new_state_collision_groups", "new_vs_legacy_entropy_collisions", "new_vs_legacy_state_collisions")
    print(json.dumps({"decision": result["decision"], "counts": result["counts"],
        "legacy_only_duplicate_count": result["legacy_only_duplicate_count"],
        "new_collisions": [len(result[k]) for k in keys], "seed_ranges_disjoint": result["seed_ranges_disjoint"],
        "identity_state_sha256": result["identity_state_sha256"], "reproduces_spec_section_3_3": result["reproduces_spec_section_3_3"],
        "probe_code_sha256": result["probe_code_sha256"], "output_sha256": result["output_sha256"]}, indent=1))

if __name__ == "__main__":
    main()
