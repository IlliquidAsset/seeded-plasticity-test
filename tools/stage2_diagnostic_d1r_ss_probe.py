#!/usr/bin/env python3
"""D1R SeedSequence collision probe (docs/STAGE2_DIAGNOSTIC_D1R_SPEC.md section 3.3).

Identity enumeration only: entropy tuples and SeedSequence(...).generate_state(8)
words. No network, stream, decoder, or D1R datum is generated.

Legacy = every r2/r3 object, tie-coin, and bootstrap identity enumerated by the
approved tools/stage2_diagnostic_ss_probe.py, plus every ebb90e74 namespace-30
identity (legacy since D1 ran), relabelled ``ns30/``. New = every D1R
namespace-31 identity for seeds 2100..2119, transcribed here independently of
snn/stage2_diagnostic_d1r.py (a test asserts the two agree).

This is the in-repo equivalent of the spec-card workspace probe (SHA-256
f042ad24...); it reproduces its counts and identity-state SHA-256 exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage2_diagnostic_ss_probe as base  # noqa: E402  approved probe, unchanged

ROOT = 20261001
D1R_NS = 31
D1R_SEEDS = tuple(range(2100, 2120))
FIXTURE_SEED = 4242
PER_SEED = (
    ("weights_w1_w2_o1", (1,)),
    ("weights_w2_o0", (21,)),
    ("A_train", (2,)),
    ("A_test", (5,)),
    ("D1R_block_bootstrap_A", (60, 1)),
    ("D1R_block_bootstrap_network_positive_control", (60, 2)),
)
FROZEN_EXPECTED = {
    "counts": {
        "legacy_records": 704,
        "legacy_unique_entropy_tuples": 698,
        "legacy_unique_states": 698,
        "new_records": 120,
        "new_unique_entropy_tuples": 120,
        "new_unique_states": 120,
    },
    "legacy_only_duplicate_groups": 6,
    "identity_state_sha256": "78fb3a3a604e5e5617cb5bb224a95e9cbf8ea0b45c77484b476fe05ae14c8706",
}


def legacy_identities() -> List[Tuple[str, Tuple[int, ...]]]:
    return list(base.old_identities()) + [(n.replace("new/", "ns30/", 1), e) for n, e in base.new_identities()]


def d1r_identities() -> List[Tuple[str, Tuple[int, ...]]]:
    return [(f"d1r/{name}/seed={seed}", (ROOT, seed, D1R_NS, *suffix)) for seed in D1R_SEEDS for name, suffix in PER_SEED]


def probe() -> Dict[str, object]:
    old = legacy_identities()
    new = d1r_identities()
    state = base.state
    res: Dict[str, object] = {
        "schema_version": 1,
        "probe": "D1R",
        "spec": "docs/STAGE2_DIAGNOSTIC_D1R_SPEC.md section 3.3",
        "numpy_version": np.__version__,
        "generate_state_words": 8,
        "seeds_new": [D1R_SEEDS[0], D1R_SEEDS[-1]],
        "seed_ranges_disjoint": not (set(D1R_SEEDS) & (set(base.OLD_SEEDS) | set(base.NEW_SEEDS) | {FIXTURE_SEED})),
        "counts": {
            "legacy_records": len(old),
            "legacy_unique_entropy_tuples": len({e for _, e in old}),
            "legacy_unique_states": len({state(e) for _, e in old}),
            "new_records": len(new),
            "new_unique_entropy_tuples": len({e for _, e in new}),
            "new_unique_states": len({state(e) for _, e in new}),
        },
        "legacy_only_entropy_duplicate_groups": base.collision_groups(old, 0),
        "legacy_only_state_duplicate_groups": base.collision_groups(old, 1),
        "new_entropy_collision_groups": base.collision_groups(new, 0),
        "new_state_collision_groups": base.collision_groups(new, 1),
        "new_vs_legacy_entropy_collisions": base.cross_collisions(old, new, 0),
        "new_vs_legacy_state_collisions": base.cross_collisions(old, new, 1),
    }
    res["legacy_only_duplicate_count"] = len(res["legacy_only_entropy_duplicate_groups"])  # type: ignore[arg-type]
    canonical = [{"scope": "legacy", "name": n, "entropy": list(e), "state": list(state(e))} for n, e in old] + [
        {"scope": "new", "name": n, "entropy": list(e), "state": list(state(e))} for n, e in new
    ]
    res["identity_state_sha256"] = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    clean = not (
        res["new_entropy_collision_groups"]
        or res["new_state_collision_groups"]
        or res["new_vs_legacy_entropy_collisions"]
        or res["new_vs_legacy_state_collisions"]
    ) and bool(res["seed_ranges_disjoint"])
    res["reproduces_spec_section_3_3"] = bool(
        res["counts"] == FROZEN_EXPECTED["counts"]
        and res["legacy_only_duplicate_count"] == FROZEN_EXPECTED["legacy_only_duplicate_groups"]
        and len(res["legacy_only_state_duplicate_groups"]) == FROZEN_EXPECTED["legacy_only_duplicate_groups"]  # type: ignore[arg-type]
        and res["identity_state_sha256"] == FROZEN_EXPECTED["identity_state_sha256"]
    )
    res["decision"] = "PASS" if clean and res["reproduces_spec_section_3_3"] else "INVALID"
    return res


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results_stage2_diagnostic_d1r/ss_probe.json")
    args = parser.parse_args()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = probe()
    repo = HERE.parent
    result["probe_code_path"] = str(Path(__file__).resolve().relative_to(repo))
    result["probe_code_sha256"] = file_sha(Path(__file__).resolve())
    result["base_probe_code_path"] = str(Path(base.__file__).resolve().relative_to(repo))
    result["base_probe_code_sha256"] = file_sha(Path(base.__file__).resolve())
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    output_sha = file_sha(path)
    (path.parent / "SHA256SUMS").write_text(
        f"{result['probe_code_sha256']}  {result['probe_code_path']}\n"
        f"{result['base_probe_code_sha256']}  {result['base_probe_code_path']}\n"
        f"{output_sha}  {path.name}\n"
    )
    print(json.dumps({
        "decision": result["decision"],
        "counts": result["counts"],
        "legacy_only_duplicate_count": result["legacy_only_duplicate_count"],
        "new_collisions": [len(result[k]) for k in ("new_entropy_collision_groups", "new_state_collision_groups", "new_vs_legacy_entropy_collisions", "new_vs_legacy_state_collisions")],  # type: ignore[arg-type]
        "seed_ranges_disjoint": result["seed_ranges_disjoint"],
        "identity_state_sha256": result["identity_state_sha256"],
        "reproduces_spec_section_3_3": result["reproduces_spec_section_3_3"],
        "probe_code_sha256": result["probe_code_sha256"],
        "output_sha256": output_sha,
    }, indent=1))


if __name__ == "__main__":
    main()
