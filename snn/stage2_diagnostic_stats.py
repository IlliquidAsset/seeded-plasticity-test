"""Deterministic D2-D4 readouts and paired seed bootstrap.

Pure functions over complete seed rows. Every metric owns its fixed namespace,
so evaluation order cannot affect a result.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from snn.stage2 import ROOT_ENTROPY
from snn.stage2_diagnostic import (
    DIAGNOSTIC_SEEDS,
    NAMESPACE,
    validate_checkpoint_rows,
)

N_BOOT = 100_000
LOWER = "one_sided_lower_95"
TWO = "two_sided_95"

# k: (name, condition(s), reduction, paired difference)
METRICS: Dict[int, Tuple[str, Tuple[str, ...], str, bool]] = {
    0: ("mean F0_lag1 accuracy", ("F0_lag1-no",), "mean", False),
    1: ("median P_lag1 accuracy", ("P_lag1-no",), "median", False),
    2: ("mean F0_A+drive accuracy", ("F0_A+drive",), "mean", False),
    3: ("median P_A+drive accuracy", ("P_A+drive",), "median", False),
    4: ("median paired P_A+drive - P_A-no accuracy", ("P_A+drive", "P_A-no"), "median", True),
    5: ("median paired P_lag1+drive - P_lag1-no accuracy", ("P_lag1+drive", "P_lag1-no"), "median", True),
    6: ("median P_lag1+drive accuracy", ("P_lag1+drive",), "median", False),
    7: ("median P_A-no accuracy", ("P_A-no",), "median", False),
    8: ("median F0_A-no accuracy", ("F0_A-no",), "median", False),
}


class IncompleteSeedRows(ValueError):
    pass


PAIR_HASH_FIELDS = (
    "initial_weights_sha256",
    "train_stream_sha256",
    "eval_stream_sha256",
    "tie_coin_sha256",
)


def pairing_integrity(
    rows: Sequence[Mapping[str, object]],
    conditions: Sequence[str],
    seeds: Sequence[int] = DIAGNOSTIC_SEEDS,
    require_common_drive: bool = False,
) -> bool:
    """Verify paired object hashes and drive-presence/common-random-number rules."""
    by = {(str(r["condition"]), int(r["seed"])): r for r in rows}
    try:
        for seed in seeds:
            paired = [by[(condition, seed)] for condition in conditions]
            reference = paired[0]
            if any(any(row[field] != reference[field] for field in PAIR_HASH_FIELDS) for row in paired[1:]):
                return False
            drive_hashes = [row["drive_sha256"] for row in paired]
            if require_common_drive:
                if any(value is None for value in drive_hashes) or len(set(drive_hashes)) != 1:
                    return False
            elif any(value is not None for value in drive_hashes):
                return False
            if any(not row["evaluation"]["weights_bitwise_constant"] for row in paired):
                return False
    except (KeyError, TypeError):
        return False
    return True


def base_pairing_integrity(
    rows: Sequence[Mapping[str, object]],
    conditions: Sequence[str],
    seeds: Sequence[int] = DIAGNOSTIC_SEEDS,
) -> bool:
    by = {(str(r["condition"]), int(r["seed"])): r for r in rows}
    try:
        for seed in seeds:
            paired = [by[(condition, seed)] for condition in conditions]
            reference = paired[0]
            if any(any(row[field] != reference[field] for field in PAIR_HASH_FIELDS) for row in paired[1:]):
                return False
    except (KeyError, TypeError):
        return False
    return True


def drive_hash_integrity(
    rows: Sequence[Mapping[str, object]],
    conditions: Sequence[str],
    seeds: Sequence[int] = DIAGNOSTIC_SEEDS,
) -> bool:
    by = {(str(r["condition"]), int(r["seed"])): r for r in rows}
    try:
        for seed in seeds:
            hashes = [by[(condition, seed)]["drive_sha256"] for condition in conditions]
            if any(value is None for value in hashes) or len(set(hashes)) != 1:
                return False
    except (KeyError, TypeError):
        return False
    return True


def complete_seed_table(
    rows: Sequence[Mapping[str, object]],
    conditions: Sequence[str],
    seeds: Sequence[int] = DIAGNOSTIC_SEEDS,
) -> Dict[str, np.ndarray]:
    """Align complete condition records; never drops or independently resamples a side."""
    by = {(str(r["condition"]), int(r["seed"])): r for r in rows}
    if len(by) != len(rows):
        raise IncompleteSeedRows("duplicate condition/seed row")
    out: Dict[str, np.ndarray] = {}
    for condition in conditions:
        missing = [s for s in seeds if (condition, s) not in by]
        if missing:
            raise IncompleteSeedRows(f"{condition} missing seeds {missing}")
        values = np.asarray([by[(condition, s)]["evaluation"]["accuracy"] for s in seeds], dtype=np.float64)
        if not np.isfinite(values).all():
            raise IncompleteSeedRows(f"{condition} has non-finite accuracy")
        out[condition] = values
    return out


def _values(table: Mapping[str, np.ndarray], spec: Tuple[str, Tuple[str, ...], str, bool]) -> np.ndarray:
    _, conditions, _, paired = spec
    if paired:
        return table[conditions[0]] - table[conditions[1]]
    return table[conditions[0]]


def bootstrap_metric(
    rows: Sequence[Mapping[str, object]],
    metric_id: int,
    n_boot: int = N_BOOT,
    seeds: Sequence[int] = DIAGNOSTIC_SEEDS,
) -> Dict[str, object]:
    if metric_id not in METRICS:
        raise ValueError(metric_id)
    spec = METRICS[metric_id]
    name, conditions, reduction, paired = spec
    table = complete_seed_table(rows, conditions, seeds)
    values = _values(table, spec)
    fn = np.mean if reduction == "mean" else np.median
    g = np.random.default_rng(np.random.SeedSequence([ROOT_ENTROPY, NAMESPACE, 7, metric_id]))
    idx = g.integers(0, len(seeds), size=(n_boot, len(seeds)))
    reps = fn(values[idx], axis=1)
    return {
        "id": metric_id,
        "name": name,
        "conditions": list(conditions),
        "paired_complete_seed_rows": bool(paired),
        "seed_sequence_entropy": [ROOT_ENTROPY, NAMESPACE, 7, metric_id],
        "seeds": list(seeds),
        "resamples": n_boot,
        "point_estimate": float(fn(values)),
        LOWER: float(np.percentile(reps, 5, method="inverted_cdf")),
        TWO: [
            float(np.percentile(reps, 2.5, method="inverted_cdf")),
            float(np.percentile(reps, 97.5, method="inverted_cdf")),
        ],
    }


def bootstrap_all(
    rows: Sequence[Mapping[str, object]],
    order: Optional[Sequence[int]] = None,
    n_boot: int = N_BOOT,
    seeds: Sequence[int] = DIAGNOSTIC_SEEDS,
) -> List[Dict[str, object]]:
    order = list(METRICS) if order is None else list(order)
    records = {k: bootstrap_metric(rows, k, n_boot=n_boot, seeds=seeds) for k in order}
    return [records[k] for k in sorted(records)]


def validate_d2_package(
    rows: Sequence[Mapping[str, object]],
    seeds: Sequence[int] = DIAGNOSTIC_SEEDS,
) -> Dict[str, object]:
    conditions = ("P_A-no", "F0_A-no")
    by = {(str(r["condition"]), int(r["seed"])): r for r in rows}
    expected = {(condition, seed) for condition in conditions for seed in seeds}
    if len(rows) != len(expected) or set(by) != expected:
        raise IncompleteSeedRows("D2 requires exactly 2 conditions x 20 unique seed rows")
    if not base_pairing_integrity(rows, conditions, seeds):
        raise IncompleteSeedRows("D2 paired object hashes differ")
    if not pairing_integrity(rows, conditions, seeds):
        raise IncompleteSeedRows("D2 no-drive/evaluation integrity failed")
    checkpoint_count = 0
    for (condition, seed), row in by.items():
        checkpoints = row["checkpoints"]
        validate_checkpoint_rows(
            checkpoints,
            seed,
            condition,
            str(row["train_stream_sha256"]),
            str(row["initial_weights_sha256"]),
        )
        checkpoint_count += len(checkpoints)
    return {
        "conditions": len(conditions),
        "seed_rows": len(rows),
        "checkpoint_rows": checkpoint_count,
        "paired_identity_integrity": True,
        "evaluation_nonmutating": True,
    }


def summarize_d2(onsets: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if sorted(int(r["seed"]) for r in onsets) != list(DIAGNOSTIC_SEEDS):
        raise ValueError("D2 requires exactly seeds 2000..2019")
    counts = {k: sum(r["label"] == k for r in onsets) for k in ("HIDDEN_FIRST", "OUTPUT_FIRST", "CO_ONSET")}
    if counts["HIDDEN_FIRST"] >= 14:
        result = "LOCALIZED_HIDDEN"
        status = "D2_PASS_LOCALIZED_HIDDEN"
    elif counts["OUTPUT_FIRST"] >= 14:
        result = "LOCALIZED_OUTPUT"
        status = "D2_PASS_LOCALIZED_OUTPUT"
    else:
        result = "CO_ONSET"
        status = "D2_FAIL_CO_ONSET"
    return {"process": "PASS", "counts": counts, "localization": result, "status": status}


def summarize_d3(rows: Sequence[Mapping[str, object]], boot: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    table = complete_seed_table(rows, ("P_lag1-no", "F0_lag1-no"))
    b = {int(x["id"]): x for x in boot}
    frozen_ci = b[0][TWO]
    frozen_constant = all(
        r["weights_bitwise_constant"] for r in rows if r["condition"] == "F0_lag1-no"
    )
    paired_ok = pairing_integrity(rows, ("P_lag1-no", "F0_lag1-no"))
    valid = frozen_ci[0] <= 0.50 <= frozen_ci[1] and frozen_constant and paired_ok
    median_p = float(np.median(table["P_lag1-no"]))
    if not valid:
        status = "INVALID"
    elif median_p >= 0.70 and b[1][LOWER] > 0.55:
        status = "PASS"
    elif b[1][TWO][0] <= 0.50 <= b[1][TWO][1]:
        status = "FAIL"
    else:
        status = "INCONCLUSIVE"
    return {
        "status": status,
        "median_P_lag1": median_p,
        "F0_mean_interval": frozen_ci,
        "P_median_lower": b[1][LOWER],
        "P_median_interval": b[1][TWO],
        "paired_identity_integrity": bool(paired_ok),
        "frozen_weights_constant_20of20": bool(frozen_constant),
    }


def summarize_d4(
    rows: Sequence[Mapping[str, object]],
    boot: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    table = complete_seed_table(rows, ("P_A-no", "F0_A-no", "P_A+drive", "F0_A+drive"))
    b = {int(x["id"]): x for x in boot}
    f0_constant = all(r["weights_bitwise_constant"] for r in rows if r["condition"] == "F0_A+drive")
    a_base_ok = base_pairing_integrity(rows, ("P_A-no", "F0_A-no", "P_A+drive", "F0_A+drive"))
    no_drive_ok = pairing_integrity(rows, ("P_A-no", "F0_A-no"))
    a_drive_ok = pairing_integrity(rows, ("P_A+drive", "F0_A+drive"), require_common_drive=True)
    drive_conditions = ["P_A+drive", "F0_A+drive"]
    if any(r["condition"] == "P_lag1+drive" for r in rows):
        drive_conditions.append("P_lag1+drive")
    common_drive_ok = drive_hash_integrity(rows, drive_conditions)
    drive_valid = (
        b[2][TWO][0] <= 0.50 <= b[2][TWO][1]
        and f0_constant
        and a_base_ok
        and no_drive_ok
        and a_drive_ok
        and common_drive_ok
    )

    p_drive = sorted((r for r in rows if r["condition"] == "P_A+drive"), key=lambda r: r["seed"])
    f_drive = sorted((r for r in rows if r["condition"] == "F0_A+drive"), key=lambda r: r["seed"])
    if len(p_drive) != 20 or len(f_drive) != 20:
        raise IncompleteSeedRows("D4 activity requires 20 complete P/F0 drive rows")
    hidden_p = np.asarray([r["checkpoints"][-1]["hidden_rate_hz"] for r in p_drive], dtype=float)
    hidden_f = np.asarray([r["checkpoints"][-1]["hidden_rate_hz"] for r in f_drive], dtype=float)
    silent = np.asarray([r["checkpoints"][-1]["both_silent_fraction"] for r in p_drive], dtype=float)
    denom = float(np.median(hidden_f))
    ratio = float(np.median(hidden_p) / denom) if denom != 0.0 else 0.0
    silent_end = float(np.median(silent))
    activity = ratio >= 0.50 and silent_end <= 0.80

    median_accuracy = float(np.median(table["P_A+drive"]))
    delta = table["P_A+drive"] - table["P_A-no"]
    median_delta = float(np.median(delta))
    if not drive_valid:
        competence = "INVALID"
    elif median_accuracy >= 0.70 and median_delta >= 0.05 and b[4][LOWER] > 0.0:
        competence = "PASS"
    elif b[4][TWO][0] <= 0.0 <= b[4][TWO][1]:
        competence = "FAIL"
    else:
        competence = "INCONCLUSIVE"

    if not drive_valid:
        overall = "INVALID"
    elif not activity or competence == "FAIL":
        overall = "FAIL"
    elif activity and competence == "PASS":
        overall = "PASS"
    else:
        overall = "INCONCLUSIVE"
    return {
        "status": overall,
        "drive_validity": "PASS" if drive_valid else "INVALID",
        "integrity": {
            "A_base_hashes_paired": bool(a_base_ok),
            "A_no_drive_absent_and_evaluation_nonmutating": bool(no_drive_ok),
            "A_drive_hashes_common_and_evaluation_nonmutating": bool(a_drive_ok),
            "drive_hash_common_across_with_drive_conditions": bool(common_drive_ok),
            "F0_weights_constant_20of20": bool(f0_constant),
        },
        "silence_prevention": "PASS" if activity else "FAIL",
        "competence": competence,
        "hidden_rate_ratio": ratio,
        "silent_end": silent_end,
        "median_P_A_drive_accuracy": median_accuracy,
        "median_delta_A": median_delta,
        "delta_lower": b[4][LOWER],
        "delta_interval": b[4][TWO],
    }
