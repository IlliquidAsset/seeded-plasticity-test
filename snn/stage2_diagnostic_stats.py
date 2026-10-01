"""Deterministic D2-D4 readouts and paired seed bootstrap.

Pure functions over complete seed rows. Every metric owns its fixed namespace,
so evaluation order cannot affect a result.

Fail-closed contract (spec sections 0.4, 5.4, 6.3, 7.5, 8): every summarize_*
function runs the strict package validator FIRST. If any invariant fails it
returns an INVALID record that contains the violation list and no accuracy,
rate, interval, or other decision statistic. No bootstrap or readout predicate
is evaluated on an invalid package.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from snn.stage2 import ROOT_ENTROPY
from snn.stage2_diagnostic import (
    CHECKPOINT_STEPS,
    DIAGNOSTIC_SEEDS,
    EVAL_LEN,
    EVAL_SCORED,
    N_CHECKPOINTS,
    NAMESPACE,
    PHASE_STEPS,
    THRESHOLDS,
    CheckpointSchemaError,
    _is_hash,
    onset_record,
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

D2_CONDITIONS = ("P_A-no", "F0_A-no")
D3_CONDITIONS = ("P_lag1-no", "F0_lag1-no")
D4_CONDITIONS = ("P_A-no", "F0_A-no", "P_A+drive", "F0_A+drive", "P_lag1-no", "P_lag1+drive")
DRIVE_CONDITIONS = ("P_A+drive", "F0_A+drive", "P_lag1+drive")


class IncompleteSeedRows(ValueError):
    pass


PAIR_HASH_FIELDS = (
    "initial_weights_sha256",
    "train_stream_sha256",
    "eval_stream_sha256",
    "tie_coin_sha256",
)

CONDITION_ROW_FIELDS = (
    "diagnostic",
    "condition",
    "seed",
    "task",
    "plastic",
    "drive",
    "initial_weights_sha256",
    "final_weights_sha256",
    "weights_bitwise_constant",
    "train_stream_sha256",
    "eval_stream_sha256",
    "tie_coin_sha256",
    "drive_sha256",
    "paired_hash_assertion_passed",
    "checkpoints",
    "evaluation",
)
EVALUATION_FIELDS = (
    "accuracy",
    "scored",
    "weights_bitwise_constant",
    "coin_reads",
    "source_start_index",
    "source_end_index",
)

# Keys that carry an outcome or decision statistic. Used to redact rows of an
# INVALID stage and to prove progress records are outcome-free.
OUTCOME_KEYS = frozenset(
    {
        "accuracy",
        "lower_95",
        "evaluation",
        "checkpoints",
        "decoders",
        "coin_reads",
        "hidden_rate_hz",
        "output_rate_hz_O1",
        "output_rate_hz_O0",
        "output_layer_rate_hz",
        "both_silent_fraction",
        "median_accuracy",
        "lower_bounds_gt_half",
    }
)


def parse_condition(condition: str) -> Tuple[bool, str, bool]:
    """'P_A+drive' -> (plastic=True, task='A', drive=True)."""
    head, _, rest = condition.partition("_")
    if head not in ("P", "F0"):
        raise ValueError(condition)
    if rest.endswith("+drive"):
        return head == "P", rest[: -len("+drive")], True
    if rest.endswith("-no"):
        return head == "P", rest[: -len("-no")], False
    raise ValueError(condition)


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


# ------------------------------------------------------- strict row validator
ExpectedHashes = Mapping[Tuple[str, bool], Mapping[int, Mapping[str, Optional[str]]]]

# The value run_condition_seed emits; any other top-level label is a schema error.
CONDITION_DIAGNOSTIC_LABEL = "D2/D3/D4"
# Fields the runner's job wrapper adds; nothing else may appear on a row.
ROW_TRANSPORT_FIELDS = ("status", "worker_pid")


def _strict_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def validate_condition_row(
    row: Mapping[str, object],
    expected: Optional[Mapping[str, Optional[str]]],
    *,
    n_checkpoints: int = N_CHECKPOINTS,
    phase_steps: int = PHASE_STEPS,
    eval_len: int = EVAL_LEN,
    eval_scored: int = EVAL_SCORED,
) -> List[str]:
    """Every spec 0.4 / 5.2 / 6.2 / 7.3 integrity invariant for one condition row.

    Length keywords default to the spec and are overridden only by short
    construction tests; the package validator always uses the spec values.
    """
    if not isinstance(row, Mapping):
        return ["condition row is not a mapping"]
    tag = f"{row.get('condition')} seed={row.get('seed')}"
    v: List[str] = []
    if row.get("status", "ok") != "ok":
        v.append(f"{tag}: job status {row.get('status')!r}")
    missing = [k for k in CONDITION_ROW_FIELDS if k not in row]
    if missing:
        return v + [f"{tag}: missing field(s) {missing}"]
    unknown = sorted(set(row) - set(CONDITION_ROW_FIELDS) - set(ROW_TRANSPORT_FIELDS), key=str)
    if unknown:
        v.append(f"{tag}: undeclared field(s) {unknown}")
    if "worker_pid" in row and not _strict_int(row["worker_pid"]):
        v.append(f"{tag}: worker_pid is not int")
    if row["diagnostic"] != CONDITION_DIAGNOSTIC_LABEL:
        v.append(f"{tag}: diagnostic label {row['diagnostic']!r}, expected {CONDITION_DIAGNOSTIC_LABEL!r}")
    seed = row["seed"]
    if not _strict_int(seed):
        v.append(f"{tag}: seed is not int")
    if not isinstance(row["condition"], str):
        return v + [f"{tag}: condition label is not a string"]
    try:
        plastic, task, drive = parse_condition(row["condition"])
    except ValueError:
        return v + [f"{tag}: unknown condition label"]
    if task not in ("A", "lag1"):
        return v + [f"{tag}: unknown task {task!r}"]
    if not (isinstance(row["plastic"], bool) and isinstance(row["drive"], bool) and isinstance(row["task"], str)):
        v.append(f"{tag}: plastic/drive must be bool and task must be str")
    if (row["plastic"], row["task"], row["drive"]) != (plastic, task, drive):
        v.append(f"{tag}: plastic/task/drive fields disagree with condition label")
    for k in ("initial_weights_sha256", "final_weights_sha256", "train_stream_sha256", "eval_stream_sha256", "tie_coin_sha256"):
        if not _is_hash(row[k]):
            v.append(f"{tag}: {k} is not a SHA-256 hex digest")
    if row["train_stream_sha256"] == row["eval_stream_sha256"]:
        v.append(f"{tag}: train and held-out stream hashes are identical")
    if drive:
        if not _is_hash(row["drive_sha256"]):
            v.append(f"{tag}: with-drive row lacks a drive hash")
    elif row["drive_sha256"] is not None:
        v.append(f"{tag}: no-drive row carries a drive hash")
    if row["paired_hash_assertion_passed"] is not True:
        v.append(f"{tag}: worker paired-hash assertion not recorded as passed")
    if expected is None:
        v.append(f"{tag}: no coordinator identity supplied")
    else:
        for k in ("initial_weights_sha256", "train_stream_sha256", "eval_stream_sha256", "tie_coin_sha256", "drive_sha256"):
            if row[k] != expected.get(k):
                v.append(f"{tag}: {k} differs from coordinator identity")
    if not plastic:
        if row["weights_bitwise_constant"] is not True or row["final_weights_sha256"] != row["initial_weights_sha256"]:
            v.append(f"{tag}: frozen weights changed")
    elif not isinstance(row["weights_bitwise_constant"], bool):
        v.append(f"{tag}: weights_bitwise_constant is not bool")
    ev = row["evaluation"]
    if not isinstance(ev, Mapping):
        return v + [f"{tag}: evaluation is not a mapping"]
    emissing = [k for k in EVALUATION_FIELDS if k not in ev]
    if emissing:
        return v + [f"{tag}: evaluation missing field(s) {emissing}"]
    eunknown = sorted(set(ev) - set(EVALUATION_FIELDS), key=str)
    if eunknown:
        v.append(f"{tag}: evaluation has undeclared field(s) {eunknown}")
    # Every evaluation field is validated for type and range (spec 0.4, 3.3, 7.3).
    scored_ok = _strict_int(ev["scored"]) and ev["scored"] == eval_scored
    if not scored_ok:
        v.append(f"{tag}: evaluation scored={ev['scored']!r}, expected int {eval_scored}")
    acc = ev["accuracy"]
    if not (isinstance(acc, float) and math.isfinite(acc) and 0.0 <= acc <= 1.0):
        v.append(f"{tag}: evaluation accuracy not a finite float in [0,1]")
    elif abs(acc * eval_scored - round(acc * eval_scored)) > 1e-6:
        v.append(f"{tag}: evaluation accuracy is not a count over {eval_scored} scored steps")
    if ev["weights_bitwise_constant"] is not True:
        v.append(f"{tag}: evaluation mutated weights")
    # Tie-coin reads: one read per tie step across the whole evaluation
    # window (warm-up included), so an int in [0, eval_len].
    reads = ev["coin_reads"]
    if not _strict_int(reads) or not 0 <= reads <= eval_len:
        v.append(f"{tag}: evaluation coin_reads={reads!r} is not an int in [0, {eval_len}]")
    start, end = ev["source_start_index"], ev["source_end_index"]
    if drive:
        if not (_strict_int(start) and _strict_int(end)) or (start, end) != (phase_steps, phase_steps + eval_len - 1):
            v.append(f"{tag}: evaluation drive indices not continuous int {phase_steps}..{phase_steps + eval_len - 1}")
    elif start is not None or end is not None:
        v.append(f"{tag}: no-drive evaluation consumed drive indices")
    cps = row["checkpoints"]
    try:
        validate_checkpoint_rows(
            cps,  # type: ignore[arg-type]
            int(seed),  # type: ignore[arg-type]
            str(row["condition"]),
            str(row["train_stream_sha256"]),
            str(row["initial_weights_sha256"]),
            expected_count=n_checkpoints,
        )
        if any(cp["drive_sha256"] != row["drive_sha256"] for cp in cps):  # type: ignore[union-attr]
            v.append(f"{tag}: checkpoint drive hash differs from row drive hash")
        if cps[-1]["weights_sha256"] != row["final_weights_sha256"]:  # type: ignore[index]
            v.append(f"{tag}: last checkpoint weights differ from final weights")
        if cps[-1]["step_index"] != n_checkpoints * CHECKPOINT_STEPS - 1:  # type: ignore[index]
            v.append(f"{tag}: last checkpoint is not step {n_checkpoints * CHECKPOINT_STEPS - 1}")
        if not plastic and any(cp["weights_sha256"] != row["initial_weights_sha256"] for cp in cps):  # type: ignore[union-attr]
            v.append(f"{tag}: frozen weights changed at a checkpoint")
    except (CheckpointSchemaError, TypeError, KeyError, IndexError, ValueError) as exc:
        v.append(f"{tag}: checkpoint schema: {exc}")
    return v


def validate_condition_package(
    rows: Sequence[Mapping[str, object]],
    conditions: Sequence[str],
    expected: Optional[ExpectedHashes],
    seeds: Sequence[int] = DIAGNOSTIC_SEEDS,
) -> List[str]:
    """Exactly one valid row per (condition, seed); paired identities; common drive."""
    v: List[str] = []
    keys = []
    for r in rows:
        if not isinstance(r, Mapping):
            v.append("row is not a mapping")
            continue
        keys.append((r.get("condition"), r.get("seed")))
    dupes = sorted({k for k in keys if keys.count(k) > 1}, key=str)
    if dupes:
        v.append(f"duplicate condition/seed rows {dupes}")
    if expected is None:
        v.append("coordinator identities not supplied")
    by = {(r.get("condition"), r.get("seed")): r for r in rows if isinstance(r, Mapping)}
    for condition in conditions:
        present = sorted(s for c, s in keys if c == condition and isinstance(s, int))
        if sorted(set(present)) != sorted(seeds) or len(present) != len(seeds):
            v.append(f"{condition}: seed rows {present} differ from predeclared {list(seeds)}")
        try:
            plastic, task, drive = parse_condition(condition)
        except ValueError:
            v.append(f"{condition}: unknown condition")
            continue
        exp_cond = None if expected is None else expected.get((task, drive))
        for seed in seeds:
            row = by.get((condition, seed))
            if row is None:
                continue
            exp = None if exp_cond is None else exp_cond.get(seed)
            v.extend(validate_condition_row(row, exp))
    # Cross-condition pairing: same task shares every paired object; every
    # with-drive condition for a seed reads the identical drive realization.
    present_conditions = [c for c in conditions if all((c, s) in by for s in seeds)]
    for seed in seeds:
        for task in ("A", "lag1"):
            same_task = [by[(c, seed)] for c in present_conditions if parse_condition(c)[1] == task]
            for field in PAIR_HASH_FIELDS:
                if len({r.get(field) for r in same_task}) > 1:
                    v.append(f"seed={seed} task={task}: paired {field} differs across conditions")
        drive_rows = [by[(c, seed)] for c in present_conditions if parse_condition(c)[2]]
        if len({r.get("drive_sha256") for r in drive_rows}) > 1:
            v.append(f"seed={seed}: with-drive conditions read different drive realizations")
        init = [by[(c, seed)].get("initial_weights_sha256") for c in present_conditions]
        if len(set(init)) > 1:
            v.append(f"seed={seed}: initial weights differ across conditions")
    return v


def redact_outcomes(obj: object) -> object:
    """Recursively drop outcome-bearing keys (used only for INVALID stages)."""
    if isinstance(obj, Mapping):
        return {k: redact_outcomes(val) for k, val in obj.items() if k not in OUTCOME_KEYS}
    if isinstance(obj, list):
        return [redact_outcomes(x) for x in obj]
    return obj


def contains_outcome(obj: object) -> bool:
    if isinstance(obj, Mapping):
        return any(k in OUTCOME_KEYS or contains_outcome(val) for k, val in obj.items())
    if isinstance(obj, list):
        return any(contains_outcome(x) for x in obj)
    return False


# ---------------------------------------------------------------- bootstrap
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


def _invalid(status: str, violations: Sequence[str]) -> Dict[str, object]:
    """Schema/integrity failure: no decision statistic is computed or returned."""
    return {"status": status, "valid": False, "rows_schema_valid": False, "violations": list(violations)}


# ---------------------------------------------------------------- readouts
def validate_d2_package(
    rows: Sequence[Mapping[str, object]],
    expected: Optional[ExpectedHashes] = None,
    seeds: Sequence[int] = DIAGNOSTIC_SEEDS,
) -> List[str]:
    v = validate_condition_package(rows, D2_CONDITIONS, expected, seeds)
    extra = sorted({str(r.get("condition")) for r in rows if isinstance(r, Mapping)} - set(D2_CONDITIONS))
    if extra:
        v.append(f"D2: unexpected conditions {extra}")
    if len(rows) != len(D2_CONDITIONS) * len(seeds):
        v.append(f"D2: {len(rows)} rows, expected {len(D2_CONDITIONS) * len(seeds)}")
    return v


def summarize_d2(
    rows: Sequence[Mapping[str, object]],
    expected: Optional[ExpectedHashes] = None,
    seeds: Sequence[int] = DIAGNOSTIC_SEEDS,
) -> Dict[str, object]:
    violations = validate_d2_package(rows, expected, seeds)
    if violations:
        out = _invalid("D2_INVALID", violations)
        out["process"] = "INVALID"
        return out
    by = {(str(r["condition"]), int(r["seed"])): r for r in rows}
    onsets = []
    try:
        for seed in seeds:
            rec = onset_record(by[("P_A-no", seed)]["checkpoints"], by[("F0_A-no", seed)]["checkpoints"])  # type: ignore[arg-type]
            rec["seed"] = seed
            onsets.append(rec)
    except (ValueError, KeyError, StopIteration) as exc:
        out = _invalid("D2_INVALID", [f"D2 onset not mechanically derivable: {exc}"])
        out["process"] = "INVALID"
        return out
    counts = {k: sum(r["label"] == k for r in onsets) for k in ("HIDDEN_FIRST", "OUTPUT_FIRST", "CO_ONSET")}
    need = THRESHOLDS["d2_localized_min_seeds"]
    if counts["HIDDEN_FIRST"] >= need:
        result, status = "LOCALIZED_HIDDEN", "D2_PASS_LOCALIZED_HIDDEN"
    elif counts["OUTPUT_FIRST"] >= need:
        result, status = "LOCALIZED_OUTPUT", "D2_PASS_LOCALIZED_OUTPUT"
    else:
        result, status = "CO_ONSET", "D2_FAIL_CO_ONSET"
    return {
        "status": status,
        "valid": True,
        "rows_schema_valid": True,
        "violations": [],
        "process": "PASS",
        "process_integrity": {
            "conditions": len(D2_CONDITIONS),
            "seed_rows": len(rows),
            "checkpoint_rows": sum(len(r["checkpoints"]) for r in rows),  # type: ignore[arg-type]
        },
        "counts": counts,
        "localization": result,
        "onsets": onsets,
    }


def summarize_d3(
    rows: Sequence[Mapping[str, object]],
    expected: Optional[ExpectedHashes] = None,
    seeds: Sequence[int] = DIAGNOSTIC_SEEDS,
    n_boot: int = N_BOOT,
) -> Dict[str, object]:
    violations = validate_condition_package(rows, D3_CONDITIONS, expected, seeds)
    extra = sorted({str(r.get("condition")) for r in rows if isinstance(r, Mapping)} - set(D3_CONDITIONS))
    if extra:
        violations.append(f"D3: unexpected conditions {extra}")
    if violations:
        return _invalid("INVALID", violations)
    t = THRESHOLDS
    boot = bootstrap_all(rows, order=(0, 1), n_boot=n_boot, seeds=seeds)
    b = {int(x["id"]): x for x in boot}
    table = complete_seed_table(rows, D3_CONDITIONS, seeds)
    by = {(str(r["condition"]), int(r["seed"])): r for r in rows}
    frozen_ci = b[0][TWO]
    chance_ok = frozen_ci[0] <= t["d3_chance"] <= frozen_ci[1]  # type: ignore[index]
    median_p = float(np.median(table["P_lag1-no"]))
    if not chance_ok:
        status = "INVALID"
    elif median_p >= t["d3_pass_median_accuracy"] and b[1][LOWER] > t["d3_pass_lower_bound_gt"]:  # type: ignore[operator]
        status = "PASS"
    elif b[1][TWO][0] <= t["d3_chance"] <= b[1][TWO][1]:  # type: ignore[index]
        status = "FAIL"
    else:
        status = "INCONCLUSIVE"
    return {
        "status": status,
        "valid": bool(chance_ok),
        "rows_schema_valid": True,
        "violations": [] if chance_ok else ["F0_lag1 mean two-sided 95% interval excludes 0.50"],
        "median_P_lag1": median_p,
        "F0_mean_interval": frozen_ci,
        "P_median_lower": b[1][LOWER],
        "P_median_interval": b[1][TWO],
        "paired_identity_integrity": True,
        "frozen_weights_constant_20of20": True,
        "generator_hashes": {
            str(s): {
                "train_stream_sha256": by[("P_lag1-no", s)]["train_stream_sha256"],
                "eval_stream_sha256": by[("P_lag1-no", s)]["eval_stream_sha256"],
                "tie_coin_sha256": by[("P_lag1-no", s)]["tie_coin_sha256"],
            }
            for s in seeds
        },
        "bootstrap": boot,
    }


def summarize_d4(
    rows: Sequence[Mapping[str, object]],
    expected: Optional[ExpectedHashes] = None,
    seeds: Sequence[int] = DIAGNOSTIC_SEEDS,
    n_boot: int = N_BOOT,
) -> Dict[str, object]:
    """D4 (i)-(iii) plus the mandatory lag-1 report. ``rows`` = D2 + D3 + drive rows."""
    violations = validate_condition_package(rows, D4_CONDITIONS, expected, seeds)
    allowed = set(D4_CONDITIONS) | {"F0_lag1-no"}
    extra = sorted({str(r.get("condition")) for r in rows if isinstance(r, Mapping)} - allowed)
    if extra:
        violations.append(f"D4: unexpected conditions {extra}")
    if violations:
        out = _invalid("INVALID", violations)
        out.update({"drive_validity": "INVALID", "silence_prevention": "NOT_EVALUATED", "competence": "INVALID"})
        return out
    t = THRESHOLDS
    boot = bootstrap_all(rows, order=(2, 3, 4, 5, 6, 7, 8), n_boot=n_boot, seeds=seeds)
    b = {int(x["id"]): x for x in boot}
    table = complete_seed_table(rows, D4_CONDITIONS, seeds)
    drive_valid = b[2][TWO][0] <= t["d4_chance"] <= b[2][TWO][1]  # type: ignore[index]

    by = {(str(r["condition"]), int(r["seed"])): r for r in rows}
    end = [by[("P_A+drive", s)]["checkpoints"][-1] for s in seeds]  # type: ignore[index]
    end_f = [by[("F0_A+drive", s)]["checkpoints"][-1] for s in seeds]  # type: ignore[index]
    hidden_p = np.asarray([cp["hidden_rate_hz"] for cp in end], dtype=float)
    hidden_f = np.asarray([cp["hidden_rate_hz"] for cp in end_f], dtype=float)
    silent = np.asarray([cp["both_silent_fraction"] for cp in end], dtype=float)
    denom = float(np.median(hidden_f))
    ratio: Optional[float] = float(np.median(hidden_p) / denom) if denom != 0.0 else None
    silent_end = float(np.median(silent))
    activity = ratio is not None and ratio >= t["d4_hidden_rate_ratio_min"] and silent_end <= t["d4_silent_end_max"]

    median_accuracy = float(np.median(table["P_A+drive"]))
    median_delta = float(np.median(table["P_A+drive"] - table["P_A-no"]))
    if not drive_valid:
        competence = "INVALID"
    elif (
        median_accuracy >= t["d4_pass_median_accuracy"]
        and median_delta >= t["d4_pass_median_delta"]
        and b[4][LOWER] > t["d4_pass_delta_lower_gt"]  # type: ignore[operator]
    ):
        competence = "PASS"
    elif b[4][TWO][0] <= 0.0 <= b[4][TWO][1]:  # type: ignore[index]
        competence = "FAIL"
    else:
        competence = "INCONCLUSIVE"

    if not drive_valid:
        overall = "INVALID"
    elif not activity or competence == "FAIL":
        overall = "FAIL"
    elif competence == "PASS":
        overall = "PASS"
    else:
        overall = "INCONCLUSIVE"
    lag = {
        "median_P_lag1_drive": float(np.median(table["P_lag1+drive"])),
        "median_P_lag1_no": float(np.median(table["P_lag1-no"])),
        "median_drive_minus_no": b[5]["point_estimate"],
        "two_sided_95": b[5][TWO],
        "report_only": True,
    }
    return {
        "status": overall,
        "valid": bool(drive_valid),
        "rows_schema_valid": True,
        "violations": [] if drive_valid else ["F0_A+drive mean two-sided 95% interval excludes 0.50"],
        "drive_validity": "PASS" if drive_valid else "INVALID",
        "integrity": {
            "paired_identities": True,
            "drive_hash_common_across_with_drive_conditions": True,
            "no_drive_conditions_drive_absent": True,
            "F0_weights_constant_20of20": True,
            "evaluation_nonmutating": True,
        },
        "drive_hashes_by_seed": {str(s): by[("P_A+drive", s)]["drive_sha256"] for s in seeds},
        "silence_prevention": "PASS" if activity else "FAIL",
        "competence": competence,
        "hidden_rate_ratio": ratio,
        "hidden_rate_ratio_denominator_zero": ratio is None,
        "silent_end": silent_end,
        "median_P_A_drive_accuracy": median_accuracy,
        "median_delta_A": median_delta,
        "delta_lower": b[4][LOWER],
        "delta_interval": b[4][TWO],
        "lag1_report_only": lag,
        "bootstrap": boot,
    }
