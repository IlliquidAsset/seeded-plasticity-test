"""Stage 2 bootstrap (sec. 9 / 9.1) and gate scoring (sec. 6, 8, 10) of docs/STAGE2_SPEC.md.

Pure functions over the 100 run rows. Scoring precedence and the handling
of the two spec points that need an explicit reading are frozen in
docs/STAGE2_IMPLEMENTATION_NOTES.md and committed before the experimental run.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from snn.stage2 import ARMS, CHECKPOINTS, EXPERIMENTAL_SEEDS, ROOT_ENTROPY

N_BOOT = 100_000
A = {a: i for i, a in enumerate(ARMS)}
C = {c: i for i, c in enumerate(CHECKPOINTS)}
LOWER, UPPER, TWO = "one_sided_lower_95", "one_sided_upper_95", "two_sided_95"


def _m(acc, arm, ck, rows):
    """m(acc[arm, ckpt]) over resampled rows; rows is (B, 20) index array or None for point."""
    v = acc[A[arm], C[ck]]
    return v.mean() if rows is None else v[rows].mean(axis=1)


def _md(acc, a1, c1, a2, c2, rows):
    d = acc[A[a1], C[c1]] - acc[A[a2], C[c2]]
    return d.mean() if rows is None else d[rows].mean(axis=1)


def _retained(acc, rows):
    num = _m(acc, "P", "A_post", rows) - 0.50
    den = _m(acc, "P", "A_pre", rows) - 0.50
    if rows is None:
        return float(num / den) if den > 0 else -math.inf
    out = np.full(den.shape, -np.inf)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def _metric_table() -> List[Tuple[int, str, Callable, Tuple[str, ...], str]]:
    t: List[Tuple[int, str, Callable, Tuple[str, ...], str]] = [
        (0, "m(acc[F0, A_pre])", lambda a, r: _m(a, "F0", "A_pre", r), (TWO,), "6.2 validity"),
        (1, "m(acc[F0, B_pre])", lambda a, r: _m(a, "F0", "B_pre", r), (TWO,), "6.2 validity"),
        (2, "m(acc[F0, B_post])", lambda a, r: _m(a, "F0", "B_post", r), (TWO,), "6.2 validity"),
        (3, "m(acc[F0, A_post])", lambda a, r: _m(a, "F0", "A_post", r), (TWO,), "6.2 validity"),
        (4, "m(acc[P, A_pre])", lambda a, r: _m(a, "P", "A_pre", r), (LOWER, TWO), "Gate 1.2"),
        (5, "m(acc[P, A_pre] - acc[F0, A_pre])", lambda a, r: _md(a, "P", "A_pre", "F0", "A_pre", r), (LOWER,), "Gate 1.4"),
        (6, "m(acc[P, B_post] - acc[P, B_pre])", lambda a, r: _md(a, "P", "B_post", "P", "B_pre", r), (LOWER, TWO), "Gate 2.2"),
        (7, "m(acc[P, B_post])", lambda a, r: _m(a, "P", "B_post", r), (LOWER, TWO), "Gate 2.4"),
        (8, "m((acc[P, B_post] - acc[P, B_pre]) - (acc[FS, B_post] - acc[FS, B_pre]))",
         lambda a, r: _dd(a, r), (LOWER,), "6.3 / Gate 2.5"),
        (9, "m(acc[P, B_post] - acc[SC, B_post])", lambda a, r: _md(a, "P", "B_post", "SC", "B_post", r), (LOWER,), "6.4 / Gate 2.5"),
        (10, "m(acc[P, A_post])", lambda a, r: _m(a, "P", "A_post", r), (LOWER, TWO), "Gate 3.3"),
        (11, "A_loss = m(acc[P, A_pre]) - m(acc[P, A_post])",
         lambda a, r: _m(a, "P", "A_pre", r) - _m(a, "P", "A_post", r), (UPPER, TWO), "Gate 3 report"),
        (12, "A_retained = (m(acc[P, A_post]) - 0.50) / (m(acc[P, A_pre]) - 0.50)", _retained, (LOWER, TWO), "Gate 3 report"),
        (13, "m(acc[P, A_post] - acc[NS, A_post])", lambda a, r: _md(a, "P", "A_post", "NS", "A_post", r), (LOWER,), "6.5 SFA-benefit claim"),
        (14, "A_learned = m(acc[P, A_pre]) - 0.50", lambda a, r: _m(a, "P", "A_pre", r) - 0.50, (TWO,), "report"),
    ]
    for ai, arm in enumerate(ARMS):
        for ci, ck in enumerate(CHECKPOINTS):
            t.append((100 + 4 * ai + ci, f"m(acc[{arm}, {ck}])",
                      (lambda arm_, ck_: (lambda a, r: _m(a, arm_, ck_, r)))(arm, ck), (TWO,), "report"))
    return t


def _dd(acc, rows):
    d = (acc[A["P"], C["B_post"]] - acc[A["P"], C["B_pre"]]) - (acc[A["FS"], C["B_post"]] - acc[A["FS"], C["B_pre"]])
    return d.mean() if rows is None else d[rows].mean(axis=1)


METRICS = _metric_table()
METRIC_IDS = [m[0] for m in METRICS]


def _pct(rep: np.ndarray, q: float) -> float:
    return float(np.percentile(rep, q, method="inverted_cdf"))


def bootstrap_metric(acc: np.ndarray, k: int, n_boot: int = N_BOOT) -> Dict[str, object]:
    """One metric: own generator [20261001, 7, k]; exactly one integers() call."""
    spec = {m[0]: m for m in METRICS}[k]
    _, name, fn, intervals, used_by = spec
    n = acc.shape[-1]
    g = np.random.default_rng(np.random.SeedSequence([ROOT_ENTROPY, 7, k]))
    idx = g.integers(0, n, size=(n_boot, n))
    rep = np.asarray(fn(acc, idx), dtype=float)
    point = float(fn(acc, None))
    rec: Dict[str, object] = {
        "id": k,
        "formula": name,
        "used_by": used_by,
        "point_estimate": point,
        "seed_sequence_entropy": [ROOT_ENTROPY, 7, k],
        "resamples": n_boot,
    }
    if LOWER in intervals:
        rec[LOWER] = _pct(rep, 5)
    if UPPER in intervals:
        rec[UPPER] = _pct(rep, 95)
    if TWO in intervals:
        rec[TWO] = [_pct(rep, 2.5), _pct(rep, 97.5)]
    if k == 12:
        rec["undefined_denominator_replicates"] = int(np.sum(~np.isfinite(rep)))
        rec["point_undefined"] = not math.isfinite(point)
    return rec


def bootstrap_all(acc: np.ndarray, order: Optional[Sequence[int]] = None, n_boot: int = N_BOOT) -> List[Dict[str, object]]:
    order = list(order) if order is not None else METRIC_IDS
    recs = {k: bootstrap_metric(acc, k, n_boot) for k in order}
    return [recs[k] for k in sorted(recs)]


# ------------------------------------------------------------------ helpers
def acc_tensor(rows: List[Dict[str, object]], seeds: Sequence[int] = EXPERIMENTAL_SEEDS) -> np.ndarray:
    """acc[arm, ckpt, seed] (5, 4, 20); raises if any arm/seed row is missing."""
    by = {(r["arm"], r["seed"]): r for r in rows}
    acc = np.empty((len(ARMS), len(CHECKPOINTS), len(seeds)))
    for ai, arm in enumerate(ARMS):
        for si, s in enumerate(seeds):
            r = by[(arm, s)]
            for ci, ck in enumerate(CHECKPOINTS):
                acc[ai, ci, si] = r["accuracy"][ck]
    return acc


def _corr(v):
    """Stage 1 convention (run_florian_core._group): undefined correlation counts as 0.0."""
    return v if v is not None else 0.0


def sign_of_life(rows: List[Dict[str, object]], phase: str, n_seeds: int = 20) -> Dict[str, object]:
    ph = [r[phase] for r in rows]
    corrs = [_corr(p["reward_signed_dw_out_corr"]) for p in ph]
    c1 = all(p["total_abs_weight_change_mv"] > 0 for p in ph)
    c2 = all(p["reward_events_positive"] >= 100 and p["reward_events_negative"] >= 100 for p in ph)
    med = float(np.median(corrs))
    n_gt = int(sum(c > 0.10 for c in corrs))
    return {
        "seeds": len(ph),
        "abs_dw_gt0_seeds": int(sum(p["total_abs_weight_change_mv"] > 0 for p in ph)),
        "min_positive_reward_events": int(min(p["reward_events_positive"] for p in ph)),
        "min_negative_reward_events": int(min(p["reward_events_negative"] for p in ph)),
        "median_reward_signed_dw_out_corr": med,
        "seeds_corr_gt_0.10": n_gt,
        "undefined_corr_seeds": int(sum(p["reward_signed_dw_out_corr"] is None for p in ph)),
        "item1_abs_dw_gt0_20of20": c1,
        "item2_ge100_pos_and_neg_each_seed": c2,
        "item3_median_corr_ge_0.30": med >= 0.30,
        "item4_ge18_seeds_corr_gt_0.10": n_gt >= 18,
        "pass": bool(c1 and c2 and med >= 0.30 and n_gt >= 18 and len(ph) == n_seeds),
    }


def score(rows: List[Dict[str, object]], boot: List[Dict[str, object]], preflight_ok: Dict[str, bool],
          seeds: Sequence[int] = EXPERIMENTAL_SEEDS, phase_steps: int = 200_000, eval_scored: int = 10_000) -> Dict[str, object]:
    """Score sec. 6 validity, sec. 8 sign-of-life, and Gates 0-4 + verdict (sec. 10).

    ``seeds``/``phase_steps``/``eval_scored`` exist only so shakedown tests can
    exercise the scorer on non-experimental fixtures; the experiment uses the defaults.
    """
    B = {b["id"]: b for b in boot}
    seeds = list(seeds)
    arm_rows = {a: sorted([r for r in rows if r["arm"] == a], key=lambda r: r["seed"]) for a in ARMS}
    complete = all([r["seed"] for r in arm_rows[a]] == seeds for a in ARMS)
    acc = acc_tensor(rows, seeds)
    mean = lambda a, c: float(acc[A[a], C[c]].mean())
    nS = len(seeds)

    # ---------------- integrity (sec. 7, 8 last paragraph, 5.3)
    by_seed_hash_ok = True
    for s in seeds:
        ref = next(r for r in rows if r["arm"] == "P" and r["seed"] == s)["hashes"]
        for a in ARMS:
            r = next(r for r in rows if r["arm"] == a and r["seed"] == s)
            if any(r["hashes"][k] != ref[k] for k in r["shared_hash_keys"]):
                by_seed_hash_ok = False
    integrity = {
        "all_100_rows_present": complete and len(rows) == 100,
        "hash_assertions_passed_all_rows": all(r["hash_assertion_passed"] for r in rows),
        "shared_hashes_equal_across_arms": by_seed_hash_ok,
        "no_nonfinite": not any(r[p]["nonfinite_detected"] for r in rows for p in ("phase_A", "phase_B"))
        and all(math.isfinite(v) for r in rows for v in r["accuracy"].values()),
        "weights_within_bounds": not any(r[p]["out_of_bounds_detected"] for r in rows for p in ("phase_A", "phase_B"))
        and all(r["final_w1_min"] >= -10.0 and r["final_w1_max"] <= 10.0 and r["final_w2_min"] >= 0.0 and r["final_w2_max"] <= 10.0 for r in rows),
        "exactly_four_evaluations_every_row": all(r["n_evaluations"] == 4 and sorted(r["evaluations"]) == sorted(CHECKPOINTS) for r in rows),
        "evaluation_isolation_byte_identical": all(r["evaluation_isolation_byte_identical"] for r in rows),
        "evaluation_weights_bitwise_constant": all(e["eval_weights_bitwise_constant"] for r in rows for e in r["evaluations"].values()),
        "evaluation_copy_start_state_ok": all(e["eval_copy_start_state_ok"] for r in rows for e in r["evaluations"].values()),
        "evaluation_scored_10000": all(e["scored"] == eval_scored for r in rows for e in r["evaluations"].values()),
    }

    # ---------------- 6.2 frozen from start
    F0 = arm_rows["F0"]
    v62 = {
        "max_abs_dW_eq_0_20of20": all(r["max_abs_dW_total"] == 0.0 for r in F0) and len(F0) == nS,
        "means_in_[0.45,0.55]": {c: (0.45 <= mean("F0", c) <= 0.55) for c in CHECKPOINTS},
        "ci95_contains_0.50": {c: (B[k][TWO][0] <= 0.50 <= B[k][TWO][1]) for k, c in zip((0, 1, 2, 3), CHECKPOINTS)},
    }
    v62["pass"] = bool(v62["max_abs_dW_eq_0_20of20"] and all(v62["means_in_[0.45,0.55]"].values()) and all(v62["ci95_contains_0.50"].values()))

    # ---------------- 6.3 freeze at shift
    FS = arm_rows["FS"]
    v63 = {
        "A_abs_dw_gt0_20of20": all(r["phase_A"]["total_abs_weight_change_mv"] > 0 for r in FS) and len(FS) == nS,
        "max_abs_dW_B_eq_0_20of20": all(r["max_abs_dW_B"] == 0.0 for r in FS) and len(FS) == nS,
        "differential_point_ge_0.10": B[8]["point_estimate"] >= 0.10,
        "differential_lower_gt_0": B[8][LOWER] > 0.0,
        "differential_point": B[8]["point_estimate"],
        "differential_lower": B[8][LOWER],
    }
    v63["pass"] = bool(v63["A_abs_dw_gt0_20of20"] and v63["max_abs_dW_B_eq_0_20of20"] and v63["differential_point_ge_0.10"] and v63["differential_lower_gt_0"])

    # ---------------- 6.4 scrambled B twin
    SC = arm_rows["SC"]
    Pr = {r["seed"]: r for r in arm_rows["P"]}
    lookups = [r["B_stream_order2_lookup_accuracy"] for r in SC]
    v64 = {
        "histogram_and_spike_count_equal_20of20": all(
            r["B_stream_symbol_counts"] == Pr[r["seed"]]["B_stream_symbol_counts"]
            and r["B_stream_input_spikes"] == Pr[r["seed"]]["B_stream_input_spikes"]
            and r["hashes"]["B_train"] == Pr[r["seed"]]["hashes"]["B_train"]
            for r in SC) and len(SC) == nS,
        "order2_lookup_le_0.55_20of20": all(x <= 0.55 for x in lookups) and len(SC) == nS,
        "order2_lookup_max": float(max(lookups)),
        "differential_point_ge_0.10": B[9]["point_estimate"] >= 0.10,
        "differential_lower_gt_0": B[9][LOWER] > 0.0,
        "differential_point": B[9]["point_estimate"],
        "differential_lower": B[9][LOWER],
    }
    v64["pass"] = bool(v64["histogram_and_spike_count_equal_20of20"] and v64["order2_lookup_le_0.55_20of20"]
                       and v64["differential_point_ge_0.10"] and v64["differential_lower_gt_0"])

    # ---------------- 6.5 SFA off
    NS = arm_rows["NS"]
    sol_ns = {"A": sign_of_life(NS, "phase_A", nS), "B": sign_of_life(NS, "phase_B", nS)}
    v65 = {
        "threshold_contribution_exactly_0_all_samples": all(
            r[p]["sfa_threshold_sampled_max_abs_mv"] == 0.0 and r[p]["sfa_threshold_contribution_max_mv"] == 0.0
            and r["beta_a_mv"] == 0.0 for r in NS for p in ("phase_A", "phase_B")),
        "abs_dw_gt0_A_and_B_20of20": all(r["phase_A"]["total_abs_weight_change_mv"] > 0 and r["phase_B"]["total_abs_weight_change_mv"] > 0 for r in NS) and len(NS) == nS,
        "sign_of_life_A": sol_ns["A"]["pass"],
        "sign_of_life_B": sol_ns["B"]["pass"],
        "sign_of_life_detail": sol_ns,
    }
    v65["pass"] = bool(v65["threshold_contribution_exactly_0_all_samples"] and v65["abs_dw_gt0_A_and_B_20of20"]
                       and v65["sign_of_life_A"] and v65["sign_of_life_B"])

    # ---------------- sec. 8 primary sign-of-life
    P = arm_rows["P"]
    sol_p = {"A": sign_of_life(P, "phase_A", nS), "B": sign_of_life(P, "phase_B", nS)}

    gate0_items = {
        **{f"integrity:{k}": v for k, v in integrity.items()},
        "6.2_frozen_from_start_zero_dW": v62["max_abs_dW_eq_0_20of20"],
        "6.2_frozen_from_start_chance_means_in_[0.45,0.55]": all(v62["means_in_[0.45,0.55]"].values()),
        "6.2_frozen_from_start_ci95_contains_0.50": all(v62["ci95_contains_0.50"].values()),
        "6.3_freeze_at_shift_A_dW_gt0": v63["A_abs_dw_gt0_20of20"],
        "6.3_freeze_at_shift_zero_dW_B": v63["max_abs_dW_B_eq_0_20of20"],
        "6.3_freeze_at_shift_B_differential_point_ge_0.10": v63["differential_point_ge_0.10"],
        "6.3_freeze_at_shift_B_differential_lower_gt_0": v63["differential_lower_gt_0"],
        "6.4_scrambled_histogram_and_spike_count_equal": v64["histogram_and_spike_count_equal_20of20"],
        "6.4_scrambled_order2_lookup_le_0.55": v64["order2_lookup_le_0.55_20of20"],
        "6.4_scrambled_B_post_differential_point_ge_0.10": v64["differential_point_ge_0.10"],
        "6.4_scrambled_B_post_differential_lower_gt_0": v64["differential_lower_gt_0"],
        "6.5_sfa_off_threshold_exactly_0": v65["threshold_contribution_exactly_0_all_samples"],
        "6.5_sfa_off_dW_gt0_A_and_B": v65["abs_dw_gt0_A_and_B_20of20"],
        "6.5_sfa_off_sign_of_life_A": v65["sign_of_life_A"],
        "6.5_sfa_off_sign_of_life_B": v65["sign_of_life_B"],
        "8_primary_sign_of_life_A": sol_p["A"]["pass"],
        "8_primary_sign_of_life_B": sol_p["B"]["pass"],
    }
    gate0 = all(gate0_items.values())

    # ---------------- Gate 1
    g1 = {
        "1_mean_A_pre_ge_0.70": mean("P", "A_pre") >= 0.70,
        "2_lower_mean_A_pre_gt_0.50": B[4][LOWER] > 0.50,
        "3_mean_P_minus_F0_A_pre_ge_0.15": B[5]["point_estimate"] >= 0.15,
        "4_lower_P_minus_F0_A_pre_gt_0": B[5][LOWER] > 0.0,
    }
    # ---------------- Gate 2
    g2 = {
        "1_B_improvement_ge_0.15": B[6]["point_estimate"] >= 0.15,
        "2_lower_B_improvement_gt_0": B[6][LOWER] > 0.0,
        "3_mean_B_post_ge_0.70": mean("P", "B_post") >= 0.70,
        "4_lower_mean_B_post_gt_0.50": B[7][LOWER] > 0.50,
        "5_freeze_and_scramble_differentials": bool(
            v63["differential_point_ge_0.10"] and v63["differential_lower_gt_0"]
            and v64["differential_point_ge_0.10"] and v64["differential_lower_gt_0"]),
    }
    # ---------------- Gate 3
    a_learned = mean("P", "A_pre") - 0.50
    a_loss = mean("P", "A_pre") - mean("P", "A_post")
    a_ret = (mean("P", "A_post") - 0.50) / a_learned if a_learned > 0 else -math.inf
    g3 = {
        "0_A_learned_gt_0": a_learned > 0,
        "1_A_retained_ge_0.80": a_learned > 0 and a_ret >= 0.80,
        "2_A_loss_le_0.05": a_loss <= 0.05,
        "3_lower_mean_A_post_gt_0.50": B[10][LOWER] > 0.50,
    }
    # ---------------- Gate 4
    Pp = arm_rows["P"]
    g4 = {
        "1_exactly_60_weights_all_arms": all(r["n_trainable_weights_before"] == 60 and r["n_trainable_weights_after"] == 60 for r in rows),
        "2_primary_one_init_zero_resets": all(
            r["live_init_counts"] == {"reset_online_state": 1, "restore_frozen_weights": 0, "plasticity_reset": 2, "sfa_reset": 1}
            and all(v == 0 for v in r["live_post_init_reset_calls"].values()) and r["live_init_start_state_ok"] for r in Pp),
        "3_primary_200k_A_then_200k_B_plastic_every_step": all(
            r["phase_A"]["steps"] == phase_steps and r["phase_B"]["steps"] == phase_steps
            and r["phase_A"]["steps_with_reward_fn"] == phase_steps and r["phase_B"]["steps_with_reward_fn"] == phase_steps
            and r["B_training_stream"] == "B_train" for r in Pp),
        "4_train_hashes_ne_heldout_and_zero_heldout_consumed": all(
            r["training_hashes_differ_from_heldout"] and r["heldout_observations_consumed_by_training"] == 0 for r in rows),
        "5_no_replay_no_A_represented_in_B": all(
            r["replay_buffer"] is None
            and {k: v for k, v in r["access_log"].items() if k.startswith("train:")}
            == {"train:A_train": phase_steps, f"train:{r['B_training_stream']}": phase_steps} for r in rows),
        "6_provenance_serialized_before_execution": bool(preflight_ok.get("provenance_before_execution", False)),
        "7_no_deviation_from_spec": bool(preflight_ok.get("params_match_spec", False) and preflight_ok.get("spec_hash_unchanged", False)
                                          and integrity["exactly_four_evaluations_every_row"]),
    }
    gates = {
        "gate0": {"items": gate0_items, "pass": gate0},
        "gate1": {"items": g1, "pass": all(g1.values())},
        "gate2": {"items": g2, "pass": all(g2.values())},
        "gate3": {"items": g3, "pass": all(g3.values()), "A_learned": a_learned, "A_loss": a_loss, "A_retained": a_ret},
        "gate4": {"items": g4, "pass": all(g4.values())},
    }
    verdict, labels = verdict_from(gates)
    # SFA-off behavioral point checks (implementation notes item 11): point thresholds of Gates 1-3 only;
    # sec. 9.1 defines no NS bootstrap metrics and forbids adding any.
    ns_learned = mean("NS", "A_pre") - 0.50
    ns_points = {
        "g1_mean_A_pre_ge_0.70": mean("NS", "A_pre") >= 0.70,
        "g1_mean_NS_minus_F0_A_pre_ge_0.15": float((acc[A["NS"], C["A_pre"]] - acc[A["F0"], C["A_pre"]]).mean()) >= 0.15,
        "g2_B_improvement_ge_0.15": float((acc[A["NS"], C["B_post"]] - acc[A["NS"], C["B_pre"]]).mean()) >= 0.15,
        "g2_mean_B_post_ge_0.70": mean("NS", "B_post") >= 0.70,
        "g3_A_retained_ge_0.80": ns_learned > 0 and (mean("NS", "A_post") - 0.50) / ns_learned >= 0.80,
        "g3_A_loss_le_0.05": mean("NS", "A_pre") - mean("NS", "A_post") <= 0.05,
    }
    label = None
    if verdict == "STAGE2_PASS":
        label = ("CONTINUAL_LEARNING; SFA_NOT_SHOWN_NECESSARY" if all(ns_points.values())
                 else "CONTINUAL_LEARNING_WITH_SFA_DEPENDENCE_AT_THIS_PARAMETERIZATION")
    sfa = {
        "sfa_benefit_claim_allowed": bool(B[13]["point_estimate"] >= 0.05 and B[13][LOWER] > 0.0),
        "metric13_point": B[13]["point_estimate"],
        "metric13_lower": B[13][LOWER],
        "sfa_off_behavioral_point_checks": ns_points,
        "interpretation_label": label,
        "note": "SFA-dependence labels (sec. 6.5) apply only if the primary arm is STAGE2_PASS; NS checks use point "
                "thresholds only (no NS bootstrap metric exists in sec. 9.1) - recorded spec gap for the reviewer.",
    }
    return {
        "integrity": integrity,
        "validity_6_2": v62,
        "validity_6_3": v63,
        "validity_6_4": v64,
        "validity_6_5": v65,
        "sign_of_life_primary": sol_p,
        "gates": gates,
        "verdict": verdict,
        "verdict_labels": labels,
        "sfa": sfa,
        "means": {a: {c: mean(a, c) for c in CHECKPOINTS} for a in ARMS},
    }


TAXONOMY = "online adaptation with forgetting — not continual learning"


def verdict_from(gates: Dict[str, Dict[str, object]]) -> Tuple[str, List[str]]:
    """Frozen precedence (implementation notes sec. 'Verdict'): every applicable sec. 10 label is listed."""
    g = {k: v["pass"] for k, v in gates.items()}
    labels: List[str] = []
    if not g["gate0"]:
        failed = [k for k, v in gates["gate0"]["items"].items() if not v]
        labels.append("INVALID_OR_MECHANISM_FAIL: " + ", ".join(failed))
    if not g["gate1"]:
        labels.append("NO_A_COMPETENCE")
    if g["gate1"] and not g["gate2"]:
        labels.append("NO_HELD_OUT_B_LEARNING")
    if g["gate2"] and not g["gate3"]:
        labels.append(TAXONOMY)
    if not g["gate4"]:
        failed = [k for k, v in gates["gate4"]["items"].items() if not v]
        labels.append("GATE4_FAIL: " + ", ".join(failed))
    if all(g.values()):
        return "STAGE2_PASS", ["STAGE2_PASS"]
    return labels[0], labels
