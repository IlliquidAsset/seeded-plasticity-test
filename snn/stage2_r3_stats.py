"""Stage 2 r3 scoring (docs/STAGE2_SPEC.md @ def4b36, sec. 6, 8, 8.1, 9, 10, 11 item 15).

The sec. 9 / 9.1 bootstrap is unchanged from r2 and is imported from
``snn/stage2_stats.py`` (same metric table, same per-metric generators
``[20261001, 7, k]``). Scoring is the r2 scorer with exactly the r3 deltas:
Gate 4 item 1 constant 80 (40 + 40), the sec. 8.1 a_end_/b_end_ schema check
as an integrity item, phase-end bounds read from both field sets, and the
sec. 11 item 15 fixture-qualification decision.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

from snn.stage2_r3 import (
    ARMS, CHECKPOINTS, EXPERIMENTAL_SEEDS, N_TRAINABLE, QUALIFICATION_SEEDS, SchemaError, validate_row,
)
from snn.stage2_stats import (  # unchanged r2 bootstrap (sec. 9 / 9.1)
    LOWER, METRIC_IDS, METRICS, N_BOOT, TAXONOMY, TWO, UPPER, A, C, acc_tensor, bootstrap_all,
    bootstrap_metric, sign_of_life, verdict_from,
)

__all__ = [
    "LOWER", "UPPER", "TWO", "N_BOOT", "METRICS", "METRIC_IDS", "TAXONOMY", "acc_tensor", "bootstrap_all",
    "bootstrap_metric", "sign_of_life", "verdict_from", "score", "qualify", "row_schema_ok",
]


def row_schema_ok(r: Dict[str, object]) -> bool:
    try:
        validate_row(r)
        return True
    except SchemaError:
        return False


# ----------------------------------------------------- sec. 11 item 15 qualification
def qualify(rows: List[Dict[str, object]], seeds: Sequence[int] = QUALIFICATION_SEEDS,
            n_boot: int = N_BOOT) -> Dict[str, object]:
    """Frozen-from-start fixture qualification: sec. 6.2's three conditions on seeds 1000..1019.

    Metrics 0-3 use the sec. 9 bootstrap with entropy [20261001, 7, k], computed on an
    acc tensor whose only populated arm is F0 (metrics 0-3 read only acc[F0, .]).
    """
    seeds = list(seeds)
    f0 = sorted([r for r in rows if r["arm"] == "F0"], key=lambda r: r["seed"])
    if any(r["arm"] != "F0" for r in rows):
        raise ValueError("qualification rows must be frozen-from-start only")
    complete = [r["seed"] for r in f0] == seeds
    acc = np.full((len(ARMS), len(CHECKPOINTS), len(seeds)), np.nan)
    by = {r["seed"]: r for r in f0}
    for si, s in enumerate(seeds):
        for ci, ck in enumerate(CHECKPOINTS):
            acc[A["F0"], ci, si] = by[s]["accuracy"][ck] if s in by else np.nan
    boot = [bootstrap_metric(acc, k, n_boot) for k in (0, 1, 2, 3)]
    means = {ck: float(acc[A["F0"], C[ck]].mean()) for ck in CHECKPOINTS}
    cond = {
        "rows_complete_20": bool(complete and len(f0) == len(seeds)),
        "weights_bitwise_constant_20of20": bool(complete and all(r["max_abs_dW_total"] == 0.0 for r in f0)),
        "mean_in_[0.45,0.55]": {ck: bool(0.45 <= means[ck] <= 0.55) for ck in CHECKPOINTS},
        "ci95_contains_0.50": {ck: bool(b[TWO][0] <= 0.50 <= b[TWO][1]) for ck, b in zip(CHECKPOINTS, boot)},
        "schema_complete_all_rows": all(row_schema_ok(r) for r in f0),
    }
    passed = bool(cond["rows_complete_20"] and cond["weights_bitwise_constant_20of20"]
                  and all(cond["mean_in_[0.45,0.55]"].values()) and all(cond["ci95_contains_0.50"].values()))
    return {
        "seeds": seeds,
        "arm": "F0",
        "means": means,
        "bootstrap": boot,
        "conditions": cond,
        "decision": "PASS" if passed else "STOP",
        "per_seed_accuracy": {ck: [by[s]["accuracy"][ck] for s in seeds if s in by] for ck in CHECKPOINTS},
    }


# ------------------------------------------------------------------ gates
def score(rows: List[Dict[str, object]], boot: List[Dict[str, object]], preflight_ok: Dict[str, bool],
          seeds: Sequence[int] = EXPERIMENTAL_SEEDS, phase_steps: int = 200_000, eval_scored: int = 10_000,
          qualification_pass: bool = False) -> Dict[str, object]:
    """Score sec. 6 validity, sec. 8 sign-of-life, and Gates 0-4 + verdict (sec. 10), r3."""
    B = {b["id"]: b for b in boot}
    seeds = list(seeds)
    nS = len(seeds)
    arm_rows = {a: sorted([r for r in rows if r["arm"] == a], key=lambda r: r["seed"]) for a in ARMS}
    complete = all([r["seed"] for r in arm_rows[a]] == seeds for a in ARMS)
    acc = acc_tensor(rows, seeds)
    mean = lambda a, c: float(acc[A[a], C[c]].mean())

    by_seed_hash_ok = True
    for s in seeds:
        ref = next(r for r in rows if r["arm"] == "P" and r["seed"] == s)["hashes"]
        for a in ARMS:
            r = next(r for r in rows if r["arm"] == a and r["seed"] == s)
            if any(r["hashes"][k] != ref[k] for k in r["shared_hash_keys"]):
                by_seed_hash_ok = False

    def in_bounds(r):
        return all(r[f"{p}_w1_min"] >= -10.0 and r[f"{p}_w1_max"] <= 10.0 and r[f"{p}_w2_min"] >= 0.0
                   and r[f"{p}_w2_max"] <= 10.0 for p in ("a_end", "b_end"))

    integrity = {
        "all_100_rows_present": complete and len(rows) == len(ARMS) * nS and nS == 20,
        "hash_assertions_passed_all_rows": all(r["hash_assertion_passed"] for r in rows),
        "shared_hashes_equal_across_arms": by_seed_hash_ok,
        "section_8_1_schema_complete_all_rows": all(row_schema_ok(r) for r in rows),
        "no_nonfinite": not any(r[p]["nonfinite_detected"] for r in rows for p in ("phase_A", "phase_B"))
        and all(math.isfinite(v) for r in rows for v in r["accuracy"].values()),
        "weights_within_bounds": not any(r[p]["out_of_bounds_detected"] for r in rows for p in ("phase_A", "phase_B"))
        and all(in_bounds(r) for r in rows),
        "exactly_four_evaluations_every_row": all(r["n_evaluations"] == 4 and sorted(r["evaluations"]) == sorted(CHECKPOINTS) for r in rows),
        "evaluation_isolation_byte_identical": all(r["evaluation_isolation_byte_identical"] for r in rows),
        "evaluation_weights_bitwise_constant": all(e["eval_weights_bitwise_constant"] for r in rows for e in r["evaluations"].values()),
        "evaluation_copy_start_state_ok": all(e["eval_copy_start_state_ok"] for r in rows for e in r["evaluations"].values()),
        "evaluation_scored_10000": all(e["scored"] == eval_scored for r in rows for e in r["evaluations"].values()),
        "tie_coin_read_only_at_scored_ties": all(e["coin_reads_scored_equal_scored_ties"] for r in rows for e in r["evaluations"].values()),
        "tie_coin_never_read_by_training": all(r["coin_reads_by_training"] == 0 for r in rows),
        "fixture_qualification_passed_before_run": bool(qualification_pass),
    }

    F0 = arm_rows["F0"]
    v62 = {
        "max_abs_dW_eq_0_20of20": all(r["max_abs_dW_total"] == 0.0 for r in F0) and len(F0) == nS,
        "means_in_[0.45,0.55]": {c: (0.45 <= mean("F0", c) <= 0.55) for c in CHECKPOINTS},
        "ci95_contains_0.50": {c: (B[k][TWO][0] <= 0.50 <= B[k][TWO][1]) for k, c in zip((0, 1, 2, 3), CHECKPOINTS)},
    }
    v62["pass"] = bool(v62["max_abs_dW_eq_0_20of20"] and all(v62["means_in_[0.45,0.55]"].values()) and all(v62["ci95_contains_0.50"].values()))

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

    g1 = {
        "1_mean_A_pre_ge_0.70": mean("P", "A_pre") >= 0.70,
        "2_lower_mean_A_pre_gt_0.50": B[4][LOWER] > 0.50,
        "3_mean_P_minus_F0_A_pre_ge_0.15": B[5]["point_estimate"] >= 0.15,
        "4_lower_P_minus_F0_A_pre_gt_0": B[5][LOWER] > 0.0,
    }
    g2 = {
        "1_B_improvement_ge_0.15": B[6]["point_estimate"] >= 0.15,
        "2_lower_B_improvement_gt_0": B[6][LOWER] > 0.0,
        "3_mean_B_post_ge_0.70": mean("P", "B_post") >= 0.70,
        "4_lower_mean_B_post_gt_0.50": B[7][LOWER] > 0.50,
        "5_freeze_and_scramble_differentials": bool(
            v63["differential_point_ge_0.10"] and v63["differential_lower_gt_0"]
            and v64["differential_point_ge_0.10"] and v64["differential_lower_gt_0"]),
    }
    a_learned = mean("P", "A_pre") - 0.50
    a_loss = mean("P", "A_pre") - mean("P", "A_post")
    a_ret = (mean("P", "A_post") - 0.50) / a_learned if a_learned > 0 else -math.inf
    g3 = {
        "0_A_learned_gt_0": a_learned > 0,
        "1_A_retained_ge_0.80": a_learned > 0 and a_ret >= 0.80,
        "2_A_loss_le_0.05": a_loss <= 0.05,
        "3_lower_mean_A_post_gt_0.50": B[10][LOWER] > 0.50,
    }
    g4 = {
        "1_exactly_80_weights_all_arms": all(r["n_trainable_weights_before"] == N_TRAINABLE and r["n_trainable_weights_after"] == N_TRAINABLE for r in rows),
        "2_primary_one_init_zero_resets": all(
            r["live_init_counts"] == {"reset_online_state": 1, "restore_frozen_weights": 0, "plasticity_reset": 2, "sfa_reset": 1}
            and all(v == 0 for v in r["live_post_init_reset_calls"].values()) and r["live_init_start_state_ok"] for r in P),
        "3_primary_200k_A_then_200k_B_plastic_every_step": all(
            r["phase_A"]["steps"] == phase_steps and r["phase_B"]["steps"] == phase_steps
            and r["phase_A"]["steps_with_reward_fn"] == phase_steps and r["phase_B"]["steps_with_reward_fn"] == phase_steps
            and r["B_training_stream"] == "B_train" for r in P),
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
                "thresholds only (no NS bootstrap metric exists in sec. 9.1) - r2 reading carried unchanged.",
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
