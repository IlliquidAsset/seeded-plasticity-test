#!/usr/bin/env python3
"""Stage 1 gate: Florian 2007 XOR on the repository core (snn/core.py).

Engine under test: ``snn.florian_bench`` (PureSNN.online_step, reward mode
``per_spike_next_step``).  For each (task, rule, seed, plastic) the core run
is paired with a fresh recorded run of the independent NumPy anchor
(``ladder.florian``) for the seed-for-seed comparison; the anchor never feeds
the core.  Gate predeclared in docs/FLORIAN_CORE_GATE.md (committed before
this run).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_florian_core"
SUMMARY = OUT / "summary.json"
ANCHOR_SUMMARY = ROOT / "results_florian_anchor" / "summary.json"
GROUPS = [(t, r) for t in ("rate", "temporal") for r in ("mstdp", "mstdpet")]


def command_from_args(args) -> str:
    return f"python run_florian_core.py --seeds {args.seeds} --epochs {args.epochs} --workers {args.workers}"


def _git(*cmd):
    return subprocess.check_output(["git", *cmd], cwd=ROOT, text=True).strip()


def wilson(k, n):
    if n == 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def fisher_p(k1, n1, k2, n2):
    return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="two-sided")[1])


def sign_test_p(k, n):
    """One-sided P(X >= k) for X ~ Binomial(n, 1/2)."""
    return float(sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def _one(job):
    import torch

    torch.set_num_threads(1)
    from snn.florian_bench import run_core_experiment
    from snn.florian_parity import compare_trains, run_anchor_recorded

    task, rule, seed, epochs, plastic = job
    t0 = time.time()
    row = run_core_experiment(task, rule, seed, epochs=epochs, plastic=plastic, record_output=True)
    row["wall_seconds_core"] = time.time() - t0
    anchor, a_res, a_train = run_anchor_recorded(task, rule, seed, epochs, plastic)
    cmp = compare_trains(row.pop("_output_train"), a_train)
    w1, w2 = row.pop("_final_w1"), row.pop("_final_w2")
    row["seed_for_seed"] = {
        **cmp,
        "anchor_rerun_success": a_res["success"],
        "anchor_rerun_margin_hz": min(a_res["last_epoch_rates_hz"]["01"], a_res["last_epoch_rates_hz"]["10"])
        - a_res["last_epoch_rates_hz"]["11"],
        "anchor_rerun_rates_hz": a_res["last_epoch_rates_hz"],
        "max_abs_w1_diff_mv": float(np.max(np.abs(w1 - anchor.w1))),
        "max_abs_w2_diff_mv": float(np.max(np.abs(w2 - anchor.w2))),
    }
    row["wall_seconds"] = time.time() - t0
    return row


def _group(rows, anchor_rows, task, rule, n_eval):
    tr = {r["seed"]: r for r in rows if r["plastic"]}
    fr = {r["seed"]: r for r in rows if not r["plastic"]}
    at = {r["seed"]: r for r in anchor_rows if r["task"] == task and r["rule"] == rule and r["plastic"]}
    af = {r["seed"]: r for r in anchor_rows if r["task"] == task and r["rule"] == rule and not r["plastic"]}
    seeds = sorted(tr)
    n = len(seeds)
    k = sum(tr[s]["success"] for s in seeds)
    kf = sum(fr[s]["success"] for s in seeds)
    ka = sum(at[s]["success"] for s in seeds if s in at)
    kaf = sum(af[s]["success"] for s in seeds if s in af)
    na = sum(1 for s in seeds if s in at)
    pub = tr[seeds[0]]["published_success_pct"]
    ci = [100 * x for x in wilson(k, n)]
    moves = [tr[s]["gate_margin_hz"] - fr[s]["gate_margin_hz"] for s in seeds]
    corrs = [tr[s]["reward_signed_dw_out_corr"] for s in seeds]
    corr_vals = [c if c is not None else 0.0 for c in corrs]
    improved = sum(m > 0 for m in moves)

    g1a = ci[0] <= pub <= ci[1]
    g1_p = fisher_p(k, n, ka, na)
    g1 = g1a and g1_p >= 0.05
    frozen_change = sum(fr[s]["total_abs_weight_change_mv"] for s in seeds)
    if task == "rate":
        g2_rule = "frozen successes <= 2/20"
        g2_stat = kf <= 2
        g2_p = None
    else:
        g2_p = fisher_p(kf, n, kaf, na)
        g2_rule = "Fisher p >= 0.05 vs anchor frozen count"
        g2_stat = g2_p >= 0.05
    g2 = frozen_change == 0.0 and g2_stat
    g3a = float(np.median(corr_vals)) >= 0.3 and sum(c > 0.1 for c in corr_vals) >= 18
    g3b = improved >= 15

    paired = []
    for s in seeds:
        t, f = tr[s], fr[s]
        paired.append(
            {
                "seed": s,
                "core_trained_success": t["success"],
                "core_frozen_success": f["success"],
                "core_trained_margin_hz": t["gate_margin_hz"],
                "core_frozen_margin_hz": f["gate_margin_hz"],
                "margin_movement_hz": t["gate_margin_hz"] - f["gate_margin_hz"],
                "reward_signed_dw_out_corr": t["reward_signed_dw_out_corr"],
                "reward_events": t["reward_events"],
                "anchor_committed_success": at.get(s, {}).get("success"),
                "anchor_committed_margin_hz": at.get(s, {}).get("gate_margin_hz"),
                "anchor_committed_frozen_success": af.get(s, {}).get("success"),
                "trained_output_train_identical_to_anchor": t["seed_for_seed"]["output_train_identical"],
                "trained_first_divergent_step": t["seed_for_seed"]["first_divergent_step"],
                "frozen_output_train_identical_to_anchor": f["seed_for_seed"]["output_train_identical"],
                "trained_max_abs_w_diff_mv": max(t["seed_for_seed"]["max_abs_w1_diff_mv"], t["seed_for_seed"]["max_abs_w2_diff_mv"]),
                "anchor_rerun_matches_committed": (
                    s in at
                    and t["seed_for_seed"]["anchor_rerun_success"] == at[s]["success"]
                    and t["seed_for_seed"]["anchor_rerun_margin_hz"] == at[s]["gate_margin_hz"]
                    and f["seed_for_seed"]["anchor_rerun_success"] == af[s]["success"]
                    and f["seed_for_seed"]["anchor_rerun_margin_hz"] == af[s]["gate_margin_hz"]
                ),
                "core_trained_retest_pass": t.get("retest_pass_count"),
                "core_frozen_retest_pass": f.get("retest_pass_count"),
                "core_trained_generalization_pass": t.get("generalization_pass_count"),
                "core_frozen_generalization_pass": f.get("generalization_pass_count"),
            }
        )
    out = {
        "seeds": n,
        "published_success_pct": pub,
        "core_trained_successes": k,
        "core_trained_wilson95_pct": ci,
        "core_frozen_successes": kf,
        "anchor_committed_trained_successes": ka,
        "anchor_committed_frozen_successes": kaf,
        "G1_wilson_contains_published": g1a,
        "G1_fisher_p_vs_anchor": g1_p,
        "G1_pass": g1,
        "G2_frozen_total_abs_weight_change_mv": frozen_change,
        "G2_rule": g2_rule,
        "G2_fisher_p_vs_anchor_frozen": g2_p,
        "G2_pass": g2,
        "G3a_median_corr": float(np.median(corr_vals)),
        "G3a_seeds_corr_gt_0.1": sum(c > 0.1 for c in corr_vals),
        "G3a_seeds_corr_undefined": sum(c is None for c in corrs),
        "G3a_pass": g3a,
        "G3b_seeds_margin_improved": improved,
        "G3b_sign_test_p_one_sided": sign_test_p(improved, n),
        "G3b_median_margin_movement_hz": float(np.median(moves)),
        "G3b_pass": g3b,
        "GROUP_PASS": bool(g1 and g2 and g3a and g3b),
        "seed_for_seed": {
            "trained_trains_identical": sum(p["trained_output_train_identical_to_anchor"] for p in paired),
            "frozen_trains_identical": sum(p["frozen_output_train_identical_to_anchor"] for p in paired),
            "success_agreement_with_committed_anchor": sum(p["core_trained_success"] == p["anchor_committed_success"] for p in paired),
            "margin_agreement_with_committed_anchor": sum(p["core_trained_margin_hz"] == p["anchor_committed_margin_hz"] for p in paired),
            "anchor_rerun_matches_committed": sum(bool(p["anchor_rerun_matches_committed"]) for p in paired),
            "max_trained_abs_w_diff_mv": max(p["trained_max_abs_w_diff_mv"] for p in paired),
        },
        "retest_pass_total_trained": sum(p["core_trained_retest_pass"] for p in paired),
        "retest_pass_total_frozen": sum(p["core_frozen_retest_pass"] for p in paired),
        "retest_denominator": n * n_eval,
        "paired": paired,
    }
    if task == "temporal":
        out["generalization_pass_total_trained"] = sum(p["core_trained_generalization_pass"] for p in paired)
        out["generalization_pass_total_frozen"] = sum(p["core_frozen_generalization_pass"] for p in paired)
    return out


def _report(payload):
    g = payload["groups"]
    L = [
        "# Stage 1 gate: Florian 2007 on the repository core (`snn/core.py`)",
        "",
        f"Generated: {payload['updated_at']}",
        f"Code commit: `{payload['code_commit']}` (core/bench files dirty: {payload['code_dirty']})",
        f"Command: `{payload['command']}` on {payload['host']}",
        f"Engine: {payload['engine']}",
        f"Predeclared gate: `docs/FLORIAN_CORE_GATE.md` (committed at `{payload['gate_commit']}` before this run)",
        "",
        f"## Verdict: {'CORE PASS' if payload['CORE_PASS'] else 'CORE_NOT_REPRODUCED'}",
        "",
        "| Task | Rule | Core trained | Wilson 95% | Paper | Anchor (committed) | Fisher p | Core frozen | Anchor frozen | G2 | Median r(reward, signed dW_out) | Seeds r>0.1 | Margin improved | G1 | G3a | G3b | Group |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|---|---|",
    ]
    ok = lambda b: "PASS" if b else "FAIL"
    for t, r in GROUPS:
        x = g[f"{t}_{r}"]
        n = x["seeds"]
        L.append(
            f"| {t} | {r.upper()} | {x['core_trained_successes']}/{n} | {x['core_trained_wilson95_pct'][0]:.1f}-{x['core_trained_wilson95_pct'][1]:.1f}% | "
            f"{x['published_success_pct']}% | {x['anchor_committed_trained_successes']}/{n} | {x['G1_fisher_p_vs_anchor']:.3f} | "
            f"{x['core_frozen_successes']}/{n} | {x['anchor_committed_frozen_successes']}/{n} | {ok(x['G2_pass'])} | {x['G3a_median_corr']:.3f} | "
            f"{x['G3a_seeds_corr_gt_0.1']}/{n} | {x['G3b_seeds_margin_improved']}/{n} | {ok(x['G1_pass'])} | {ok(x['G3a_pass'])} | {ok(x['G3b_pass'])} | {ok(x['GROUP_PASS'])} |"
        )
    L += [
        "",
        "G0 (engineering precondition, tests): " + payload["G0"],
        "",
        "## Seed-for-seed against the independent NumPy anchor",
        "",
        "| Task | Rule | Trained 400 s output trains identical | Frozen identical | Success agrees w/ committed anchor | Margin agrees | Anchor re-run reproduces committed | Max final dW (mV) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for t, r in GROUPS:
        s = g[f"{t}_{r}"]["seed_for_seed"]
        n = g[f"{t}_{r}"]["seeds"]
        L.append(
            f"| {t} | {r.upper()} | {s['trained_trains_identical']}/{n} | {s['frozen_trains_identical']}/{n} | {s['success_agreement_with_committed_anchor']}/{n} | "
            f"{s['margin_agreement_with_committed_anchor']}/{n} | {s['anchor_rerun_matches_committed']}/{n} | {s['max_trained_abs_w_diff_mv']:.2e} |"
        )
    L += [
        "",
        "## Frozen post-training evaluation (reported, not gated)",
        "",
        "| Task | Rule | Retest trained | Retest frozen | Generalization trained | Generalization frozen |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for t, r in GROUPS:
        x = g[f"{t}_{r}"]
        d = x["retest_denominator"]
        gt = f"{x['generalization_pass_total_trained']}/{d}" if t == "temporal" else "n/a"
        gf = f"{x['generalization_pass_total_frozen']}/{d}" if t == "temporal" else "n/a"
        L.append(f"| {t} | {r.upper()} | {x['retest_pass_total_trained']}/{d} | {x['retest_pass_total_frozen']}/{d} | {gt} | {gf} |")
    L += [
        "",
        "## Claim boundary",
        "",
        "Supported: the repository substrate (`LIFNeuron`, `Synapse`, `RSTDPPlasticity`, `PureSNN.online_step`) learns Florian's rate-coded and temporal XOR with per-output-spike next-step reward. Results are compatible with the paper's published rates and with the independent anchor, under the predeclared gate.",
        "",
        "Not supported: equality with the paper's 1000-run percentages (20 seeds cannot resolve 98% vs 100%); any claim about the legacy end-of-trial mode, which is preserved unchanged and was not re-run; continual learning (Stage 2); any claim about AIB developmental runs.",
        "",
        "No oracle: the target enters only as the sign of the per-output-spike reward. Frozen twins receive no weight change.",
    ]
    (OUT / "FLORIAN_CORE_REPORT.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--gate-commit", required=True, help="commit that froze docs/FLORIAN_CORE_GATE.md")
    ap.add_argument("--g0", default="not recorded")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    anchor = json.loads(ANCHOR_SUMMARY.read_text())
    start = time.time()
    payload = {
        "schema_version": 1,
        "code_commit": _git("rev-parse", "HEAD"),
        "code_dirty": bool(_git("status", "--porcelain", "--", "snn", "ladder", "run_florian_core.py")),
        "gate_commit": args.gate_commit,
        "G0": args.g0,
        "command": command_from_args(args),
        "host": "Mac Mini",
        "engine": "snn/core.py PureSNN.online_step, reward_mode=per_spike_next_step (snn/florian_bench.py); comparison anchor ladder/florian.py",
        "anchor_summary_commit_of_results": anchor.get("code_commit"),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    jobs = [(t, r, s, args.epochs, p) for t, r in GROUPS for s in range(args.seeds) for p in (True, False)]
    rows = []
    partial = OUT / "runs.partial.jsonl"
    partial.write_text("")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(_one, jobs):
            rows.append(row)
            with partial.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            sfs = row["seed_for_seed"]
            print(
                f"{row['task']}/{row['rule']} seed={row['seed']} plastic={row['plastic']} success={row['success']} "
                f"margin={row['gate_margin_hz']:+.0f} corr={row['reward_signed_dw_out_corr']} identical={sfs['output_train_identical']} "
                f"core={row['wall_seconds_core']:.0f}s",
                flush=True,
            )
    n_eval = rows[0].get("n_eval", 10)
    payload["runs"] = rows
    payload["groups"] = {
        f"{t}_{r}": _group([x for x in rows if x["task"] == t and x["rule"] == r], anchor["runs"], t, r, n_eval) for t, r in GROUPS
    }
    payload["CORE_PASS"] = all(v["GROUP_PASS"] for v in payload["groups"].values())
    payload["wall_seconds"] = time.time() - start
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    SUMMARY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    partial.unlink()
    _report(payload)
    print(json.dumps({k: {kk: v[kk] for kk in ("core_trained_successes", "core_frozen_successes", "G1_pass", "G2_pass", "G3a_pass", "G3b_pass", "GROUP_PASS")} for k, v in payload["groups"].items()}, indent=2))
    print("CORE_PASS", payload["CORE_PASS"])


if __name__ == "__main__":
    main()
