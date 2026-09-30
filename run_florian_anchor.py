#!/usr/bin/env python3
"""Florian 2007 XOR anchor with paired frozen controls, margins, and generalization.

Each (task, rule, seed) runs twice with the identical RNG stream, inputs,
pattern order, and initial weights: once plastic (learning) and once frozen
(plasticity off = no-reward baseline).  After training both copies receive the
same frozen evaluation: a 10x retest of the training code and, for the temporal
task, 10 newly generated symbol pairs (paper section 4.3 generalization claim).

Engine: independent NumPy implementation (``ladder.florian``), NOT the
repository core ``snn/core.py``.
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

from ladder.florian import FlorianConfig, run_experiment
from ladder.florian_check import run_equation_checks

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_florian_anchor"
SUMMARY = OUT / "summary.json"


def command_from_args(args) -> str:
    """Record the command actually executed, derived from the parsed arguments."""
    return f"python run_florian_anchor.py --seeds {args.seeds} --epochs {args.epochs} --workers {args.workers}"


def _sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _dirty():
    out = subprocess.check_output(["git", "status", "--porcelain", "--", "ladder", "run_florian_anchor.py"], cwd=ROOT, text=True)
    return bool(out.strip())


def _wilson(successes, total):
    if total == 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [center - half, center + half]


def _one(job):
    task, rule, seed, epochs, plastic = job
    start = time.time()
    row = run_experiment(task, rule, seed, epochs=epochs, plastic=plastic)
    row["wall_seconds"] = time.time() - start
    return row


def _group(rows_trained, rows_frozen, task, rule):
    by_seed_f = {r["seed"]: r for r in rows_frozen}
    paired = []
    for r in sorted(rows_trained, key=lambda x: x["seed"]):
        f = by_seed_f[r["seed"]]
        entry = {
            "seed": r["seed"],
            "trained_success": r["success"],
            "frozen_success": f["success"],
            "trained_margin_hz": r["gate_margin_hz"],
            "frozen_margin_hz": f["gate_margin_hz"],
            "margin_movement_hz": r["gate_margin_hz"] - f["gate_margin_hz"],
            "trained_retest_pass": r["retest_pass_count"],
            "frozen_retest_pass": f["retest_pass_count"],
            "trained_retest_median_margin_hz": r["retest_median_margin_hz"],
            "frozen_retest_median_margin_hz": f["retest_median_margin_hz"],
        }
        if task == "temporal":
            entry.update(
                {
                    "trained_generalization_pass": r["generalization_pass_count"],
                    "frozen_generalization_pass": f["generalization_pass_count"],
                    "trained_generalization_median_margin_hz": r["generalization_median_margin_hz"],
                    "frozen_generalization_median_margin_hz": f["generalization_median_margin_hz"],
                }
            )
        paired.append(entry)
    n = len(paired)
    hits = sum(p["trained_success"] for p in paired)
    fhits = sum(p["frozen_success"] for p in paired)
    moves = [p["margin_movement_hz"] for p in paired]
    n_eval = rows_trained[0]["n_eval"] if rows_trained else 0
    out = {
        "seeds_completed": n,
        "published_success_pct_1000_runs": FlorianConfig(task, rule).published_success_pct,
        "trained_successes": hits,
        "trained_success_pct": 100.0 * hits / n if n else None,
        "trained_wilson_95_pct": [100.0 * x for x in _wilson(hits, n)],
        "frozen_successes": fhits,
        "frozen_success_pct": 100.0 * fhits / n if n else None,
        "seeds_margin_improved": sum(m > 0 for m in moves),
        "seeds_margin_unchanged": sum(m == 0 for m in moves),
        "seeds_margin_worse": sum(m < 0 for m in moves),
        "median_margin_movement_hz": float(np.median(moves)) if moves else None,
        "min_margin_movement_hz": float(min(moves)) if moves else None,
        "seeds_trained_pass_frozen_fail": sum(p["trained_success"] and not p["frozen_success"] for p in paired),
        "seeds_trained_fail_frozen_pass": sum((not p["trained_success"]) and p["frozen_success"] for p in paired),
        "retest_n_eval_per_seed": n_eval,
        "trained_retest_pass_total": sum(p["trained_retest_pass"] for p in paired),
        "frozen_retest_pass_total": sum(p["frozen_retest_pass"] for p in paired),
        "retest_denominator": n * n_eval,
        "paired": paired,
    }
    if task == "temporal":
        out.update(
            {
                "trained_generalization_pass_total": sum(p["trained_generalization_pass"] for p in paired),
                "frozen_generalization_pass_total": sum(p["frozen_generalization_pass"] for p in paired),
                "generalization_denominator": n * n_eval,
                "seeds_generalization_majority_trained": sum(p["trained_generalization_pass"] > n_eval / 2 for p in paired),
                "seeds_generalization_majority_frozen": sum(p["frozen_generalization_pass"] > n_eval / 2 for p in paired),
            }
        )
    return out


def _write_report(payload):
    g = payload["groups"]
    lines = [
        "# Florian 2007 XOR anchor (paired frozen control, independent NumPy engine)",
        "",
        f"Generated: {payload['updated_at']}",
        f"Code commit: `{payload['code_commit']}` (working tree dirty for ladder/anchor: {payload['code_dirty']})",
        f"Command: `{payload['command']}`",
        f"Engine: {payload['engine']}",
        "",
        "## Claim boundary",
        "",
        "This is an independent NumPy reproduction of Florian's equations. It is not execution of the repository core (`snn/core.py`). A match here shows the paper's protocol works when implemented as written; it says nothing yet about the repository's own network code.",
        "",
        "No oracle: the target enters only as the sign of the per-output-spike reward. Frozen twins receive the identical RNG stream, input, order, and initial weights, with plasticity off.",
        "",
        "## Final-epoch gate (paper criterion) vs paired frozen twin",
        "",
        "| Task | Rule | Trained | Frozen | Paper | Margin improved / same / worse | Median margin move (Hz) | Trained pass & frozen fail |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in ("rate", "temporal"):
        for rule in ("mstdp", "mstdpet"):
            r = g[f"{task}_{rule}"]
            n = r["seeds_completed"]
            lines.append(
                f"| {task} | {rule.upper()} | {r['trained_successes']}/{n} | {r['frozen_successes']}/{n} | "
                f"{r['published_success_pct_1000_runs']:.1f}% | {r['seeds_margin_improved']}/{r['seeds_margin_unchanged']}/{r['seeds_margin_worse']} | "
                f"{r['median_margin_movement_hz']:+.1f} | {r['seeds_trained_pass_frozen_fail']} |"
            )
    lines += [
        "",
        "Margin = min(rate01, rate10) - rate11 on the last training epoch; the paper gate is margin > 0. Movement = trained margin - frozen margin for the same seed.",
        "",
        "## Frozen post-training evaluation",
        "",
        "| Task | Rule | Retest trained | Retest frozen | New-symbol generalization trained | New-symbol generalization frozen |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task in ("rate", "temporal"):
        for rule in ("mstdp", "mstdpet"):
            r = g[f"{task}_{rule}"]
            d = r["retest_denominator"]
            gen_t = f"{r['trained_generalization_pass_total']}/{r['generalization_denominator']}" if task == "temporal" else "n/a"
            gen_f = f"{r['frozen_generalization_pass_total']}/{r['generalization_denominator']}" if task == "temporal" else "n/a"
            lines.append(
                f"| {task} | {rule.upper()} | {r['trained_retest_pass_total']}/{d} | {r['frozen_retest_pass_total']}/{d} | {gen_t} | {gen_f} |"
            )
    lines += [
        "",
        "Retest = the training code presented again, ten times, with synapses fixed (paper section 4.2). Generalization = ten newly generated pairs of 50-spike symbol trains per seed, with synapses fixed (paper section 4.3: \"for any pair of random input signals similarly generated\"). Each count is presentations that pass the gate.",
        "",
        "## Equation and reward-order check",
        "",
        f"`ladder/florian_check.py` recomputes the final weights from recorded spikes and targets using explicit pair sums. It shares no update code with the anchor. It covers the 2-epoch runs for both tasks and both rules; worst error {payload['equation_check']['max_abs_error_mv']:.2e} mV; all pass: {payload['equation_check']['all_pass']}. `tests/test_florian_equations.py` shows the check fails under three deliberate mutations: flipped sign, reward delivered one step late, and reward applied before the eligibility update.",
        "",
        "Remaining implementation differences: `docs/FLORIAN_DIFFERENCE_TABLE.md`. Twenty seeds cannot resolve 98% vs 100%. They can only detect a gross failure.",
    ]
    (OUT / "FLORIAN_ANCHOR_REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    start = time.time()
    eq_rows = run_equation_checks(epochs=2, seeds=(0, 1))
    payload = {
        "schema_version": 2,
        "status": "RUNNING",
        "code_commit": _sha(),
        "code_dirty": _dirty(),
        "command": command_from_args(args),
        "host": "Mac Mini",
        "engine": "independent NumPy reference implementation of Florian 2007 (ladder/florian.py); not snn/core.py",
        "difference_table": "docs/FLORIAN_DIFFERENCE_TABLE.md",
        "no_oracle_statement": "Targets affect reward sign only; no target, expected answer, or post-hoc success value enters weights or spikes by another path. Frozen twins receive no weight change at all.",
        "equation_check": {
            "rows": eq_rows,
            "max_abs_error_mv": max(r["max_abs_error_mv"] for r in eq_rows),
            "all_pass": all(r["pass"] for r in eq_rows),
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    jobs = [
        (task, rule, seed, args.epochs, plastic)
        for task in ("rate", "temporal")
        for rule in ("mstdp", "mstdpet")
        for seed in range(args.seeds)
        for plastic in (True, False)
    ]
    runs = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(_one, jobs):
            runs.append(row)
            print(
                f"{row['task']}/{row['rule']} seed={row['seed']} plastic={row['plastic']} "
                f"success={row['success']} margin={row['gate_margin_hz']:+.0f} wall={row['wall_seconds']:.1f}s",
                flush=True,
            )
    payload["runs"] = runs
    payload["groups"] = {
        f"{task}_{rule}": _group(
            [r for r in runs if r["task"] == task and r["rule"] == rule and r["plastic"]],
            [r for r in runs if r["task"] == task and r["rule"] == rule and not r["plastic"]],
            task,
            rule,
        )
        for task in ("rate", "temporal")
        for rule in ("mstdp", "mstdpet")
    }
    payload["status"] = "COMPLETE"
    payload["wall_seconds"] = time.time() - start
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    SUMMARY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_report(payload)
    print(json.dumps({k: {kk: v[kk] for kk in ("trained_successes", "frozen_successes", "seeds_margin_improved")} for k, v in payload["groups"].items()}, indent=2))
    print(f"summary: {SUMMARY}")


if __name__ == "__main__":
    main()
