#!/usr/bin/env python3
"""Run the paper-faithful Florian XOR anchors after ladder rungs 0-3 pass."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from ladder.florian import FlorianConfig, run_experiment

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_florian_anchor"
SUMMARY = OUT / "summary.json"


def _sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _wilson(successes, total):
    if total == 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [center - half, center + half]


def _require_ladder_pass():
    ladder_path = ROOT / "results_ladder" / "summary.json"
    if not ladder_path.exists():
        raise RuntimeError("results_ladder/summary.json is missing; run the ladder first")
    ladder = json.loads(ladder_path.read_text())
    if ladder["status"] != "PASS":
        raise RuntimeError("ladder did not pass; Florian anchor is fail-closed")
    for rung in range(4):
        if ladder["results"][f"rung_{rung}"]["status"] != "PASS":
            raise RuntimeError(f"rung {rung} did not pass")
    return ladder_path


def _summarize(payload):
    groups = {}
    for task in ("rate", "temporal"):
        for rule in ("mstdp", "mstdpet"):
            rows = [x for x in payload["runs"] if x["task"] == task and x["rule"] == rule]
            hits = sum(bool(x["success"]) for x in rows)
            config = FlorianConfig(task, rule)
            groups[f"{task}_{rule}"] = {
                "seeds_completed": len(rows),
                "successes": hits,
                "observed_success_pct": 100.0 * hits / len(rows) if rows else None,
                "wilson_95_interval_pct": [100.0 * x for x in _wilson(hits, len(rows))],
                "published_success_pct_1000_runs": config.published_success_pct,
                "claim": "finite replication estimate; not proof of equality to the paper",
            }
    payload["groups"] = groups


def _write(payload):
    _summarize(payload)
    SUMMARY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_report(payload):
    lines = [
        "# Florian 2007 XOR anchor",
        "",
        f"Generated: {payload['updated_at']}",
        f"Baseline commit before anchor: `{payload['baseline_commit']}`",
        f"Command: `{payload['command']}`",
        "",
        "The old Track 1 NULL was a result for our old protocol, not for Florian's protocol. This run uses online reward on each output spike, one output neuron, continuous state, the paper's network sizes, input codes, bounds, and learning rates.",
        "",
        "| Task | Rule | This run | 95% Wilson interval | Paper (1000 runs) |",
        "|---|---|---:|---:|---:|",
    ]
    for task in ("rate", "temporal"):
        for rule in ("mstdp", "mstdpet"):
            row = payload["groups"][f"{task}_{rule}"]
            lo, hi = row["wilson_95_interval_pct"]
            lines.append(
                f"| {task} | {rule.upper()} | {row['successes']}/{row['seeds_completed']} "
                f"({row['observed_success_pct']:.1f}%) | {lo:.1f}-{hi:.1f}% | "
                f"{row['published_success_pct_1000_runs']:.1f}% |"
            )
    lines.extend(
        [
            "",
            "Twenty seeds are enough to catch a gross failure, but not enough to distinguish 98% from 100%. The table must therefore be read as a finite replication estimate, not proof that the rates are identical.",
            "",
            "Remaining implementation differences were frozen before execution in `docs/FLORIAN_DIFFERENCE_TABLE.md`.",
        ]
    )
    (OUT / "FLORIAN_ANCHOR_REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--tasks", nargs="+", choices=("rate", "temporal"), default=("rate", "temporal"))
    parser.add_argument("--rules", nargs="+", choices=("mstdp", "mstdpet"), default=("mstdp", "mstdpet"))
    args = parser.parse_args()
    ladder_path = _require_ladder_pass()
    OUT.mkdir(exist_ok=True)
    start = time.time()
    payload = {
        "schema_version": 1,
        "status": "RUNNING",
        "baseline_commit": _sha(),
        "ladder_gate_path": str(ladder_path.relative_to(ROOT)),
        "difference_table": "docs/FLORIAN_DIFFERENCE_TABLE.md",
        "command": "python run_florian_anchor.py --seeds 20 --epochs 200",
        "host": "Mac Mini",
        "engine": "independent NumPy reference implementation of Florian 2007",
        "no_oracle_statement": "Targets affect reward sign only; no target, expected answer, or post-hoc success value enters weights or spikes by another path.",
        "runs": [],
        "groups": {},
        "started_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write(payload)
    for task in args.tasks:
        for rule in args.rules:
            for seed in range(args.seeds):
                print(f"{task}/{rule} seed={seed}", flush=True)
                one_start = time.time()
                row = run_experiment(task, rule, seed, epochs=args.epochs)
                row["wall_seconds"] = time.time() - one_start
                payload["runs"].append(row)
                payload["updated_at"] = datetime.now(timezone.utc).isoformat()
                _write(payload)
                print(
                    f"  success={row['success']} rates={row['last_epoch_rates_hz']} "
                    f"wall={row['wall_seconds']:.2f}s",
                    flush=True,
                )
    payload["status"] = "COMPLETE"
    payload["wall_seconds"] = time.time() - start
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write(payload)
    _write_report(payload)
    print(json.dumps(payload["groups"], indent=2))
    print(f"summary: {SUMMARY}")


if __name__ == "__main__":
    main()
