#!/usr/bin/env python3
"""Run rungs 0-3, stop at the first failure, and retain review artifacts."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

from ladder.experiments import run_all_rungs

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_ladder"


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _write_delay_csv(payload):
    path = OUT / "rung0_delay_credit.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("tau_elig_ms", "delay_ms", "observed_credit", "theory_credit", "absolute_error"),
            lineterminator="\n",
        )
        writer.writeheader()
        for tau, curve in payload["results"]["rung_0"]["curves"].items():
            for row in curve["points"]:
                writer.writerow({"tau_elig_ms": tau, **row})


def _plot_delay(payload):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for axis, (tau, curve) in zip(axes, payload["results"]["rung_0"]["curves"].items()):
        x = [row["delay_ms"] for row in curve["points"]]
        y = [row["observed_credit"] for row in curve["points"]]
        theory = [row["theory_credit"] for row in curve["points"]]
        noise = curve["noise_floor"]["noise_rms_normalized_to_unit_tag"]
        axis.semilogy(x, y, "o", label="measured")
        axis.semilogy(x, theory, "-", label="exp(-delay/tau)")
        axis.axhline(noise, color="tab:red", linestyle=":", label="5 Hz background RMS")
        axis.set_title(f"tau = {tau} ms")
        axis.set_xlabel("reward delay (ms)")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("credited fraction (log scale)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("One tag fades continuously; it does not switch off")
    fig.tight_layout()
    fig.savefig(OUT / "rung0_delay_vs_credit.png", dpi=180)
    plt.close(fig)


def _write_report(payload):
    r = payload["results"]
    lines = [
        "# R-STDP learning ladder: rungs 0-3",
        "",
        f"Generated: {payload['generated_at']}",
        f"Baseline commit: `{payload['baseline_commit']}`",
        f"Command: `{payload['command']}`",
        "",
        "## Result table",
        "",
        "| Rung | Mechanism | Outcome | Gate |",
        "|---|---:|---:|---:|",
    ]
    for i in range(4):
        row = r.get(f"rung_{i}")
        if row is None:
            lines.append(f"| {i} | NOT RUN | NOT RUN | STOPPED |")
        else:
            lines.append(
                f"| {i} | {'PASS' if row['mechanism_pass'] else 'FAIL'} | "
                f"{'PASS' if row['outcome_pass'] else 'FAIL'} | {row['status']} |"
            )

    lines.extend(
        [
            "",
            "## In plain language",
            "",
            "- Rung 0: One recorded event fades exponentially. It never abruptly stops; a delayed reward gets `exp(-delay/tau)` of the original tag.",
            "- Rung 1: Rewarding every output spike drives the one synapse to its cap. Subtracting expected reward makes the drive die away and prevents saturation in this test.",
            "- Rung 2: With two synapses and no background pairings, only the synapse paired with the rewarded output changes. Background pairings can contaminate credit; the JSON maps that contamination by delay and firing rate.",
            "- Rung 3: Reward changes every eligible link in a causal series, only the paired link in a parallel motif, and never an unconnected link with zero eligibility.",
            "",
            "Learning here means both tests passed: (1) every weight change had the predicted sign and size, and (2) firing moved in the rewarded direction versus an identical frozen copy.",
            "",
            "## Important boundary",
            "",
            "A positive reward delivered only when the output is silent cannot change a pair-based R-STDP synapse, because silence creates no postsynaptic tag. To teach silence, this rung punishes an unwanted emitted spike (`reward=-1`); that shrinks the weight and reduces firing.",
            "",
            "The red line in the plot is not a universal noise floor. It is the measured RMS trace from independent 5 Hz pre- and postsynaptic background spikes with the repository's default STDP amplitudes. Exact values and crossings are in `summary.json`.",
            "",
            "## Track 1 correction",
            "",
            "Track 1's NULL result applies to our protocol, not to Florian's published protocol. Our run rewarded once after a 500 ms trial, used two output neurons and an argmax, redrew its input trains, reset traces each trial, filtered synaptic currents, and used different bounds and STDP scales. Florian rewarded each output spike on the following 1 ms step, used one output, fixed the temporal code for the whole experiment, and ran continuously.",
            "",
            "See `docs/FLORIAN_DIFFERENCE_TABLE.md` for the complete pre-run difference table.",
        ]
    )
    (OUT / "LADDER_REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(exist_ok=True)
    payload = run_all_rungs()
    payload.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "baseline_commit": _git_sha(),
            "command": "python run_ladder.py",
            "host": "Mac Mini",
            "engine": "numpy reference ladder; torch core is cross-checked by tests",
        }
    )
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_delay_csv(payload)
    _plot_delay(payload)
    _write_report(payload)
    for name, result in payload["results"].items():
        print(f"{name}: mechanism={result['mechanism_pass']} outcome={result['outcome_pass']} gate={result['status']}")
    print(f"overall: {payload['status']}")
    print(f"artifacts: {OUT}")
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
