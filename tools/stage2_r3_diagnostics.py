#!/usr/bin/env python3
"""Report-only diagnostics for the Stage 2 r3 result package (no gate, no re-scoring).

Reads results_stage2_r3/runs.jsonl after all 100 jobs completed and writes
results_stage2_r3/DIAGNOSTICS.json: per-arm medians of the sec. 8 / 5.3 report-only
fields (output spikes, both-silent/both-fire fractions, reward events, non-tie counts).
"""

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results_stage2_r3")


def med(v):
    return float(st.median(v))


def main():
    rows = [json.loads(line) for line in (OUT / "runs.jsonl").read_text().splitlines()]
    assert len(rows) == 100 and all(r["status"] == "ok" for r in rows)
    d = {"source": "runs.jsonl", "note": "report only; medians across 20 seeds; not gate quantities", "arms": {}}
    for arm in ("P", "F0", "FS", "SC", "NS"):
        R = [r for r in rows if r["arm"] == arm]
        a = {}
        for ph in ("phase_A", "phase_B"):
            x = [r[ph] for r in R]
            a[ph] = {
                "median_output_spikes_O1": med([p["output_spikes_O1"] for p in x]),
                "median_output_spikes_O0": med([p["output_spikes_O0"] for p in x]),
                "median_both_silent_fraction": med([p["both_silent_fraction"] for p in x]),
                "median_both_fire_fraction": med([p["both_fire_fraction"] for p in x]),
                "median_reward_events_positive": med([p["reward_events_positive"] for p in x]),
                "median_reward_events_negative": med([p["reward_events_negative"] for p in x]),
                "seeds_with_lt100_pos_or_neg_events": sum(min(p["reward_events_positive"], p["reward_events_negative"]) < 100 for p in x),
                "seeds_with_zero_reward_events": sum(p["reward_events_positive"] + p["reward_events_negative"] == 0 for p in x),
                "median_hidden_rate_hz": med([p["hidden_rate_hz"] for p in x]),
            }
        for ck in ("A_pre", "B_pre", "B_post", "A_post"):
            e = [r["evaluations"][ck] for r in R]
            a[ck] = {
                "median_accuracy": med([v["accuracy"] for v in e]),
                "median_both_silent_fraction": med([v["both_silent_fraction"] for v in e]),
                "median_both_fire_fraction": med([v["both_fire_fraction"] for v in e]),
                "median_nontie_count": med([v["nontie_count"] for v in e]),
                "seeds_all_tie": sum(v["nontie_count"] == 0 for v in e),
                "nontie_count_per_seed": [v["nontie_count"] for v in e],
            }
        d["arms"][arm] = a
    (OUT / "DIAGNOSTICS.json").write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT / 'DIAGNOSTICS.json'}")


if __name__ == "__main__":
    main()
