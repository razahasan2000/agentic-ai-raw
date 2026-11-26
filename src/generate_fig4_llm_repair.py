#!/usr/bin/env python3
"""
Figure 4: Loop 2 repair progression, 1.5B vs 4-bit 7B.

Reads the raw trace files and plots cumulative pass rate after attempt 1, 2
and 3 for the full agentic variant. No values are hardcoded: if the 7B trace
file is missing the figure falls back to 1.5B only and says so.

Outputs (repo root + results/figures):
  fig4_llm_repair.png / .pdf
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_DIRS = [ROOT, RESULTS / "figures"]


def load_full_agentic(path):
    """Return list of records for the full_agentic variant."""
    path = Path(path)
    if not path.exists():
        return None
    recs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("variant") == "full_agentic" and not r.get("mock"):
                recs[(r.get("patient"), r.get("variant"))] = r
    return list(recs.values()) if recs else None


def cumulative(records):
    n = len(records)
    if n == 0:
        return None
    passed = [r for r in records if r.get("passed")]
    return [
        100.0 * sum(1 for r in passed if int(r.get("attempts", 1)) <= k) / n
        for k in (1, 2, 3)
    ], n


def main():
    series = []

    r15 = load_full_agentic(RESULTS / "llm_traces.jsonl")
    if r15:
        vals, n = cumulative(r15)
        series.append(("Qwen2.5-1.5B", vals, n, "#4C72B0"))

    r7 = load_full_agentic(RESULTS / "llm_traces_7b.jsonl")
    if r7:
        vals, n = cumulative(r7)
        series.append(("Qwen2.5-7B (4-bit)", vals, n, "#DD8452"))

    if not series:
        print("[fig4] no trace files found; nothing to plot", file=sys.stderr)
        return 1

    labels = ["Attempt 1", "Attempt 2", "Attempt 3"]
    x = np.arange(len(labels))
    width = 0.8 / len(series)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for i, (name, vals, n, colour) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=f"{name} (N={n})",
                      color=colour, edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Cumulative pass rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 108)
    ax.axhline(100, color="grey", linestyle=":", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=len(series), fontsize=9)
    ax.set_title("Loop 2 verifier repair: cumulative pass rate", fontsize=11)
    fig.tight_layout()

    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig4_llm_repair.png", dpi=300, bbox_inches="tight")
        fig.savefig(d / "fig4_llm_repair.pdf", bbox_inches="tight")

    print("[fig4] wrote fig4_llm_repair.png/.pdf")
    for name, vals, n, _ in series:
        print(f"  {name}: " + " -> ".join(f"{v:.1f}%" for v in vals) + f" (N={n})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
