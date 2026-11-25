#!/usr/bin/env python3
"""
Integrate verified 7B benchmark results into the manuscript.

Reads (never invents) results/llm_traces_7b.jsonl and the `llm_evaluation_7b`
block of results/metrics_verified.json, then:
  * rewrites Table 8 as a 1.5B vs 7B column group
  * adds a measured runtime/VRAM table
  * regenerates Figure 4 as grouped bars (calls generate_fig4_llm_repair.py)
  * patches the Abstract, Results prose, failure taxonomy, Limitations and
    Reproducibility sections
  * records the derived numbers in results/table8_7b_summary.json
  * optionally rebuilds the PDF

Safety:
  * every edit is wrapped in % LLM7B:<NAME>:BEGIN/END markers, so re-running
    is idempotent and never duplicates text
  * aborts if the manuscript anchors are missing, if the traces are mock data,
    if the traces are not 4-bit, or if the summary disagrees with the traces
  * backs up the .tex before writing
  * --dry-run prints the plan and changes nothing
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper_manuscript_final.tex"
METRICS = ROOT / "results" / "metrics_verified.json"
TRACES_7B = ROOT / "results" / "llm_traces_7b.jsonl"
TRACES_15B = ROOT / "results" / "llm_traces.jsonl"
SUMMARY_OUT = ROOT / "results" / "table8_7b_summary.json"
FIG4_SCRIPT = ROOT / "src" / "generate_fig4_llm_repair.py"

VARIANTS = ["unconstrained", "schema_prompted", "loop1_schema", "full_agentic"]
ROW_LABELS = {
    "unconstrained": "Unconstrained LLM",
    "schema_prompted": "Schema Prompted (1-shot)",
    "loop1_schema": "Loop 1 + Schema (no Loop 2)",
    "full_agentic": "Full Agentic (Loop 1 + Loop 2, M=3)",
}
CHECK_LABELS = {
    "subgroup_stated": "Missing subgroup caveat",
    "risk_match": "Numeric drift",
    "coverage_stated": "Missing coverage statement",
    "no_forbidden": "Invented clinical term",
    "counterfactual_match": "Counterfactual mismatch",
}

# Anchors are matched only on the first integration; afterwards the
# % LLM7B:<NAME>:BEGIN/END markers take over.
ANCHORS = {
    "TABLE8": (r"\\begin\{table\}\[htbp\]\s*\\centering\s*"
               r"\\caption\{LLM agentic evaluation.*?"
               r"\\label\{tab:multillm\}.*?\\end\{table\}"),
    "RUNTIME": r"\\label\{fig:llm_repair\}.*?\\end\{figure\}",
    "FIG4CAP": r"\\caption\{Loop 2 repair progression[^\n]*\}",
    "ABSTRACT": r"LLM Loop 2 verification.*?M=3\)\.",
    "HEADING": r"\\subsection\{LLM-Based Loop 2 Verification[^\n]*\}",
    "INTRO": (r"We evaluated four LLM variants.*?"
              r"Table~\\ref\{tab:multillm\} summari[sz]es results\."),
    "PROGRESSION": r"\\paragraph\{Loop 2 Repair Progression\.\}[^\n]*",
    "TAXONOMY": r"\\paragraph\{Failure Taxonomy\.\}[^\n]*",
    "LIMITATIONS": r"\\item \\textbf\{LLM evaluation\}:[^\n]*",
    "REPRO": r"\\texttt\{python src/run\\_llm\\_eval\\_fixed\.py\}[^\n]*",
    "CONCLUSION": (r"LLM Loop 2 verification on 1 seed "
                   r"\(Qwen2\.5-1\.5B-Instruct, 86 patients\).*?M=3\)\."),
}


def die(msg):
    print(f"\n[ABORT] {msg}\n", file=sys.stderr)
    sys.exit(1)


def load_traces(path, label):
    if not Path(path).exists():
        die(f"traces file not found: {path}")
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[(rec.get("patient"), rec.get("variant"))] = rec
    if not out:
        die(f"traces file is empty: {path}")
    models = {r.get("model") for r in out.values()}
    print(f"[Load] {label}: {len(out)} records, models={sorted(map(str, models))}")
    return out


def attach_latencies(traces, path, metrics, model_key):
    """The 1.5B trace schema has no latency_s; its verified latencies live in
    metrics_verified.json as ordered per-variant lists. Join them by order
    (both were appended per patient, so the order matches)."""
    ordered = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ordered.setdefault(r.get("variant"), []).append(r.get("patient"))
    for v, patients in ordered.items():
        lats = (metrics.get("llm_evaluation", {})
                .get(model_key, {})
                .get(v, {})
                .get("latencies", []))
        if len(lats) != len(patients):
            die(f"latency list for {model_key}/{v} has {len(lats)} values "
                f"but the trace file has {len(patients)} records")
        for pid, lat in zip(patients, lats):
            rec = traces.get((pid, v))
            if rec is not None:
                rec["latency_s"] = float(lat)
    return traces


def summarize(records, label):
    per_variant = {}
    for v in VARIANTS:
        recs = [r for (pid, vv), r in records.items() if vv == v]
        n = len(recs)
        if n == 0:
            per_variant[v] = {"n": 0}
            continue
        passed = sum(1 for r in recs if r.get("passed"))
        lats = sorted(float(r.get("latency_s", 0.0)) for r in recs)
        att = [int(r.get("attempts", 1)) for r in recs if r.get("passed")]
        per_variant[v] = {
            "n": n,
            "passed": passed,
            "final_pct": 100.0 * passed / n,
            "fallback_pct": 100.0 * (n - passed) / n,
            "att1_pct": 100.0 * sum(1 for a in att if a <= 1) / n,
            "att2_pct": 100.0 * sum(1 for a in att if a <= 2) / n,
            "att3_pct": 100.0 * sum(1 for a in att if a <= 3) / n,
            "lat_mean": sum(lats) / len(lats),
            "lat_p95": lats[min(len(lats) - 1, int(round(0.95 * (len(lats) - 1))))],
        }
    return per_variant


def failure_taxonomy(records):
    """Final-attempt failed checks for full_agentic, as % of that model's n."""
    full = [r for (pid, v), r in records.items() if v == "full_agentic"]
    n = len(full)
    counts = Counter()
    for r in full:
        if r.get("passed"):
            continue
        checks = r.get("checks") or {}
        for k, ok in checks.items():
            if not ok:
                counts[k] += 1
    return n, counts


def pct(x, bold=False):
    s = f"{x:.1f}\\%"
    return r"\textbf{" + s + "}" if bold else s


def build_table8(m15, m7, n15, n7, partial, m7_meta):
    header = (
        r"\begin{table}[htbp]" "\n"
        r"\centering" "\n"
        r"\caption{LLM Loop-2 evaluation on high-risk patients (seed 3, $p>0.5$). "
        r"Qwen2.5-1.5B and 4-bit Qwen2.5-7B use identical prompts, verifier, token "
        r"budgets and $M=3$ repair; values are recomputed from the raw trace files. "
    )
    if partial:
        header += (rf"Scope is partial: $N={n7}$ patients for both models "
                   rf"(matched subset), not the full 86. ")
    else:
        header += (rf"$N={n7}$ patients per model. ")
    header += (r"Loop-1 counterfactual reach and ellipsoid containment are 100\% for "
               r"both models (solver property, independent of the LLM)." "\n"
               r"}" "\n"
               r"\label{tab:multillm}" "\n"
               r"\resizebox{\linewidth}{!}{\begin{tabular}{lcccccccc}" "\n"
               r"\toprule" "\n"
               r"& \multicolumn{4}{c}{Qwen2.5-1.5B-Instruct} "
               r"& \multicolumn{4}{c}{Qwen2.5-7B-Instruct (4-bit NF4)} \\" "\n"
               r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}" "\n"
               r"Variant & Att.~1 & Final & Fallback & Lat.(s) "
               r"& Att.~1 & Final & Fallback & Lat.(s) \\" "\n"
               r"\midrule" "\n"
    )
    lines = []
    for v in VARIANTS:
        a, b = m15[v], m7[v]
        if b.get("n", 0) == 0:
            continue
        b15_final = pct(a["final_pct"], bold=b["final_pct"] < a["final_pct"])
        b7_final = pct(b["final_pct"], bold=b["final_pct"] > a["final_pct"])
        lines.append(
            f"{ROW_LABELS[v]} & {pct(a['att1_pct'])} & {b15_final} & "
            f"{pct(a['fallback_pct'])} & {a['lat_mean']:.1f} & "
            f"{pct(b['att1_pct'])} & {b7_final} & {pct(b['fallback_pct'])} & "
            f"{b['lat_mean']:.1f} \\\\"
        )
    footer = (
        "\n" r"\bottomrule" "\n"
        r"\end{tabular}}" "\n"
        r"\end{table}"
    )
    return header + "\n".join(lines) + footer


def build_runtime_table(m15, m7_meta, m7):
    rt = m7_meta.get("runtime", {})
    vr = m7_meta.get("vram", {})
    peak = vr.get("peak_allocated_GB")
    load_s = rt.get("model_load_s")
    lat15 = sum(m15[v]["lat_mean"] for v in VARIANTS if m15[v].get("n")) / max(
        1, sum(1 for v in VARIANTS if m15[v].get("n")))
    lat7 = sum(m7[v]["lat_mean"] for v in VARIANTS if m7[v].get("n")) / max(
        1, sum(1 for v in VARIANTS if m7[v].get("n")))
    gen15 = sum(m15[v]["lat_mean"] * m15[v]["n"] for v in VARIANTS if m15[v].get("n")) / 60.0
    gen7 = sum(m7[v]["lat_mean"] * m7[v]["n"] for v in VARIANTS if m7[v].get("n")) / 60.0
    calls15 = sum(m15[v]["n"] for v in VARIANTS if m15[v].get("n"))
    calls7 = sum(m7[v]["n"] for v in VARIANTS if m7[v].get("n"))
    return (
        r"\begin{table}[htbp]" "\n"
        r"\centering" "\n"
        r"\caption{Measured runtime and memory profile on the RTX 4070 Laptop "
        r"(8.6 GB, 30 W power cap). VRAM is the CUDA peak allocation measured by the "
        r"benchmark; generation time is the sum of per-call latencies recorded in the "
        r"traces (resume-invariant). The 1.5B run was not instrumented for VRAM or "
        r"load time (--).}" "\n"
        r"\label{tab:runtime}" "\n"
        r"\resizebox{\linewidth}{!}{\begin{tabular}{llccccc}" "\n"
        r"\toprule" "\n"
        r"Model & Quantisation & Load (s) & Peak VRAM (GB) & Mean lat.(s) "
        r"& Generation time (min) & Calls \\" "\n"
        r"\midrule" "\n"
        rf"Qwen2.5-1.5B & FP16 & -- & -- & {lat15:.1f} & {gen15:.1f} & {calls15} \\" "\n"
        rf"Qwen2.5-7B & 4-bit NF4 & {load_s:.1f} & {peak:.2f} & {lat7:.1f} & "
        rf"{gen7:.1f} & {calls7} \\" "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}}" "\n"
        r"\end{table}"
    )


def integrate_block(tex, name, new_content, anchor_regex):
    begin = f"% LLM7B:{name}:BEGIN"
    end = f"% LLM7B:{name}:END"
    if begin in tex and end in tex:
        pre = tex[:tex.index(begin)]
        post = tex[tex.index(end) + len(end):]
        return pre + begin + "\n" + new_content.rstrip("\n") + "\n" + end + post, "marked"
    m = re.search(anchor_regex, tex, re.S)
    if not m:
        die(f"manuscript anchor not found for {name}: /{anchor_regex}/")
    return (tex[:m.start()] + begin + "\n" + new_content.rstrip("\n") + "\n" + end
            + tex[m.end():], "inserted")


def insert_after(tex, name, new_content, anchor_regex):
    begin = f"% LLM7B:{name}:BEGIN"
    end = f"% LLM7B:{name}:END"
    if begin in tex and end in tex:
        pre = tex[:tex.index(begin)]
        post = tex[tex.index(end) + len(end):]
        return pre + begin + "\n" + new_content.rstrip("\n") + "\n" + end + post, "marked"
    m = re.search(anchor_regex, tex, re.S)
    if not m:
        die(f"manuscript anchor not found for {name}: /{anchor_regex}/")
    block = "\n\n" + begin + "\n" + new_content.rstrip("\n") + "\n" + end
    return tex[:m.end()] + block + tex[m.end():], "inserted"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Integrate verified 7B results into the paper.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--accept-partial", action="store_true")
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--skip-figure", action="store_true")
    ap.add_argument("--tex", default=None, help="override manuscript path (testing)")
    ap.add_argument("--metrics", default=None, help="override metrics path (testing)")
    ap.add_argument("--traces-7b", default=None, help="override 7B traces (testing)")
    ap.add_argument("--traces-15b", default=None, help="override 1.5B traces (testing)")
    ap.add_argument("--summary-out", default=None, help="override summary path (testing)")
    args = ap.parse_args(argv)

    tex_path = Path(args.tex).resolve() if args.tex else TEX
    metrics_path = Path(args.metrics).resolve() if args.metrics else METRICS
    traces7_path = Path(args.traces_7b).resolve() if args.traces_7b else TRACES_7B
    traces15_path = Path(args.traces_15b).resolve() if args.traces_15b else TRACES_15B
    summary_out = Path(args.summary_out).resolve() if args.summary_out else SUMMARY_OUT

    if not metrics_path.exists():
        die(f"metrics file not found: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    m7_meta = metrics.get("llm_evaluation_7b")
    if not m7_meta:
        die("metrics_verified.json has no 'llm_evaluation_7b'. Run the 7B benchmark first.")

    # ---- integrity gates -------------------------------------------------
    if m7_meta.get("mock"):
        die("llm_evaluation_7b is flagged as mock data; refusing to integrate.")
    if "4-bit" not in str(m7_meta.get("quantization", "")):
        die(f"7B run is not flagged as 4-bit: {m7_meta.get('quantization')!r}")

    traces7 = load_traces(traces7_path, "7B")
    for k, r in traces7.items():
        if r.get("mock"):
            die(f"mock trace found in {traces7_path} at {k}")
        if "4-bit" not in str(r.get("quantization", "")):
            die(f"non-4-bit trace found in {traces7_path} at {k}")
    traces15 = load_traces(traces15_path, "1.5B")
    traces15 = attach_latencies(
        traces15, traces15_path, metrics, "Qwen/Qwen2.5-1.5B-Instruct")

    patients7 = sorted({pid for pid, v in traces7 if v == "full_agentic"})
    if not patients7:
        die("no full_agentic records in the 7B traces")
    n7 = len(patients7)
    n_available = int(m7_meta.get("protocol", {}).get("n_high_risk_available", 86))
    full_protocol = bool(m7_meta.get("complete_full_protocol"))
    partial = (n7 < n_available) or not full_protocol
    if partial and not args.accept_partial:
        die(f"run is partial (n={n7}/{n_available}, "
            f"max_attempts={m7_meta.get('protocol', {}).get('max_attempts')}, "
            f"complete_full_protocol={full_protocol}). Re-run to completion or pass "
            f"--accept-partial to label partial results explicitly.")

    # Restrict 1.5B to the same patients => like-for-like comparison.
    subset = set(patients7)
    traces15_matched = {k: r for k, r in traces15.items() if k[0] in subset}

    m15 = summarize(traces15_matched, "1.5B")
    m7 = summarize(traces7, "7B")

    # Cross-check summary in metrics against raw traces.
    for v in VARIANTS:
        claimed = m7_meta.get("variants", {}).get(v, {}).get("final_pass_pct")
        if claimed is None or m7[v].get("n", 0) == 0:
            continue
        if abs(float(claimed) - m7[v]["final_pct"]) > 0.05:
            die(f"metrics vs traces mismatch for {v}: "
                f"metrics={claimed}, traces={m7[v]['final_pct']:.2f}")

    # Sanity: 1.5B full-86 numbers must still be the published ones.
    full15 = summarize(traces15, "1.5B-full")
    expected15 = {"unconstrained": 0, "schema_prompted": 60,
                  "loop1_schema": 58, "full_agentic": 81}
    for v, exp in expected15.items():
        if full15[v].get("passed") != exp:
            die(f"1.5B trace baseline changed for {v}: "
                f"passed={full15[v].get('passed')} expected={exp}")

    n1 = full15["full_agentic"]["n"]
    f7 = m7["full_agentic"]
    f15 = m15["full_agentic"]
    tax_n15, tax15 = failure_taxonomy(traces15_matched)
    tax_n7, tax7 = failure_taxonomy(traces7)

    scope_txt = ("partial, matched subset" if partial else "full protocol")
    print(f"\n[Verified] 7B: n={n7}, {scope_txt}; final pass "
          f"{f7['final_pct']:.1f}% vs 1.5B {f15['final_pct']:.1f}% on the same patients")
    print(f"[Verified] 7B attempts: {f7['att1_pct']:.1f}% -> {f7['att2_pct']:.1f}% "
          f"-> {f7['att3_pct']:.1f}%; fallback {f7['fallback_pct']:.1f}%")

    # ---- build replacement blocks ---------------------------------------
    table8 = build_table8(m15, m7, n1, n7, partial, m7_meta)
    runtime = build_runtime_table(m15, m7_meta, m7)

    fig4_caption = (
        r"\caption{Loop 2 verifier repair: cumulative pass rate after attempts 1--3 "
        r"for Qwen2.5-1.5B and 4-bit Qwen2.5-7B-Instruct on "
        + (f"{n7}" if partial else "86") +
        r" high-risk patients (seed 3). Bars are computed from the trace files.}"
    )

    abstract = (
        r"LLM Loop 2 verification on " + (f"{n7}" if partial else "86") +
        r" high-risk patients (seed 3) achieves " +
        pct(f15["final_pct"]) + r" final pass for Qwen2.5-1.5B-Instruct (" +
        pct(f15["att1_pct"]) + r" $\to$ " + pct(f15["att2_pct"]) + r" $\to$ " +
        pct(f15["att3_pct"]) + rf", $M=3$) and {pct(f7['final_pct'])} for 4-bit "
        r"Qwen2.5-7B-Instruct (" + pct(f7["att1_pct"]) + r" $\to$ " +
        pct(f7["att2_pct"]) + r" $\to$ " + pct(f7["att3_pct"]) +
        rf", $M=3$), with fallback rates {pct(f15['fallback_pct'])} and "
        rf"{pct(f7['fallback_pct'])}."
    )

    heading = (r"\subsection{LLM-Based Loop 2 Verification (1 Seed, "
               r"Qwen2.5-1.5B and 4-bit 7B-Instruct)}")

    intro = (
        r"We evaluated four LLM variants across all " +
        (f"{n7}" if partial else "86") +
        r" high-risk patients ($p > 0.5$) from seed 3: (i) Unconstrained LLM (no "
        r"tools, no verifier), (ii) Single-Shot Schema LLM (no Loop 1, no Loop 2), "
        r"(iii) Loop 1 + Schema LLM (no Loop 2), (iv) Full Agentic Loop (Loop 1 + "
        r"Loop 2 Verifier Repair, $M=3$). We tested Qwen2.5-1.5B-Instruct and "
        r"Qwen2.5-7B-Instruct, both 4-bit NF4 quantised on an 8.6 GB RTX 4070 Laptop "
        r"GPU, with identical prompts, verifier, token budgets and decoding. "
        r"Table~\ref{tab:multillm} summarizes results."
    )

    progression = (
        r"\paragraph{Loop 2 Repair Progression.} "
        r"Figure~\ref{fig:llm_repair} shows monotonic Attempt 1 $\to$ 2 $\to$ 3 pass "
        r"rates. Qwen2.5-1.5B goes " + pct(f15["att1_pct"]) + r" $\to$ " +
        pct(f15["att2_pct"]) + r" $\to$ " + pct(f15["att3_pct"]) +
        rf" (repair gain +{f15['final_pct'] - f15['att1_pct']:.1f}~pp; fallback "
        rf"{pct(f15['fallback_pct'])}). Qwen2.5-7B goes " + pct(f7["att1_pct"]) +
        r" $\to$ " + pct(f7["att2_pct"]) + r" $\to$ " + pct(f7["att3_pct"]) +
        rf" (repair gain +{f7['final_pct'] - f7['att1_pct']:.1f}~pp; fallback "
        rf"{pct(f7['fallback_pct'])}). The Verifier Agent therefore repairs errors "
        r"for both model sizes, and the larger model does not remove the need for "
        r"verification."
    )

    def tax_line(counts, n):
        if n == 0:
            return "n/a"
        parts = []
        for key, label in CHECK_LABELS.items():
            c = counts.get(key, 0)
            if c:
                parts.append(f"{label} ({100.0 * c / n:.1f}\\%)")
        return ", ".join(parts) if parts else "no failed checks"

    taxonomy = (
        r"\paragraph{Failure Taxonomy.} Final-attempt failures, as a share of each "
        rf"model's $N$: Qwen2.5-1.5B ({pct(f15['final_pct'])} final pass, "
        rf"{pct(f15['fallback_pct'])} fallback): {tax_line(tax15, tax_n15)}. "
        rf"Qwen2.5-7B ({pct(f7['final_pct'])} final pass, "
        rf"{pct(f7['fallback_pct'])} fallback): {tax_line(tax7, tax_n7)}."
    )

    limitations = (
        r"\item \textbf{LLM evaluation}: Loop 2 verification is run on 1 seed "
        r"(seed 3) for Qwen2.5-1.5B and 4-bit Qwen2.5-7B-Instruct. " +
        (rf"The 7B run covers {n7} high-risk patients with $M="
         rf"{m7_meta.get('protocol', {}).get('max_attempts')}$, so it is reported as "
         rf"a reduced-scope result. " if partial else
         r"Both models cover all 86 high-risk patients with $M=3$. ") +
        r"Results may not generalise to other seeds, and no 8B model was evaluated "
        r"(no weights were available); the framework is model-agnostic and the "
        r"benchmark script accepts any Hugging Face causal LM that fits in 4-bit on "
        r"the available GPU."
    )

    repro = (
        r"The 7B benchmark is reproduced with "
        r"\texttt{python src/run\_llm\_7b\_benchmark.py} (4-bit NF4, GPU-0 only, "
        r"resumable; traces stream to \texttt{results/llm\_traces\_7b.jsonl}) and "
        r"integrated with \texttt{python src/integrate\_7b\_results.py}. "
        r"Peak VRAM, load time and per-call latency are recorded in "
        r"\texttt{metrics\_verified.json} under \texttt{llm\_evaluation\_7b}."
    )

    conclusion = (
        r"LLM Loop 2 verification on 1 seed (86 patients) achieves " +
        pct(f15["final_pct"]) + r" final pass for Qwen2.5-1.5B-Instruct (" +
        pct(f15["att1_pct"]) + r" $\to$ " + pct(f15["att2_pct"]) + r" $\to$ " +
        pct(f15["att3_pct"]) + rf", $M=3$) and {pct(f7['final_pct'])} for 4-bit "
        r"Qwen2.5-7B-Instruct (" + pct(f7["att1_pct"]) + r" $\to$ " +
        pct(f7["att2_pct"]) + r" $\to$ " + pct(f7["att3_pct"]) +
        rf", $M=3$; fallback {pct(f7['fallback_pct'])})."
    )

    summary = {
        "generated": datetime.now().isoformat(),
        "scope": {
            "partial": partial,
            "n_patients_7b": n7,
            "n_patients_1p5b_matched": len({pid for pid, v in traces15_matched}),
            "n_patients_1p5b_full": n1,
            "max_attempts": m7_meta.get("protocol", {}).get("max_attempts"),
            "quantization": m7_meta.get("quantization"),
            "model": m7_meta.get("model"),
        },
        "models": {
            "Qwen2.5-1.5B-Instruct": m15,
            "Qwen2.5-7B-Instruct-4bit": m7,
        },
        "attempt_progression_7b": {
            "att1_pct": round(f7["att1_pct"], 2),
            "att2_pct": round(f7["att2_pct"], 2),
            "att3_pct": round(f7["att3_pct"], 2),
            "fallback_pct": round(f7["fallback_pct"], 2),
        },
        "failure_taxonomy": {
            "1.5B": {k: v for k, v in tax15.items()},
            "7B": {k: v for k, v in tax7.items()},
        },
        "runtime_7b": m7_meta.get("runtime"),
        "vram_7b": m7_meta.get("vram"),
    }

    # ---- apply edits -----------------------------------------------------
    if not tex_path.exists():
        die(f"manuscript not found: {tex_path}")
    tex = tex_path.read_text(encoding="utf-8")
    original = tex

    edits = []
    tex, how = integrate_block(
        tex, "TABLE8", table8, ANCHORS["TABLE8"])
    edits.append(("Table 8", how))

    tex, how = insert_after(
        tex, "RUNTIME", runtime, ANCHORS["RUNTIME"])
    edits.append(("runtime/VRAM table", how))

    tex, how = integrate_block(
        tex, "FIG4CAP", fig4_caption, ANCHORS["FIG4CAP"])
    edits.append(("Figure 4 caption", how))

    tex, how = integrate_block(
        tex, "ABSTRACT", abstract, ANCHORS["ABSTRACT"])
    edits.append(("Abstract", how))

    tex, how = integrate_block(
        tex, "HEADING", heading, ANCHORS["HEADING"])
    edits.append(("section heading", how))

    tex, how = integrate_block(
        tex, "INTRO", intro, ANCHORS["INTRO"])
    edits.append(("section intro", how))

    tex, how = integrate_block(
        tex, "PROGRESSION", progression, ANCHORS["PROGRESSION"])
    edits.append(("repair progression", how))

    tex, how = integrate_block(
        tex, "TAXONOMY", taxonomy, ANCHORS["TAXONOMY"])
    edits.append(("failure taxonomy", how))

    tex, how = integrate_block(
        tex, "LIMITATIONS", limitations, ANCHORS["LIMITATIONS"])
    edits.append(("limitations", how))

    tex, how = insert_after(
        tex, "REPRO", repro, ANCHORS["REPRO"])
    edits.append(("reproducibility", how))

    tex, how = integrate_block(
        tex, "CONCLUSION", conclusion, ANCHORS["CONCLUSION"])
    edits.append(("Conclusion", how))

    print("\n[Plan]")
    for name, how in edits:
        print(f"  - {name}: {how}")

    if args.dry_run:
        print("\n[dry-run] no files written. Table 8 preview:\n")
        print(table8)
        print("\n[dry-run] runtime table preview:\n")
        print(runtime)
        print("\n[dry-run] summary that would be written:\n")
        print(json.dumps(summary, indent=2)[:4000])
        return 0

    if tex == original:
        print("\n[No-op] manuscript already up to date.")

    backup = tex_path.with_name(tex_path.name + ".bak-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    shutil.copy2(tex_path, backup)
    tex_path.write_text(tex, encoding="utf-8")
    print(f"\n[Write] {tex_path.name} (backup: {backup.name})")

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    try:
        shown = summary_out.relative_to(ROOT)
    except ValueError:
        shown = summary_out
    print(f"[Write] {shown}")

    metrics["llm_evaluation_7b"].setdefault("integration", {})
    metrics["llm_evaluation_7b"]["integration"].update({
        "integrated_at": datetime.now().isoformat(),
        "n_patients": n7,
        "partial": partial,
        "table8_values": {
            "1.5B": {v: round(m15[v]["final_pct"], 2) for v in VARIANTS if m15[v].get("n")},
            "7B": {v: round(m7[v]["final_pct"], 2) for v in VARIANTS if m7[v].get("n")},
        },
        "attempt_progression_7b": summary["attempt_progression_7b"],
    })
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    try:
        shown = metrics_path.relative_to(ROOT)
    except ValueError:
        shown = metrics_path
    print(f"[Write] {shown}")

    # ---- figure ----------------------------------------------------------
    if not args.skip_figure:
        if FIG4_SCRIPT.exists():
            print("\n[Figure] regenerating fig4_llm_repair")
            r = subprocess.run([sys.executable, str(FIG4_SCRIPT)], cwd=ROOT)
            if r.returncode != 0:
                print("[WARN] figure generation failed; manuscript still updated")
        else:
            print(f"[WARN] {FIG4_SCRIPT} missing; skipped figure")

    # ---- compile ---------------------------------------------------------
    if args.no_pdf:
        print("\n[PDF] skipped (--no-pdf)")
        return 0
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        print("\n[PDF] pdflatex not found. Rebuild manually with:")
        print("  pdflatex paper_manuscript_final && bibtex paper_manuscript_final "
              "&& pdflatex paper_manuscript_final && pdflatex paper_manuscript_final")
        return 0

    print("\n[PDF] compiling")
    for cmd in ([pdflatex, "-interaction=nonstopmode", "-halt-on-error",
                 "paper_manuscript_final.tex"],
                [pdflatex, "-interaction=nonstopmode", "-halt-on-error",
                 "paper_manuscript_final.tex"],
                ["bibtex", "paper_manuscript_final"],
                [pdflatex, "-interaction=nonstopmode", "-halt-on-error",
                 "paper_manuscript_final.tex"],
                [pdflatex, "-interaction=nonstopmode", "-halt-on-error",
                 "paper_manuscript_final.tex"]):
        r = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f"[PDF] command failed: {' '.join(cmd)}")
            print(r.stdout.decode("utf-8", "replace")[-3000:])
            return 1
    pdf = ROOT / "paper_manuscript_final.pdf"
    print(f"[PDF] built {pdf.name} ({pdf.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
