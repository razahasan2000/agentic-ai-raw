# Certifiable Agentic Risk Explanation

Reproducibility package for:

> **Certifiable Agentic Risk Explanation: Subgroup-Calibrated Conformal Guarantees, Plausible Counterfactual Recourse, and Zero-Hallucination Clinical Narratives**

> Anonymous Authors — Under Review — 2026

The package contains two manuscripts:

- **`paper/main.tex` / `paper/main.pdf`** — Main manuscript (IMRaD format, 12 pages)
- **`paper/supplementary.tex` / `paper/supplementary.pdf`** — Comprehensive Supplementary Information (11 pages, 8 notes)

Every number in both documents is produced by the scripts in `src/`; the canonical metrics are in `results/metrics_verified.json`.

---

## 1. Quick Start

```bash
# Core + LLM dependencies
pip install -r requirements.txt -r requirements-llm.txt

# For an NVIDIA GPU, install the CUDA build of torch first, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu128

# 20-seed statistical pipeline (conformal coverage, baselines, counterfactuals,
# attribution, auditor, drift). Writes results/metrics_verified.json.
python src/run_full_verified_pipeline.py

# 1.5B LLM Loop-2 evaluation (seed 3, 86 high-risk patients).
# Writes results/llm_traces.jsonl and updates metrics_verified.json.
python src/run_llm_eval_fixed.py

# 4-bit 7B LLM benchmark (seed 3, same patients/prompts/verifier, resumable).
export HF_TOKEN=...            # PowerShell: $env:HF_TOKEN="..."
python src/run_llm_7b_benchmark.py

# Manuscript integration (Table 8, Figure 4, Abstract, Conclusion, Limitations,
# Reproducibility — idempotent, recomputes all values from raw traces).
python src/integrate_7b_results.py --dry-run   # show the plan
python src/integrate_7b_results.py             # patch the manuscript
```

Tested with **Python 3.14**, **PyTorch 2.11.0+cu128**, **Transformers 5.12**, **bitsandbytes 0.50.2** on an **RTX 4070 Laptop GPU** (8.6 GB VRAM, driver 616.87, CUDA 12.8).

---

## 2. Repository Layout

```
reproducibility_package/
├── paper/                          # Manuscripts
│   ├── main.tex / main.pdf         # Main paper (IMRaD, 12 pp)
│   ├── supplementary.tex / supplementary.pdf  # Supplementary Information (11 pp)
│   └── references_final.bib
├── src/                            # Canonical scripts (9 scripts)
│   ├── run_full_verified_pipeline.py   # 20-seed statistical pipeline
│   ├── run_llm_eval_fixed.py           # 1.5B LLM Loop-2 evaluation
│   ├── run_llm_7b_benchmark.py         # 4-bit 7B LLM benchmark (resumable)
│   ├── integrate_7b_results.py         # Manuscript integration (Table 8, Fig. 4)
│   ├── generate_fig1_architecture.py   # Figure 1 — architecture (TikZ)
│   ├── generate_fig2_conformal.py      # Figure 2 — conformal parity
│   ├── generate_fig3_counterfactual.py # Figure 3 — counterfactual Pareto
│   ├── generate_fig4_llm_repair.py     # Figure 4 — LLM repair progression
│   └── generate_fig5_drift.py          # Figure 5 — drift auditor
├── legacy/                         # Superseded scripts (provenance only)
│   └── run_llm_7b_gpu_fp16_superseded.py
├── data/
│   └── nhanes_diabetes_cohort.csv  # NHANES 2017–2018 (5,533 adults)
├── results/
│   ├── metrics_verified.json       # Canonical metrics (20 seeds + LLM)
│   ├── llm_traces.jsonl            # 1.5B traces (344 records)
│   ├── llm_traces_7b.jsonl         # 7B traces (344 records)
│   ├── table8_7b_summary.json      # Derived 7B summary
│   ├── figures/                    # Copies of paper figures
│   └── tables/
├── fig1_architecture.png/.pdf      # Paper figures (also in results/figures/)
├── fig2_conformal_parity.png
├── fig3_counterfactual_pareto.png
├── fig4_llm_repair.png
├── fig5_drift_auditor.png
├── requirements.txt
├── requirements-llm.txt
└── README.md
```

---

## 3. Canonical Scripts

| Task | Script |
|------|--------|
| 20-seed statistical pipeline | `src/run_full_verified_pipeline.py` |
| 1.5B LLM Loop-2 evaluation | `src/run_llm_eval_fixed.py` |
| 4-bit 7B LLM benchmark (resumable) | `src/run_llm_7b_benchmark.py` |
| Manuscript integration | `src/integrate_7b_results.py` |
| Figure 1 — architecture (TikZ) | `src/generate_fig1_architecture.py` |
| Figure 2 — conformal parity | `src/generate_fig2_conformal.py` |
| Figure 3 — counterfactual Pareto | `src/generate_fig3_counterfactual.py` |
| Figure 4 — LLM repair progression | `src/generate_fig4_llm_repair.py` |
| Figure 5 — drift auditor | `src/generate_fig5_drift.py` |

Older variants (`clean_pipeline_v2..v8`, `agent_evaluation_v2`, etc.) are not used by the paper. The FP16 7B script that caused the original OOM is archived at `legacy/run_llm_7b_gpu_fp16_superseded.py`.

---

## 4. Data

Only **NHANES 2017–2018** is included (`data/nhanes_diabetes_cohort.csv`, 5,533 adults, 18.6% diabetes prevalence). Features deliberately exclude HbA1c and blood glucose to avoid circularity; the target is `known_diabetes` (self-reported physician diagnosis).

MIMIC-IV and BRFSS are **not** part of this repository. Multi-cycle NHANES temporal validation was not possible (CDC `.XPT` downloads were unavailable); this is stated in the paper's Limitations.

---

## 5. Verified Results

From `results/metrics_verified.json` (20 stratified seeds, seeds 0–19; mean with 95% CI where applicable):

| Metric | Value |
|--------|-------|
| Logistic-regression AUC (no HbA1c/glucose) | 0.770 |
| Overall conformal test coverage (target 90%) | 91.3% |
| Worst-subgroup test coverage | 84.4% |
| Max coverage gap | 11.1% |
| Validation ECE (Platt-calibrated) | 3.9% |
| Counterfactual target reach | 100% |
| Ellipsoid containment | 100% |
| Mean ‖δ‖₂ | 1.38 SD |
| Tier distribution (Tier 1 / 2 / 3) | 31.2% / 24.4% / 0.0% |

Conformal baselines, worst-subgroup coverage: **EB-MinCov 84.4%**, Mondrian 81.1%, Standard EB 80.9%, Marginal 82.3%, Complete Pooling 82.3%.

**LLM Loop-2 evaluation (seed 3, 86 high-risk patients, greedy decoding):**

| Variant | Qwen2.5-1.5B final pass | Qwen2.5-7B (4-bit NF4) final pass |
|---------|------------------------|----------------------------------|
| Unconstrained | 0/86 (0.0%) | 0/86 (0.0%) |
| Schema prompted | 60/86 (69.8%) | 46/86 (53.5%) |
| Loop 1 + schema | 58/86 (67.4%) | 29/86 (33.7%) |
| Full agentic (M=3) | 81/86 (94.2%) | **86/86 (100.0%)** |

Cumulative repair progression (Attempt 1 → 2 → 3):
- **Qwen2.5-1.5B:** 67.4% → 91.9% → 94.2% (fallback 5.8%)
- **Qwen2.5-7B (4-bit NF4):** 33.7% → 87.2% → 100.0% (fallback 0.0%)

**7B runtime/VRAM (RTX 4070 Laptop):** load 35.6 s, peak VRAM 5.74 GB, mean per-call latency 49.1 s, total generation time 281.4 min (344 calls).

---

## 6. 7B Benchmark: Memory Design and Resumability

The original 7B script loaded **FP16** weights (`dtype=torch.float16`, `device_map="auto"`), which needs ~15.2 GB and silently offloaded to CPU on an 8.6 GB GPU; it died after ~55 minutes with nothing saved. The current script:

* loads genuine **4-bit NF4** weights (double quantisation, FP16 compute), measured at **5.56 GB allocated / 5.69 GB peak** on the RTX 4070;
* pins `device_map={"": 0}` and asserts every parameter is on `cuda:0` (no CPU/meta offload);
* uses SDPA attention and one sequence at a time;
* retries CUDA OOM by clearing the cache, then halving the token budget and **recording** the reduction in the trace (`oom_retries`, `max_new_tokens`);
* appends + `fsync`s every trace, repairs a torn final line on restart, skips completed `(patient, variant)` pairs, and removes duplicates;
* writes metrics atomically.

Useful flags:

```bash
python src/run_llm_7b_benchmark.py --limit 43 --max-attempts 1
python src/run_llm_7b_benchmark.py --variants full_agentic
python src/run_llm_7b_benchmark.py --mock --limit 3 \
  --traces-file results/mock_smoke.jsonl --metrics-file results/mock_metrics.json
```

`--mock` is for pipeline smoke tests only and is refused by the integrator. Reduced-scope runs must be integrated with `--accept-partial`, which labels the table and prose as a partial/matched-subset result.

---

## 7. Manuscript Integration

```bash
python src/integrate_7b_results.py --dry-run   # show the plan
python src/integrate_7b_results.py             # patch the manuscript
python src/integrate_7b_results.py --no-pdf    # skip LaTeX compilation
```

The integrator recomputes every value from the raw traces, aborts on mock/non-4-bit data or metrics/trace mismatches, backs up the `.tex`, wraps each edit in `% LLM7B:<NAME>:BEGIN/END` markers (idempotent), regenerates Figure 4, and writes `results/table8_7b_summary.json`.

Manual PDF rebuild:

```bash
pdflatex paper_manuscript_final
bibtex  paper_manuscript_final
pdflatex paper_manuscript_final
pdflatex paper_manuscript_final
```

To rebuild both main and supplementary:

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
pdflatex supplementary.tex && bibtex supplementary.tex && pdflatex supplementary.tex && pdflatex supplementary.tex
```

---

## 8. Known Limitations

* **Single seed:** LLM evaluation uses one seed (seed 3, 86 high-risk patients); it is not a 20-seed estimate.
* **No 8B model:** No weights were available; the benchmark accepts any Hugging Face causal LM that fits in 4-bit on the available GPU.
* **Single dataset:** Results on NHANES 2017–2018 only; external validation needed. Multi-cycle NHANES temporal validation was not performed (CDC `.XPT` downloads were unavailable).
* **Diabetes definition:** Self-reported physician diagnosis (DIQ010) misses undiagnosed cases and conflates T1D/T2D.
* **Throughput:** On the tested laptop the 4-bit 7B path runs at ~3 tokens/s (GPU at ~600 MHz, ~17 W); the full 86-patient, 4-variant, M=3 run takes hours. This is a throughput limitation, not a memory limitation.

---

## 9. Citation

```bibtex
@article{conformal_plausibility_agent_2026,
  title={Certifiable Agentic Risk Explanation: Subgroup-Calibrated Conformal Guarantees, Plausible Counterfactual Recourse, and Zero-Hallucination Clinical Narratives},
  author={Anonymous Authors},
  journal={Under Review},
  year={2026}
}
```

---

## 10. License

MIT License.