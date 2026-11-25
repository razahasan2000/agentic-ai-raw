# legacy/

Superseded scripts kept for provenance. Nothing here is imported by the
canonical pipeline.

| File | Why it was superseded |
|------|-----------------------|
| `run_llm_7b_gpu_fp16_superseded.py` | Loaded the 7B model in **FP16** (`dtype=torch.float16`, `device_map="auto"`), which needs ~15.2 GB and silently offloaded to CPU on the 8.6 GB RTX 4070, dying after ~55 min without saving traces. Replaced by `src/run_llm_7b_benchmark.py` (4-bit NF4, GPU-0 pin, resumable). |
