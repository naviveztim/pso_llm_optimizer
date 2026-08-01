# Family PSO Kaufland Optimizer

This project includes a family-style shopping optimizer that uses a PSO loop and optional local Qwen model guidance.

## What it does

- Loads product offers from `data/kaufland_prices_by_category_various_countries.json`.
- Creates five competing personas:
  - father
  - mother
  - daughter
  - son
  - mother_in_law
- Runs PSO updates where each persona iteratively improves its shopping proposal.
- Uses fitness = sum of selected prices + penalties for missing persona preferences.
- Stops after `--iterations` (default 30) or earlier when improvement stagnates.

## Files

- `family_pso_optimizer.py` - main optimizer script
- `run_family_pso_smoke_test.py` - validation for generated output
- `data/family_pso_plan.json` - generated best plan (after run)

## Quick start (PowerShell)

```powershell
py -m pip install -r D:\Research\pso_llm_optimizer\data\requirements.txt
py -u D:\Research\pso_llm_optimizer\family_pso_optimizer.py --dry-run
py -u D:\Research\pso_llm_optimizer\run_family_pso_smoke_test.py
```

## Run with Qwen model (no dry-run)

```powershell
py -u D:\Research\pso_llm_optimizer\family_pso_optimizer.py --iterations 30 --model-dir "D:\Research\pso_llm_optimizer\models\ Qwen2.5-1.5B-Instruct" --device auto
```

## GPU-optimized run (CUDA)

```powershell
py -u D:\Research\pso_llm_optimizer\family_pso_optimizer.py --iterations 30 --device cuda --compile-model --llm-refresh-every 2
```

## Notes

- `--dry-run` uses heuristic proposals and does not load `transformers`/`torch`.
- LLM mode uses one loaded model backend and five persona agents to reduce memory usage.
- With `--device auto`, the script prefers CUDA and uses mixed precision (`bfloat16` when supported, otherwise `float16`).
- `--compile-model` can improve throughput on some GPUs/PyTorch versions.
- You can tune convergence with `--stagnation-window` and `--min-delta`.

