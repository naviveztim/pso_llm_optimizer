# Family PSO Kaufland Optimizer

This project builds a family shopping plan with Particle Swarm Optimization (PSO), optionally guided by Copilot API calls.

## What it does

- Loads product offers from `data/kaufland_prices_by_category_various_countries.json`.
- Creates five personas (`father`, `mother`, `daughter`, `son`, `mother_in_law`) with different preferences.
- Optimizes one shopping basket per persona under a shared PSO process.
- Scores each proposal by `fitness = total_price + penalties`.
- Stops when either max iterations are reached or global fitness stops improving.

## Main structures and their meaning

The optimizer logic in `family_pso_optimizer.py` is centered around these structures:

- `OfferItem`: one candidate product (id, product name, category, price, currency).
- `PersonaProfile`: persona definition (keywords, basket size, minimum preferred matches).
- `Particle`: PSO state for one persona:
  - `position`: continuous preference vector over all candidate products.
  - `velocity`: movement vector used by PSO to update position.
  - `best_position` / `best_selection` / `best_fitness`: that persona's personal best memory.
- `IterationRecord`: audit row for each persona update in each iteration (selection, price, penalty, fitness).
- `FamilyLLM`: shared optional LLM backend that proposes item ids from each persona shortlist.

## How the solution is updated (principles)

Each optimization iteration follows the same update principles:

1. Build shortlist from current `position`
   - Convert position values with sigmoid.
   - Blend with preference keyword score and price pressure to rank products.
   - Keep top `--shortlist-size` items per persona.

2. Generate a proposal for each persona
   - Dry run or non-refresh iteration: use heuristic proposer.
   - Refresh iteration (`iteration == 1` or `iteration % --llm-refresh-every == 0`): try LLM proposal.
   - If LLM output is empty/invalid, fallback to heuristic.

3. Blend proposal into particle state
   - Transform selected ids into a one-hot vector.
   - Blend with existing position: `position = 0.75 * position + 0.25 * proposal_vector`.

4. Apply PSO dynamics
   - Update velocity with inertia + cognitive + social terms.
   - Update position using new velocity.
   - This combines:
     - current motion (`inertia`),
     - pull toward persona's personal best (`cognitive`),
     - pull toward global best found by any persona (`social`).

5. Decode position into a concrete basket and evaluate
   - Select top `basket_size` indices from sigmoid(position).
   - Compute:
     - `total_price` (sum of selected items),
     - penalty for missing preferred items,
     - penalty for under-filled basket.
   - Final objective remains `fitness = total_price + penalty` (lower is better).

6. Update best memories
   - If current fitness improves persona best, update particle personal best.
   - If it improves global best, update swarm global best.

7. Check convergence / stagnation
   - Measure global best improvement against `--min-delta`.
   - If improvement is too small for `--stagnation-window` rounds, stop early.

## Key files

- `family_pso_optimizer.py`: main optimizer and CLI entry point.
- `run_family_pso_smoke_test.py`: smoke test for generated output JSON.
- `data/family_pso_plan.json`: generated optimization result (after run).
- `data/family_pso_plan_smoke.json`: sample smoke-test output.

## Quick start (PowerShell)

```powershell
py -m pip install -r D:\Research\pso_llm_optimizer\data\requirements.txt
py -u D:\Research\pso_llm_optimizer\family_pso_optimizer.py --dry-run
py -u D:\Research\pso_llm_optimizer\run_family_pso_smoke_test.py
```

## Run with Copilot API mode

```powershell
$env:COPILOT_GITHUB_TOKEN = "<your_github_token>"
py -u D:\Research\pso_llm_optimizer\family_pso_optimizer.py --iterations 30 --copilot-model auto
```

## Run with explicit token environment variable name

```powershell
$env:MY_TOKEN = "<your_github_token>"
py -u D:\Research\pso_llm_optimizer\family_pso_optimizer.py --iterations 30 --copilot-token-env MY_TOKEN
```

## Notes

- `--dry-run` skips API calls and uses only heuristic proposals.
- API mode requires a token in `COPILOT_GITHUB_TOKEN` (or custom name via `--copilot-token-env`).
- `FamilyLLM` now uses one shared Copilot session across all personas.
- Convergence behavior is mainly controlled by `--stagnation-window` and `--min-delta`.

