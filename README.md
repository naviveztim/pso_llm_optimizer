# Family PSO LLM Optimizer
This project builds a complete grocery plan for a family by combining a particle-swarm-like search loop with Copilot LLM proposals. The optimizer selects products from a Kaufland offer catalog, evaluates every proposed plan with a deterministic objective, and writes the best plan to JSON.
The implementation is in `family_pso_optimizer.py`, with the LLM integration in `particle_llm.py`. The project does not implement local remove/add swaps or vector-valued particle velocities: each LLM call proposes a complete family position.
## What is optimized
The configured family has five contiguous basket blocks:
| Member | Items | Preference keywords |
| --- | ---: | --- |
| father | 12 | beer, meat, grill, sausage |
| mother | 12 | vegetables, fruit, salad, organic produce |
| daughter | 10 | chocolate, candy, biscuits, dessert |
| son | 10 | chocolate, snacks, chips, cola |
| mother-in-law | 11 | detergent, cleaning, household products |
Together these blocks form one flat position of **55 item IDs** (12 + 12 + 10 + 10 + 11). IDs may repeat because the same catalog product can be selected for more than one family member. The LLM is shown the block layout so it can keep each part of the flat position associated with the appropriate member.
Before optimization, `load_offers()` filters offers by country, ignores offers without a valid price, sorts the remaining offers by price, keeps at most `MAX_CANDIDATES` offers, and renumbers them from zero.
## What “PSO” means in this project
This is **PSO-inspired LLM search**, not classical mathematical PSO. The implementation keeps the swarm concepts of particles, personal bests, a collective best, fitness comparison, inertia/cognitive/social/exploration coefficients, and stagnation-based stopping. However, it does **not** maintain a velocity vector, calculate `v = v + ...`, or update a numeric position with `x = x + v`. A position is a discrete list of product IDs, so ordinary continuous PSO equations do not apply directly.
The actual update algorithm is:
```text
initialize each particle with a random complete family basket
evaluate each basket
set particle.best_selection and particle.best_fitness
set swarm.global_best_position and swarm.global_best_fitness
repeat for each iteration:
    for each particle:
        calculate four normalized influence shares
        build a prompt containing:
            - the particle's current basket
            - its personal-best basket
            - the swarm's collective-best basket
            - all three fitness values
            - the complete candidate catalog
        ask the LLM to generate a complete next basket
        parse and validate the returned item IDs
        if invalid, keep the current basket
        evaluate the resulting basket
        replace the particle's personal best if the score improves
        replace the swarm's collective best if the score improves
    stop early if the swarm has stagnated
```
Therefore, the LLM is the **position-update operator**. The PSO-inspired values do not modify item IDs mathematically; they are converted into textual percentage targets that tell the LLM how strongly to consider the current position, personal best, collective best, and new exploration. The search can be described as a swarm of LLM-guided particles with PSO-style memory, rather than as a full classical PSO implementation.## How the PSO-like prompts work
The search state is maintained independently for each particle:
- **Current position:** the particle's latest complete 55-ID family basket.
- **Personal best:** the best position found by that particle so far.
- **Collective best:** the best complete position found by any particle.
- **Fitness values:** the scores for the current, personal-best, and collective-best positions.
For every particle and iteration, `ParticleLLM.propose()` creates a prompt containing those three positions and their scores, followed by the complete candidate catalog. Each catalog row includes the item ID, price, currency, category, product name, and health penalty. The model is asked to construct a new complete basket rather than to return a single swap.
The prompt translates the PSO-inspired coefficients into four normalized, randomized influence shares:
1. **Current position** — controlled by `PSO_INERTIA` (`0.7` by default).
2. **Personal best** — based on `PSO_COGNITIVE` (`0.8`) multiplied by a per-call random factor.
3. **Collective best** — based on `PSO_SOCIAL` (`1.8`) multiplied by a per-call random factor.
4. **Exploration** — based on `PSO_EXPLORATION` (`0.2`) multiplied by a per-call random factor.
The normalized shares are presented as soft targets such as “retain about 40%” or “use about 15% for exploration.” They are guidance, not quotas. The model may preserve current products, borrow from either best position, or introduce products that appear in neither best position. The prompt asks it to balance price, product type, health characteristics, and product interactions.
The response protocol is deliberately strict:
```json
{"item_ids": [0, 17, 42]}
```
The actual array must contain exactly 55 integer IDs, and every ID must be in the current catalog range. Duplicate IDs are valid. If JSON is missing, malformed, out of range, or has the wrong length, the proposal is rejected and the particle keeps its current position.
## LLM loop and conversation context
Each particle receives its own `ParticleLLM` instance and therefore its own Copilot session. The session is created lazily on the first proposal using the `COPILOT_GITHUB_TOKEN` environment variable (or the variable named by `--copilot-token-env`). A background asyncio event loop lets the synchronous optimizer call the asynchronous Copilot client.
The first message for a particle establishes persistent conversation context:
- the total basket size;
- the fact that duplicate IDs are allowed;
- the fixed family-member block layout; and
- every member's basket size, role, preference keywords, and minimum preferred item target.
The first message also explains the high-level goal: prefer healthier products while keeping the budget low. After that initial context has been sent, later messages do not repeat the setup. Each subsequent message asks for the next basket and supplies the changing optimization state: influence shares, the current position, personal best, collective best, their scores, a variation token, and the full catalog. Thus, initial context is one-time per particle session, while the positions and scores are refreshed on every question.
Copilot is expected to return only the JSON object. Responses are parsed by `_extract_json_ids()`, validated, and then evaluated locally. A Copilot timeout or unavailable dependency/token raises an operational error; an invalid model response specifically falls back to the current position.
## Optimization loop
`run_optimizer()` performs the following steps:
1. Initialize independently seeded random positions and evaluate them.
2. Set each particle's personal best and choose the initial collective best.
3. For each iteration, ask every particle's LLM for a complete next position.
4. Evaluate the returned position and update the particle's personal best when its fitness improves.
5. Update the collective best when any proposal improves the swarm score.
6. Record the proposal in `IterationRecord` and stop after the requested number of iterations or after the configured stagnation window.
7. Close all Copilot clients, including when the loop fails.
The command-line default is **3 particles** and **10 maximum iterations**. The `run_optimizer()` function itself has a default of 7 particles when called directly, but the CLI passes the value from `--num-particles`.
## Objective and health proxy
The objective is minimized:
```text
fitness = total_price + 1.2 * total_health_penalty
```
`total_price` is the sum of each selected occurrence. The health penalty is computed per item from configured healthy keywords matched in its product and category text. Each item contributes a value from `0.0` to `1.0`; more healthy keyword matches reduce that value. Repeated IDs contribute repeatedly to both price and penalty.
The health score is a lightweight keyword proxy, not a nutritional or medical assessment. It never subtracts money from the price.
## Output
The optimizer writes `data/family_pso_plan.json` by default. The output contains:
- run metadata and completed iterations;
- the global best complete position;
- the winning particle; and
- each particle's best proposal split into the five member baskets, including product, category, price, and currency details.
## Setup and usage
Install the dependencies listed in `data/requirements.txt`:
```powershell
py -m pip install -r D:\Research\pso_llm_optimizer\data\requirements.txt
```
Set a GitHub token for Copilot API access, then run the optimizer from the repository root:
```powershell
$env:COPILOT_GITHUB_TOKEN = "<your-token>"
py -u .\family_pso_optimizer.py
```
Useful options include:
```powershell
py -u .\family_pso_optimizer.py `
  --country DE `
  --iterations 10 `
  --num-particles 3 `
  --max-candidates 220 `
  --output .\data\family_pso_plan.json
```
Use `py -u .\family_pso_optimizer.py --help` for the complete option list.
## Repository files
- `family_pso_optimizer.py` — CLI, swarm initialization, evaluation, loop, and JSON report generation.
- `particle_llm.py` — per-particle Copilot session, prompt construction, response parsing, and validation.
- `utils.py` — data classes, catalog loading, family profiles, and health scoring.
- `config.py` — paths, defaults, PSO coefficients, and health keywords.
- `data/kaufland_prices_by_category_various_countries.json` — catalog input.
- `test_API_calls.py` — standalone Copilot connectivity smoke script; it is not the optimizer test suite.
- `test_semantic_similrity.py` — standalone embedding comparison experiment; it is not used by the optimizer.
The scraper and its documentation are in the `data/` directory. The optimizer does not use sentence-transformer embeddings or a separate `family_llm.py` module.

