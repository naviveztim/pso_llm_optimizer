seven complete-family particles by default (`--num-particles`).
- The swarm best is one complete family proposal: the lowest-fitness flat list found by any particle.
This project builds a family shopping plan with iterative optimization and optional Copilot API guidance.
- `ParticleLLM`: shared optional LLM backend that updates item-id positions from each persona shortlist.
- `ParticleLLM`: shared optional LLM backend that updates item-id positions from each persona shortlist.
## How the solution is updated (principles)
## How the solution is updated (principles)
- Creates five family personas with different preferences.
- Represents each candidate solution as one complete flat family basket. Duplicate item IDs are allowed because multiple family members may select the same product.
- Uses the minimized objective `fitness = total_price + 1.2 * total_health_penalty`.
- Gives the LLM the full relevant product catalog and asks it to construct the next complete basket.

## LLM composition prompt
2. Generate the next position for each particle
For every update, the LLM receives the current basket, the previous best solution, the best solution found across candidates, their scores, and the full catalog with product metadata.
   - The prompt identifies the consecutive member blocks so changes stay aligned with the correct family member.
The prompt gives these soft influence targets:
4. Evaluate and update PSO memories
- retain approximately 40% of useful current items;
- obtain approximately 15% from useful information in the previous best solution;
- obtain approximately 30% from useful information in the best solution found across candidates;
- use approximately 15% for entirely new exploration.
   - Update the particle's personal best and the single best complete family position found by any particle.
These are influence targets, not mandatory quotas. The LLM chooses actual IDs using price, product type, health characteristics, and interactions among products. It must return exactly `K` IDs, where `K` is the total family basket size.

The program validates IDs and basket length, calculates fitness deterministically, updates the best memories, and retains valid complete positions even when they are temporarily worse than the previous current position.
     - `total_price` (sum of selected items),
## LLM response protocol

`ParticleLLM.propose()` returns a complete position parsed from JSON:
   - Only IDs in the corresponding removal/addition pools are accepted.
```json
{"item_ids":[0,17,42]}
```
      - `total_health_penalty` is derived from health-related product/category keywords; healthy keyword matches reduce the penalty.
The array must contain exactly `K` integer IDs in the catalog range. Duplicate IDs are permitted. Missing, malformed, out-of-range, or incorrectly sized responses are rejected and the current position is retained.
     - Final objective is `fitness = total_price + health_weight * total_health_penalty` (lower is better).
## Objective and health penalty
The health proxy counts configured healthy-keyword matches in the product/category text. Each item receives a bounded penalty:
The health proxy counts configured healthy-keyword matches in product/category text. Each item receives a bounded penalty:
The family objective is therefore price plus a positive health penalty. Healthiness reduces the penalty; it is never subtracted from price.
## Main structures
The family objective is therefore price plus a positive health penalty. Healthiness reduces the penalty; it is never subtracted from price.
- `IterationRecord`: audit row with selection, price, penalty, and fitness.
- `ParticleLLM`: optional stateful Copilot client that generates local swaps.
Healthiness reduces the penalty; it is never subtracted from price.
- `Particle`: current flat basket and personal-best memory.
- `PSO_INERTIA = 0.7`
- `SwapProposal`: one `{remove, add}` replacement.
- `ParticleLLM`: optional stateful Copilot client that generates local swaps.
- `ALPHA_MOVE_SIZE = 0.3`
- `Particle`: current flat basket and previous-best memory.
- `PSO_INERTIA = 0.7`
- `ParticleLLM`: optional stateful Copilot client that generates complete positions.
- `PSO_EXPLORATION = 0.2`
- `ALPHA_MOVE_SIZE = 0.3`
- `CANDIDATE_BUDGET = 20`
The internal guidance values are in `config.py`:
Install the data dependencies:
- `PSO_INERTIA = 0.7` — moderate preservation of the current position.
- `PSO_COGNITIVE = 0.8` — weak previous-best influence.
- `PSO_SOCIAL = 1.8` — strong collective-best influence.
- `PSO_EXPLORATION = 0.2` — new-product exploration influence.

These names are implementation details and are not shown to the LLM; the prompt uses the neutral composition targets above.
- `family_llm.py`: Copilot lifecycle, neutral composition prompt, and position parser.
