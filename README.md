# Family PSO Kaufland Optimizer

This project builds a family shopping plan with Particle Swarm Optimization (PSO), optionally guided by Copilot API calls.

## What it does

- Loads product offers from `data/kaufland_prices_by_category_various_countries.json`.
- Creates five personas (`father`, `mother`, `daughter`, `son`, `mother_in_law`) with different preferences.
- Optimizes complete family proposals under a shared PSO-like process.
- Scores each proposal by `fitness = total_price - health_weight * health_value`.
- Stops when either max iterations are reached or global fitness stops improving.

## Main structures and their meaning

The optimizer logic in `family_pso_optimizer.py` is centered around these structures:

- `OfferItem`: one candidate product (id, product name, category, price, currency).
- `PersonaProfile`: persona definition (keywords, basket size, minimum preferred matches).
- `Particle`: PSO state for one complete family proposal:
  - `position`: one flat list containing all family-member item indexes; duplicate indexes are allowed.
  - The list uses fixed consecutive blocks: father first, then mother, daughter, son, and mother-in-law. Each block length is that member's `basket_size`.
  - `best_position` / `best_selection` / `best_fitness`: that particle's best complete family proposal.
- The swarm contains seven complete-family particles by default (`--num-particles`).
- The swarm best is one complete family proposal: the lowest-fitness flat list found by any particle.
- `IterationRecord`: audit row for each persona update in each iteration (selection, price, penalty, fitness).
- `FamilyLLM`: shared optional LLM backend that updates item-id positions from each persona shortlist.

## How the solution is updated (principles)

Each optimization iteration follows the same update principles:

1. Build a shortlist from each particle's flat `position`
   - Rank item indexes by frequency in the position, all family preferences, and price.
   - Keep top `--shortlist-size` products for the family-level prompt.

2. Generate the next position for each particle
   - Send the current, personal-best, and swarm-best flat proposals to the LLM.
   - The prompt identifies the consecutive member blocks so changes stay aligned with the correct family member.
   - The LLM returns one complete updated `item_ids` list every time.
   - `FamilyLLM` keeps one Copilot conversation for the lifetime of the optimizer run, across all particles and iterations.
   - The first message establishes the durable context: the total basket size, duplicate-index rule, family-member position blocks, every family preference, the affordability/health objectives, and the meaning of fitness values (`fitness = total basket price - 1.2 * total health value`; lower is better).
   - Later messages are follow-up questions containing only the changing PSO positions, fitness values, and current available-item shortlist. The LLM can therefore use the remembered family context instead of receiving the same instructions repeatedly.
   - If the response is empty/invalid, retain the current selection.

3. Store the LLM-updated flat position
   - Preserve repeated indexes because multiple family members may want the same item.
   - No proposal blending or numeric post-proposal PSO movement is applied.

4. Evaluate and update PSO memories
   - Evaluate the LLM-selected basket.
   - Update the particle's personal best and the single best complete family position found by any particle.
   - Both best selections are included in subsequent LLM prompts.

5. Evaluate the concrete basket
   - Use the LLM-selected flat ids directly (rather than decoding a newly moved vector).
   - Compute:
     - `total_price` (sum of selected items),
      - health value derived from health-related product/category keywords.
    - Final objective is `fitness = total_price - health_weight * health_value` (lower is better).

6. Check convergence / stagnation
   - Measure global best improvement against `--min-delta`.
   - If improvement is too small for `--stagnation-window` rounds, stop early.

## Conversational LLM approach

`family_llm.py` uses the Copilot SDK as a stateful chat rather than creating an
independent chat for every proposal. The session is created lazily, on the
first call that has candidates, and is then reused for every subsequent call
to `propose()`.

The conversation has two types of messages:

1. **Initialization message** — sent once. It tells the model what it is
   optimizing, the exact total number of items, the fixed contiguous index
   blocks for each family member, all persona preferences, and the general
   selection objectives. It also explains that lower fitness is better and
   defines fitness as `total basket price - 1.2 * total health value`, so the
   personal-best and global-best fitness values have a clear meaning.
2. **Step message** — sent once per particle update. It supplies the current
   position, personal-best position, global-best position, their fitness values,
   and the current shortlist with prices, categories, and product names. It
   asks for exactly one JSON object containing a complete position.

The optimizer itself is synchronous while the Copilot SDK is asynchronous. The
implementation uses a **dedicated background event-loop thread** owned by
`FamilyLLM`:

- The loop is started lazily when the first request is needed.
- Each synchronous `_send_prompt()` call submits its coroutine to that same loop
  with `asyncio.run_coroutine_threadsafe()` and waits for its result.
- The Copilot client and session therefore remain attached to one event loop for
  the entire optimizer run.

This is important because calling `asyncio.run()` for every prompt creates and
then closes a new event loop each time. An asynchronous Copilot session tied to
the previous loop cannot reliably be reused, and the intended chat history can
be lost. The dedicated loop lets repeated `propose()` calls continue the same
conversation instead.

`close()` now performs the matching shutdown: it stops the shared Copilot
client, stops the dedicated event loop, and joins its thread. The optimizer
calls it from its existing `finally` block, including when optimization exits
because of an error.

The session lifecycle can be tested without API credentials using a local
mocked `copilot` module. The mock should verify that two or more calls to
`propose()` create only one client and one session, that the initialization
context appears only in the first prompt, that later prompts contain the new
PSO state, and that `close()` stops the client and loop. If the real model
returns malformed JSON, an incomplete list, or no response, the existing
optimizer behavior remains unchanged: it keeps the particle's current
selection.

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
- Use `--cognitive-coeff` and `--social-coeff` to tune the LLM's personal-best and swarm-best influence (both default to `1.42`).
- `FamilyLLM` uses one shared Copilot session across all particles and personas; the session remembers the initial family context throughout the run.
- Convergence behavior is mainly controlled by `--stagnation-window` and `--min-delta`.

