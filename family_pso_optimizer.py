#!/usr/bin/env python3
"""Family shopping optimizer using PSO and optional Copilot API proposals."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, cast

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_FILE_PATH = REPO_ROOT / "data" / "kaufland_prices_by_category_various_countries.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "family_pso_plan.json"
DEFAULT_COPILOT_MODEL = "auto"
DEFAULT_COPILOT_TOKEN_ENV = "COPILOT_GITHUB_TOKEN"
DEFAULT_COUNTRY = "DE"
DEFAULT_NUM_ITERATIONS = 30
MAX_CANDIDATES = 220 # Total amount of considered products
DEFAULT_SHORT_LIST_SIZE = 70 # Per-persona shortlist size

@dataclass
class OfferItem:
    item_id: int
    product: str
    category: str
    amount: float
    currency: str


@dataclass
class PersonaProfile:
    name: str
    role: str
    keywords: List[str]
    basket_size: int
    min_preferred_items: int


@dataclass
class Particle:
    profile: PersonaProfile
    position: List[float]
    velocity: List[float]
    best_position: List[float]
    best_selection: List[int] = field(default_factory=list)
    best_fitness: float = float("inf")


@dataclass
class IterationRecord:
    iteration: int
    persona: str
    selection: List[int]
    total_price: float
    penalty: float
    fitness: float


def normalize(text: str) -> str:
    # Canonicalize free text for robust keyword matching.
    return re.sub(r"\s+", " ", text.lower()).strip()


def load_offers(dataset_path: Path, country_code: str, max_candidates: int) -> List[OfferItem]:
    # Parse source JSON and walk each category entry.
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    categories = payload.get("categories") or {}
    offers: List[OfferItem] = []
    current_id = 0

    for category, entries in categories.items():
        for entry in entries:
            # Filter by country availability and valid numeric price.
            price = entry.get("price") or {}
            meta = entry.get("meta") or {}
            countries = meta.get("countries_valid") or []
            country_codes = {str(c.get("country_code")) for c in countries if isinstance(c, dict)}
            if country_code and country_code not in country_codes:
                continue

            raw_amount = price.get("amount")
            if raw_amount is None:
                continue
            try:
                amount = float(str(raw_amount))
            except (TypeError, ValueError):
                continue

            # Convert each valid row into a normalized OfferItem.
            offers.append(
                OfferItem(
                    item_id=current_id,
                    product=str(entry.get("product") or "Unknown"),
                    category=str(category),
                    amount=amount,
                    currency=str(price.get("currency") or "EUR"),
                )
            )
            current_id += 1

    # Keep the cheapest candidates and reindex to dense ids.
    offers.sort(key=lambda x: x.amount)
    trimmed = offers[:max_candidates]
    for idx, item in enumerate(trimmed):
        item.item_id = idx
    return trimmed


def keyword_score(item: OfferItem, keywords: Sequence[str]) -> float:
    # Count how many profile keywords appear in product/category text.
    text = normalize(f"{item.product} {item.category}")
    score = 0.0
    for kw in keywords:
        if kw in text:
            score += 1.0
    return score


def health_score(item: OfferItem) -> float:
    # Lightweight health proxy from product/category keywords.
    healthy_keywords = (
        "bio",
        "gemuese",
        "gemuse",
        "obst",
        "salat",
        "tomaten",
        "gurke",
        "vollkorn",
        "hafer",
        "natur",
        "fresh",
        "frisch",
    )
    return keyword_score(item, healthy_keywords)


def sigmoid(value: float) -> float:
    # Numerically stable sigmoid used for position-to-probability mapping.
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def top_indices(values: Sequence[float], k: int) -> List[int]:
    # Return indices of the largest k values.
    indexed = list(enumerate(values))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in indexed[:k]]


class FamilyLLM:
    """Single Copilot API backend used by all family personas."""

    def __init__(
        self,
        copilot_model: str,
        token_env_var: str,
        seed: int,
        max_new_tokens: int = 384,
        max_prompt_items: int = 24,
    ) -> None:
        self.copilot_model = copilot_model
        self.token_env_var = token_env_var
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.max_prompt_items = max_prompt_items

    def _prepare_shortlist_ids(
        self,
        candidates: Sequence[OfferItem],
        shortlist_ids: Sequence[int],
    ) -> List[int]:
        # Remove near-duplicate rows and cap prompt size to stabilize JSON generation.
        unique_ids: List[int] = []
        seen_keys = set()
        for item_id in shortlist_ids:
            item = candidates[item_id]
            key = (normalize(item.product), normalize(item.category), round(item.amount, 2))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_ids.append(item_id)
            if len(unique_ids) >= self.max_prompt_items:
                break
        return unique_ids

    def close(self) -> None:
        # Kept for compatibility with existing call sites; no persistent resources remain.
        return

    async def _async_send_prompt(self, prompt: str) -> str:
        # Keep request lifecycle in one coroutine to avoid cross-loop session issues.
        try:
            from copilot import CopilotClient
        except ImportError as exc:  # pragma: no cover - runtime environment dependent
            raise RuntimeError("Missing dependency 'copilot'. Install it to use Copilot API mode.") from exc

        token = os.environ.get(self.token_env_var)
        if not token:
            raise RuntimeError(f"Missing environment variable {self.token_env_var} for Copilot API authentication.")

        client = CopilotClient({
            "github_token": token,
            "use_logged_in_user": False,
        })  # type: ignore[arg-type]
        await cast(Any, client).start()
        try:
            session = await cast(Any, client).create_session(cast(Any, {"model": self.copilot_model}))
            response = await cast(Any, session).send_and_wait({
                "prompt": (
                    f"{prompt}\n"
                    "Return exactly one JSON object and nothing else."
                )
            })
            return str(cast(Any, response).data.content or "")
        finally:
            await cast(Any, client).stop()

    def _send_prompt(self, prompt: str) -> str:
        return str(asyncio.run(self._async_send_prompt(prompt)))

    def propose(
        self,
        profile: PersonaProfile,
        candidates: Sequence[OfferItem],
        shortlist_ids: Sequence[int],
        personal_best_selection: Sequence[int],
        personal_best_fitness: float,
    ) -> List[int]:

        shortlist_for_prompt = self._prepare_shortlist_ids(candidates, shortlist_ids)
        if not shortlist_for_prompt:
            return []

        # Build persona-scoped prompt with strict JSON response contract.
        prompt_lines = [
            f"You are {profile.role} in a family shopping discussion.",
            "Pick affordable monthly shopping items from shortlist.",
            f"You must choose exactly {profile.basket_size} distinct item IDs.",
            f"At least {profile.min_preferred_items} chosen items should match preferences.",
            f"Preference keywords: {', '.join(profile.keywords)}",
            "Objective: minimize fitness = total_price + penalties + duplicate_penalty - health_bonus - variety_bonus.",
            "Prefer healthier and category-diverse baskets; avoid many near-duplicate products.",
            "Return STRICT JSON only: {\"item_ids\":[...]}.",
            "Return exactly one JSON object and nothing else.",
            "Shortlist:",
        ]

        if personal_best_selection and math.isfinite(personal_best_fitness):
            prompt_lines.insert(
                5,
                f"Current best for this persona: item_ids={list(personal_best_selection)}, fitness={personal_best_fitness:.2f}.",
            )

        for item_id in shortlist_for_prompt:
            item = candidates[item_id]
            prompt_lines.append(
                f"- id={item.item_id} | product={item.product} | category={item.category} | price={item.amount:.2f} {item.currency}"
            )

        prompt = "\n".join(prompt_lines)

        text = self._send_prompt(prompt)

        # Parse and sanitize ids to shortlist scope and basket size.
        parsed = self._extract_json_ids(text)
        if not parsed:
            # Recovery pass: ask model to emit strict JSON from its own previous output.
            repair_prompt = (
                "Convert the following assistant response into STRICT JSON only "
                "with format {\"item_ids\":[...]}. Use only IDs present in the shortlist.\n"
                f"Shortlist IDs: {shortlist_for_prompt}\n"
                f"Assistant response:\n{text}"
            )
            repaired = self._send_prompt(repair_prompt)
            parsed = self._extract_json_ids(repaired)

        valid = [item_id for item_id in parsed if item_id in shortlist_for_prompt]
        return list(dict.fromkeys(valid))[: profile.basket_size]

    @staticmethod
    def _extract_json_ids(text: str) -> List[int]:
        # Prefer strict JSON extraction from generated text.
        matches = re.findall(r"\{[^{}]*}", text, flags=re.DOTALL)
        for blob in reversed(matches):
            try:
                data = json.loads(blob)
            except json.JSONDecodeError:
                continue
            raw_ids = data.get("item_ids")
            if isinstance(raw_ids, list):
                cleaned = []
                for value in raw_ids:
                    try:
                        cleaned.append(int(value))
                    except (TypeError, ValueError):
                        continue
                return cleaned

        # Strict mode: if no valid JSON object is found, return empty result.
        return []


def heuristic_propose(
    profile: PersonaProfile,
    items: Sequence[OfferItem],
    shortlist_ids: Sequence[int],
    rng: random.Random,
) -> List[int]:
    # Score shortlisted items by preference hit + affordability + small random tie-breaker.
    scored: List[Tuple[int, float]] = []
    for item_id in shortlist_ids:
        item = items[item_id]
        pref = keyword_score(item, profile.keywords)
        affordability = 1.0 / max(item.amount, 0.01)
        noise = rng.random() * 0.2
        scored.append((item_id, pref * 2.0 + affordability + noise))

    # Return the highest-scoring fixed-size basket proposal.
    scored.sort(key=lambda x: x[1], reverse=True)
    return [item_id for item_id, _ in scored[: profile.basket_size]]


def evaluate_selection(
    selection: Sequence[int],
    items: Sequence[OfferItem],
    profile: PersonaProfile,
) -> Tuple[float, float, float]:
    # Compute primary objective terms for the chosen basket.
    total_price = sum(items[i].amount for i in selection)
    preferred_hits = sum(1 for i in selection if keyword_score(items[i], profile.keywords) > 0)
    penalty = 0.0

    # Penalize preference shortfall and under-filled baskets.
    missing_preferred = max(0, profile.min_preferred_items - preferred_hits)
    if missing_preferred:
        penalty += missing_preferred * 7.5

    if len(selection) < profile.basket_size:
        penalty += (profile.basket_size - len(selection)) * 5.0

    # Add soft balancing terms so optimization is not driven only by cheapest repeated items.
    health_hits = sum(1 for i in selection if health_score(items[i]) > 0)
    unique_categories = len({normalize(items[i].category) for i in selection})
    unique_products = len({normalize(items[i].product) for i in selection})
    duplicate_count = max(0, len(selection) - unique_products)

    duplicate_penalty = duplicate_count * 2.0
    health_bonus = min(health_hits, max(1, profile.basket_size // 3)) * 1.2
    variety_bonus = min(unique_categories, max(1, profile.basket_size // 2)) * 0.6

    fitness = total_price + penalty + duplicate_penalty - health_bonus - variety_bonus
    return total_price, penalty, fitness


def selection_to_vector(selection: Sequence[int], dimension: int) -> List[float]:
    # Convert selected ids to a dense one-hot-like vector in optimizer space.
    vector = [0.0] * dimension
    for idx in selection:
        if 0 <= idx < dimension:
            vector[idx] = 1.0
    return vector


def update_particle(
    particle: Particle,
    social_best_position: Sequence[float],
    inertia: float,
    cognitive: float,
    social: float,
    rng: random.Random,
    velocity_clip: float = 0.5,
    position_clip: float = 4.0,
) -> None:
    # Standard PSO update with clipping to prevent unstable oscillation.
    for i in range(len(particle.position)):
        r1 = rng.random()
        r2 = rng.random()
        cognitive_term = cognitive * r1 * (particle.best_position[i] - particle.position[i])
        social_term = social * r2 * (social_best_position[i] - particle.position[i])
        velocity = inertia * particle.velocity[i] + cognitive_term + social_term
        velocity = max(-velocity_clip, min(velocity_clip, velocity))
        particle.velocity[i] = velocity
        position = particle.position[i] + velocity
        particle.position[i] = max(-position_clip, min(position_clip, position))


def shortlist_from_position(
    particle: Particle,
    items: Sequence[OfferItem],
    shortlist_size: int,
) -> List[int]:
    # Rank items from particle position with preference boost and price pressure.
    # Add a small duplicate penalty to reduce repeated near-identical products.
    key_counts: Dict[Tuple[str, str], int] = {}
    for item in items:
        key = (normalize(item.product), normalize(item.category))
        key_counts[key] = key_counts.get(key, 0) + 1

    scores = []
    for idx, item in enumerate(items):
        key = (normalize(item.product), normalize(item.category))
        duplicate_penalty = 0.02 * max(0, key_counts.get(key, 1) - 1)
        pref = keyword_score(item, particle.profile.keywords)
        score = sigmoid(particle.position[idx]) + 0.4 * pref - 0.03 * item.amount - duplicate_penalty
        scores.append(score)
    return top_indices(scores, shortlist_size)


def make_profiles() -> List[PersonaProfile]:
    # Define family personas and their shopping preference constraints.
    return [
        PersonaProfile(
            name="father",
            role="father",
            keywords=["beer", "bier", "meat", "fleisch", "grill", "sausage", "wurst"],
            basket_size=12,
            min_preferred_items=4,
        ),
        PersonaProfile(
            name="mother",
            role="mother",
            keywords=["gemuese", "gemuse", "obst", "salat", "bio", "tomaten", "gurke"],
            basket_size=12,
            min_preferred_items=4,
        ),
        PersonaProfile(
            name="daughter",
            role="daughter",
            keywords=["chocolate", "schokolade", "candy", "bonbon", "keks", "ice", "dessert"],
            basket_size=10,
            min_preferred_items=3,
        ),
        PersonaProfile(
            name="son",
            role="son",
            keywords=["chocolate", "schokolade", "snack", "chips", "candy", "cola"],
            basket_size=10,
            min_preferred_items=3,
        ),
        PersonaProfile(
            name="mother_in_law",
            role="mother in law",
            keywords=["detergent", "clean", "reiniger", "spul", "putz", "haushalt", "wasch"],
            basket_size=11,
            min_preferred_items=3,
        ),
    ]


def run_optimizer(
    items: Sequence[OfferItem],
    profiles: Sequence[PersonaProfile],
    iterations: int,
    stagnation_window: int,
    min_delta: float,
    shortlist_size: int,
    dry_run: bool,
    copilot_model: str,
    copilot_token_env: str,
    seed: int,
    llm_refresh_every: int,
) -> Tuple[List[IterationRecord], Dict[str, Particle], float]:
    """Run the multi-persona PSO loop over the candidate item space.

    Each particle coordinate corresponds to one candidate item index in `items`
    (typically 220 after `--max-candidates`). These coordinates are optimized as
    continuous, probability-like selection preferences: higher values (after
    sigmoid) make an item more likely to appear in shortlist/selection steps.
    """
    if not items:
        raise ValueError("No items loaded from dataset.")

    # Initialize random state, optional LLM backend, and one particle per persona.
    rng = random.Random(seed)
    dimension = len(items)
    llm = None if dry_run else FamilyLLM(
        copilot_model=copilot_model,
        token_env_var=copilot_token_env,
        seed=seed,
    )

    particles: Dict[str, Particle] = {}
    for profile in profiles:
        position = [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
        velocity = [rng.uniform(-0.3, 0.3) for _ in range(dimension)]
        particle = Particle(profile=profile, position=position, velocity=velocity, best_position=position.copy())
        particles[profile.name] = particle

    history: List[IterationRecord] = []
    best_global_fitness = float("inf")
    best_global_position = [0.0] * dimension
    best_family_fitness = float("inf")
    stagnation_counter = 0

    # Main PSO loop: propose -> blend -> move -> evaluate -> update bests.
    try:
        for iteration in range(1, iterations + 1):
            previous_best = best_family_fitness

            # In each iteration, each persona items are chosen
            for persona_name, particle in particles.items():
                # Build a persona-specific candidate shortlist from current position.
                # Higher score means item idx is more likely to be selected into the basket.
                shortlist_ids = shortlist_from_position(particle, items, shortlist_size=shortlist_size)

                # Use LLM only on refresh cadence; otherwise use heuristic proposals.
                use_llm = (iteration == 1) or (iteration % llm_refresh_every == 0)
                if llm is None or not use_llm:
                    proposal = heuristic_propose(particle.profile, items, shortlist_ids, rng)
                else:
                    proposal = llm.propose(
                        profile=particle.profile,
                        candidates=items,
                        shortlist_ids=shortlist_ids,
                        personal_best_selection=particle.best_selection,
                        personal_best_fitness=particle.best_fitness,
                    )
                    if not proposal:
                        proposal = heuristic_propose(particle.profile, items, shortlist_ids, rng)

                # Mix proposal signal into position before PSO dynamics.
                proposal_vector = selection_to_vector(proposal, dimension)
                particle.position = [0.75 * p + 0.25 * q for p, q in zip(particle.position, proposal_vector)]

                # Move particle according to PSO update rule.
                update_particle(
                    particle=particle,
                    social_best_position=particle.best_position,
                    inertia=0.62,
                    cognitive=1.42,
                    social=0.0,
                    rng=rng,
                )

                # Decode continuous position to concrete basket and evaluate fitness.
                selection_scores = [sigmoid(x) for x in particle.position]
                selection = top_indices(selection_scores, particle.profile.basket_size)
                total_price, penalty, fitness = evaluate_selection(selection, items, particle.profile)

                # Persist iteration trace for downstream analysis/output.
                history.append(
                    IterationRecord(
                        iteration=iteration,
                        persona=persona_name,
                        selection=selection,
                        total_price=total_price,
                        penalty=penalty,
                        fitness=fitness,
                    )
                )

                # Update persona-level best memory.
                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()
                    particle.best_selection = selection

                # Update swarm-level best memory.
                if fitness < best_global_fitness:
                    best_global_fitness = fitness
                    best_global_position = particle.position.copy()

                # Per-step trace: current fitness and current global best fitness.
                print(
                    f"iter={iteration} persona={persona_name} "
                    f"fitness={fitness:.2f} best_global_fitness={best_global_fitness:.2f}"
                )

            # Early-stop bookkeeping based on family-wide progress across personas.
            current_family_fitness = sum(p.best_fitness for p in particles.values()) / max(1, len(particles))
            if current_family_fitness < best_family_fitness:
                best_family_fitness = current_family_fitness

            improvement = abs(previous_best - best_family_fitness)
            if improvement < min_delta:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            if stagnation_counter >= stagnation_window:
                break
    finally:
        if llm is not None:
            llm.close()

    return history, particles, best_global_fitness


def build_result(
    items: Sequence[OfferItem],
    history: Sequence[IterationRecord],
    particles: Dict[str, Particle],
    best_global_fitness: float,
    iterations_requested: int,
    dry_run: bool,
) -> Dict[str, Any]:
    # Report only the latest completed iteration across personas.
    latest_iteration = max((h.iteration for h in history), default=0)

    # Materialize each persona's best basket into JSON-ready objects.
    proposals = {}
    for persona, particle in particles.items():
        selected = [items[i] for i in particle.best_selection]
        proposals[persona] = {
            "fitness": round(particle.best_fitness, 2),
            "basket_size": len(selected),
            "items": [
                {
                    "item_id": item.item_id,
                    "product": item.product,
                    "category": item.category,
                    "price": item.amount,
                    "currency": item.currency,
                }
                for item in selected
            ],
        }

    # Return metadata plus best proposals grouped by persona.
    return {
        "metadata": {
            "optimizer": "family-pso-llm",
            "dry_run": dry_run,
            "iterations_requested": iterations_requested,
            "iterations_completed": latest_iteration,
            "global_best_fitness": round(best_global_fitness, 2),
        },
        "best_proposals_by_persona": proposals,
    }


def parse_args() -> argparse.Namespace:
    # Build CLI with dataset/model/runtime tuning parameters.
    parser = argparse.ArgumentParser(description="Run family PSO shopping optimization with optional Copilot API proposals.")
    parser.add_argument(
        "--data-file",
        type=Path,
        default= DEFAULT_DATA_FILE_PATH,
        help="Path to Kaufland offers JSON file.",
    )
    parser.add_argument(
        "--copilot-model",
        type=str,
        default=DEFAULT_COPILOT_MODEL,
        help="Copilot model session identifier (e.g. auto).",
    )
    parser.add_argument(
        "--copilot-token-env",
        type=str,
        default=DEFAULT_COPILOT_TOKEN_ENV,
        help="Environment variable name containing GitHub token for CopilotClient.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output file for best plan.",
    )
    parser.add_argument("--country", type=str, default=DEFAULT_COUNTRY,
                        help="Country code to filter offers (e.g. DE).")
    parser.add_argument("--iterations", type=int, default=DEFAULT_NUM_ITERATIONS, help="Maximum PSO iterations.")
    parser.add_argument("--stagnation-window", type=int, default=5,
                        help="Stop if no significant improvement for this many rounds.")
    parser.add_argument("--min-delta", type=float, default=0.15,
                        help="Minimum global fitness improvement considered significant.")
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES,
                        help="How many cheapest offers to include in optimization space.")
    parser.add_argument("--shortlist-size", type=int, default=DEFAULT_SHORT_LIST_SIZE,
                        help="Per-persona shortlist size sent to proposer.")
    parser.add_argument("--llm-refresh-every", type=int, default=1,
                        help="Run LLM proposal every N iterations; heuristic used between.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip Copilot API calls and use heuristic proposals only.")

    return parser.parse_args()


def main() -> None:
    # Parse inputs and load optimization universe.
    args = parse_args()

    offers = load_offers(args.data_file, country_code=args.country.strip().upper(), max_candidates=args.max_candidates)
    profiles = make_profiles()

    # Run optimizer with selected mode (dry-run or LLM-assisted).
    history, particles, best_global_fitness = run_optimizer(
        items=offers,
        profiles=profiles,
        iterations=args.iterations,
        stagnation_window=args.stagnation_window,
        min_delta=args.min_delta,
        shortlist_size=args.shortlist_size,
        dry_run=args.dry_run,
        copilot_model=args.copilot_model,
        copilot_token_env=args.copilot_token_env,
        seed=args.seed,
        llm_refresh_every=max(1, args.llm_refresh_every),
    )

    # Build result payload and write output file.
    result = build_result(
        items=offers,
        history=history,
        particles=particles,
        best_global_fitness=best_global_fitness,
        iterations_requested=args.iterations,
        dry_run=args.dry_run,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print concise execution summary.
    print(f"Wrote plan to {args.output}")
    print(f"Iterations completed: {result['metadata']['iterations_completed']}")
    print(f"Global best fitness: {result['metadata']['global_best_fitness']}")


if __name__ == "__main__":
    main()

