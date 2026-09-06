#!/usr/bin/env python3
"""Family shopping optimizer using PSO and optional Copilot API proposals."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from family_llm import FamilyLLM
from utils import (
    IterationRecord,
    OfferItem,
    Particle,
    PersonaProfile,
    load_offers,
    health_score,
    shortlist_from_position,
)
from config import make_profiles
from config import (
    DEFAULT_DATA_FILE_PATH,
    DEFAULT_OUTPUT,
    DEFAULT_COPILOT_MODEL,
    DEFAULT_COPILOT_TOKEN_ENV,
    DEFAULT_NUM_ITERATIONS,
    DEFAULT_COUNTRY,
    MAX_CANDIDATES,
    DEFAULT_SHORT_LIST_SIZE,
    SEED,
)

def evaluate_selection(
    selection: Sequence[int],
    items: Sequence[OfferItem],
) -> Tuple[float, float]:
    """The fitness function: Evaluate a basket using only price and the health of its items.
    """
    total_price = sum(items[i].amount for i in selection)
    health_value = sum(health_score(items[i]) for i in selection)
    health_weight = 1.2
    fitness = total_price - health_weight * health_value
    return total_price, fitness


def run_optimizer(
    items: Sequence[OfferItem],
    profiles: Sequence[PersonaProfile],
    iterations: int,
    stagnation_window: int,
    min_delta: float,
    shortlist_size: int,
    copilot_model: str,
    copilot_token_env: str,
    num_particles: int = 7,
) -> Tuple[List[IterationRecord], Dict[str, Particle], float]:
    """Run the multi-persona PSO loop over the candidate item space.

    Each particle retains one flat combined item-index list containing all family
    baskets, with repeated indexes allowed. The LLM performs one complete-family
    PSO-inspired update using current, personal-best, and swarm-best lists.
    """
    if not items:
        raise ValueError("No items loaded from dataset.")
    if not profiles:
        raise ValueError("At least one family profile is required.")
    if num_particles < 1:
        raise ValueError("num_particles must be at least 1.")

    # Initialize.
    rng = random.Random(SEED)
    llm = FamilyLLM(
        copilot_model=copilot_model,
        token_env_var=copilot_token_env,
        profiles=profiles,
    )
    num_all_items = len(items)
    profile_map = {profile.name: profile for profile in profiles}
    total_basket_size = sum(profile.basket_size for profile in profiles)
    particles: Dict[str, Particle] = {}
    for particle_index in range(num_particles):
        position = [rng.randrange(num_all_items) for _ in range(total_basket_size)]
        particles[f"particle_{particle_index}"] = Particle(
            profiles=profile_map,
            position=position,
            best_position=position.copy(),
        )
    history: List[IterationRecord] = []
    best_global_selection: List[int] = []
    best_global_fitness = float("inf")
    stagnation_counter = 0

    # Main loop: shortlist -> LLM PSO-inspired discrete update -> evaluate -> update bests.
    try:
        for iteration in range(1, iterations + 1):
            previous_best = best_global_fitness

            # Each particle is one complete family plan represented by a flat
            # list. One LLM call updates that entire list.
            for particle_name, particle in particles.items():
                shortlist_ids = shortlist_from_position(
                    particle.position, profiles, items, shortlist_size
                )
                proposal = llm.propose(
                    candidates=items,
                    shortlist_ids=shortlist_ids,
                    current_position=particle.position,
                    personal_best_position=particle.best_selection,
                    personal_best_fitness=particle.best_fitness,
                    global_best_position=best_global_selection,
                    global_best_fitness=best_global_fitness,
                )
                selection = [item_id for item_id in proposal if 0 <= item_id < num_all_items]
                if len(selection) != total_basket_size:
                    selection = particle.position.copy()
                particle.position = selection[:total_basket_size]

                # Apply fitness function
                family_total_price, family_fitness = evaluate_selection(
                    particle.position, items
                )

                # Update history
                history.append(
                    IterationRecord(
                        iteration=iteration,
                        persona=particle_name,
                        selection=particle.position.copy(),
                        total_price=family_total_price,
                        penalty=0.0,
                        fitness=family_fitness,
                    )
                )

                if family_fitness < particle.best_fitness:
                    particle.best_fitness = family_fitness
                    particle.best_position = particle.position.copy()
                    particle.best_selection = particle.position.copy()

                if family_fitness < best_global_fitness:
                    best_global_fitness = family_fitness
                    best_global_selection = particle.position.copy()

                print(
                    f"iter={iteration} particle={particle_name} "
                    f"family_price={family_total_price:.2f} "
                    f"family_fitness={family_fitness:.2f} "
                    f"swarm_best={best_global_fitness:.2f}"
                )

            # Stop when the single best complete family plan stops improving.
            improvement = (
                float("inf")
                if not math.isfinite(previous_best)
                else previous_best - best_global_fitness
            )
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
) -> Dict[str, Any]:
    # Report only the latest completed iteration across personas.
    latest_iteration = max((h.iteration for h in history), default=0)

    def materialize(selection: Sequence[int]) -> Dict[str, Any]:
        selected = [items[i] for i in selection]
        return {
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

    def split_by_member(selection: Sequence[int], particle: Particle) -> Dict[str, List[int]]:
        blocks: Dict[str, List[int]] = {}
        start = 0
        for name, profile in particle.profiles.items():
            blocks[name] = list(selection[start:start + profile.basket_size])
            start += profile.basket_size
        return blocks

    particle_results = {}
    for particle_name, particle in particles.items():
        particle_results[particle_name] = {
            "combined_fitness": round(particle.best_fitness, 2),
            "proposal": list(particle.best_selection),
            "baskets_by_member": {
                name: materialize(selection)
                for name, selection in split_by_member(particle.best_selection, particle).items()
            },
        }

    # The reported family plan is the best complete plan from one particle.
    winning_particle_name = min(
        particles, key=lambda name: particles[name].best_fitness, default=None
    )
    winning_particle = particles.get(winning_particle_name) if winning_particle_name else None
    winning_selection = winning_particle.best_selection if winning_particle else []
    proposals = {
        "item_ids": list(winning_selection),
        "baskets_by_member": {
            name: materialize(selection)
            for name, selection in split_by_member(winning_selection, winning_particle).items()
        } if winning_particle else {},
    }

    # Return metadata plus best proposals grouped by persona.
    return {
        "metadata": {
            "optimizer": "family-pso-llm",
            "iterations_requested": iterations_requested,
            "iterations_completed": latest_iteration,
            "global_best_fitness": round(best_global_fitness, 2),
            "global_best_particle": winning_particle_name,
        },
        "global_best_proposal": proposals,
        "best_proposals_by_particle": particle_results,
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
    parser.add_argument("--min-delta", type=float, default=0.05,
                        help="Minimum global fitness improvement considered significant.")
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES,
                        help="How many cheapest offers to include in optimization space.")
    parser.add_argument("--shortlist-size", type=int, default=DEFAULT_SHORT_LIST_SIZE,
                        help="Per-persona shortlist size sent to proposer.")
    parser.add_argument(
        "--num-particles",
        type=int,
        default=3, # Increase that number to around 10- in order PSO like decisions to working properly.
        help="Number of complete-family particles in the swarm (5-7 recommended).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Parse inputs and load optimization universe.
    offers = load_offers(args.data_file,
                         country_code=args.country.strip().upper(),
                         max_candidates=args.max_candidates)

    # Run the LLM-assisted PSO like optimizer.
    history, particles, best_global_fitness = run_optimizer(
        items=offers,
        profiles=make_profiles(),
        iterations=args.iterations,
        stagnation_window=args.stagnation_window,
        min_delta=args.min_delta,
        shortlist_size=args.shortlist_size,
        copilot_model=args.copilot_model,
        copilot_token_env=args.copilot_token_env,
        num_particles=args.num_particles,
    )

    # Build result payload and write output file.
    result = build_result(
        items=offers,
        history=history,
        particles=particles,
        best_global_fitness=best_global_fitness,
        iterations_requested=args.iterations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print concise execution summary.
    print(f"Wrote plan to {args.output}")
    print(f"Iterations completed: {result['metadata']['iterations_completed']}")
    print(f"Global best fitness: {result['metadata']['global_best_fitness']}")


if __name__ == "__main__":
    main()

