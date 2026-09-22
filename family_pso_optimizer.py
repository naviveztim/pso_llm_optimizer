#!/usr/bin/env python3
"""Family shopping optimizer using PSO and optional Copilot API proposals."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from family_llm import FamilyLLM
from utils import (
    IterationRecord,
    OfferItem,
    Particle,
    PersonaProfile,
    health_penalty,
    load_offers,
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
    SEED,
    DEFAULT_NUMBER_OF_PARTICLES,
    DEFAULT_STAGNATION_WINDOW,
    DEFAULT_MIN_DELTA,
    PSO_INERTIA,
)


@dataclass
class SwarmState:
    """Mutable state shared by the particles during one optimizer run."""

    particles: Dict[str, Particle]
    particle_llms: Dict[str, FamilyLLM]
    global_best_position: List[int]
    global_best_fitness: float
    stagnation_counter: int = 0


def _initialize_swarm(
    items: Sequence[OfferItem],
    profiles: Sequence[PersonaProfile],
    num_particles: int,
    copilot_model: str,
    copilot_token_env: str,
) -> SwarmState:
    """Create particles, LLM sessions, and the initial global best."""
    item_count = len(items)
    basket_size = sum(profile.basket_size for profile in profiles)
    particles: Dict[str, Particle] = {}
    particle_llms: Dict[str, FamilyLLM] = {}
    used_positions = set()

    try:
        for particle_index in range(num_particles):

            # Select randomly distinct initial positions for each particle.
            rng = random.Random(SEED + particle_index)
            position = [rng.randrange(item_count) for _ in range(basket_size)]

            # Keep initial particles distinct while preserving deterministic seeds.
            position_key = tuple(position)
            while position_key in used_positions:
                position[-1] = (position[-1] + 1) % item_count
                position_key = tuple(position)
            used_positions.add(position_key)

            # Initial call of the fitness function to set the particle's best known selection and fitness.
            _, fitness = evaluate_selection(position, items)

            particle_name = f"particle_{particle_index}"
            particles[particle_name] = Particle(
                position=position,
                best_selection=position.copy(),
                best_fitness=fitness,
            )
            # Represents the particle’s individual search experience,
            # that's why we keep a separate LLM instance per particle.
            particle_llms[particle_name] = FamilyLLM(
                copilot_model=copilot_model,
                token_env_var=copilot_token_env,
                profiles=profiles,
                random_seed=SEED + particle_index,
            )
    except Exception:
        for particle_llm in particle_llms.values():
            particle_llm.close()
        raise

    # Define best particle as the one with the lowest fitness value among all particles.
    best_particle = min(particles.values(), key=lambda particle: particle.best_fitness)

    return SwarmState(
        particles=particles,
        particle_llms=particle_llms,
        global_best_position=best_particle.best_selection.copy(),
        global_best_fitness=best_particle.best_fitness,
    )


def evaluate_selection(
    selection: Sequence[int],
    items: Sequence[OfferItem],
) -> Tuple[float, float]:
    """Fitness function - evaluate the basket, where lower fitness_score means a better basket.

    ``health_score`` measures positive health signals, so it must be converted
    to a penalty before being combined with price.  Normalizing by the number
    of configured health keywords keeps the penalty bounded per item and
    prevents the raw keyword count from overwhelming the price term.
    """
    # Compute the total price of the selected items.
    total_price = sum(items[i].amount for i in selection)

    # Compute the total health penalty of the selected items.
    total_health_penalty = sum(health_penalty(items[i]) for i in selection)
    health_weight = 1.2

    # Combine price and health penalty into a single fitness_score score.
    fitness_score = total_price + health_weight * total_health_penalty
    return total_price, fitness_score



def run_optimizer(
    items: Sequence[OfferItem],
    profiles: Sequence[PersonaProfile],
    iterations: int,
    stagnation_window: int,
    min_delta: float,
    copilot_model: str,
    copilot_token_env: str,
    num_particles: int = 7,
) -> Tuple[List[IterationRecord], Dict[str, Particle], float]:
    """Run the multi-persona PSO loop over the candidate item space."""
    if not items:
        raise ValueError("No items loaded from dataset.")
    if not profiles:
        raise ValueError("At least one family profile is required.")
    if num_particles < 1:
        raise ValueError("num_particles must be at least 1.")

    # Define the number of items to be purchased
    basket_size = sum(profile.basket_size for profile in profiles)
    if basket_size < 1:
        raise ValueError("The family must require at least one basket item.")

    # Initialize the swarm with particles and their LLMs.
    swarm = _initialize_swarm(
        items=items,
        profiles=profiles,
        num_particles=num_particles,
        copilot_model=copilot_model,
        copilot_token_env=copilot_token_env,
    )
    history: List[IterationRecord] = []

    # Main loop: full catalog -> LLM proposal -> evaluate -> update memories.
    try:
        for iteration in range(1, iterations + 1):
            # Synchronous PSO: every particle reads the same previous global best.
            previous_global_best_position = swarm.global_best_position.copy()
            previous_global_best_fitness = swarm.global_best_fitness
            next_global_best_position = previous_global_best_position.copy()
            next_global_best_fitness = previous_global_best_fitness
            swarm_positions = [
                particle.position.copy()
                for particle in swarm.particles.values()
            ]

            for particle_name, particle in swarm.particles.items():
                _, current_fitness = evaluate_selection(particle.position, items)
                particle.position = swarm.particle_llms[particle_name].propose(
                    candidates=items,
                    current_position=particle.position,
                    personal_best_position=particle.best_selection,
                    personal_best_fitness=particle.best_fitness,
                    global_best_position=previous_global_best_position,
                    global_best_fitness=previous_global_best_fitness,
                    current_fitness=current_fitness,
                    inertia=PSO_INERTIA,
                    iteration=iteration,
                    exploration_rate=max(0.10, 1.0 - iteration / max(iterations, 1)),
                    particle_id=particle_name,
                    swarm_positions=swarm_positions,
                )

                # Apply fitness function to proposed selection
                family_total_price, family_fitness = evaluate_selection(
                    particle.position, items
                )

                # Update particle's best known selection if improved.
                if family_fitness < particle.best_fitness:
                    particle.best_fitness = family_fitness
                    particle.best_selection = particle.position.copy()

                # Update global best selection if improved.
                if family_fitness < next_global_best_fitness:
                    next_global_best_fitness = family_fitness
                    next_global_best_position = particle.position.copy()

                history.append(
                    IterationRecord(
                        iteration=iteration,
                        persona=particle_name,
                        selection=particle.position.copy(),
                        total_price=family_total_price,
                        penalty=family_fitness - family_total_price,
                        fitness=family_fitness,
                    )
                )

                print(
                    f"iter={iteration} particle={particle_name} "
                    f"family_price={family_total_price:.2f} "
                    f"family_fitness={family_fitness:.2f} "
                    f"swarm_best={next_global_best_fitness:.2f}"
                )

            swarm.global_best_fitness = next_global_best_fitness
            swarm.global_best_position = next_global_best_position
            improvement = previous_global_best_fitness - swarm.global_best_fitness
            if improvement < min_delta:
                swarm.stagnation_counter += 1
            else:
                swarm.stagnation_counter = 0

            if swarm.stagnation_counter >= stagnation_window:
                break
    finally:
        for particle_llm in swarm.particle_llms.values():
            particle_llm.close()

    return history, swarm.particles, swarm.global_best_fitness


def build_result(
    items: Sequence[OfferItem],
    profiles: Sequence[PersonaProfile],
    history: Sequence[IterationRecord],
    particles: Dict[str, Particle],
    global_best_fitness: float,
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

    def split_by_member(selection: Sequence[int]) -> Dict[str, List[int]]:
        blocks: Dict[str, List[int]] = {}
        start = 0
        for profile in profiles:
            blocks[profile.name] = list(selection[start:start + profile.basket_size])
            start += profile.basket_size
        return blocks

    particle_results = {}
    for particle_name, particle in particles.items():
        particle_results[particle_name] = {
            "combined_fitness": round(particle.best_fitness, 2),
            "proposal": list(particle.best_selection),
            "baskets_by_member": {
                name: materialize(selection)
                for name, selection in split_by_member(particle.best_selection).items()
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
            for name, selection in split_by_member(winning_selection).items()
        } if winning_particle else {},
    }

    # Return metadata plus best proposals grouped by persona.
    return {
        "metadata": {
            "optimizer": "family-pso-llm",
            "iterations_requested": iterations_requested,
            "iterations_completed": latest_iteration,
            "global_best_fitness": round(global_best_fitness, 2),
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
    parser.add_argument("--stagnation-window", type=int, default=DEFAULT_STAGNATION_WINDOW,
                        help="Stop if no significant improvement for this many rounds.")
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA,
                        help="Minimum global fitness improvement considered significant.")
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES,
                        help="How many cheapest offers to include in optimization space.")
    parser.add_argument(
        "--num-particles",
        type=int,
        default=DEFAULT_NUMBER_OF_PARTICLES,
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

    profiles = make_profiles()

    # Run the LLM-assisted PSO-like optimizer.
    history, particles, global_best_fitness = run_optimizer(
        items=offers,
        profiles=profiles,
        iterations=args.iterations,
        stagnation_window=args.stagnation_window,
        min_delta=args.min_delta,
        copilot_model=args.copilot_model,
        copilot_token_env=args.copilot_token_env,
        num_particles=args.num_particles,
    )

    # Build result payload and write output file.
    result = build_result(
        items=offers,
        profiles=profiles,
        history=history,
        particles=particles,
        global_best_fitness=global_best_fitness,
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

