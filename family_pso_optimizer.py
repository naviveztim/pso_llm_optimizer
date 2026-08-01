#!/usr/bin/env python3
"""Family shopping optimizer using PSO and optional local Qwen proposals."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, cast


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
    return re.sub(r"\s+", " ", text.lower()).strip()


def load_offers(dataset_path: Path, country_code: str, max_candidates: int) -> List[OfferItem]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    categories = payload.get("categories") or {}
    offers: List[OfferItem] = []
    current_id = 0

    for category, entries in categories.items():
        for entry in entries:
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

    offers.sort(key=lambda x: x.amount)
    trimmed = offers[:max_candidates]
    for idx, item in enumerate(trimmed):
        item.item_id = idx
    return trimmed


def keyword_score(item: OfferItem, keywords: Sequence[str]) -> float:
    text = normalize(f"{item.product} {item.category}")
    score = 0.0
    for kw in keywords:
        if kw in text:
            score += 1.0
    return score


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def top_indices(values: Sequence[float], k: int) -> List[int]:
    indexed = list(enumerate(values))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in indexed[:k]]


class FamilyLLM:
    """Single model backend used by all family personas."""

    def __init__(
        self,
        model_dir: Path,
        seed: int,
        max_new_tokens: int = 280,
        device_preference: str = "auto",
        use_compile: bool = False,
    ) -> None:
        self.model_dir = model_dir
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.device_preference = device_preference
        self.use_compile = use_compile
        self._tokenizer: Any = None
        self._model: Any = None
        self._device = "cpu"
        self._autocast_dtype = None

    @staticmethod
    def _resolve_device(torch: Any, device_preference: str) -> str:
        if device_preference == "cpu":
            return "cpu"
        if device_preference == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("--device cuda requested, but CUDA is not available.")
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - runtime environment dependent
            raise RuntimeError(
                "Missing dependencies. Install transformers and torch to use LLM mode."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self._device = self._resolve_device(torch, self.device_preference)

        if self._device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            self._autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        model_kwargs: Dict[str, Any] = {"low_cpu_mem_usage": True}
        if self._device == "cuda":
            model_kwargs["torch_dtype"] = self._autocast_dtype
            if self.device_preference == "auto":
                model_kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(self.model_dir, **model_kwargs)
        if self._device == "cuda" and self.device_preference != "auto":
            self._model.to("cuda")

        if self.use_compile and hasattr(torch, "compile"):
            try:
                self._model = torch.compile(self._model, mode="reduce-overhead", fullgraph=False)
            except Exception:
                pass

        self._model.eval()
        torch.manual_seed(self.seed)
        if self._device == "cuda":
            torch.cuda.manual_seed_all(self.seed)

    def propose(
        self,
        profile: PersonaProfile,
        candidates: Sequence[OfferItem],
        shortlist_ids: Sequence[int],
    ) -> List[int]:
        self._ensure_loaded()
        prompt_lines = [
            f"You are {profile.role} in a family shopping discussion.",
            "Pick affordable monthly shopping items from shortlist.",
            f"You must choose exactly {profile.basket_size} distinct item IDs.",
            f"At least {profile.min_preferred_items} chosen items should match preferences.",
            f"Preference keywords: {', '.join(profile.keywords)}",
            "Return STRICT JSON only: {\"item_ids\":[...]}.",
            "Shortlist:",
        ]

        for item_id in shortlist_ids:
            item = candidates[item_id]
            prompt_lines.append(
                f"- id={item.item_id} | product={item.product} | category={item.category} | price={item.amount:.2f} {item.currency}"
            )

        prompt = "\n".join(prompt_lines)

        # Deferred imports to keep dry-run mode dependency-light.
        import torch

        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Model was not initialized correctly.")

        tokenizer = cast(Any, self._tokenizer)
        model = cast(Any, self._model)

        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self._autocast_dtype)
            if self._device == "cuda" and self._autocast_dtype is not None
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            generated = model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        parsed = self._extract_json_ids(text)
        valid = [item_id for item_id in parsed if item_id in shortlist_ids]
        return list(dict.fromkeys(valid))[: profile.basket_size]

    @staticmethod
    def _extract_json_ids(text: str) -> List[int]:
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

        # Fallback: parse any integer IDs if strict JSON wasn't produced.
        return [int(x) for x in re.findall(r"\b\d+\b", text)]


def heuristic_propose(
    profile: PersonaProfile,
    items: Sequence[OfferItem],
    shortlist_ids: Sequence[int],
    rng: random.Random,
) -> List[int]:
    scored: List[Tuple[int, float]] = []
    for item_id in shortlist_ids:
        item = items[item_id]
        pref = keyword_score(item, profile.keywords)
        affordability = 1.0 / max(item.amount, 0.01)
        noise = rng.random() * 0.2
        scored.append((item_id, pref * 2.0 + affordability + noise))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [item_id for item_id, _ in scored[: profile.basket_size]]


def evaluate_selection(
    selection: Sequence[int],
    items: Sequence[OfferItem],
    profile: PersonaProfile,
) -> Tuple[float, float, float]:
    total_price = sum(items[i].amount for i in selection)
    preferred_hits = sum(1 for i in selection if keyword_score(items[i], profile.keywords) > 0)
    penalty = 0.0

    missing_preferred = max(0, profile.min_preferred_items - preferred_hits)
    if missing_preferred:
        penalty += missing_preferred * 7.5

    if len(selection) < profile.basket_size:
        penalty += (profile.basket_size - len(selection)) * 5.0

    fitness = total_price + penalty
    return total_price, penalty, fitness


def selection_to_vector(selection: Sequence[int], dimension: int) -> List[float]:
    vector = [0.0] * dimension
    for idx in selection:
        if 0 <= idx < dimension:
            vector[idx] = 1.0
    return vector


def update_particle(
    particle: Particle,
    global_best_position: Sequence[float],
    inertia: float,
    cognitive: float,
    social: float,
    rng: random.Random,
) -> None:
    for i in range(len(particle.position)):
        r1 = rng.random()
        r2 = rng.random()
        cognitive_term = cognitive * r1 * (particle.best_position[i] - particle.position[i])
        social_term = social * r2 * (global_best_position[i] - particle.position[i])
        particle.velocity[i] = inertia * particle.velocity[i] + cognitive_term + social_term
        particle.position[i] += particle.velocity[i]


def shortlist_from_position(
    particle: Particle,
    items: Sequence[OfferItem],
    shortlist_size: int,
) -> List[int]:
    scores = []
    for idx, item in enumerate(items):
        pref = keyword_score(item, particle.profile.keywords)
        score = sigmoid(particle.position[idx]) + 0.4 * pref - 0.03 * item.amount
        scores.append(score)
    return top_indices(scores, shortlist_size)


def make_profiles() -> List[PersonaProfile]:
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
    model_dir: Path,
    seed: int,
    llm_refresh_every: int,
    llm_device: str,
    llm_compile: bool,
) -> Tuple[List[IterationRecord], Dict[str, Particle], float]:
    if not items:
        raise ValueError("No items loaded from dataset.")

    rng = random.Random(seed)
    dimension = len(items)
    llm = None if dry_run else FamilyLLM(
        model_dir=model_dir,
        seed=seed,
        device_preference=llm_device,
        use_compile=llm_compile,
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
    stagnation_counter = 0

    for iteration in range(1, iterations + 1):
        previous_best = best_global_fitness

        for persona_name, particle in particles.items():
            shortlist_ids = shortlist_from_position(particle, items, shortlist_size=shortlist_size)

            use_llm = (iteration == 1) or (iteration % llm_refresh_every == 0)
            if llm is None or not use_llm:
                proposal = heuristic_propose(particle.profile, items, shortlist_ids, rng)
            else:
                proposal = llm.propose(particle.profile, items, shortlist_ids)
                if not proposal:
                    proposal = heuristic_propose(particle.profile, items, shortlist_ids, rng)

            proposal_vector = selection_to_vector(proposal, dimension)
            particle.position = [0.75 * p + 0.25 * q for p, q in zip(particle.position, proposal_vector)]

            update_particle(
                particle=particle,
                global_best_position=best_global_position,
                inertia=0.62,
                cognitive=1.42,
                social=1.35,
                rng=rng,
            )

            selection_scores = [sigmoid(x) for x in particle.position]
            selection = top_indices(selection_scores, particle.profile.basket_size)
            total_price, penalty, fitness = evaluate_selection(selection, items, particle.profile)

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

            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()
                particle.best_selection = selection

            if fitness < best_global_fitness:
                best_global_fitness = fitness
                best_global_position = particle.position.copy()

        improvement = abs(previous_best - best_global_fitness)
        if improvement < min_delta:
            stagnation_counter += 1
        else:
            stagnation_counter = 0

        if stagnation_counter >= stagnation_window:
            break

    return history, particles, best_global_fitness


def build_result(
    items: Sequence[OfferItem],
    history: Sequence[IterationRecord],
    particles: Dict[str, Particle],
    best_global_fitness: float,
    iterations_requested: int,
    dry_run: bool,
) -> Dict[str, Any]:
    latest_iteration = max((h.iteration for h in history), default=0)

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
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run family PSO shopping optimization with optional Qwen LLM proposals.")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=repo_root / "data" / "kaufland_prices_by_category_various_countries.json",
        help="Path to Kaufland offers JSON file.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=repo_root / "models" / " Qwen2.5-1.5B-Instruct",
        help="Path to local Qwen model directory.",
    )
    parser.add_argument("--country", type=str, default="DE", help="Country code to filter offers (e.g. DE).")
    parser.add_argument("--iterations", type=int, default=30, help="Maximum PSO iterations.")
    parser.add_argument("--stagnation-window", type=int, default=5, help="Stop if no significant improvement for this many rounds.")
    parser.add_argument("--min-delta", type=float, default=0.15, help="Minimum global fitness improvement considered significant.")
    parser.add_argument("--max-candidates", type=int, default=220, help="How many cheapest offers to include in optimization space.")
    parser.add_argument("--shortlist-size", type=int, default=40, help="Per-persona shortlist size sent to proposer.")
    parser.add_argument("--llm-refresh-every", type=int, default=3, help="Run LLM proposal every N iterations; heuristic used between.")
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="LLM execution device. Use auto to prefer GPU when available.",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Enable torch.compile for the local model when supported.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--dry-run", action="store_true", help="Skip model loading and use heuristic proposals only.")
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "data" / "family_pso_plan.json",
        help="Output file for best plan.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    offers = load_offers(args.data_file, country_code=args.country.strip().upper(), max_candidates=args.max_candidates)
    profiles = make_profiles()

    history, particles, best_global_fitness = run_optimizer(
        items=offers,
        profiles=profiles,
        iterations=args.iterations,
        stagnation_window=args.stagnation_window,
        min_delta=args.min_delta,
        shortlist_size=args.shortlist_size,
        dry_run=args.dry_run,
        model_dir=args.model_dir,
        seed=args.seed,
        llm_refresh_every=max(1, args.llm_refresh_every),
        llm_device=args.device,
        llm_compile=args.compile_model,
    )

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

    print(f"Wrote plan to {args.output}")
    print(f"Iterations completed: {result['metadata']['iterations_completed']}")
    print(f"Global best fitness: {result['metadata']['global_best_fitness']}")


if __name__ == "__main__":
    main()

