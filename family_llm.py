#!/usr/bin/env python3
"""Copilot-backed complete-position proposal generation."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import threading
import time
from concurrent.futures import Future
from typing import Any, List, Sequence, cast

from utils import OfferItem, PersonaProfile, health_penalty
from config import PSO_COGNITIVE, PSO_EXPLORATION, PSO_SOCIAL


class FamilyLLM:
    """LLM backend that generates one complete family position per call."""

    def __init__(
        self,
        copilot_model: str,
        token_env_var: str,
        profiles: Sequence[PersonaProfile] = (),
        max_new_tokens: int = 384,
        temperature: float = 0.85,
        random_seed: int = 0,
    ) -> None:
        self.copilot_model = copilot_model
        self.token_env_var = token_env_var
        self.profiles = tuple(profiles)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._influence_rng = random.Random(random_seed)
        self._conversation_initialized = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_ready = threading.Event()
        self._client: Any = None
        self._session: Any = None

    def close(self) -> None:
        """Close the shared Copilot session and its background event loop."""
        loop = self._loop
        thread = self._loop_thread
        if loop is None or thread is None:
            return
        if loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._close_async(), loop)
            try:
                future.result(timeout=30)
            finally:
                loop.call_soon_threadsafe(lambda _unused: loop.stop(), None)
        thread.join(timeout=30)
        self._loop = None
        self._loop_thread = None
        self._loop_ready.clear()

    async def _close_async(self) -> None:
        if self._client is not None:
            await cast(Any, self._client).stop()
        self._client = None
        self._session = None

    def _start_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            def run_loop() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                self._loop_ready.set()
                loop.run_forever()
                loop.close()

            self._loop_ready.clear()
            self._loop_thread = threading.Thread(target=run_loop, daemon=True)
            self._loop_thread.start()
            self._loop_ready.wait()
        assert self._loop is not None
        return self._loop

    async def _ensure_session(self) -> Any:
        if self._session is not None:
            return self._session
        try:
            from copilot import CopilotClient
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Missing dependency 'copilot'. Install it to use Copilot API mode.") from exc
        token = os.environ.get(self.token_env_var)
        if not token:
            raise RuntimeError(f"Missing environment variable {self.token_env_var} for Copilot API authentication.")
        self._client = cast(Any, CopilotClient)({
            "github_token": token,
            "use_logged_in_user": False,
        })
        await cast(Any, self._client).start()
        session_config = {"model": self.copilot_model, "temperature": self.temperature}
        try:
            self._session = await cast(Any, self._client).create_session(
                cast(Any, session_config)
            )
        except (TypeError, ValueError) as exc:
            if "temper" not in str(exc).lower():
                raise
            self._session = await cast(Any, self._client).create_session(
                cast(Any, {"model": self.copilot_model})
            )
        return self._session

    async def _async_send_prompt(self, prompt: str) -> str:
        session = await self._ensure_session()
        try:
            response = await cast(Any, session).send_and_wait(cast(Any, {
                "prompt": prompt,
                "timeout": 180,
            }))
            return str(cast(Any, response).data.content or "")
        except asyncio.TimeoutError:
            print("[FamilyLLM] Copilot API timeout — no proposal returned for this step.")
            return ""

    def _send_prompt(self, prompt: str) -> str:
        loop = self._start_loop()
        future: Future[str] = asyncio.run_coroutine_threadsafe(self._async_send_prompt(prompt), loop)
        return str(future.result())

    def propose(
        self,
        candidates: Sequence[OfferItem],
        current_position: Sequence[int],
        personal_best_position: Sequence[int],
        personal_best_fitness: float,
        global_best_position: Sequence[int],
        global_best_fitness: float,
        current_fitness: float = float("nan"),
        inertia: float = 0.7,
    ) -> List[int]:
        """Generate a complete next position from the full product catalog."""
        total_size = sum(profile.basket_size for profile in self.profiles)
        candidate_ids = list(range(len(candidates)))
        if not candidate_ids or total_size <= 0:
            return []

        cognitive_weight = PSO_COGNITIVE * self._influence_rng.random()
        social_weight = PSO_SOCIAL * self._influence_rng.random()
        exploration_weight = PSO_EXPLORATION * self._influence_rng.random()
        raw_influences = (
            max(0.0, float(inertia)),
            max(0.0, cognitive_weight),
            max(0.0, social_weight),
            max(0.0, exploration_weight),
        )
        influence_total = sum(raw_influences)
        if influence_total > 0.0:
            current_share, previous_best_share, collective_best_share, exploration_share = (
                strength / influence_total for strength in raw_influences
            )
        else:
            # A neutral fallback keeps the prompt meaningful if all strengths
            # are disabled by configuration.
            current_share = previous_best_share = collective_best_share = exploration_share = 0.25

        prompt_lines: List[str] = []
        # Set initial context to particle instance of LLM
        if not self._conversation_initialized:
            prompt_lines.extend([
                "You are optimizing a complete family grocery basket over a sequence of steps.",
                f"Remember this shared context for all later messages: the basket always contains exactly {total_size} product IDs.",
                "Duplicate indexes are allowed because multiple family members may want the same product.",
                "Family member position blocks:",
                *self._family_layout_lines(),
                "Family preferences (consider as many as possible):",
                *[
                    f"- {profile.name}: {profile.basket_size} items, at least {profile.min_preferred_items} preferred items, keywords={profile.keywords}"
                    for profile in self.profiles
                ],
                "The deterministic objective is fitness = total basket price + 1.2 * total health penalty; lower fitness is better.",
                "Health penalty is lower for products matching more healthy-keyword signals.",
                "Use product semantics to construct a promising complete basket; do not calculate or report fitness yourself.",
            ])

        # Prepare the prompt for the next basket proposal
        prompt_lines.extend([
            "Construct the next basket.",
            "Target composition for the next basket:",
            f"- retain approximately {current_share:.0%} of useful current items",
            f"- obtain approximately {previous_best_share:.0%} of the basket from useful information in the previous best solution",
            f"- obtain approximately {collective_best_share:.0%} of the basket from useful information in the best solution found across candidates",
            f"- approximately {exploration_share:.0%} may be entirely new exploration",
            "These are influence targets, not mandatory quotas.",
            "You are allowed to:",
            "- preserve products from the current position,",
            "- adopt products from the previous best solution,",
            "- adopt products from the best solution found across candidates,",
            "- introduce products not present in either best solution,",
            "- use price, product type, health characteristics, and interactions among products to construct a promising basket.",
            "Do not simply copy a best solution mechanically; use the targets as flexible guidance.",
            f"Current basket (score {current_fitness:.4f}): {list(current_position)}",
            f"Previous best solution (score {personal_best_fitness}): {list(personal_best_position)}",
            f"Best solution found across candidates (score {global_best_fitness}): {list(global_best_position)}",
            f"Variation token for this call: {time.time_ns() % 1000000}.",
            "Relevant product catalog:",
        ])
        prompt_lines.extend(
            f"- id={item_id} | price={candidates[item_id].amount:.2f} {candidates[item_id].currency} | "
            f"category={candidates[item_id].category} | health_penalty={health_penalty(candidates[item_id]):.3f} | "
            f"product={candidates[item_id].product}"
            for item_id in candidate_ids
        )
        prompt_lines.extend([
            "Return exactly one JSON object and nothing else: {\"item_ids\":[...]}",
            f"The item_ids array must contain exactly {total_size} IDs.",
            f"Every ID must be an integer from 0 through {len(candidates) - 1}.",
        ])

        # Send request to LLM
        parsed = self._extract_json_ids(self._send_prompt("\n".join(prompt_lines)))

        # Parse the response and validate the proposed basket
        if len(parsed) != total_size or any(item_id not in candidate_ids for item_id in parsed):
            print(
                f"[FamilyLLM] Ignoring incomplete/invalid position "
                f"({len(parsed)}/{total_size} IDs); keeping current position."
            )
            return list(current_position)

        self._conversation_initialized = True
        print(f"[FamilyLLM] proposal: {parsed}")

        return parsed

    def _family_layout_lines(self) -> List[str]:
        """Describe the fixed contiguous index blocks in a flat family position."""
        lines: List[str] = []
        start = 0
        for profile in self.profiles:
            end = start + profile.basket_size - 1
            lines.append(
                f"- positions {start}..{end}: {profile.name} ({profile.role}), "
                f"{profile.basket_size} items"
            )
            start = end + 1
        return lines

    @staticmethod
    def _extract_json_ids(text: str) -> List[int]:
        """Extract an item_ids array from a model response."""
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                data, _ = decoder.raw_decode(text[match.start():])
            except (json.JSONDecodeError, TypeError):
                continue
            raw_ids = data.get("item_ids") if isinstance(data, dict) else None
            if isinstance(raw_ids, list):
                result: List[int] = []
                for value in raw_ids:
                    try:
                        result.append(int(value))
                    except (TypeError, ValueError):
                        continue
                return result
        return []


