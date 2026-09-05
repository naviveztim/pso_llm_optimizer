#!/usr/bin/env python3
"""Copilot-backed family proposal generation."""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from concurrent.futures import Future
from typing import Any, List, Sequence, cast

from utils import OfferItem, PersonaProfile, normalize


class FamilyLLM:
    """LLM backend that updates one complete flat family position per call."""

    def __init__(
        self,
        copilot_model: str,
        token_env_var: str,
        seed: int,
        profiles: Sequence[PersonaProfile] = (),
        cognitive_coeff: float = 1.42,
        social_coeff: float = 1.42,
        max_new_tokens: int = 384,
        max_prompt_items: int = 24,
    ) -> None:
        self.copilot_model = copilot_model
        self.token_env_var = token_env_var
        self.seed = seed
        self.profiles = tuple(profiles)
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.max_new_tokens = max_new_tokens
        self.max_prompt_items = max_prompt_items
        self._conversation_initialized = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_ready = threading.Event()
        self._client: Any = None
        self._session: Any = None

    def _prepare_shortlist_ids(self, candidates: Sequence[OfferItem], shortlist_ids: Sequence[int]) -> List[int]:
        unique_ids: List[int] = []
        seen_keys = set()
        for item_id in shortlist_ids:
            if not 0 <= item_id < len(candidates):
                continue
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
        shortlist_ids: Sequence[int],
        current_position: Sequence[int],
        personal_best_position: Sequence[int],
        personal_best_fitness: float,
        global_best_position: Sequence[int],
        global_best_fitness: float,
    ) -> List[int]:
        """Generate a complete next family position from the three PSO positions."""
        total_size = sum(profile.basket_size for profile in self.profiles)
        candidate_ids = self._prepare_shortlist_ids(
            candidates,
            list(dict.fromkeys([
                *current_position,
                *personal_best_position,
                *global_best_position,
                *shortlist_ids,
            ])),
        )
        if not candidate_ids or total_size <= 0:
            return []

        prompt_lines: List[str] = []
        if not self._conversation_initialized:
            prompt_lines.extend([
                "You are optimizing a complete family grocery basket over a sequence of steps.",
                f"Remember this shared context for all later messages: the basket always contains exactly {total_size} items.",
                "Duplicate indexes are allowed because multiple family members may want the same item.",
                "Family member position blocks:",
                *self._family_layout_lines(),
                "Family preferences (consider as many as possible):",
                *[
                    f"- {profile.name} ({profile.role}): {profile.basket_size} items, at least {profile.min_preferred_items} preferred items, keywords={profile.keywords}"
                    for profile in self.profiles
                ],
                "Prefer affordable, healthy items while satisfying the family preferences.",
                "Fitness definition: lower is better. fitness = total basket price - 1.2 * total health value.",
                "Fitness combines the basket price with a health-value bonus; it is not a price-only value.",
                "Use only indexes listed in each step's Available items section.",
            ])

        prompt_lines.extend([
            "For this optimization step, return a complete next family position as JSON.",
            f"Current position: {list(current_position)}",
            f"Personal-best position (fitness {personal_best_fitness}): {list(personal_best_position)}",
            f"Global-best position (fitness {global_best_fitness}): {list(global_best_position)}",
        ])

        # Available items
        prompt_lines.append("Available items (use only these indexes):")
        prompt_lines.extend(
            f"- id={item_id} | price={candidates[item_id].amount:.2f} {candidates[item_id].currency} | category={candidates[item_id].category} | product={candidates[item_id].product}"
            for item_id in candidate_ids
        )

        # Return format
        prompt_lines.append("Return exactly one JSON object and nothing else: {\"item_ids\":[...]}")
        prompt_lines.append(f"The item_ids array must contain exactly {total_size} indexes.")

        # Run the prompt
        parsed = self._extract_json_ids(self._send_prompt("\n".join(prompt_lines)))
        self._conversation_initialized = True

        # Parse the answers
        valid = [item_id for item_id in parsed if item_id in candidate_ids]
        print(valid[:total_size])
        return valid[:total_size]

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
        for blob in reversed(re.findall(r"\{[^{}]*}", text, flags=re.DOTALL)):
            try:
                data = json.loads(blob)
            except json.JSONDecodeError:
                continue
            raw_ids = data.get("item_ids")
            if isinstance(raw_ids, list):
                result: List[int] = []
                for value in raw_ids:
                    try:
                        result.append(int(value))
                    except (TypeError, ValueError):
                        continue
                return result
        return []

