"""Stateless per-item wording.

Each call sends exactly one ``BriefItem`` to the model and reads back wording
fields: a one-line summary, an urgency, and an action prompt. The request body
is built by ``single_item_payload``, which has no slot for a second item, a
count, or a conflict. Dropping another item is impossible inside this call
because the other item is not in the prompt. Inventing a day-level count is
outside the schema the model is allowed to return.

When ``OPENAI_API_KEY`` is unset, or the call fails schema checks, the same
function returns ``engine.deterministic_wording`` for that one item. The
pipeline still has one wording object per item, and the coverage guardrail
remains the backstop if a line never makes it into the buffer.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from engine import BriefItem, deterministic_wording

logger = logging.getLogger(__name__)

WordingSource = Literal["llm", "deterministic"]

# A model must not be able to satisfy coverage for a different item by writing
# that item's token into this item's sentence. Code adds the real token later,
# as a line prefix, after this scrub.
_FORGED_ID = re.compile(r"\[(?:evt|eml|rng)_\d{2,}\]")

_SYSTEM_PROMPT = """You write spoken wording for exactly one agenda item in an Alexa day brief.

You will receive a JSON object with a locale and a single item. That item is the only fact you know.
Return JSON with:
- item_id: copy the item's id exactly
- one_line_summary: one spoken sentence using only facts present on this item
- urgency: one of low, normal, high, judged only from this item
- action_prompt: a short imperative the listener can say next, grounded in this item

Write one_line_summary and action_prompt in the requested locale.
Do not invent attendees, times, devices, or senders.
Do not mention how many items exist, and do not mention other events, emails, or alerts.
"""

_STRICT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "item_id": {"type": "string"},
        "one_line_summary": {"type": "string"},
        "urgency": {"type": "string", "enum": ["low", "normal", "high"]},
        "action_prompt": {"type": "string"},
    },
    "required": ["item_id", "one_line_summary", "urgency", "action_prompt"],
}


class ItemWording(BaseModel):
    """Wording fields for one item. Counts and membership are not fields here."""

    item_id: str = Field(min_length=1)
    one_line_summary: str = Field(min_length=1, max_length=400)
    urgency: Literal["low", "normal", "high"]
    action_prompt: str = Field(min_length=1, max_length=200)


class MappedItem(BaseModel):
    """A wording result plus which path produced it. The source flag is code-owned."""

    wording: ItemWording
    source: WordingSource


def single_item_payload(item: BriefItem, locale: str) -> dict[str, object]:
    """The only user payload a wording call is allowed to send."""

    return {
        "locale": locale,
        "item": item.model_dump(mode="json", exclude_none=True),
    }


async def map_item(item: BriefItem, locale: str) -> MappedItem:
    """Map one item to wording. The payload contains that item and nothing else."""

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return _from_deterministic(item, reason="OPENAI_API_KEY is unset")

    payload = single_item_payload(item, locale)
    try:
        raw = await _call_llm(payload, api_key)
        wording = ItemWording.model_validate(raw)
    except Exception:
        logger.warning("Wording call failed for %s; using the deterministic sentence", item.id)
        return _from_deterministic(item, reason="model call failed")

    if wording.item_id != item.id:
        logger.warning(
            "Wording call for %s echoed id %s; using the deterministic sentence",
            item.id,
            wording.item_id,
        )
        return _from_deterministic(item, reason="echoed id did not match")

    summary = _one_line(wording.one_line_summary)
    action = _one_line(wording.action_prompt)
    if not summary:
        return _from_deterministic(item, reason="summary empty after sanitizing")
    if not action:
        action = deterministic_wording(item)[1]

    cleaned = ItemWording(
        item_id=item.id,
        one_line_summary=summary,
        urgency=wording.urgency,
        action_prompt=action,
    )
    return MappedItem(wording=cleaned, source="llm")


async def map_items(items: list[BriefItem], locale: str) -> list[MappedItem | None]:
    """Map each item in its own call, in order.

    The loop is sequential on purpose. There is no combined prompt and no
    shared chat history between iterations. A failure on one item becomes
    ``None`` so the coverage guardrail can inject a backup line; siblings
    still map.
    """

    mapped: list[MappedItem | None] = []
    for item in items:
        try:
            mapped.append(await map_item(item, locale))
        except Exception:
            logger.exception("Unexpected wording failure for %s", item.id)
            mapped.append(None)
    return mapped


def _from_deterministic(item: BriefItem, *, reason: str) -> MappedItem:
    logger.info("Deterministic wording for %s (%s)", item.id, reason)
    summary, action, urgency = deterministic_wording(item)
    wording = ItemWording(
        item_id=item.id,
        one_line_summary=summary,
        urgency=urgency,
        action_prompt=action,
    )
    return MappedItem(wording=wording, source="deterministic")


def _one_line(text: str) -> str:
    cleaned = _FORGED_ID.sub("", text.replace("\n", " ").replace("\r", " "))
    return " ".join(cleaned.split())


async def _call_llm(payload: dict[str, object], api_key: str) -> object:
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    request = {
        "model": model,
        "temperature": 0,
        "max_tokens": 300,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "item_wording",
                "strict": True,
                "schema": _STRICT_SCHEMA,
            },
        },
    }
    timeout = httpx.Timeout(8.0, connect=3.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=request,
        )
    if response.status_code != 200:
        raise RuntimeError(f"wording model returned HTTP {response.status_code}")
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return json.loads(content)
    return content
