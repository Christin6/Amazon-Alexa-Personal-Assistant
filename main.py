"""Alexa day-brief server.

Voice clients and MCP clients share one pipeline:

1. ``engine.build_timeline`` ingests the day, sorts it, stamps ``evt_`` / ``eml_`` /
   ``rng_`` ids, and computes counts, overlaps, and back-to-back gaps.
2. ``session.SessionManager`` keeps that timeline for 20 minutes and hands out
   three items at a time.
3. ``mapper.map_item`` phrases each active item in its own stateless call.
   The model returns wording fields. It does not choose membership or counts.
4. This module writes the page buffer: a code-rendered header (exact counts and
   constraint sentences) plus one ``[id]`` line per active item.
5. ``guardrails.ensure_coverage`` scans line prefixes. Any active id that is
   absent is injected from the deterministic backup sentence before the page
   is returned.

``spoken_text`` is what Alexa should read. ``total_count``, ``ordered_ids``,
and ``rendered_text`` are the code-owned record of the same page.

Run locally:

    uvicorn main:app --host 127.0.0.1 --port 8000

Environment:

    SESSION_TTL_SECONDS     default 1200
    BRIEF_BATCH_SIZE        default 3
    TIGHT_GAP_MINUTES       default 10
    OPENAI_API_KEY          optional; unset uses deterministic per-item wording
    OPENAI_BASE_URL         default https://api.openai.com/v1
    OPENAI_MODEL            default gpt-4o-mini
    MCP_ALLOWED_HOSTS       comma-separated extra Host values for Streamable HTTP
    MCP_ALLOWED_ORIGINS     comma-separated extra Origin values
    CORS_ORIGINS            comma-separated browser origins, or * (default)
    HOST / PORT             used by ``python main.py``
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field

from actions import ActionKind, ActionResult, perform_action
from engine import (
    BriefItem,
    DayConstraints,
    Overlap,
    SourceKind,
    TightGap,
    build_timeline,
    conflict_ids,
    describe_constraints,
    deterministic_wording,
)
from guardrails import backup_line, ensure_coverage
from mapper import MappedItem, map_items
from session import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_TTL_SECONDS,
    BriefSessionError,
    PageSlice,
    SessionManager,
)

logger = logging.getLogger(__name__)

_HTTP_STATUS: dict[str, int] = {
    "session_not_found": 404,
    "session_expired": 410,
    "invalid_timezone": 422,
    "item_not_found": 404,
    "action_not_supported": 422,
}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc


TTL_SECONDS = _env_int("SESSION_TTL_SECONDS", DEFAULT_TTL_SECONDS)
BATCH_SIZE = _env_int("BRIEF_BATCH_SIZE", DEFAULT_BATCH_SIZE)
TIGHT_GAP_MINUTES = _env_int("TIGHT_GAP_MINUTES", 10)
if TTL_SECONDS <= 0 or BATCH_SIZE <= 0 or TIGHT_GAP_MINUTES < 0:
    raise RuntimeError("SESSION_TTL_SECONDS and BRIEF_BATCH_SIZE must be positive")

sessions = SessionManager(ttl_seconds=TTL_SECONDS, batch_size=BATCH_SIZE)

mcp = MCPServer(
    "Alexa Day Brief",
    title="Alexa Day Brief",
    instructions=(
        "Call get_day_brief to start a spoken day brief. Read spoken_text aloud. "
        "total_count, remaining_count, ordered_ids, and the item ids are authoritative. "
        "When the user says next, call next_items with the same session_id. "
        "To act on an item, call find_item to turn a reference like 'email 2' into an "
        "item_id, then read_item, reply_to_item, open_item, or show_item. "
        "Use only the tool result for events, emails, alerts, and counts."
    ),
    version="0.1.0",
)


class GetDayBriefRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    timezone: str = "America/Los_Angeles"
    locale: str = Field(default="en-US", min_length=2, max_length=35)
    day: date | None = None


class NextItemsRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    locale: str = Field(default="", max_length=35)


class FindItemRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    kind: SourceKind
    ordinal: int = Field(ge=1)


class ActionRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    item_id: str = Field(min_length=1, max_length=64)


class ResolvedItem(BaseModel):
    """A spoken reference such as "email 2" resolved to a code-owned item id."""

    ok: bool
    error_code: str | None = None
    item_id: str | None = None
    kind: SourceKind | None = None
    ordinal: int | None = None
    one_line_summary: str = ""
    spoken_text: str = ""


class SpokenItem(BaseModel):
    id: str
    kind: SourceKind
    timestamp: datetime
    urgency: Literal["low", "normal", "high"]
    one_line_summary: str
    action_prompt: str
    wording_source: Literal["llm", "deterministic", "coverage_backup"]
    in_conflict: bool


class VoiceBrief(BaseModel):
    """One spoken page. Numeric fields are copied from the engine and the session cursor."""

    ok: bool
    error_code: str | None = None
    spoken_text: str
    rendered_text: str = ""
    session_id: str | None = None
    day: str | None = None
    timezone: str | None = None
    expires_at: datetime | None = None
    expires_in_seconds: int | None = None
    items: list[SpokenItem] = Field(default_factory=list)
    active_ids: list[str] = Field(default_factory=list)
    ordered_ids: list[str] = Field(default_factory=list)
    injected_ids: list[str] = Field(default_factory=list)
    coverage_complete: bool = False
    batch_number: int = 0
    batch_size: int = BATCH_SIZE
    total_count: int = 0
    counts_by_kind: dict[str, int] = Field(default_factory=dict)
    remaining_count: int = 0
    has_more: bool = False
    overlaps: list[Overlap] = Field(default_factory=list)
    tight_gaps: list[TightGap] = Field(default_factory=list)


def _error_brief(code: str, spoken_text: str) -> VoiceBrief:
    return VoiceBrief(ok=False, error_code=code, spoken_text=spoken_text, coverage_complete=False)


def _status_for(brief: VoiceBrief) -> int:
    if brief.ok:
        return 200
    return _HTTP_STATUS.get(brief.error_code or "", 400)


async def open_day_brief(
    *,
    user_id: str,
    timezone_name: str,
    locale: str,
    day: date | None,
) -> VoiceBrief:
    """Build a timeline, open a session, and render the first page."""

    try:
        tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        return _error_brief("invalid_timezone", "I couldn't use that timezone.")

    local_day = day or datetime.now(tz).date()
    timeline = build_timeline(
        user_id=user_id,
        tz=tz,
        day=local_day,
        tight_gap_minutes=TIGHT_GAP_MINUTES,
    )
    page = sessions.create(timeline, locale=locale)
    return await _render(page)


async def advance_brief(*, session_id: str, locale: str) -> VoiceBrief:
    """Render the next batch of three, or say the brief is finished."""

    try:
        page = sessions.view(session_id)
    except BriefSessionError as exc:
        return _error_brief(exc.code, exc.spoken_text)
    if locale.strip():
        page = _with_locale(page, locale.strip())
    return await _render(page)


async def _render(page: PageSlice) -> VoiceBrief:
    """Render one page. Counts and IDs are written by code; wording is per item.

    The cursor advances only after coverage succeeds. A terminal page (the user
    said next after the last item) does not call the model.
    """

    constraints = page.constraints
    if page.is_terminal:
        spoken = "That's everything on your brief."
        still_live = sessions.complete_page(page.session_id, page.cursor_after)
        if not still_live:
            spoken = "That's everything on your brief. The session has expired."
        return _brief_shell(page, spoken_text=spoken, rendered_text="", items=[], injected_ids=[])

    mapped = await map_items(list(page.active_items), page.locale)
    buffer = _compose_buffer(page, mapped)
    # Active ids are the page membership from the session cursor, not a model guess.
    coverage = ensure_coverage(buffer, list(page.active_items))
    if coverage.injected_ids:
        logger.warning("Coverage backup injected ids: %s", ", ".join(coverage.injected_ids))
    spoken_items = _spoken_items(page, mapped, coverage.injected_ids)
    spoken = _compose_spoken(page, spoken_items)
    still_live = sessions.complete_page(page.session_id, page.cursor_after)
    if not still_live:
        spoken = spoken + " This brief just expired, so ask me to start over if you want the rest."
        page = _closed(page)

    return _brief_shell(
        page,
        spoken_text=spoken,
        rendered_text=coverage.rendered_text,
        items=spoken_items,
        injected_ids=coverage.injected_ids,
    )


def _compose_buffer(page: PageSlice, mapped: list[MappedItem | None]) -> str:
    """Header from engine numbers, then one prefixed line per successfully mapped item.

    An item that failed to map is left out on purpose. ``ensure_coverage`` sees
    the missing prefix and injects the backup sentence, in cursor order.
    """

    constraints = page.constraints
    lines = [
        f"total_count={constraints.total_count}",
        (
            "counts "
            f"calendar={constraints.counts_by_kind.get('calendar', 0)} "
            f"email={constraints.counts_by_kind.get('email', 0)} "
            f"ring={constraints.counts_by_kind.get('ring', 0)}"
        ),
        f"ordered_ids={','.join(constraints.ordered_ids)}",
        (
            f"batch={page.batch_number} active={len(page.active_items)} "
            f"remaining={page.remaining_after}"
        ),
    ]
    for overlap in constraints.overlaps:
        lines.append(
            f"overlap {overlap.first_id} {overlap.second_id} minutes={overlap.overlap_minutes}"
        )
    for gap in constraints.tight_gaps:
        lines.append(f"tight_gap {gap.first_id} {gap.second_id} minutes={gap.gap_minutes}")

    for item, mapped_item in zip(page.active_items, mapped):
        if mapped_item is None or mapped_item.wording.item_id != item.id:
            continue
        summary = mapped_item.wording.one_line_summary
        action = mapped_item.wording.action_prompt
        lines.append(f"[{item.id}] {summary} {action}")
    return "\n".join(lines)


def _spoken_items(
    page: PageSlice,
    mapped: list[MappedItem | None],
    injected_ids: list[str],
) -> list[SpokenItem]:
    conflicts = conflict_ids(page.constraints)
    injected = set(injected_ids)
    spoken: list[SpokenItem] = []
    for item, mapped_item in zip(page.active_items, mapped):
        if item.id in injected or mapped_item is None:
            summary, action, urgency = _backup_fields(item)
            source: Literal["llm", "deterministic", "coverage_backup"] = "coverage_backup"
        else:
            summary = mapped_item.wording.one_line_summary
            action = mapped_item.wording.action_prompt
            urgency = mapped_item.wording.urgency
            source = mapped_item.source
        spoken.append(
            SpokenItem(
                id=item.id,
                kind=item.kind,
                timestamp=item.timestamp,
                urgency=urgency,
                one_line_summary=summary,
                action_prompt=action,
                wording_source=source,
                in_conflict=item.id in conflicts,
            )
        )
    return spoken


def _backup_fields(item: BriefItem) -> tuple[str, str, Literal["low", "normal", "high"]]:
    """Speech fields for a coverage backup. The line prefix is checked, then discarded."""

    summary, action, urgency = deterministic_wording(item)
    if urgency == "high":
        narrowed: Literal["low", "normal", "high"] = "high"
    elif urgency == "low":
        narrowed = "low"
    else:
        narrowed = "normal"
    line = backup_line(item)
    if not line.startswith(f"[{item.id}] "):
        raise RuntimeError(f"backup line for {item.id} lost its id token")
    return summary, action, narrowed


def _compose_spoken(page: PageSlice, items: list[SpokenItem]) -> str:
    """Assemble speech from code-owned counts plus each item's wording fields."""

    total = page.constraints.total_count
    parts: list[str] = []
    if page.batch_number == 1:
        parts.append(_count_sentence(page.constraints))
        parts.extend(describe_constraints(page.constraints, page.timeline))
        if page.has_more:
            parts.append(f"Here are the first {len(items)}.")
    elif items:
        start = page.cursor_after - len(items) + 1
        end = page.cursor_after
        parts.append(f"Items {start} through {end} of {total}.")

    for item in items:
        parts.append(item.one_line_summary)
        parts.append(item.action_prompt)

    if page.has_more:
        noun = "item" if page.remaining_after == 1 else "items"
        parts.append(f"{page.remaining_after} more {noun}. Say next when you want them.")
    else:
        parts.append("That's the full brief.")
    return " ".join(part.strip() for part in parts if part.strip())


def _count_sentence(constraints: DayConstraints) -> str:
    if constraints.total_count == 0:
        return "You have nothing on today's brief."
    calendar = constraints.counts_by_kind.get("calendar", 0)
    email = constraints.counts_by_kind.get("email", 0)
    ring = constraints.counts_by_kind.get("ring", 0)
    return (
        f"You have {constraints.total_count} {_noun(constraints.total_count, 'item')} today: "
        f"{calendar} calendar {_noun(calendar, 'event')}, "
        f"{email} priority {_noun(email, 'email')}, "
        f"and {ring} Ring {_noun(ring, 'alert')}."
    )


def _noun(count: int, singular: str) -> str:
    if count == 1:
        return singular
    if singular == "email":
        return "emails"
    return singular + "s"


def _brief_shell(
    page: PageSlice,
    *,
    spoken_text: str,
    rendered_text: str,
    items: list[SpokenItem],
    injected_ids: list[str],
) -> VoiceBrief:
    constraints = page.constraints
    return VoiceBrief(
        ok=True,
        spoken_text=spoken_text,
        rendered_text=rendered_text,
        session_id=page.session_id,
        day=page.day_iso,
        timezone=page.timezone,
        expires_at=page.expires_at,
        expires_in_seconds=page.expires_in_seconds,
        items=items,
        active_ids=[item.id for item in items],
        ordered_ids=list(constraints.ordered_ids),
        injected_ids=injected_ids,
        coverage_complete=True,
        batch_number=page.batch_number,
        batch_size=page.batch_size,
        total_count=constraints.total_count,
        counts_by_kind=dict(constraints.counts_by_kind),
        remaining_count=page.remaining_after,
        has_more=page.has_more,
        overlaps=list(constraints.overlaps),
        tight_gaps=list(constraints.tight_gaps),
    )


def _with_locale(page: PageSlice, locale: str) -> PageSlice:
    return PageSlice(
        session_id=page.session_id,
        user_id=page.user_id,
        day_iso=page.day_iso,
        timezone=page.timezone,
        locale=locale,
        active_items=page.active_items,
        timeline=page.timeline,
        constraints=page.constraints,
        batch_number=page.batch_number,
        batch_size=page.batch_size,
        cursor_after=page.cursor_after,
        remaining_after=page.remaining_after,
        has_more=page.has_more,
        is_terminal=page.is_terminal,
        expires_at=page.expires_at,
        expires_in_seconds=page.expires_in_seconds,
    )


def _closed(page: PageSlice) -> PageSlice:
    """A page whose session died during rendering cannot offer another batch."""

    return PageSlice(
        session_id=page.session_id,
        user_id=page.user_id,
        day_iso=page.day_iso,
        timezone=page.timezone,
        locale=page.locale,
        active_items=page.active_items,
        timeline=page.timeline,
        constraints=page.constraints,
        batch_number=page.batch_number,
        batch_size=page.batch_size,
        cursor_after=page.cursor_after,
        remaining_after=0,
        has_more=False,
        is_terminal=page.is_terminal,
        expires_at=page.expires_at,
        expires_in_seconds=0,
    )


def _json(brief: VoiceBrief) -> JSONResponse:
    return JSONResponse(status_code=_status_for(brief), content=brief.model_dump(mode="json"))


def _json_action(result: ActionResult | ResolvedItem) -> JSONResponse:
    return JSONResponse(
        status_code=200 if result.ok else _HTTP_STATUS.get(result.error_code or "", 400),
        content=result.model_dump(mode="json"),
    )


def resolve_item(session_id: str, kind: SourceKind, ordinal: int) -> ResolvedItem:
    """Resolve a spoken reference such as "email 2" to its code-owned item id.

    The session supplies the live timeline; ``engine.find_by_ordinal`` picks the
    item. The caller (a model) never counts or chooses membership.
    """

    try:
        item = sessions.lookup_ordinal(session_id, kind, ordinal)
    except BriefSessionError as exc:
        return ResolvedItem(ok=False, error_code=exc.code, spoken_text=exc.spoken_text)
    summary, _action, _urgency = deterministic_wording(item)
    return ResolvedItem(
        ok=True,
        item_id=item.id,
        kind=item.kind,
        ordinal=ordinal,
        one_line_summary=summary,
        spoken_text=f"{kind.value.capitalize()} {ordinal} is {summary}",
    )


def act_on_item(session_id: str, item_id: str, action: ActionKind) -> ActionResult:
    """Execute one triage action against a resolved item in a live brief.

    Session lookup and action semantics are code-owned; ``perform_action`` is a
    pure function of the item and the action.
    """

    try:
        item = sessions.lookup(session_id, item_id)
    except BriefSessionError as exc:
        return ActionResult(ok=False, error_code=exc.code, spoken_text=exc.spoken_text)
    return perform_action(item, action)


@mcp.tool(
    description=(
        "Start today's brief. Code loads, sorts, numbers, and counts every calendar "
        "event, priority email, and Ring alert, then phrases the first three items "
        "one at a time. Read spoken_text. Keep session_id and call next_items when "
        "the user says next."
    )
)
async def get_day_brief(
    user_id: str,
    timezone: str = "America/Los_Angeles",
    locale: str = "en-US",
) -> VoiceBrief:
    return await open_day_brief(user_id=user_id, timezone_name=timezone, locale=locale, day=None)


@mcp.tool(
    description=(
        "Speak the next three items for an existing brief. Pass the session_id "
        "returned by get_day_brief. Read spoken_text. Stop when has_more is false."
    )
)
async def next_items(session_id: str, locale: str = "") -> VoiceBrief:
    return await advance_brief(session_id=session_id, locale=locale)


@mcp.tool(
    description=(
        "Resolve a spoken reference like 'email 2' to its item id in a live brief. "
        "kind is calendar, email, or ring; ordinal is 1-based. Returns item_id and a "
        "spoken summary. Use the returned item_id with the action tools; never guess an id."
    )
)
async def find_item(session_id: str, kind: SourceKind, ordinal: int) -> ResolvedItem:
    return resolve_item(session_id=session_id, kind=kind, ordinal=ordinal)


@mcp.tool(
    description=(
        "Read a priority email aloud. Pass the session_id and the exact item_id from "
        "get_day_brief, next_items, or find_item. Returns spoken_text and the email body."
    )
)
async def read_item(session_id: str, item_id: str) -> ActionResult:
    return act_on_item(session_id=session_id, item_id=item_id, action=ActionKind.READ)


@mcp.tool(
    description=(
        "Prepare a reply to a priority email. Pass the session_id and item_id. Returns a "
        "draft; nothing is sent. Read spoken_text to confirm."
    )
)
async def reply_to_item(session_id: str, item_id: str) -> ActionResult:
    return act_on_item(session_id=session_id, item_id=item_id, action=ActionKind.REPLY)


@mcp.tool(
    description=(
        "Open a calendar event's details. Pass the session_id and item_id. Returns "
        "spoken_text with the time and location."
    )
)
async def open_item(session_id: str, item_id: str) -> ActionResult:
    return act_on_item(session_id=session_id, item_id=item_id, action=ActionKind.OPEN)


@mcp.tool(
    description=(
        "Show a Ring alert. Pass the session_id and item_id. Returns spoken_text with the "
        "device and what it detected."
    )
)
async def show_item(session_id: str, item_id: str) -> ActionResult:
    return act_on_item(session_id=session_id, item_id=item_id, action=ActionKind.SHOW)


def _transport_security() -> TransportSecuritySettings:
    """DNS-rebinding allowlist. Local hosts are always included; add public hosts via env."""

    hosts = [
        "127.0.0.1",
        "127.0.0.1:*",
        "localhost",
        "localhost:*",
        "[::1]",
        "[::1]:*",
        "testserver",
        "testserver:*",
    ]
    origins = [
        "http://127.0.0.1",
        "http://127.0.0.1:*",
        "http://localhost",
        "http://localhost:*",
    ]
    for host in _csv("MCP_ALLOWED_HOSTS"):
        hosts.append(host)
        if not host.endswith(":*"):
            hosts.append(f"{host}:*")
    for origin in _csv("MCP_ALLOWED_ORIGINS"):
        origins.append(origin)
        if not origin.endswith(":*"):
            origins.append(f"{origin}:*")
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts,
        allowed_origins=origins,
    )


def _csv(name: str) -> list[str]:
    return [part.strip() for part in os.getenv(name, "").split(",") if part.strip()]


# Built at import so ``mcp.session_manager`` exists before the FastAPI lifespan starts.
# The inner route stays ``/mcp``. Mounting at ``/`` keeps that public path, and the
# FastAPI routes registered below are matched first.
mcp_app = mcp.streamable_http_app(transport_security=_transport_security())


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # A broken fixture should fail boot, before the first voice request.
    build_timeline(
        user_id="startup-check",
        tz=ZoneInfo("America/Los_Angeles"),
        day=date(2026, 10, 13),
        tight_gap_minutes=TIGHT_GAP_MINUTES,
    )
    logger.info("Day-brief pipeline ready (batch=%s ttl=%ss)", BATCH_SIZE, TTL_SECONDS)
    async with mcp.session_manager.run():
        yield
    # ``app`` is part of the FastAPI lifespan contract.
    del app


app = FastAPI(
    title="Alexa Day Brief",
    version="0.1.0",
    summary="Code-owned day brief with per-item wording, served over HTTP and MCP Streamable HTTP.",
    lifespan=_lifespan,
)

_cors = _csv("CORS_ORIGINS") or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Mcp-Session-Id"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/get_day_brief")
async def get_day_brief_route(body: GetDayBriefRequest) -> JSONResponse:
    brief = await open_day_brief(
        user_id=body.user_id,
        timezone_name=body.timezone,
        locale=body.locale,
        day=body.day,
    )
    return _json(brief)


@app.post("/next_items")
async def next_items_route(body: NextItemsRequest) -> JSONResponse:
    brief = await advance_brief(session_id=body.session_id, locale=body.locale)
    return _json(brief)


@app.post("/find_item")
async def find_item_route(body: FindItemRequest) -> JSONResponse:
    return _json_action(resolve_item(session_id=body.session_id, kind=body.kind, ordinal=body.ordinal))


@app.post("/read_item")
async def read_item_route(body: ActionRequest) -> JSONResponse:
    return _json_action(
        act_on_item(session_id=body.session_id, item_id=body.item_id, action=ActionKind.READ)
    )


@app.post("/reply_to_item")
async def reply_to_item_route(body: ActionRequest) -> JSONResponse:
    return _json_action(
        act_on_item(session_id=body.session_id, item_id=body.item_id, action=ActionKind.REPLY)
    )


@app.post("/open_item")
async def open_item_route(body: ActionRequest) -> JSONResponse:
    return _json_action(
        act_on_item(session_id=body.session_id, item_id=body.item_id, action=ActionKind.OPEN)
    )


@app.post("/show_item")
async def show_item_route(body: ActionRequest) -> JSONResponse:
    return _json_action(
        act_on_item(session_id=body.session_id, item_id=body.item_id, action=ActionKind.SHOW)
    )


# Mounted last so ``/get_day_brief``, ``/next_items``, ``/health``, and the action
# routes win first. A request to ``/mcp`` is forwarded to the Streamable HTTP app.
app.mount("/", mcp_app)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=_env_int("PORT", 8000),
        reload=False,
    )


if __name__ == "__main__":
    main()
