"""Per-item triage actions (FR-4).

A listener names an item and an action; this module turns that pair into a
spoken confirmation plus a structured payload. It owns the whole triage surface:

* which actions each source kind allows (a code-owned allowlist),
* what each action returns (payload shape and spoken wording),
* the error mode for an action a kind does not support.

The module is **pure**: ``perform_action`` takes one ``BriefItem`` and one
``ActionKind`` and returns an ``ActionResult``. It never touches a session, the
network, or a model. Session access (is this item in a live brief?) lives in
``session.py``. That seam keeps the entire action surface testable by calling
``perform_action`` with a hand-built item.

Resolution of a spoken reference such as "email 2" to an item is **not** here:
``engine.find_by_ordinal`` owns it, so a model never counts items.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from engine import BriefItem, SourceKind, format_clock


class ActionKind(str, Enum):
    READ = "read"
    REPLY = "reply"
    OPEN = "open"
    SHOW = "show"


# Which actions each source kind supports. Code-owned: a caller cannot widen
# this set by asking for a different verb.
_ALLOWED: dict[SourceKind, tuple[ActionKind, ...]] = {
    SourceKind.EMAIL: (ActionKind.READ, ActionKind.REPLY),
    SourceKind.CALENDAR: (ActionKind.OPEN,),
    SourceKind.RING: (ActionKind.SHOW,),
}

_KIND_NOUN: dict[SourceKind, str] = {
    SourceKind.EMAIL: "email",
    SourceKind.CALENDAR: "event",
    SourceKind.RING: "alert",
}


class ActionResult(BaseModel):
    """The outcome of one triage action.

    ``ok`` False carries a stable ``error_code`` and a spoken sentence; the
    item/kind/action fields are present whenever the reference resolved.
    """

    model_config = ConfigDict(frozen=True)

    ok: bool
    spoken_text: str
    error_code: str | None = None
    item_id: str | None = None
    kind: SourceKind | None = None
    action: ActionKind | None = None
    payload: dict[str, Any] = {}


def perform_action(item: BriefItem, action: ActionKind) -> ActionResult:
    """Execute one action against one resolved item.

    Total by design: an unsupported action is a returned error result, not an
    exception, so callers hand the same object back to the client regardless of
    outcome.
    """

    allowed = _ALLOWED[item.kind]
    if action not in allowed:
        return ActionResult(
            ok=False,
            item_id=item.id,
            kind=item.kind,
            action=action,
            error_code="action_not_supported",
            spoken_text=(
                f"I can't {action.value} that {_KIND_NOUN[item.kind]}. "
                f"Try {', '.join(a.value for a in allowed)}."
            ),
        )

    if action is ActionKind.READ:
        return _read(item)
    if action is ActionKind.REPLY:
        return _reply(item)
    if action is ActionKind.OPEN:
        return _open(item)
    return _show(item)


def _read(item: BriefItem) -> ActionResult:
    sender = item.sender or "an unknown sender"
    body = item.detail or "There is no further detail on this email."
    return ActionResult(
        ok=True,
        item_id=item.id,
        kind=item.kind,
        action=ActionKind.READ,
        spoken_text=f"From {sender}, about {item.title}. {body}",
        payload={
            "sender": item.sender,
            "subject": item.title,
            "body": item.detail,
        },
    )


def _reply(item: BriefItem) -> ActionResult:
    sender = item.sender or "the sender"
    return ActionResult(
        ok=True,
        item_id=item.id,
        kind=item.kind,
        action=ActionKind.REPLY,
        spoken_text=(
            f"I've drafted a reply to {sender} about {item.title}. "
            "Say send it to send, or say cancel."
        ),
        payload={
            "to": item.sender,
            "subject": f"Re: {item.title}",
            "draft": _draft_reply(item),
        },
    )


def _draft_reply(item: BriefItem) -> str:
    # Deterministic draft: the hackathon MVP never sends mail, it only prepares
    # the reply from the item's own fields.
    greeting = (item.sender or "there").split()[0]
    return (
        f"Hi {greeting}, thanks for the note about \"{item.title}\". "
        "I'll take a look and follow up shortly."
    )


def _open(item: BriefItem) -> ActionResult:
    start = format_clock(item.timestamp)
    end = format_clock(item.end_timestamp) if item.end_timestamp else None
    when = f"{start} to {end}" if end else start
    where = f" at {item.location}" if item.location else ""
    return ActionResult(
        ok=True,
        item_id=item.id,
        kind=item.kind,
        action=ActionKind.OPEN,
        spoken_text=f"{item.title}, {when}{where}.",
        payload={
            "title": item.title,
            "start": item.timestamp.isoformat(),
            "end": item.end_timestamp.isoformat() if item.end_timestamp else None,
            "location": item.location,
        },
    )


def _show(item: BriefItem) -> ActionResult:
    device = item.device or "a Ring device"
    return ActionResult(
        ok=True,
        item_id=item.id,
        kind=item.kind,
        action=ActionKind.SHOW,
        spoken_text=f"{device}: {item.title}.",
        payload={
            "device": item.device,
            "alert_type": item.alert_type,
            "summary": item.title,
            "detail": item.detail,
        },
    )
