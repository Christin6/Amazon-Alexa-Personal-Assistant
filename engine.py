"""Deterministic day-brief engine.

Code owns ingestion, chronological order, immutable IDs, and every numeric
claim (counts, overlap minutes, back-to-back gaps). A later wording step may
phrase a single item. It never receives this module's full timeline, and it
is never asked to count.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from enum import Enum
from typing import Sequence
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# Back-to-back meetings at or under this gap are reported as a hard constraint.
DEFAULT_TIGHT_GAP_MINUTES = 10


class SourceKind(str, Enum):
    CALENDAR = "calendar"
    EMAIL = "email"
    RING = "ring"


_ID_PREFIX: dict[SourceKind, str] = {
    SourceKind.CALENDAR: "evt",
    SourceKind.EMAIL: "eml",
    SourceKind.RING: "rng",
}

# Tie-break when two sources share a timestamp. IDs stay the primary stable key.
_KIND_ORDER: dict[SourceKind, int] = {
    SourceKind.CALENDAR: 0,
    SourceKind.EMAIL: 1,
    SourceKind.RING: 2,
}


class TimelineIntegrityError(RuntimeError):
    """Raised when a timeline violates an invariant the rest of the server trusts."""


class BriefItem(BaseModel):
    """One code-owned fact. Frozen so an ID cannot be rewritten downstream."""

    model_config = ConfigDict(frozen=True)

    id: str
    kind: SourceKind
    timestamp: datetime
    title: str
    detail: str = ""
    end_timestamp: datetime | None = None
    location: str | None = None
    sender: str | None = None
    device: str | None = None
    alert_type: str | None = None


class Overlap(BaseModel):
    """Two calendar events whose intervals intersect. Minutes are computed in code."""

    model_config = ConfigDict(frozen=True)

    first_id: str
    second_id: str
    overlap_start: datetime
    overlap_end: datetime
    overlap_minutes: int = Field(ge=1)


class TightGap(BaseModel):
    """Successive calendar events separated by a short, non-negative gap."""

    model_config = ConfigDict(frozen=True)

    first_id: str
    second_id: str
    gap_minutes: int = Field(ge=0)


class DayConstraints(BaseModel):
    """Hard facts about the whole day. Serialized to clients as-is."""

    model_config = ConfigDict(frozen=True)

    total_count: int = Field(ge=0)
    counts_by_kind: dict[str, int]
    ordered_ids: list[str]
    overlaps: list[Overlap]
    tight_gaps: list[TightGap]
    tight_gap_threshold_minutes: int = Field(ge=0)


class DayTimeline(BaseModel):
    """The full day after sort, ID assignment, and the integrity self-check."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    day: date
    timezone: str
    items: list[BriefItem]
    constraints: DayConstraints


class _RawCalendar(BaseModel):
    title: str
    start: datetime
    end: datetime
    location: str | None = None


class _RawEmail(BaseModel):
    subject: str
    sender: str
    received_at: datetime
    snippet: str


class _RawRing(BaseModel):
    device: str
    alert_type: str
    happened_at: datetime
    summary: str


def build_timeline(
    *,
    user_id: str,
    tz: ZoneInfo,
    day: date,
    tight_gap_minutes: int = DEFAULT_TIGHT_GAP_MINUTES,
) -> DayTimeline:
    """Ingest raw records, sort them, stamp IDs, and compute constraints.

    ``user_id`` is the fetch key. The bundled simulator returns one household
    fixture; a live adapter would replace ``_fetch_*`` and return the same
    raw shapes.
    """

    safe_user = _one_line(user_id) or "anonymous"
    logger.info(
        "Building timeline user_id=%s day=%s tz=%s",
        safe_user,
        day.isoformat(),
        tz.key,
    )

    calendar = _assign_calendar(_fetch_calendar(safe_user, tz, day))
    email = _assign_email(_fetch_email(safe_user, tz, day))
    ring = _assign_ring(_fetch_ring(safe_user, tz, day))

    # Merge only after each source has immutable IDs. Sorting cannot renumber them.
    items = sorted(
        (*calendar, *email, *ring),
        key=lambda item: (item.timestamp, _KIND_ORDER[item.kind], item.id),
    )
    constraints = DayConstraints(
        total_count=len(items),
        counts_by_kind=_counts(items),
        ordered_ids=[item.id for item in items],
        overlaps=_calendar_overlaps(items),
        tight_gaps=_calendar_tight_gaps(items, tight_gap_minutes),
        tight_gap_threshold_minutes=tight_gap_minutes,
    )
    timeline = DayTimeline(
        user_id=safe_user,
        day=day,
        timezone=tz.key,
        items=items,
        constraints=constraints,
    )
    validate_timeline(timeline)
    return timeline


def validate_timeline(timeline: DayTimeline) -> None:
    """Reject a timeline whose IDs, order, or counts disagree with each other.

    This runs before any wording call. Downstream code treats ``total_count``,
    ``ordered_ids``, and per-kind sequences as already proven.
    """

    items = timeline.items
    constraints = timeline.constraints
    if constraints.total_count != len(items):
        raise TimelineIntegrityError("total_count does not match the item list")
    if constraints.ordered_ids != [item.id for item in items]:
        raise TimelineIntegrityError("ordered_ids drifted from chronological order")

    counts = _counts(items)
    if constraints.counts_by_kind != counts:
        raise TimelineIntegrityError("counts_by_kind does not match the item list")
    if sum(counts.values()) != constraints.total_count:
        raise TimelineIntegrityError("per-kind counts do not sum to total_count")

    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            raise TimelineIntegrityError(f"duplicate id {item.id}")
        seen.add(item.id)
        prefix = _ID_PREFIX[item.kind]
        if not item.id.startswith(prefix + "_"):
            raise TimelineIntegrityError(f"{item.id} does not match kind {item.kind.value}")
        if item.timestamp.tzinfo is None:
            raise TimelineIntegrityError(f"{item.id} is missing a timezone")

    for kind, prefix in _ID_PREFIX.items():
        group = [item for item in items if item.kind is kind]
        for index, item in enumerate(group, start=1):
            expected = f"{prefix}_{index:02d}"
            if item.id != expected:
                raise TimelineIntegrityError(
                    f"expected {expected} for {kind.value} in timestamp order, found {item.id}"
                )
        for earlier, later in zip(group, group[1:]):
            if later.timestamp < earlier.timestamp:
                raise TimelineIntegrityError(f"{kind.value} ids are not in timestamp order")


def deterministic_wording(item: BriefItem) -> tuple[str, str, str]:
    """Phrase one item from its own fields.

    Returns ``(summary, action_prompt, urgency)``. Used when a live model call
    is unavailable, and as the coverage backup so a failed wording call still
    speaks the underlying fact. The sentence names this item only.
    """

    when = format_clock(item.timestamp)
    if item.kind is SourceKind.CALENDAR:
        end = format_clock(item.end_timestamp) if item.end_timestamp else None
        if end and item.location:
            summary = f"At {when}, {item.title} runs until {end} at {item.location}."
        elif end:
            summary = f"At {when}, {item.title} runs until {end}."
        else:
            summary = f"At {when}, {item.title}."
        return summary, "Say open it if you want the details.", "normal"

    if item.kind is SourceKind.EMAIL:
        sender = item.sender or "an unknown sender"
        summary = f"Priority email from {sender}: {item.title}."
        return summary, "Say read it if you want the body.", "high"

    device = item.device or "a Ring device"
    summary = f"At {when}, {device} reported {item.title}."
    urgency = "high" if item.alert_type in {"person", "doorbell"} else "normal"
    return summary, "Say show the clip if you want the video.", urgency


def describe_constraints(constraints: DayConstraints, items: Sequence[BriefItem]) -> list[str]:
    """Speak overlaps and short gaps from computed minutes, looking up titles by ID."""

    by_id = {item.id: item for item in items}
    sentences: list[str] = []
    for overlap in constraints.overlaps:
        first = by_id[overlap.first_id]
        second = by_id[overlap.second_id]
        sentences.append(
            f"{first.title} and {second.title} overlap for {overlap.overlap_minutes} minutes, "
            f"from {format_clock(overlap.overlap_start)} to {format_clock(overlap.overlap_end)}."
        )
    for gap in constraints.tight_gaps:
        first = by_id[gap.first_id]
        second = by_id[gap.second_id]
        if gap.gap_minutes == 0:
            sentences.append(f"{first.title} ends as {second.title} starts, with no break between them.")
        else:
            minutes = "minute" if gap.gap_minutes == 1 else "minutes"
            sentences.append(
                f"There are {gap.gap_minutes} {minutes} between {first.title} and {second.title}."
            )
    if not sentences:
        sentences.append("No overlapping or back-to-back meetings.")
    return sentences


def format_clock(moment: datetime) -> str:
    """Format an aware timestamp in its own zone as a 12-hour clock."""

    local = moment.astimezone(moment.tzinfo) if moment.tzinfo else moment
    hour = local.hour % 12 or 12
    suffix = "AM" if local.hour < 12 else "PM"
    return f"{hour}:{local.minute:02d} {suffix}"


def conflict_ids(constraints: DayConstraints) -> set[str]:
    """IDs that participate in an overlap or a tight gap."""

    found: set[str] = set()
    for overlap in constraints.overlaps:
        found.add(overlap.first_id)
        found.add(overlap.second_id)
    for gap in constraints.tight_gaps:
        found.add(gap.first_id)
        found.add(gap.second_id)
    return found


def _fetch_calendar(user_id: str, tz: ZoneInfo, day: date) -> list[_RawCalendar]:
    """Simulated calendar. Returned out of order so the sort step is observable.

    A production adapter would call the user's calendar API with ``user_id``
    and map each event into ``_RawCalendar``.
    """

    del user_id  # The fixture is shared; live fetchers would filter on this.
    return [
        _RawCalendar(
            title="Lunch with the demo crew",
            start=_at(day, 12, 0, tz),
            end=_at(day, 12, 45, tz),
            location="Cafeteria",
        ),
        _RawCalendar(
            title="Daily standup",
            start=_at(day, 8, 30, tz),
            end=_at(day, 9, 0, tz),
            location="Platform room",
        ),
        _RawCalendar(
            title="Hackathon dry run",
            start=_at(day, 16, 0, tz),
            end=_at(day, 17, 0, tz),
            location="Lab",
        ),
        _RawCalendar(
            title="1:1 with Jordan",
            start=_at(day, 9, 30, tz),
            end=_at(day, 10, 15, tz),
            location="Jordan's office",
        ),
        _RawCalendar(
            title="Alexa+ design review",
            start=_at(day, 9, 0, tz),
            end=_at(day, 9, 45, tz),
            location="Studio",
        ),
    ]


def _fetch_email(user_id: str, tz: ZoneInfo, day: date) -> list[_RawEmail]:
    """Simulated priority inbox. Snippets stay on the item; they are not pre-summarized."""

    del user_id
    return [
        _RawEmail(
            subject="Front door battery is at 8 percent",
            sender="Ring Support",
            received_at=_at(day, 14, 20, tz),
            snippet="Charge the front door battery before tonight.",
        ),
        _RawEmail(
            subject="Submission checklist due by 5 PM",
            sender="Alexa+ Judges",
            received_at=_at(day, 7, 10, tz),
            snippet="The written brief must list every demo step you will show.",
        ),
        _RawEmail(
            subject="Blocking comment on the brief pipeline",
            sender="Priya Shah",
            received_at=_at(day, 11, 5, tz),
            snippet="The coverage check has to run before we speak a page.",
        ),
    ]


def _fetch_ring(user_id: str, tz: ZoneInfo, day: date) -> list[_RawRing]:
    """Simulated Ring notifications."""

    del user_id
    return [
        _RawRing(
            device="Front door",
            alert_type="doorbell",
            happened_at=_at(day, 17, 55, tz),
            summary="Doorbell pressed",
        ),
        _RawRing(
            device="Front door",
            alert_type="package",
            happened_at=_at(day, 6, 42, tz),
            summary="Package detected",
        ),
        _RawRing(
            device="Garage",
            alert_type="person",
            happened_at=_at(day, 10, 12, tz),
            summary="Person detected",
        ),
    ]


def _assign_calendar(raw_items: Sequence[_RawCalendar]) -> list[BriefItem]:
    ordered = sorted(raw_items, key=lambda raw: raw.start)
    items: list[BriefItem] = []
    for index, raw in enumerate(ordered, start=1):
        if raw.end <= raw.start:
            logger.warning("Skipping calendar event with a non-positive duration: %s", raw.title)
            continue
        items.append(
            BriefItem(
                id=f"evt_{index:02d}",
                kind=SourceKind.CALENDAR,
                timestamp=raw.start,
                end_timestamp=raw.end,
                title=raw.title,
                detail=raw.location or "",
                location=raw.location,
            )
        )
    return items


def _assign_email(raw_items: Sequence[_RawEmail]) -> list[BriefItem]:
    ordered = sorted(raw_items, key=lambda raw: raw.received_at)
    return [
        BriefItem(
            id=f"eml_{index:02d}",
            kind=SourceKind.EMAIL,
            timestamp=raw.received_at,
            title=raw.subject,
            detail=raw.snippet,
            sender=raw.sender,
        )
        for index, raw in enumerate(ordered, start=1)
    ]


def _assign_ring(raw_items: Sequence[_RawRing]) -> list[BriefItem]:
    ordered = sorted(raw_items, key=lambda raw: raw.happened_at)
    return [
        BriefItem(
            id=f"rng_{index:02d}",
            kind=SourceKind.RING,
            timestamp=raw.happened_at,
            title=raw.summary,
            detail=f"{raw.alert_type} at {raw.device}",
            device=raw.device,
            alert_type=raw.alert_type,
        )
        for index, raw in enumerate(ordered, start=1)
    ]


def _counts(items: Sequence[BriefItem]) -> dict[str, int]:
    counts = {kind.value: 0 for kind in SourceKind}
    for item in items:
        counts[item.kind.value] += 1
    return counts


def _calendar_overlaps(items: Sequence[BriefItem]) -> list[Overlap]:
    """Every pair of calendar intervals where each starts before the other ends.

    A meeting that ends as the next one starts is a zero-minute gap, not an overlap.
    """

    calendar = [item for item in items if item.kind is SourceKind.CALENDAR and item.end_timestamp]
    overlaps: list[Overlap] = []
    for index, left in enumerate(calendar):
        assert left.end_timestamp is not None
        for right in calendar[index + 1 :]:
            assert right.end_timestamp is not None
            if left.timestamp < right.end_timestamp and right.timestamp < left.end_timestamp:
                start = max(left.timestamp, right.timestamp)
                end = min(left.end_timestamp, right.end_timestamp)
                seconds = (end - start).total_seconds()
                if seconds <= 0:
                    continue
                minutes = max(1, int(seconds // 60))
                first, second = (left, right) if left.id < right.id else (right, left)
                overlaps.append(
                    Overlap(
                        first_id=first.id,
                        second_id=second.id,
                        overlap_start=start,
                        overlap_end=end,
                        overlap_minutes=minutes,
                    )
                )
    overlaps.sort(key=lambda item: (item.overlap_start, item.first_id, item.second_id))
    return overlaps


def _calendar_tight_gaps(items: Sequence[BriefItem], threshold_minutes: int) -> list[TightGap]:
    """Gaps between successive calendar events in start order.

    Overlapping successors are skipped here; ``_calendar_overlaps`` already records them.
    """

    calendar = [item for item in items if item.kind is SourceKind.CALENDAR and item.end_timestamp]
    gaps: list[TightGap] = []
    for left, right in zip(calendar, calendar[1:]):
        assert left.end_timestamp is not None
        delta = right.timestamp - left.end_timestamp
        seconds = delta.total_seconds()
        if seconds < 0:
            continue
        minutes = int(seconds // 60)
        if minutes <= threshold_minutes:
            gaps.append(TightGap(first_id=left.id, second_id=right.id, gap_minutes=minutes))
    return gaps


def _at(day: date, hour: int, minute: int, tz: ZoneInfo) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=tz)


def _one_line(value: str) -> str:
    return " ".join(value.replace("\n", " ").replace("\r", " ").split())
