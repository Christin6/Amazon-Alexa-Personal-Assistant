"""Ephemeral voice session memory.

The timeline produced by the engine is stored here, in process, and forgotten
after a TTL (20 minutes by default). Pagination is a cursor over that list in
batches of 3. "Next" advances the cursor. Nothing about the conversation is
written to disk, and expired records are deleted on the next access so the
process does not accumulate briefs.

The cursor is an integer owned by this module. Wording calls do not choose
which items are on the page.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from engine import BriefItem, DayConstraints, DayTimeline, SourceKind, find_by_ordinal

DEFAULT_TTL_SECONDS = 20 * 60
DEFAULT_BATCH_SIZE = 3


class BriefSessionError(Exception):
    """A voice-facing session failure with a stable ``code`` and a spoken sentence."""

    def __init__(self, code: str, spoken_text: str) -> None:
        super().__init__(spoken_text)
        self.code = code
        self.spoken_text = spoken_text


class SessionNotFoundError(BriefSessionError):
    def __init__(self) -> None:
        super().__init__(
            "session_not_found",
            "I don't have that brief anymore. Ask me for today's brief.",
        )


class SessionExpiredError(BriefSessionError):
    def __init__(self, ttl_phrase: str) -> None:
        super().__init__(
            "session_expired",
            f"That brief expired after {ttl_phrase}. Ask me for today's brief again.",
        )


class ItemNotFoundError(BriefSessionError):
    def __init__(self, reference: str) -> None:
        super().__init__(
            "item_not_found",
            f"I don't have {reference} in this brief. Ask me to read the brief again.",
        )


@dataclass(frozen=True)
class PageSlice:
    """One page of a live session. Item objects are the engine's frozen records."""

    session_id: str
    user_id: str
    day_iso: str
    timezone: str
    locale: str
    active_items: tuple[BriefItem, ...]
    timeline: tuple[BriefItem, ...]
    constraints: DayConstraints
    batch_number: int
    batch_size: int
    cursor_after: int
    remaining_after: int
    has_more: bool
    is_terminal: bool
    expires_at: datetime
    expires_in_seconds: int


@dataclass
class _Record:
    session_id: str
    user_id: str
    day_iso: str
    timezone: str
    locale: str
    items: tuple[BriefItem, ...]
    constraints: DayConstraints
    cursor: int
    batch_size: int
    deadline_monotonic: float
    expires_at: datetime


class SessionManager:
    """In-memory TTL map from session id to a timeline plus a pagination cursor."""

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.ttl_seconds = ttl_seconds
        self.batch_size = batch_size
        self._records: dict[str, _Record] = {}
        self._lock = threading.Lock()

    def create(self, timeline: DayTimeline, *, locale: str) -> PageSlice:
        """Store a timeline and return the first page without advancing the cursor.

        The caller advances with ``complete_page`` after the page has been rendered,
        so a failed wording pass can be retried on the same items.
        """

        now_mono = time.monotonic()
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.ttl_seconds)
        record = _Record(
            session_id="brief_" + secrets.token_hex(8),
            user_id=timeline.user_id,
            day_iso=timeline.day.isoformat(),
            timezone=timeline.timezone,
            locale=locale,
            items=tuple(timeline.items),
            constraints=timeline.constraints,
            cursor=0,
            batch_size=self.batch_size,
            deadline_monotonic=now_mono + self.ttl_seconds,
            expires_at=expires_at,
        )
        with self._lock:
            self._purge_locked(now_mono)
            self._records[record.session_id] = record
            return self._slice_locked(record, now_mono)

    def view(self, session_id: str) -> PageSlice:
        """Return the current page. Does not move the cursor."""

        now_mono = time.monotonic()
        with self._lock:
            # Distinguish "expired" from "never existed" before the sweep deletes the record.
            self._raise_if_expired_locked(session_id, now_mono)
            self._purge_locked(now_mono)
            record = self._live_locked(session_id, now_mono)
            return self._slice_locked(record, now_mono)

    def lookup(self, session_id: str, item_id: str) -> BriefItem:
        """Return one item by its immutable id. Does not move the cursor.

        Raises ``SessionNotFoundError`` / ``SessionExpiredError`` for a dead
        session and ``ItemNotFoundError`` when the id is not in this brief.
        """

        now_mono = time.monotonic()
        with self._lock:
            self._raise_if_expired_locked(session_id, now_mono)
            self._purge_locked(now_mono)
            record = self._live_locked(session_id, now_mono)
            for item in record.items:
                if item.id == item_id:
                    return item
        raise ItemNotFoundError(item_id)

    def lookup_ordinal(self, session_id: str, kind: SourceKind, ordinal: int) -> BriefItem:
        """Return one item by kind and 1-based ordinal. Does not move the cursor.

        Ordinal resolution is delegated to ``engine.find_by_ordinal`` so the
        session never counts items itself.
        """

        now_mono = time.monotonic()
        with self._lock:
            self._raise_if_expired_locked(session_id, now_mono)
            self._purge_locked(now_mono)
            record = self._live_locked(session_id, now_mono)
            item = find_by_ordinal(record.items, kind, ordinal)
        if item is None:
            raise ItemNotFoundError(f"{kind.value} {ordinal}")
        return item

    def complete_page(self, session_id: str, cursor_after: int) -> bool:
        """Advance the cursor to ``cursor_after`` if the session is still live.

        The update is monotonic: a late completion cannot move the cursor
        backwards. Returns False when the session has already expired, in which
        case the rendered page can still be spoken and further pages cannot.
        """

        now_mono = time.monotonic()
        with self._lock:
            self._purge_locked(now_mono)
            record = self._records.get(session_id)
            if record is None or now_mono >= record.deadline_monotonic:
                if record is not None:
                    del self._records[session_id]
                return False
            if cursor_after > record.cursor:
                record.cursor = min(cursor_after, len(record.items))
            return True

    def _raise_if_expired_locked(self, session_id: str, now_mono: float) -> None:
        record = self._records.get(session_id)
        if record is not None and now_mono >= record.deadline_monotonic:
            del self._records[session_id]
            raise SessionExpiredError(self.ttl_phrase())

    def _live_locked(self, session_id: str, now_mono: float) -> _Record:
        record = self._records.get(session_id)
        if record is None:
            raise SessionNotFoundError()
        if now_mono >= record.deadline_monotonic:
            del self._records[session_id]
            raise SessionExpiredError(self.ttl_phrase())
        return record

    def _slice_locked(self, record: _Record, now_mono: float) -> PageSlice:
        total = len(record.items)
        start = record.cursor
        if start >= total:
            batch_number = 0
            if total:
                batch_number = (total + record.batch_size - 1) // record.batch_size
            return PageSlice(
                session_id=record.session_id,
                user_id=record.user_id,
                day_iso=record.day_iso,
                timezone=record.timezone,
                locale=record.locale,
                active_items=(),
                timeline=record.items,
                constraints=record.constraints,
                batch_number=batch_number,
                batch_size=record.batch_size,
                cursor_after=start,
                remaining_after=0,
                has_more=False,
                is_terminal=True,
                expires_at=record.expires_at,
                expires_in_seconds=max(0, int(record.deadline_monotonic - now_mono)),
            )

        end = min(start + record.batch_size, total)
        remaining_after = total - end
        return PageSlice(
            session_id=record.session_id,
            user_id=record.user_id,
            day_iso=record.day_iso,
            timezone=record.timezone,
            locale=record.locale,
            active_items=record.items[start:end],
            timeline=record.items,
            constraints=record.constraints,
            batch_number=(start // record.batch_size) + 1,
            batch_size=record.batch_size,
            cursor_after=end,
            remaining_after=remaining_after,
            has_more=remaining_after > 0,
            is_terminal=False,
            expires_at=record.expires_at,
            expires_in_seconds=max(0, int(record.deadline_monotonic - now_mono)),
        )

    def _purge_locked(self, now_mono: float) -> None:
        expired = [key for key, record in self._records.items() if now_mono >= record.deadline_monotonic]
        for key in expired:
            del self._records[key]

    def ttl_phrase(self) -> str:
        seconds = self.ttl_seconds
        if seconds % 60 == 0:
            minutes = seconds // 60
            unit = "minute" if minutes == 1 else "minutes"
            return f"{minutes} {unit}"
        return f"{seconds} seconds"
