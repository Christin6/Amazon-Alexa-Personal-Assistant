"""Coverage guardrail.

After code renders a page, every active item must occupy its own line whose
first token is ``[item_id]``. The scan looks at line prefixes only, so a model
mention of some other id inside a sentence cannot satisfy coverage for an item
that was never rendered.

A missing prefix is replaced with ``engine.deterministic_wording`` for that
item, wrapped in the same ``[id]`` token. The function returns only after a
second scan sees every active id. That is the zero-dropped-detail guarantee
for the page the user is about to hear.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from engine import BriefItem, deterministic_wording

# Prefixes are the coverage token. A bracketed id later in the line does not count.
_LINE_ID = re.compile(r"^\[((?:evt|eml|rng)_\d{2,})\](?:\s|$)")


class CoverageError(RuntimeError):
    """Raised when a backup line itself failed to carry its item id."""

    def __init__(self, missing_ids: list[str]) -> None:
        super().__init__("coverage backup did not include: " + ", ".join(missing_ids))
        self.missing_ids = missing_ids


@dataclass(frozen=True)
class CoverageResult:
    """A buffer that has passed the second coverage scan."""

    rendered_text: str
    injected_ids: list[str]


def line_prefix_ids(rendered: str) -> list[str]:
    """IDs that begin a line, in buffer order, first occurrence only."""

    found: list[str] = []
    seen: set[str] = set()
    for raw_line in rendered.splitlines():
        match = _LINE_ID.match(raw_line.strip())
        if match and match.group(1) not in seen:
            seen.add(match.group(1))
            found.append(match.group(1))
    return found


def backup_line(item: BriefItem) -> str:
    """Deterministic sentence for one item, prefixed with its code-owned id."""

    summary, action, _urgency = deterministic_wording(item)
    return f"[{item.id}] {summary} {action}"


def ensure_coverage(rendered: str, active_items: list[BriefItem]) -> CoverageResult:
    """Guarantee every active id is a line prefix in ``rendered``.

    Header lines (counts, overlaps, the ordered-id index) are preserved.
    Item lines are rewritten in active-item order. An active id with no prefix
    line is injected from ``backup_line``. The result is scanned again, and a
    still-missing id raises ``CoverageError`` instead of being returned.
    """

    header: list[str] = []
    item_lines: dict[str, str] = {}
    for raw_line in rendered.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        match = _LINE_ID.match(stripped)
        if match and match.group(1) not in item_lines:
            item_lines[match.group(1)] = stripped
        elif match is None:
            header.append(stripped)

    injected: list[str] = []
    ordered_item_lines: list[str] = []
    for item in active_items:
        existing = item_lines.get(item.id)
        if existing is None:
            ordered_item_lines.append(backup_line(item))
            injected.append(item.id)
        else:
            ordered_item_lines.append(existing)

    guarded = "\n".join([*header, *ordered_item_lines])
    covered = line_prefix_ids(guarded)
    covered_set = set(covered)
    missing = [item.id for item in active_items if item.id not in covered_set]
    if missing:
        raise CoverageError(missing)

    return CoverageResult(
        rendered_text=guarded,
        injected_ids=injected,
    )
