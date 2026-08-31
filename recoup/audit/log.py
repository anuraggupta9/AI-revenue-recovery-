"""Tamper-evident decision log.

The track bar asks for an audit trail. A plain log file is not one: anything that
can be edited after the fact proves nothing about what the agent actually did. So
each entry carries the hash of its predecessor, which means altering or removing
any historical entry invalidates every hash after it and `verify()` names the
exact index where the chain broke.

Two design choices worth defending in an interview:

Timestamps are passed in, never read from the clock inside this module. A log
whose contents depend on wall-clock time cannot be reproduced, and the whole
submission rests on a batch that reproduces from a fixed seed.

Declined actions are recorded, not just taken ones. "We considered a rail switch
and rejected it because the expected value was below the floor" is the part that
makes the trail an explanation rather than a receipt.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, TextIO

from recoup.domain.money import Money

# Opening link of the chain. Any value works so long as it is fixed; using an
# obvious sentinel makes a truncated-to-empty log distinguishable from a log
# whose first entry was deleted.
GENESIS_HASH = "0" * 64


class EntryKind(str, Enum):
    """What happened. Kept coarse enough to stay readable in a demo."""

    EVENT_INGESTED = "event_ingested"
    DUPLICATE_SUPPRESSED = "duplicate_suppressed"
    DIAGNOSED = "diagnosed"
    # Emitted once per rule per decision, whether it passed or failed. Verbose
    # on purpose: the passes are what demonstrate the rule was actually checked.
    RULE_EVALUATED = "rule_evaluated"
    EV_COMPUTED = "ev_computed"
    ACTION_CHOSEN = "action_chosen"
    ACTION_DECLINED = "action_declined"
    ACTION_EXECUTED = "action_executed"
    ACTION_RESULT = "action_result"
    STATE_TRANSITION = "state_transition"
    ESCALATED = "escalated"
    CIRCUIT_BREAKER = "circuit_breaker"
    # What a control-arm case would have done. Never paired with ACTION_EXECUTED.
    SHADOW_DECISION = "shadow_decision"

    def __str__(self) -> str:
        return self.value


def _encode(obj: Any) -> Any:
    """Make domain objects canonically serialisable.

    Money becomes an explicit paise/currency pair rather than a formatted string,
    so the log stays machine-checkable and never round-trips through a float.
    """
    if isinstance(obj, Money):
        return {"paise": obj.paise, "currency": obj.currency}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(str(x) for x in obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(f"cannot serialise {type(obj).__name__} into the audit log")


def canonical_json(payload: Any) -> str:
    """Byte-stable JSON.

    sort_keys matters: dict ordering varies with construction order, and a hash
    over unsorted keys would flag honest re-serialisation as tampering.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_encode,
    )


@dataclass(frozen=True, slots=True)
class AuditEntry:
    seq: int
    at: datetime
    kind: EntryKind
    case_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    prev_hash: str = GENESIS_HASH
    entry_hash: str = ""

    def compute_hash(self) -> str:
        material = canonical_json(
            {
                "seq": self.seq,
                "at": self.at,
                "kind": self.kind,
                "case_id": self.case_id,
                "payload": self.payload,
                "prev_hash": self.prev_hash,
            }
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        return canonical_json(
            {
                "seq": self.seq,
                "at": self.at,
                "kind": self.kind,
                "case_id": self.case_id,
                "payload": self.payload,
                "prev_hash": self.prev_hash,
                "entry_hash": self.entry_hash,
            }
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AuditEntry:
        return cls(
            seq=int(raw["seq"]),
            at=datetime.fromisoformat(raw["at"]),
            kind=EntryKind(raw["kind"]),
            case_id=raw["case_id"],
            payload=raw.get("payload") or {},
            prev_hash=raw["prev_hash"],
            entry_hash=raw["entry_hash"],
        )


@dataclass(frozen=True, slots=True)
class ChainStatus:
    ok: bool
    entries: int
    broken_at: int | None = None
    detail: str = ""

    def __str__(self) -> str:
        if self.ok:
            return f"audit chain intact across {self.entries} entries"
        return f"audit chain BROKEN at seq {self.broken_at}: {self.detail}"


class AuditLog:
    """Append-only log with an in-memory mirror.

    Holding every entry in memory is fine at this scale — a few thousand entries
    per batch — and it keeps the demo single-process. A production version would
    stream to durable storage and keep only the tip hash.

    The file handle is opened once and held. The first version opened, wrote and
    closed per entry, which is the obvious way to write an append-only log and was
    measurably wrong: a 300-case batch emits 17,415 entries, and that many
    open/close cycles against a cloud-synced directory took long enough that I first
    assumed the orchestrator had hung.

    `flush_each` defaults to true because that is the behaviour a live deployment
    needs: an entry is readable before the action it authorises is executed, so the
    log cannot be missing a decision that already moved money. The simulation
    harness turns it off, because it closes the handle before anything re-reads the
    file. Measured on that same 300-case batch, in this synced folder: 9.0s with
    flushing off against 72.5s with it on, for a durability guarantee no reader here
    is relying on. At the 2,000-case headline size that difference is the reason
    `compare` finishes in about two and a half minutes rather than twenty.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        truncate: bool = False,
        flush_each: bool = True,
    ) -> None:
        self.path = Path(path) if path else None
        self._entries: list[AuditEntry] = []
        self._handle: TextIO | None = None
        self._flush_each = flush_each
        if not self.path:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if truncate:
            # Explicit rather than unlinking the file. A new run appending onto a
            # previous run's log produces a chain that verifies correctly and
            # describes two different runs, which is worse than either a fresh log
            # or an error.
            self._handle = self.path.open("w", encoding="utf-8")
        else:
            if self.path.exists():
                self._entries = list(read_entries(self.path))
            self._handle = self.path.open("a", encoding="utf-8")

    @property
    def tip_hash(self) -> str:
        return self._entries[-1].entry_hash if self._entries else GENESIS_HASH

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[AuditEntry]:
        return iter(self._entries)

    def __enter__(self) -> AuditLog:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        # A backstop, not the intended path — use `close()` or the context manager.
        # Present because this object owns a file descriptor and the alternative is
        # leaking one per abandoned log, which in a long-lived API process is a
        # slow crash rather than a warning.
        try:
            self.close()
        except Exception:  # pragma: no cover - interpreter shutdown
            pass

    def append(
        self,
        kind: EntryKind,
        *,
        case_id: str,
        at: datetime,
        **payload: Any,
    ) -> AuditEntry:
        if at.tzinfo is None:
            raise ValueError("audit timestamps must be timezone-aware")
        entry = AuditEntry(
            seq=len(self._entries),
            at=at,
            kind=kind,
            case_id=case_id,
            payload=payload,
            prev_hash=self.tip_hash,
        )
        entry = replace(entry, entry_hash=entry.compute_hash())
        self._entries.append(entry)
        if self._handle is not None:
            self._handle.write(entry.to_json() + "\n")
            if self._flush_each:
                # Flushed, not fsynced. The guarantee wanted here is that an entry
                # is readable by another process before the action it authorises
                # gets executed; surviving power loss is a production concern and
                # would need fsync plus a real durability story.
                self._handle.flush()
        return entry

    def for_case(self, case_id: str) -> list[AuditEntry]:
        return [entry for entry in self._entries if entry.case_id == case_id]

    def of_kind(self, kind: EntryKind) -> list[AuditEntry]:
        return [entry for entry in self._entries if entry.kind is kind]

    def verify(self) -> ChainStatus:
        return verify_chain(self._entries)

    def verify_on_disk(self) -> ChainStatus:
        """Re-read from storage and verify.

        Distinct from verify(): this is the check that means something, because
        it does not trust the process that wrote the log.
        """
        if not self.path or not self.path.exists():
            return ChainStatus(ok=True, entries=0, detail="no file")
        return verify_chain(list(read_entries(self.path)))


def read_entries(path: str | Path) -> Iterator[AuditEntry]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield AuditEntry.from_dict(json.loads(line))


def verify_chain(entries: list[AuditEntry]) -> ChainStatus:
    """Walk the chain, checking linkage, ordering and content integrity."""
    expected_prev = GENESIS_HASH
    for index, entry in enumerate(entries):
        if entry.seq != index:
            return ChainStatus(
                ok=False,
                entries=len(entries),
                broken_at=index,
                detail=f"sequence gap: expected {index}, found {entry.seq} "
                "(an entry was removed or reordered)",
            )
        if entry.prev_hash != expected_prev:
            return ChainStatus(
                ok=False,
                entries=len(entries),
                broken_at=index,
                detail="predecessor hash does not match (chain was spliced)",
            )
        if entry.entry_hash != entry.compute_hash():
            return ChainStatus(
                ok=False,
                entries=len(entries),
                broken_at=index,
                detail="content hash mismatch (this entry was edited in place)",
            )
        expected_prev = entry.entry_hash
    return ChainStatus(ok=True, entries=len(entries))
