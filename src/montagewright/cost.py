"""What the run has spent, against one ceiling.

A single global cap in dollars, and no per-stage quotas. Quotas make a budget
into a behaviour modifier: a stage told it has little left starts choosing
cheaper answers, and cheaper answers to editorial questions are worse ones.
Running out of money is a reason to stop with the best cut so far, never a
reason to do the next step badly.

Pricing lives in a table rather than in the arithmetic, because rates change
and a hard-coded rate is a silent error the day they do.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Mapping, NotRequired, TypedDict

# USD per million tokens. Thinking tokens bill at the output rate. Google
# publishes an introductory 3.7 Flash price through 2026-12-31, inclusive;
# the transition is evaluated in UTC every time money is reserved or settled,
# so a long-lived Web process does not need a restart on New Year's Day.
GEMINI_37_PROMO_END = date(2026, 12, 31)
PRICING: dict[str, dict[str, dict[str, float]]] = {
    "gemini-3.6-flash": {
        # 3.6 has no temporary introductory rate.  Keeping both periods in
        # the same table lets one ledger price mixed-model calls without
        # special-case arithmetic at the 3.7 transition date.
        "promotional": {
            "input": 1.50,
            "cached_input": 0.15,
            "output": 7.50,
        },
        "standard": {
            "input": 1.50,
            "cached_input": 0.15,
            "output": 7.50,
        },
    },
    "gemini-3.7-flash": {
        "promotional": {
            "input": 0.75,
            "cached_input": 0.075,
            "output": 3.75,
        },
        "standard": {
            "input": 1.50,
            "cached_input": 0.15,
            "output": 7.50,
        },
    },
}


def pricing_for(
    model_id: str, *, at: datetime | date | None = None,
) -> dict[str, float]:
    """The published Standard API rates in force at a UTC instant."""

    if model_id not in PRICING:
        raise ValueError(f"no production pricing for {model_id}")
    when = datetime.now(timezone.utc).date() if at is None else (
        at.date() if isinstance(at, datetime) else at
    )
    period = "promotional" if when <= GEMINI_37_PROMO_END else "standard"
    return PRICING[model_id][period]


class BudgetSpent(RuntimeError):
    """The cap is reached. Deliver what exists; do not degrade to continue."""


@dataclass(frozen=True)
class Reservation:
    reservation_id: str
    stage: str
    input_tokens: int
    output_tokens: int
    usd: float
    model_id: str


class Spend(TypedDict):
    """What a run cost, in the shape three places read it in.

    It was `dict[str, object]`, which is true and useless: the CLI prints
    by_stage sorted by value, the interface shows it per stage, and the
    report writes it out. All three were indexing into something declared to
    hold anything.
    """

    cap_usd: float
    spent_usd: float
    remaining_usd: float
    calls: int
    by_stage: dict[str, float]
    uncertain_attempts: NotRequired[int]
    cost_warning: NotRequired[str | None]


@dataclass
class Ledger:
    cap_usd: float
    model_id: str = "gemini-3.7-flash"
    journal_path: Path | None = None
    run_id: str = field(default_factory=lambda: uuid4().hex)
    entries: list[dict[str, float | str]] = field(default_factory=list)
    uncertain_attempts: list[dict[str, int | str]] = field(
        default_factory=list
    )
    reservations: dict[str, Reservation] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model_id not in PRICING:
            raise ValueError(
                f"no production pricing for {self.model_id}"
            )

    @property
    def spent_usd(self) -> float:
        return sum(float(entry["usd"]) for entry in self.entries)

    @property
    def remaining_usd(self) -> float:
        return max(
            0.0,
            self.cap_usd - self.spent_usd - self.reserved_usd,
        )

    @property
    def reserved_usd(self) -> float:
        return sum(one.usd for one in self.reservations.values())

    def _usd(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        rates: dict[str, float] | None = None,
    ) -> float:
        rates = rates or pricing_for(self.model_id)
        fresh = max(0, input_tokens - cached_tokens)
        return (
            fresh * rates["input"]
            + cached_tokens * rates["cached_input"]
            + output_tokens * rates["output"]
        ) / 1_000_000

    def reserve(
        self,
        stage: str,
        *,
        input_tokens: int,
        max_output_tokens: int,
        model_id: str | None = None,
    ) -> str:
        """Reserve the worst case before dispatching a paid interaction."""

        charged_model = model_id or self.model_id
        usd = self._usd(
            input_tokens=input_tokens,
            output_tokens=max_output_tokens,
            rates=pricing_for(charged_model),
        )
        available = self.cap_usd - self.spent_usd - self.reserved_usd
        if usd > available + 1e-9:
            raise BudgetSpent(
                f"{stage} could cost up to ${usd:.4f}, but only "
                f"${max(0.0, available):.4f} remains of the "
                f"${self.cap_usd:.2f} cap; it was not sent"
            )
        reservation_id = uuid4().hex
        self.reservations[reservation_id] = Reservation(
            reservation_id=reservation_id,
            stage=stage,
            input_tokens=input_tokens,
            output_tokens=max_output_tokens,
            usd=usd,
            model_id=charged_model,
        )
        return reservation_id

    def cancel(self, reservation_id: str) -> None:
        self.reservations.pop(reservation_id, None)

    def settle(
        self,
        reservation_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        reservation = self.reservations.pop(reservation_id)
        return self.record(
            reservation.stage,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            model_id=reservation.model_id,
        )

    def record(
        self,
        stage: str,
        *,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        model_id: str | None = None,
    ) -> float:
        now = datetime.now(timezone.utc)
        charged_model = model_id or self.model_id
        rates = pricing_for(charged_model, at=now)
        usd = self._usd(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            rates=rates,
        )
        entry: dict[str, float | str] = {
            "stage": stage,
            "input": input_tokens,
            "cached": cached_tokens,
            "output": output_tokens,
            "usd": round(usd, 6),
            "pricing_period": (
                "promotional" if now.date() <= GEMINI_37_PROMO_END
                else "standard"
            ),
            "input_rate": rates["input"],
            "cached_input_rate": rates["cached_input"],
            "output_rate": rates["output"],
            "pricing_at": now.isoformat(),
            "model_id": charged_model,
        }
        self.entries.append(entry)
        self._journal(entry)
        return usd

    def note_uncertain_attempt(self, stage: str, *, status: int) -> None:
        """Record a retry whose provider-side billing cannot be known.

        A 5xx can mean the request failed before inference, or that inference
        completed and only the response was lost.  Without usage metadata we
        must not invent a dollar amount, but omitting the attempt entirely
        makes the local report look more exact than it is.
        """

        entry: dict[str, int | str] = {
            "event": "uncertain_provider_attempt",
            "stage": stage,
            "status": int(status),
        }
        self.uncertain_attempts.append(entry)
        self._journal(entry)

    def _journal(self, entry: Mapping[str, object]) -> None:
        """Persist each settled paid call before the next stage can crash."""

        if self.journal_path is None:
            return
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **entry,
        }
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            handle.flush()

    def cumulative_summary(self) -> "Spend":
        """All attempts sharing this output folder, without charging twice."""

        if self.journal_path is None or not self.journal_path.exists():
            return self.summary()
        entries: list[dict[str, object]] = []
        for line in self.journal_path.read_text(encoding="utf-8").splitlines():
            try:
                one = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(one, dict):
                entries.append(one)
        by_stage: dict[str, float] = {}
        spent = 0.0
        uncertain = 0
        settled_calls = 0
        for entry in entries:
            if entry.get("event") == "uncertain_provider_attempt":
                uncertain += 1
                continue
            raw_usd = entry.get("usd")
            if not isinstance(raw_usd, (int, float, str)):
                continue
            usd = float(raw_usd)
            stage = str(entry.get("stage") or "unknown")
            spent += usd
            settled_calls += 1
            by_stage[stage] = round(by_stage.get(stage, 0.0) + usd, 6)
        return {
            "cap_usd": self.cap_usd,
            "spent_usd": round(spent, 6),
            # The cap is per invocation; cumulative history may exceed it.
            "remaining_usd": round(self.remaining_usd, 6),
            "calls": settled_calls,
            "by_stage": by_stage,
            "uncertain_attempts": uncertain,
            "cost_warning": self._uncertain_warning(uncertain),
        }

    def check(self) -> None:
        """Call before dispatching, so the cap stops work rather than paying for it."""

        if self.spent_usd >= self.cap_usd:
            raise BudgetSpent(
                f"spent ${self.spent_usd:.4f} of ${self.cap_usd:.2f}; "
                "delivering the best cut reached so far"
            )

    def summary(self) -> "Spend":
        by_stage: dict[str, float] = {}
        for entry in self.entries:
            by_stage[str(entry["stage"])] = round(
                by_stage.get(str(entry["stage"]), 0.0) + float(entry["usd"]), 6
            )
        uncertain = len(self.uncertain_attempts)
        return {
            "cap_usd": self.cap_usd,
            "spent_usd": round(self.spent_usd, 6),
            "remaining_usd": round(self.remaining_usd, 6),
            "calls": len(self.entries),
            "by_stage": by_stage,
            "uncertain_attempts": uncertain,
            "cost_warning": self._uncertain_warning(uncertain),
        }

    @staticmethod
    def _uncertain_warning(count: int) -> str | None:
        if not count:
            return None
        return (
            f"{count} provider retry attempt(s) may have been billed, but "
            "returned no usage metadata; the known USD total excludes them"
        )
