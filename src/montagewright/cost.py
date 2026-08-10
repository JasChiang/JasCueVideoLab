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

from dataclasses import dataclass, field
from uuid import uuid4
from typing import TypedDict

# USD per million tokens. Thinking tokens bill at the output rate.
PRICING: dict[str, dict[str, float]] = {
    "gemini-3.6-flash": {
        "input": 1.50,
        "cached_input": 0.15,
        "output": 7.50,
    },
    "gemini-3.5-flash": {
        "input": 1.50,
        "cached_input": 0.15,
        "output": 9.00,
    },
}


class BudgetSpent(RuntimeError):
    """The cap is reached. Deliver what exists; do not degrade to continue."""


@dataclass(frozen=True)
class Reservation:
    reservation_id: str
    stage: str
    input_tokens: int
    output_tokens: int
    usd: float


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


@dataclass
class Ledger:
    cap_usd: float
    model_id: str = "gemini-3.6-flash"
    entries: list[dict[str, float | str]] = field(default_factory=list)
    reservations: dict[str, Reservation] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model_id != "gemini-3.6-flash":
            raise ValueError(
                "MontageWright production pricing is fixed to "
                "gemini-3.6-flash"
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
    ) -> float:
        rates = PRICING[self.model_id]
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
    ) -> str:
        """Reserve the worst case before dispatching a paid interaction."""

        usd = self._usd(
            input_tokens=input_tokens,
            output_tokens=max_output_tokens,
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
        )

    def record(
        self,
        stage: str,
        *,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        usd = self._usd(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
        )
        self.entries.append(
            {
                "stage": stage,
                "input": input_tokens,
                "cached": cached_tokens,
                "output": output_tokens,
                "usd": round(usd, 6),
            }
        )
        return usd

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
        return {
            "cap_usd": self.cap_usd,
            "spent_usd": round(self.spent_usd, 6),
            "remaining_usd": round(self.remaining_usd, 6),
            "calls": len(self.entries),
            "by_stage": by_stage,
        }
