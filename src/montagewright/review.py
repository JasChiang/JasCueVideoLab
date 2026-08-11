"""Look at the finished cut and say whether it is done.

The reviewer may approve. That is the loop's main way of converging: a system
whose critic can only ever ask for another round will keep asking until
something else stops it, and the something else is usually money.

It sees the cut, the brief, and the direction the cut set for itself. It does
not see manifests, ledgers, or degradation tables -- those are for the local
gates that already ran, and handing them over invites a reviewer to audit
paperwork instead of watching the film.

Rounds stop on the first of: approval, the round cap, no progress, or the
budget. Whichever fires, there is a finished cut in hand, because every round
renders before it reviews.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from montagewright.capabilities import describe_limits_for_prompt
from montagewright.gemini import structured_json
from montagewright.cost import BudgetSpent, Ledger
from montagewright.planner import ask
from montagewright.schema import looks_of, move_of_shot, must_be_whole_of, DegradationStep, Issue, ReviewVerdict

from montagewright.planner import MAX_OUTPUT_TOKENS, MODEL_ID

from montagewright.uploads import upload_now

PROMPTS = Path(__file__).resolve().parent / "prompts"
MAX_ROUNDS = 3


def _verdict_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["verdict", "overall", "issues"],
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["approve", "revise"],
                "description": (
                    "revise needs at least one issue at major or blocking. "
                    "Minor notes alone are recorded and do not start another "
                    "render, so a verdict of revise carrying only minor "
                    "issues is rejected -- say approve and leave the minor "
                    "notes, or name what actually has to change."
                ),
            },
            "overall": {"type": "string"},
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["severity", "description", "fix"],
                    "properties": {
                        "clip_id": {"type": "string"},
                        "at_seconds": {
                            "type": "string",
                            "description": (
                                "片子裡大概哪個位置，寫成 MM:SS（`0:12`）。"
                                "大概就好，本機會對到最近的一顆上。"
                                "不要寫成秒數——一分鐘以上的片子，`1:12` "
                                "會變成 112 或 1.12，兩種讀法都落在片長內。"
                            ),
                        },
                        "issue_type": {
                            "type": "string",
                            "enum": [
                                "pacing", "framing", "music_sync", "coverage",
                                "named_fact", "continuity", "audio_content", "other",
                            ],
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["minor", "major", "blocking"],
                        },
                        "description": {"type": "string"},
                        "fix": {
                            "type": "string",
                            "description": (
                                "What to change. A note nobody can act on "
                                "does not start another round."
                            ),
                        },
                    },
                },
            },
        },
    }


@dataclass
class Round:
    index: int
    verdict: ReviewVerdict
    actionable: tuple[str, ...]


@dataclass
class Outcome:
    """Why the loop stopped, and what it left."""

    stopped_because: str
    rounds: list[Round] = field(default_factory=list)
    unadjudicated: list[DegradationStep] = field(default_factory=list)

    @property
    def approved(self) -> bool:
        return bool(self.rounds) and self.rounds[-1].verdict.verdict == "approve"


def review_cut(
    preview: Path,
    *,
    brief: str,
    direction: str,
    client: Any,
    wanted_seconds: float = 0.0,
    delivered_seconds: float = 0.0,
    cache: Any = None,
    ledger: Ledger | None = None,
    model_id: str = MODEL_ID,
) -> ReviewVerdict:
    """One pass over a finished cut.

    The length is given because nothing else was checking it. Three planning
    layers each made a defensible call and delivered 17.9 seconds against 30,
    and the only trace was a number in the report that no stage read. The
    reviewer already judges whether this works as a film; whether it is the
    film that was asked for is the same question.
    """

    if ledger is not None:
        ledger.check()

    instruction = (PROMPTS / "review_zh-TW.txt").read_text(
        encoding="utf-8"
    ).replace("{limits}", describe_limits_for_prompt())
    if wanted_seconds > 0 and delivered_seconds > 0:
        off = delivered_seconds - wanted_seconds
        instruction += (
            f"\n\n## 長度\n\n這支片要 {wanted_seconds:.0f} 秒，"
            f"交出來是 {delivered_seconds:.1f} 秒"
            + (
                f"，差 {off:+.1f} 秒。差得多就是一個要修的問題——"
                "不是把每顆按比例縮放，是有顆不該在裡面，或有顆給得不夠。"
                if abs(off) > max(1.5, wanted_seconds * 0.12)
                else "，在範圍內。"
            )
            + "\n"
        )
    if cache is None:
        uri = upload_now(preview, client).uri
    else:
        uri, _ = cache.uri_for(preview, client, mime_type="video/mp4")

    interaction = ask(
        client,
        model=model_id,
        store=False,
        input=[
        # The video first, the question after it: Google's guidance for a
        # single video is to put the text last.
            {
                "type": "video",
                "mime_type": "video/mp4",
                "uri": uri,
                "resolution": "low",
            },
            {
                "type": "text",
                "text": (
                    f"{instruction}\n\n## 剪輯 brief\n\n{brief}\n\n"
                    f"## 這支片的創意定調\n\n{direction}\n"
                ),
            },
        ],
        generation_config={"thinking_level": "high", "max_output_tokens": MAX_OUTPUT_TOKENS},
        response_format=structured_json(_verdict_schema()),
        ledger=ledger,
        budget_stage="review",
    )
    from montagewright.planner import Usage, _parse

    payload = _parse(interaction, what="review")
    # A point in the finished film, so it comes back as a clock reading and
    # is resolved here rather than by the model that has to write it.
    from montagewright.spans import seconds_of

    for issue in payload.get("issues") or []:
        if isinstance(issue, dict) and "at_seconds" in issue:
            issue["at_seconds"] = seconds_of(issue["at_seconds"])
    return ReviewVerdict.model_validate(payload)


def _shot_schema(clip_ids: list[str]) -> dict[str, Any]:
    """One verdict per shot, bound to the ids that were sent.

    One array of small flat objects. The reasoning that would nest lives in
    `note`, written once per shot, because a clause asked for inside a
    repeated item is what overran two output ceilings before.
    """

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["shots"],
        "properties": {
            "shots": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "clip_id", "delivered", "note", "degradation_verdict"
                    ],
                    "properties": {
                        "clip_id": {"type": "string", "enum": clip_ids},
                        "delivered": {
                            "type": "boolean",
                            "description": (
                                "Whether the shot did what its plan said it "
                                "would."
                            ),
                        },
                        "note": {
                            "type": "string",
                            "description": (
                                "What is on screen, in one sentence. "
                                "'the push stops at the whole handset, never "
                                "reaching the camera module' beats 'poorly "
                                "executed'."
                            ),
                        },
                        "degradation_verdict": {
                            "type": "string",
                            "enum": ["none_recorded", "acceptable", "replan"],
                        },
                    },
                },
            }
        },
    }


def _describe_plan(shot: dict[str, Any], seconds: float, steps: list) -> str:
    """What this shot promised, so the promise can be checked."""

    # The looks themselves, not a move name and one subject. A reviewer
    # checking whether the shot did what it promised needs to know it
    # promised three stops, not that it promised "pan".
    looks = looks_of(shot)
    lines = [
        f"運鏡：{move_of_shot(shot)}",
        "畫面停在：" + "　→　".join(
            f"「{one.at}」（{one.framing}"
            + (f"，停 {one.seconds:g}s" if one.seconds else "")
            + ("，必須完整" if one.must_be_whole else "")
            + "）"
            for one in looks
        ) if looks else "畫面停在：（沒有指定主體）",
        f"長度：{seconds:.2f}s（選片估要 "
        f"{float(shot.get('seconds_needed') or 0):.1f}s）",
        f"為什麼挑這顆：{shot.get('why', '')}",
    ]
    if must_be_whole_of(shot):
        lines.append("這顆的主體被裁掉一部分就失去意義，字要讀得完。")
    for step in steps:
        measured = "、".join(f"{k}={v}" for k, v in step.measured.items())
        lines.append(
            f"降級紀錄：{step.ladder_other or step.ladder} —— "
            f"{step.trigger}（{measured}）"
        )
    return "\n".join(lines)


def review_shots(
    segments: dict[str, Path],
    shots: list[dict[str, Any]],
    *,
    seconds: dict[str, float],
    degradations: list[DegradationStep],
    client: Any,
    cache: Any = None,
    ledger: Ledger | None = None,
    model_id: str = MODEL_ID,
) -> dict[str, dict[str, Any]]:
    """Check each rendered shot against the plan that asked for it.

    A finished cut answers whether this is a film. It does not answer whether
    any one shot came out as planned, and it turns out it cannot: a wordmark
    cropped to "Galaxy Unpac", a pan ending on background wall, a push whose
    first frame is empty -- six in a row survived review of the whole thing,
    every one of them found by opening a single shot. At thirty seconds and
    low resolution the next shot arrives before the fault registers.
    """

    if not segments:
        return {}
    if ledger is not None:
        ledger.check()

    by_clip: dict[str, list[DegradationStep]] = {}
    for step in degradations:
        by_clip.setdefault(step.clip_id, []).append(step)

    instruction = (PROMPTS / "shotreview_zh-TW.txt").read_text(encoding="utf-8")
    sent = [
        (f"k{index:02d}", shot)
        for index, shot in enumerate(shots)
        if f"k{index:02d}" in segments
    ]
    body: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
    for clip_id, shot in sent:
        plan = _describe_plan(
            shot, seconds.get(clip_id, 0.0), by_clip.get(clip_id, [])
        )
        body.append({"type": "text", "text": f"\n## {clip_id}\n\n{plan}\n"})
        path = segments[clip_id]
        if cache is None:
            uri = upload_now(path, client).uri
        else:
            uri, _ = cache.uri_for(path, client, mime_type="video/mp4")
        body.append(
            {
                "type": "video",
                "mime_type": "video/mp4",
                "uri": uri,
                "resolution": "low",
            }
        )

    interaction = ask(
        client,
        model=model_id,
        store=False,
        input=body,
        generation_config={
            "thinking_level": "high",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        response_format=structured_json(
            _shot_schema([clip_id for clip_id, _ in sent])
        ),
        ledger=ledger,
        budget_stage="shot_review",
    )
    from montagewright.planner import Usage, _parse

    payload = _parse(interaction, what="shot review")
    return {entry["clip_id"]: entry for entry in payload.get("shots", [])}


def actionable_keys(verdict: ReviewVerdict) -> tuple[str, ...]:
    """What this round is actually asking to change.

    Keyed so two rounds can be compared. A round that asks for the same things
    as the last one has made no progress, and another render will not change
    that -- which is a stopping condition, not a reason to try harder.
    """

    return tuple(
        sorted(
            f"{issue.clip_id or issue.at_seconds}:{issue.issue_type}"
            for issue in verdict.issues
            if issue.severity in {"major", "blocking"}
        )
    )


def should_continue(
    rounds: list[Round], *, ledger: Ledger | None = None, undelivered: int = 0
) -> tuple[bool, str]:
    """Decide whether another round is worth rendering.

    Two reviewers report here and they answer different questions. The shot
    reviewer watches one shot against its own plan; the film reviewer watches
    the whole cut. An approval from the second used to end the loop before
    the first was consulted, so a run finished with three shots that did not
    do what they said they would, forty-five degradations and five seconds
    missing off the length -- and a verdict of "approve (0 issues)".

    That is not the film reviewer being wrong. It is watching a finished cut
    and cannot see a promise it was never told about; a shot that was meant
    to read a wordmark and instead held on half of one looks like a
    considered composition from the outside. The two verdicts are both
    correct and only one of them was being read.
    """

    if not rounds:
        return True, "not started"
    latest = rounds[-1]

    # The limits come first, because everything below them is a reason to go
    # round again and a run has to be able to stop. Ordering the shot
    # reviewer above the round cap would let a shot nobody can fix spend the
    # budget one replan at a time.
    if len(rounds) >= MAX_ROUNDS:
        return False, f"reached the {MAX_ROUNDS}-round cap"
    if ledger is not None and ledger.remaining_usd <= 0:
        return False, "budget spent"
    if len(rounds) >= 2:
        previous = set(rounds[-2].actionable)
        current = set(latest.actionable)
        if current and current.issubset(previous):
            # Including the case where they are equal: the same complaint
            # twice means the change did not land, and a third render will
            # not make it land either.
            return False, "no progress between rounds"

    if undelivered:
        return True, f"{undelivered} shots did not do what they planned"
    if latest.verdict.verdict == "approve":
        return False, "approved"
    if not latest.actionable:
        return False, "nothing actionable was raised"
    return True, "continuing"


def adjudicate(
    degradations: list[DegradationStep],
    verdict: ReviewVerdict,
    shot_verdicts: dict[str, dict[str, Any]] | None = None,
) -> list[DegradationStep]:
    """Settle each degradation against whoever actually saw the shot.

    This used to rest entirely on the whole-cut reviewer, who never saw the
    shot in question -- they had a thirty-second film and a line of numbers.
    "The subject is 0.88 of frame wide and can show 36% of itself" is not
    judgeable from that; it is judgeable from the shot. So the shot reviewer
    settles it when there is one, and silence from the whole-cut reviewer
    remains the fallback.

    A degradation the whole-cut reviewer raised still goes back for a replan,
    because the way to answer a bad fallback is a different plan rather than
    a better fallback. Anything left unsettled is labelled as such rather
    than quietly shipped.
    """

    shot_verdicts = shot_verdicts or {}
    touched = {issue.clip_id for issue in verdict.issues if issue.clip_id}
    settled: list[DegradationStep] = []
    for step in degradations:
        seen = shot_verdicts.get(step.clip_id, {}).get("degradation_verdict")
        if seen in {"acceptable", "replan"}:
            settled.append(
                step.model_copy(
                    update={
                        "adjudication": (
                            "accept" if seen == "acceptable" else "replan"
                        ),
                        "adjudication_reason": (
                            "the shot reviewer watched this shot: "
                            + shot_verdicts[step.clip_id].get("note", "")
                        )[:300],
                    }
                )
            )
            continue
        if step.clip_id in touched:
            settled.append(
                step.model_copy(
                    update={
                        "adjudication": "replan",
                        "adjudication_reason": "the reviewer raised this shot",
                    }
                )
            )
        else:
            settled.append(
                step.model_copy(
                    update={
                        "adjudication": "accept",
                        "adjudication_reason": (
                            "the reviewer watched the cut and did not raise it"
                        ),
                    }
                )
            )
    return settled
