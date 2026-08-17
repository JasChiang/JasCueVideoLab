"""Per-asset facts, written once and reused forever.

A card describes one clip in isolation: what is in it, where the subjects sit,
how the frame is composed, whether the take failed. Everything here is true
without knowing what the clip will be used for, which is exactly why the brief
is not an input. A card written against one brief would have to be rewritten
for the next deliverable made from the same shoot; a card written against the
material alone is good until the material changes.

Subject boxes live here for the same reason. They are expensive to compute,
they never change, and asking for them once per render instead means paying
for the same answer on every rhythm tweak, every second aspect, and every
review round -- eleven calls a run for something that was already known.

What a card deliberately cannot carry is anything comparative or sequential.
Whether this take beats the other two, whether it serves the brief, how long
it should hold: none of that is visible from inside one clip.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from montagewright.planner import MAX_OUTPUT_TOKENS, ask
from montagewright.gemini import structured_json
from montagewright.spans import seconds_of

from montagewright.uploads import upload_now

PROMPTS = Path(__file__).resolve().parent / "prompts"

def _card_version() -> str:
    """A version that changes when the card's shape does.

    Cards are keyed by the bytes they describe, so adding a field does not
    invalidate them -- only the version does. Two required fields were added
    without touching it, so every cached card stayed and the fields were
    silently absent from every listing that had asked for them.

    Deriving it from the required fields removes the step somebody has to
    remember. Adding, removing or renaming one rewrites the library on the
    next run; changing only a description does not, which is right, because
    a description change does not make an old card wrong.
    """

    import hashlib
    import json

    # The whole schema and the prompt that goes with it, not the top-level
    # required names. Those caught a field being added and nothing else:
    # segments could change shape, an enum could gain a value, a unit could
    # flip from seconds to a clock reading, and every cached card stayed
    # valid while meaning something different. All of those happened.
    # The measurement is an input to the card as much as the prompt is. When
    # the same clip started being described as "not a shift" rather than
    # "unmeasurable", every cached card held an answer to a question that is
    # no longer asked -- and cards outlive runs, so it would never have
    # resurfaced on its own.
    from montagewright.motion import READING

    shape = json.dumps(card_schema(), sort_keys=True, ensure_ascii=False)
    prompt = (PROMPTS / "clipcard_zh-TW.txt").read_text(encoding="utf-8")
    said = hashlib.sha256((shape + prompt + READING).encode("utf-8"))
    return f"montagewright-clip-card-{said.hexdigest()[:8]}"


CARD_VERSION = ""  # set below, once card_schema is defined


class CardLibraryEmpty(RuntimeError):
    """Every clip failed to describe, so there is nothing to plan from."""


@dataclass(frozen=True)
class Beat:
    """One movement in a clip, and when it runs."""

    beat_id: str
    what: str
    starts_seconds: float
    ends_seconds: float


_NEAREST_ACTION = object()


def action_beats(card: dict[str, Any]) -> list[Beat]:
    beats: list[Beat] = []
    for entry in card.get("action", []) or []:
        # Clock readings, like the segments above them. The card asked for
        # bare seconds and repaired the collisions afterwards, which worked
        # and left the same card carrying two notations -- one where `1:53`
        # is unwritable and one where it has to be guessed at.
        start = seconds_of(entry.get("from"))
        end = seconds_of(entry.get("to"))
        if start is None or end is None:
            continue
        # A span shorter than a few frames is not a span. Twenty-six of
        # forty-one beats in one library came back under a quarter of a
        # second -- "the models rotate their phones, 0.02 to 0.06s" -- and
        # every consumer downstream treated them as real: in-points were
        # snapped onto them and the rhythm pass was shown them as how long
        # the action takes. A guessed timestamp is worse than none.
        if end - start >= 0.25:
            beats.append(
                Beat(
                    str(entry.get("id", "") or f"a{len(beats):02d}"),
                    str(entry.get("what", "")),
                    start,
                    end,
                )
            )
    return sorted(beats, key=lambda beat: beat.starts_seconds)


def snap_to_action(
    card: dict[str, Any],
    wanted_start: float,
    duration: float,
    *,
    action_id: object = _NEAREST_ACTION,
    within: tuple[float, float] | None = None,
    focus: "list[Any] | None" = None,
) -> tuple[float, str | None]:
    """Compatibility wrapper around the contract-producing action snap."""

    start, _contract, note = snap_to_action_contract(
        card, wanted_start, duration, action_id=action_id,
        within=within, focus=focus
    )
    return start, note


def snap_to_action_contract(
    card: dict[str, Any],
    wanted_start: float,
    duration: float,
    *,
    action_id: object = _NEAREST_ACTION,
    within: tuple[float, float] | None = None,
    focus: "list[Any] | None" = None,
) -> "tuple[float, Any | None, str | None]":
    """Resolve one explicitly selected action into a completion contract.

    A nearby action is descriptive card evidence, not an instruction to show
    it. New planning callers pass ``action_id`` explicitly; ``none`` leaves a
    static/detail shot untouched. Omitting the keyword retains only the legacy
    helper behaviour for old callers and cached tests.

    `within` is the stretch the card said was worth cutting into. Actions are
    recorded across the whole take, including the parts nobody should use --
    the camera being repositioned is a movement, and so is somebody walking
    in to reset a prop -- so without this the correction that exists to land
    a cut on a gesture could land it on the wrong side of the take's own
    boundary, and moved it there on purpose.
    """

    from montagewright.schema import ActionContract

    legacy_nearest = action_id is _NEAREST_ACTION
    selected = str(action_id or "none").strip()
    if not legacy_nearest and selected in {"", "none"}:
        return wanted_start, None, None
    # The id is scoped by the shot's selected source; the caller validates that
    # pairing before this card resolves it.
    local_id = selected.rsplit(":", 1)[-1]
    beats = action_beats(card)
    if not legacy_nearest:
        beats = [beat for beat in beats if beat.beat_id == local_id]
    if within is not None:
        first, last = within
        beats = [
            beat for beat in beats
            if beat.starts_seconds >= first - 1e-6
            and beat.ends_seconds <= last + 1e-6
            and (
                # The requested window may already contain the whole action.
                # In that case keep its earlier context instead of snapping
                # to the action start and pushing the out-point past the span.
                (
                    wanted_start >= first - 1e-6
                    and wanted_start + duration <= last + 1e-6
                    and beat.starts_seconds >= wanted_start - 1e-6
                    and beat.ends_seconds <= wanted_start + duration + 1e-6
                )
                # Otherwise snapping to the action start is safe only when
                # both the requested hold and real completion still fit.
                or max(
                    beat.starts_seconds + duration, beat.ends_seconds
                ) <= last + 1e-6
            )
        ]
    if not beats:
        return wanted_start, None, None

    nearest = (
        min(beats, key=lambda beat: abs(beat.starts_seconds - wanted_start))
        if legacy_nearest else beats[0]
    )
    already_contained = (
        not legacy_nearest
        and nearest.starts_seconds >= wanted_start - 1e-6
        and nearest.ends_seconds <= wanted_start + duration + 1e-6
    )
    resolved_start = wanted_start if already_contained else nearest.starts_seconds
    drift = resolved_start - wanted_start
    if legacy_nearest and abs(drift) > max(0.5, duration / 2.0):
        return wanted_start, None, None
    # Landing on the gesture is worth moving for; landing on the gesture
    # while the lens is still hunting is not. A cut that was planned on
    # sharp footage stays where it was planned rather than being dragged
    # into the soft part -- the take that prompted this had its in-point
    # pushed a second earlier onto a heart gesture, from 80% of its own
    # best focus down to 33%, with the sharp half inside the same span.
    if focus:
        from montagewright.focus import moving_into_softer

        if moving_into_softer(
            focus, wanted_start, resolved_start, duration
        ):
            return wanted_start, None, None
    contract = ActionContract(
        action_id=nearest.beat_id,
        what=nearest.what,
        source_start_seconds=nearest.starts_seconds,
        source_complete_seconds=nearest.ends_seconds,
        # Coarse MM:SS cannot prove a sub-second settle.  Keep completion as
        # the safe boundary until a decoded-PTS refinement supplies one.
        safe_cut_after_seconds=nearest.ends_seconds,
        completion_policy="must_complete",
        timing_basis="coarse_mmss",
    )
    return resolved_start, contract, (
        f"moved {drift:+.2f}s onto '{nearest.what}'" if abs(drift) > 0.05 else None
    )


@dataclass(frozen=True)
class SubjectBox:
    """One nameable thing in the frame, with where it sits."""

    label: str
    centre_x: float
    centre_y: float
    width: float
    height: float
    moves: bool
    entity_id: str | None = None
    # When this position was true. A box is a moment, and a moment is the
    # whole answer only for a locked-off frame.
    at_seconds: float = 0.0

    @property
    def is_horizontal(self) -> bool:
        return self.width > self.height


def card_schema() -> dict[str, Any]:
    """Flat, because a nested one was what made the old plan unservable."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "summary",
            "usable",
            "composition",
            "subjects",
            "action",
            "camera_motion",
            "shot_size",
            "facing",
            "speech",
            "segments",
            "needs",
        ],
        "properties": {
            "summary": {
                "type": "string",
                "description": "What happens in this clip, in a line or two.",
            },
            "usable": {
                "type": "boolean",
                "description": (
                    "False only for a take that failed outright -- an action "
                    "cut off, a screen that slept, a misfire. Handheld "
                    "movement, soft focus and unusual framing are styles "
                    "available to an edit, not defects."
                ),
            },
            "unusable_reason": {"type": "string"},
            "composition": {
                "type": "string",
                "enum": ["horizontal", "vertical", "square", "mixed"],
                "description": (
                    "How the content is laid out in the frame. A row of "
                    "handsets across a table is horizontal and will fight a "
                    "vertical crop; a standing person is vertical and will "
                    "sit in one happily. Recorded here so the edit knows "
                    "before it commits an aspect."
                ),
            },
            "segments": {
                "type": "array",
                "minItems": 1,
                "description": (
                    "把這支素材從頭到尾切成連續、不重疊的片段，每一段說它"
                    "能不能剪進片子。第一段從 0:00 開始，最後一段到素材"
                    "結尾，中間首尾相接不留空隙。\n"
                    "這裡取代了「可用起點／可用終點」那一組數字，因為一組"
                    "起訖只能說「前面不能用、中間可以、後面不能用」，而毛"
                    "片常見的樣子是：晃動 → 一次好的 → 有人喊卡 → 重新"
                    "構圖 → 又一次好的 → 收器材。那是兩座可用的島，中間"
                    "隔著不能用的水，一組起訖表達不出來——只能把喊卡那段"
                    "包進去，或者丟掉一個好的。\n"
                    "整支從頭到尾都能用，就給一段涵蓋全部。"
                ),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["from", "to", "status", "why", "motion_role"],
                    "properties": {
                        "from": {
                            "type": "string",
                            "description": (
                                "這一段從哪裡開始，寫成 MM:SS 或 MM:SS，"
                                "例如 0:00、1:07。**不要寫成小數秒。**"
                                "你看到的影片時間就是這個格式，換算成 73.5 "
                                "這種數字反而會出錯——之前 1:10 被寫成 110、"
                                "1:53 被寫成 1.53，兩支超過一分鐘的素材都錯了。"
                                "而且影片是一秒一格給你的，你本來就分辨不到"
                                "比一秒更細，寫小數只是在編造精度。"
                            ),
                        },
                        "to": {
                            "type": "string",
                            "description": "這一段到哪裡結束，同樣是 MM:SS。",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["eligible", "reject"],
                            "description": (
                                "`eligible`：這一段本身成立，可以剪進片子。"
                                "`reject`：不能用——動作做到一半中斷、有人"
                                "說重來、攝影機還在甩或還在對焦、工作人員"
                                "入鏡收東西、講到一半停下來。\n"
                                "判斷的是**這一段本身**成不成立，不是它好不"
                                "好看，也不是別的 take 有沒有比它好——那是"
                                "後面比較過所有素材才能回答的問題。手持晃"
                                "動、淺景深、構圖不完整都是風格，不是缺陷。"
                            ),
                        },
                        "why": {
                            "type": "string",
                            "description": (
                                "reject 的說明為什麼不能用；eligible 說這一"
                                "段裡發生了什麼。兩種都要寫，因為後面挑片"
                                "的人只看得到這句話。"
                            ),
                        },
                        "motion_role": {
                            "type": "string",
                            "enum": [
                                "locked", "authored", "subject_follow",
                                "handheld_texture", "setup_reframe",
                                "disturbance", "unknown",
                            ],
                            "description": (
                                "這一段的攝影機運動是什麼意思。上面附了本機"
                                "量到的位移區間——**位移有沒有發生是量出來的，"
                                "不用你判斷；你要判斷的是它為什麼在動。**\n"
                                "`locked`：沒有位移。\n"
                                "`authored`：刻意的運鏡，移動本身帶出新東西"
                                "——揭示畫面外的第二樣東西、沿著產品看過去、"
                                "推進到細節。**觀眾因為這個移動多看到了什麼**"
                                "就是判準。\n"
                                "`subject_follow`：相機在跟住一個會動的主體，"
                                "構圖大致維持。\n"
                                "`handheld_texture`：整段一致的小幅手持感，"
                                "那是質感不是缺陷。\n"
                                "`setup_reframe`：攝影機從一個還沒完成的構圖"
                                "移到準備好的構圖，或從完成的構圖離開。"
                                "**觀眾沒有因此多看到任何東西**——那是拍攝"
                                "準備，不是內容。\n"
                                "`disturbance`：碰撞、突發甩動、失控後恢復。\n"
                                "`unknown`：證據不足以判斷。\n"
                                "`setup_reframe` 跟 `disturbance` 的片段後面"
                                "選片會拿不到，所以那兩個等於「這段剪不進去」。"
                                "拿不準就填 `unknown`，不要猜。"
                            ),
                        },
                    },
                },
            },
            "needs": {
                "type": "array",
                "description": (
                    "這支素材要用的話需要什麼處理。這是你看過畫面之後的"
                    "理解，不是規則：主體太小就要推近，重點偏在一側就要"
                    "裁切重新構圖，內容橫向鋪開就要橫搖帶過，"
                    "只有一段可用就要修剪。什麼都不需要就留空。"
                ),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["what", "why"],
                    "properties": {
                        "what": {
                            "type": "string",
                            "enum": ["trim", "crop", "zoom", "pan", "tilt"],
                        },
                        "why": {
                            "type": "string",
                            "description": (
                                "用畫面上看得到的東西說明。「主體只佔畫面"
                                "一小塊，直式輸出要推近才看得清楚」"
                                "比「需要 zoom」有用。"
                            ),
                        },
                    },
                },
            },
            "action": {
                "type": "array",
                "description": (
                    "Where things actually happen in this clip. An editor "
                    "cuts on action -- into a gesture as it begins, out as it "
                    "completes -- and a cut placed by arithmetic lands "
                    "mid-movement, which reads as a mistake even to someone "
                    "who cannot say why. Empty for a static shot."
                ),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "what", "from", "to"],
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": (
                                "這個動作的短名字，`a01`、`a02` 這樣照順序編。"
                                "後面排節奏的時候要用它指名『這個動作要落在"
                                "哪個拍點上』——一段描述指不準，一個 id 可以。"
                            ),
                        },
                        "what": {
                            "type": "string",
                            "description": (
                                "The movement, in a few words: 'the hand "
                                "reaches in', 'the phone opens', 'the watch "
                                "is lowered'."
                            ),
                        },
                        "from": {
                            "type": "string",
                            "description": (
                                "動作開始的那一刻，寫成 MM:SS——手還沒碰到"
                                "之前、東西還沒開始動之前。跟上面的片段一樣"
                                "用時鐘寫法，不要寫小數秒。"
                            ),
                        },
                        "to": {
                            "type": "string",
                            "description": (
                                "動作完成的那一秒。這是一段時間，不是一個"
                                "瞬間：起訖相同或只差零點幾秒，等於沒有指出"
                                "任何東西，後面就沒辦法把畫面切在動作上。"
                                "看不出來動作在哪裡結束，就不要寫這一筆——"
                                "留空比給一個猜的時間有用。"
                            ),
                        },
                    },
                },
            },
            "shot_size": {
                "type": "string",
                "enum": ["wide", "medium", "close", "extreme_close"],
                "description": (
                    "主體在畫面裡佔多大。`wide`：主體連同它所在的環境，"
                    "人是全身或更遠。`medium`：主體是畫面主角但還看得到周圍，"
                    "人約半身。`close`：主體填滿大部分畫面，人是肩上。"
                    "`extreme_close`：只有一個局部——鏡頭模組、鉸鏈、"
                    "螢幕上的一個數字、眼睛。\n"
                    "這是剪接會用到的事實：兩顆景別太接近接在一起會跳，"
                    "而一段戲通常要從遠往近推進。看的是主體佔畫面的比例，"
                    "不是攝影機離它多遠。"
                ),
            },
            "facing": {
                "type": "string",
                "enum": ["left", "right", "toward", "away", "flat"],
                "description": (
                    "主體朝向或移動的方向，從觀眾的角度看。`left`／`right`："
                    "人面向那一側、或東西往那一側走。`toward`：朝鏡頭來。"
                    "`away`：離鏡頭去。`flat`：正對鏡頭、對稱擺放、"
                    "或看不出方向。\n"
                    "這是銀幕方向：兩顆都朝右的對談鏡頭接在一起，"
                    "觀眾會以為兩個人在對同一邊說話；一個往右走的東西"
                    "下一顆變成往左，會讀成它掉頭了。這件事只有看畫面"
                    "才知道，量不出來。"
                ),
            },
            "speech": {
                "type": "string",
                "enum": ["none", "ambient", "content"],
                "description": (
                    "這支素材裡的說話是不是內容本身。`content`：訪談、"
                    "受訪者的回答、對鏡頭講話、旁白——把聲音拿掉這顆就"
                    "沒有意義了。`ambient`：現場環境音、旁邊路人的交談、"
                    "聽不清楚的背景人聲，剪掉不影響。`none`：沒有人聲。"
                    "填 content 的素材後面才會去做逐字稿，那要花錢也花"
                    "時間，所以不確定的時候看的是「這顆的意思靠不靠聲音"
                    "成立」，不是「有沒有人在講話」。"
                ),
            },
            "camera_motion": {
                "type": "string",
                "description": (
                    "攝影機怎麼動，以及動了之後畫面裡多了什麼、少了什麼。"
                    "「往右平移，右邊會有第三台白色手機進畫面」、"
                    "「緩慢推近到機身鉸鏈」、「繞著產品轉，背面轉出來」。"
                    "這決定的是要不要再加一層數位運鏡：素材自己會把東西"
                    "帶進來時，框住不動讓它演完就好，兩個運鏡疊在一起會"
                    "打架。攝影機不動就留空。"
                ),
            },
            "subjects": {
                "type": "array",
                "description": (
                    "The things worth framing, described so each can be told "
                    "from anything similar beside it, with where it sits."
                ),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "label",
                        "centre_x",
                        "centre_y",
                        "width",
                        "height",
                        "moves",
                        "seen_at",
                    ],
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": (
                                "Stable id supplied by an approved grounding "
                                "spec. Leave absent for an ordinary descriptive "
                                "subject; never invent one."
                            ),
                        },
                        "label": {
                            "type": "string",
                            "description": (
                                "'the left, grey handset', not 'the handset'."
                            ),
                        },
                        "centre_x": {
                            "type": "number",
                            "description": (
                                "Fraction of frame width, 0.0 left to 1.0 "
                                "right. Never pixels."
                            ),
                        },
                        "centre_y": {"type": "number"},
                        "width": {"type": "number"},
                        "height": {"type": "number"},
                        "seen_at": {
                            "type": "string",
                            "description": (
                                "你是看第幾秒說出這個位置的，寫成 MM:SS。"
                                "攝影機或主體"
                                "在動的時候，位置只在那一刻成立——後面要拿"
                                "這個框去框別的時間點，得先知道它是什麼時候"
                                "量的。"
                            ),
                        },
                        "moves": {
                            "type": "boolean",
                            "description": (
                                "Whether this subject moves within the shot."
                            ),
                        },
                    },
                },
            },
        },
    }


def load_card(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if payload.get("version") == CARD_VERSION else None


def save_card(path: Path, card: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({**card, "version": CARD_VERSION}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


def subjects_from_card(card: dict[str, Any]) -> list[SubjectBox]:
    boxes: list[SubjectBox] = []
    for entry in card.get("subjects", []):
        try:
            boxes.append(
                SubjectBox(
                    label=str(entry["label"]),
                    entity_id=(
                        str(entry["entity_id"])
                        if entry.get("entity_id") else None
                    ),
                    centre_x=float(entry["centre_x"]),
                    centre_y=float(entry["centre_y"]),
                    width=float(entry["width"]),
                    height=float(entry["height"]),
                    moves=bool(entry.get("moves", False)),
                    at_seconds=seconds_of(entry.get("seen_at")) or 0.0,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return boxes


def find_subject(
    card: dict[str, Any], description: str, *, entity_id: str | None = None
) -> SubjectBox | None:
    """Match a planner's subject description to a box the card already holds.

    Exact wording will not match, so this looks for the card's label inside
    the description or the other way round. A miss returns None and the caller
    grounds the subject the expensive way -- a card that cannot answer is not
    a reason to fail, only a reason to pay.
    """

    boxes = subjects_from_card(card)
    if entity_id is not None:
        matched = [box for box in boxes if box.entity_id == entity_id]
        if len(matched) == 1:
            return matched[0]
        if matched:
            boxes = matched
        # Legacy cards predate grounding identities and therefore have no
        # entity_id on otherwise useful measured boxes.  Falling through to
        # the description keeps those local coordinates available; identity
        # authority still comes from exact grounding, never from this match.
    lowered = description.lower()
    exactish = []
    for box in boxes:
        label = box.label.lower()
        if label and (label in lowered or lowered in label):
            exactish.append(box)
    if len(exactish) == 1:
        return exactish[0]
    if exactish:
        return None
    # Selection may describe a measured box in the brief's language while a
    # reusable card was written in another one. Relative composition remains
    # unambiguous across languages, and is geometry rather than identity.
    positional = {
        "left": ("left", "左"),
        "centre": ("middle", "center", "centre", "中間", "中央"),
        "right": ("right", "右"),
    }
    requested = [
        name for name, words in positional.items()
        if any(word in lowered for word in words)
    ]
    if len(requested) == 1 and boxes:
        name = requested[0]
        ranked = (
            sorted(boxes, key=lambda box: box.centre_x)
            if name == "left"
            else sorted(boxes, key=lambda box: box.centre_x, reverse=True)
            if name == "right"
            else sorted(boxes, key=lambda box: abs(box.centre_x - 0.5))
        )
        if len(ranked) == 1:
            return ranked[0]
        first_distance = (
            ranked[0].centre_x if name == "left"
            else 1.0 - ranked[0].centre_x if name == "right"
            else abs(ranked[0].centre_x - 0.5)
        )
        second_distance = (
            ranked[1].centre_x if name == "left"
            else 1.0 - ranked[1].centre_x if name == "right"
            else abs(ranked[1].centre_x - 0.5)
        )
        if second_distance - first_distance > 0.05:
            return ranked[0]
    # Fall back only when one candidate shares a distinguishing word. A
    # common token matching two boxes is ambiguity, not permission to pick
    # whichever the card happened to list first.
    candidates = []
    for box in boxes:
        for word in box.label.split():
            if len(word) >= 3 and word.casefold() in lowered:
                candidates.append(box)
                break
    return candidates[0] if len(candidates) == 1 else None


def clip_seconds(path: Path) -> float:
    """How long this clip is, measured rather than asked about."""

    import subprocess

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


# How far past the measured end a timestamp may land and still be believed.
#
# Gemini samples video at one frame a second, so what it can say is roughly
# integer seconds -- on a clip lasting 12.012s the last frame it holds is at
# 12, and "the action ends at 13" is a rounding artefact rather than a
# mistake. A tolerance of half a second, which is what this had first, threw
# that away and deleted a real action.
#
# There is a lot of room to be generous here: the smallest possible MM:SS
# collision is 1:01 written as 101 on a clip just past a minute, which
# overshoots by forty seconds. Anything between one second and forty is not a
# notation problem, and nothing has produced one yet.
#
# The proxy is also about 0.1s longer than the file it was made from, since
# re-encoding pads the tail. Cards describe the proxy and the edit cuts the
# original, so a usable_to taken at face value can sit a frame past the end
# of the source. The renderer clamps there; it is noted here because this is
# where the two clocks are closest to being confused for each other.
SLOP = 1.5


def _as_mmss(value: float) -> float | None:
    """Read a number back as the MM:SS it was probably written from.

    `1:53` comes back either as `1.53` or as `153`, depending on whether the
    colon became a decimal point or vanished. Both are recoverable, and both
    are only worth trying when the plain reading has already failed.
    """

    for minutes, seconds in (
        (int(value), round((value - int(value)) * 100)),   # 1.53 -> 1m 53s
        (int(value) // 100, int(value) % 100),             # 110  -> 1m 10s
    ):
        if 0 < minutes < 60 and 0 <= seconds < 60:
            return minutes * 60 + seconds
    return None


def times_on_receipt(card: dict[str, Any], duration: float) -> dict[str, Any]:
    """Put the card's seconds back on a clock that matches the file.

    Gemini reads video in MM:SS, and the card asks for plain seconds -- so on
    any clip past a minute the two notations collide. Both clips over a minute
    in this library came back wrong, in opposite directions: a 71.1s take said
    `usable_to: 110.0`, which is 1:10 with the colon dropped, and a 113.4s take
    said `usable_to: 1.53`, which is 1:53 with the colon turned into a decimal
    point. The second is the dangerous one, because 1.53 is smaller than the
    duration and so passes every range check while claiming a two-minute take
    is usable for a second and a half.

    This is the same shape as the subject boxes arriving in 0..1000 space
    whatever the field says, and it gets the same treatment: the model answers
    in its own units, and the conversion happens here, against a duration that
    was measured locally rather than asked for.

    Anything still out of range afterwards is dropped rather than clamped. A
    missing action is a static shot, which is a fine thing to be; an action at
    a wrong second puts a cut in the wrong place.
    """

    if duration <= 0:
        return card

    def readings(value: Any) -> list[float]:
        """Every way of reading this number that lands inside the clip.

        Plain seconds first, so an unambiguous number keeps its obvious
        meaning and only a number that cannot be what it says gets reread.
        """

        try:
            plain = float(value)
        except (TypeError, ValueError):
            return []
        out = []
        if 0 <= plain <= duration + SLOP:
            out.append(min(plain, duration))
        again = _as_mmss(plain)
        if again is not None and again <= duration + SLOP and again not in out:
            out.append(min(again, duration))
        return out

    def span(
        raw_start: Any, raw_end: Any, least: float
    ) -> tuple[float, float] | None:
        """The start and end of one interval, read the same way at both ends.

        A span is what makes the notation visible: 1.1 and 1.13 are both
        readable as plain seconds, and read that way they describe a
        thirty-millisecond action, which is not a thing that happens. Read as
        MM:SS they are 1:10 to 1:13, which is. So the readings are scored
        together, mixed ones are penalised, and a degenerate span loses to a
        real one -- but if the plain reading is the only one available it is
        kept whatever its length, because a genuinely short window is the
        model's to report.
        """

        starts = readings(raw_start) or [0.0]
        ends = readings(raw_end)
        pairs = [
            ((i != j, max(i, j), i + j), s, e)
            for i, s in enumerate(starts)
            for j, e in enumerate(ends)
            if e > s
        ]
        if not pairs:
            return None
        real = [one for one in pairs if one[2] - one[1] >= least]
        return min(real or pairs)[1:]

    # Segments are read rather than repaired. They are asked for as clock
    # readings, so `1:53` has one meaning and the notation this whole
    # function exists to undo cannot be written down. Everything below it is
    # still a bare number and still needs the guesswork.
    edges: list[dict[str, Any]] = []
    for entry in card.get("segments") or []:
        first = seconds_of(entry.get("from"))
        last = seconds_of(entry.get("to"))
        if first is None or last is None:
            continue
        first = max(0.0, min(first, duration))
        last = max(0.0, min(last, duration))
        if last <= first:
            continue
        edges.append(dict(entry, **{
            "from": round(first, 3), "to": round(last, 3),
        }))
    card["segments"] = edges or [{
        "from": 0.0, "to": round(duration, 3), "status": "eligible",
        "why": "no segments were written, so the whole take stands",
    }]

    # Actions and subject moments are clock readings too now, so the whole
    # guessing apparatus above is gone -- what is left is range checking. A
    # beat outside the clip is dropped rather than clamped: a missing action
    # is a static shot, which is a fine thing to be, while an action at a
    # wrong second puts a cut in the wrong place.
    kept = []
    for beat in card.get("action") or []:
        start = seconds_of(beat.get("from"))
        end = seconds_of(beat.get("to"))
        if start is None or end is None:
            continue
        if not (0 <= start < end <= duration + SLOP):
            continue
        # The same floor `action_beats` uses. Below it a beat cannot place a
        # cut anyway, so there is nothing to preserve by keeping it.
        if end - start < 0.25:
            continue
        kept.append(dict(beat, **{
            "from": round(start, 3), "to": round(min(end, duration), 3),
        }))
    card["action"] = kept

    for subject in card.get("subjects") or []:
        seen = seconds_of(subject.get("seen_at"))
        subject["seen_at"] = round(
            seen if seen is not None and 0 <= seen <= duration + SLOP else 0.0, 3
        )
    return card


CARD_VERSION = _card_version()


def describe_clip(
    proxy: Path,
    *,
    client,
    cache=None,
    model_id: str | None = None,
    thinking: str = "low",
    motion: "list[Any] | None" = None,
    ledger: Any | None = None,
) -> tuple[dict[str, Any], Any]:
    """Watch one clip and write its card.

    Called once per asset, ever. The result is content-addressed, so a card
    survives every rerun, every second aspect and every review round -- which
    is the whole reason the subject boxes belong here rather than being
    grounded again on each render.
    """

    from montagewright.planner import MODEL_ID, Usage, _parse, _to_frame_fractions

    prompts = Path(__file__).resolve().parent / "prompts"
    instruction = (prompts / "clipcard_zh-TW.txt").read_text(encoding="utf-8")

    # The card asks for "the whole length" without ever saying what it is, so
    # the total is stated here rather than guessed at.
    #
    # This used to end by demanding plain seconds -- "1 分 53 秒要寫 113.0" --
    # which was right when the card took numbers and became the exact
    # opposite of the truth the day it took clock readings. It is appended
    # after the prompt file, so it was the last thing the model read, and a
    # model that obeyed it would have had every segment refused by
    # `seconds_of` (a decimal with no colon is two readings, so it is not
    # taken) and the card would have fallen back to "the whole take stands".
    # Segmentation would have switched itself off in silence. It did not
    # happen because the field descriptions won, which is luck rather than
    # design.
    duration = clip_seconds(proxy)
    if duration > 0:
        minutes = int(duration) // 60
        instruction += (
            f"\n\n## 這支素材的長度\n\n"
            f"{minutes}:{duration - 60 * minutes:04.1f}"
            f"（{duration:.1f} 秒）。\n"
            f"時間一律寫成 MM:SS，最後一段要到這裡為止。\n"
        )
        # The half it cannot see. Video reaches it at a frame a second, and
        # camera shake is what happens between frames, so "is this stable"
        # is not a question the picture it was given can answer -- it
        # answered "固定鏡頭，畫面穩定清晰" about a take whose first three
        # seconds are the operator still finding the frame. Measured here
        # and handed over as intervals with ids, so the model is asked what
        # the movement means and never whether it happened.
        if motion:
            from montagewright.motion import describe as describe_motion

            instruction += f"\n\n## 攝影機運動\n\n{describe_motion(motion)}\n"

    if cache is None:
        uploaded = upload_now(proxy, client)
        uri = uploaded.uri
    else:
        uri, _ = cache.uri_for(proxy, client, mime_type="video/mp4")

    interaction = ask(
        client,
        upload_cache=cache,
        model=model_id or MODEL_ID,
        store=False,
        # The video first, the question after it. Google's own guidance for
        # a single video is to put the text last, and this had it the other
        # way round since the card writer was first written.
        input=[
            {
                "type": "video",
                "mime_type": "video/mp4",
                "uri": uri,
                "resolution": "low",
            },
            {"type": "text", "text": instruction},
        ],
        # One short clip. Anything past this is not a slow answer, it is a
        # call that has stopped coming back -- and seventy-four of these run
        # in a row, so the cost of finding that out late is paid once per
        # clip. The library is written as it goes, so the retry after a
        # timeout starts from the one that hung.
        patience_seconds=300.0,
        generation_config={
            # Describing what is in a frame is recognition and needs little
            # deliberation. Deciding whether a take failed -- whether an
            # action finished, whether "again" was a line or an instruction --
            # is a judgement, and would want more. Parameterised so the two
            # can be measured against each other rather than argued about.
            "thinking_level": thinking,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        response_format=structured_json(card_schema()),
        ledger=ledger,
        budget_stage="clip_cards",
    )
    card = _parse(interaction, what="clip card")
    # The model answers boxes in its native 0..1000 space for some clips
    # whatever the field says, so the conversion happens on receipt.
    card["subjects"] = _to_frame_fractions(card.get("subjects", []))
    card = times_on_receipt(card, duration)
    return card, Usage.from_interaction(interaction)


def card_map(proxies: Path, card_dir: Path) -> dict[str, Path]:
    """Which card describes which source, rebuilt after the fact.

    `build_library` hands this map back when it writes the cards, and anything
    asking for it later has to derive it the same way -- a card is named for
    the bytes it describes, not for the source those bytes came from. Two
    callers took the filename to be the source id instead. Every lookup
    missed, so every shot was reframed with no subject to aim at, so every
    crop sat dead centre while the report went on describing the subject it
    had followed. It looked like a working cut. That is the whole reason this
    is a function and not two dict comprehensions.
    """

    from montagewright.uploads import content_hash

    found: dict[str, Path] = {}
    for proxy in sorted(proxies.glob("*.mp4")):
        card = card_dir / f"{content_hash(proxy)[:20]}.json"
        if card.exists():
            found[proxy.stem] = card
    return found


def build_library(
    proxies: dict[str, Path],
    card_dir: Path,
    *,
    client,
    cache=None,
    model_id: str | None = None,
    progress=None,
    motion_of=None,
    ledger=None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Write a card for every asset that does not already have one.

    Resumable by construction: an existing card of the current version is left
    alone. A run interrupted halfway costs nothing to restart, which matters
    when the alternative is seventy-four paid calls.
    """

    card_dir.mkdir(parents=True, exist_ok=True)
    # Named by what they describe, not by where the file happened to sit. A
    # card is only worth caching because it stays true, and it stays true for
    # the same bytes under any name in any folder -- keeping them per run
    # meant "content-addressed" and "written once per output directory" at
    # the same time, so a second cut of the same rushes rewrote all of them.
    from montagewright.cost import BudgetSpent
    from montagewright.uploads import content_hash
    paths: dict[str, Path] = {}
    stats: dict[str, Any] = {
        "written": 0, "reused": 0, "failed": 0, "input": 0, "output": 0,
    }
    failures: list[str] = []

    total = len(proxies)
    for index, (source_id, proxy) in enumerate(sorted(proxies.items()), start=1):
        destination = card_dir / f"{content_hash(proxy)[:20]}.json"
        if load_card(destination) is not None:
            paths[source_id] = destination
            stats["reused"] += 1
            continue
        try:
            card, usage = describe_clip(
                proxy, client=client, cache=cache, model_id=model_id,
                motion=motion_of(source_id) if motion_of else None,
                ledger=ledger,
                # Cutting a take into what survives and what does not is a
                # judgement -- whether an action finished, whether "again"
                # was a line or an instruction -- and low turns out to mean
                # off: measured across five clips it produced exactly zero
                # thought tokens. Medium is 795 to 1,398 each, roughly flat
                # against duration, which put the library at $0.91 instead
                # of $0.43. Once per batch of rushes, since cards are
                # content addressed.
                thinking="medium",
            )
        except BudgetSpent:
            # Not a fact about this clip. The money ran out, so every clip
            # after it fails for the same reason, and none of those failures
            # says anything about the material. Swallowing them produced a
            # library covering twenty-one of seventy-four rushes that looked
            # exactly like a library of rushes where fifty-three were
            # unreadable -- and the run went on to freeze that inventory as
            # revision zero of the planning authority, which the next run
            # (after the credits were topped up) could no longer agree with.
            # Stop here instead: what is described is cached and costs
            # nothing to resume.
            raise
        except Exception as error:
            # One unreadable clip is not a reason to abandon the library.
            # Losing why it failed is a different matter: a NameError in the
            # request took all seventy-four down and reported it as the
            # routine "74 failed ($0.0000)" line, and the run went on to
            # plan a film from an empty library.
            failures.append(f"{source_id}: {type(error).__name__}: {error}")
            stats["failed"] += 1
            continue
        save_card(destination, card)
        paths[source_id] = destination
        stats["written"] += 1
        stats["input"] += usage.input_tokens
        stats["output"] += usage.output_tokens + usage.thought_tokens
        # Seventy-four of these is four minutes with nothing on screen, which
        # is indistinguishable from a hang to anyone watching.
        if progress is not None:
            progress(index, total, source_id)
    stats["failures"] = failures
    if failures and not paths:
        # Not a degradation. A library with nothing in it is the input
        # missing, and everything downstream reads its absence as "these
        # clips have no description" rather than as an error -- one run
        # picked an aspect, chose sixteen shots and spent $1.80 planning a
        # film out of empty cards before anything said so.
        raise CardLibraryEmpty(
            f"every one of the {len(failures)} clips failed to describe; "
            f"first: {failures[0]}"
        )
    return paths, stats
