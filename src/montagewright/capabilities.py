"""What the execution layer can actually do, in one place.

The planner cannot ask for a move it has never been told exists. Selection was
returning a subject and a position and nothing else, so a row of three
handsets -- which wants a sweep across them, not a follow of any one -- came
back as a single group subject whose centre never moves, and rendered as a
hold. The capability was missing from the menu, not from the material.

This list is the menu. It is rendered into the prompt so the model chooses
from what exists, and it is the vocabulary the executor dispatches on, so the
two cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

# The shortest rest that reads as the frame having stopped rather than
# slowed. Mirrors reframe.SETTLE_SECONDS, stated here because this is what
# the planner is told and the two must not drift.
SETTLE_SECONDS = 0.35


@dataclass(frozen=True)
class Capability:
    name: str
    when: str
    needs_subject: bool
    # The shortest this move can be and still register as a move at all --
    # a technical floor, not an answer to "how long should this shot be".
    # How long a sweep needs depends on how far it travels and how much has
    # to be taken in on the way, and that is visible in the material: the
    # same 2.5 seconds that reads a two-word logo is not enough for three
    # handsets, and is twice what a short push needs. The planner watches the
    # shot, so the planner sets the length; this is what the executor reports
    # against when the length it was given cannot contain the move.
    min_seconds: float = 0.0


CAMERA_MOVES: tuple[Capability, ...] = (
    Capability(
        name="hold",
        when=(
            "畫面本身已經成立，或主體靜止且資訊集中。"
            "素材本身已經有攝影機運動時也用這個——讓它自己的推軌或搖攝演完，"
            "再疊一層數位運鏡兩者會打架。不動是一個決定，不是沒有決定。"
        ),
        needs_subject=True,
        min_seconds=0.8,
    ),
    Capability(
        name="pan",
        when=(
            "水平帶過去。指定一個主體就跟著它走；指定兩個就從第一個帶到"
            "第二個；一排東西（三台手機、一列手錶）就從頭掃到尾。"
            "主體不動又只有一個時會變成 hold，因為沒有東西可跟。"
        ),
        needs_subject=True,
        min_seconds=2.5,
    ),
    Capability(
        name="tilt",
        when=(
            "垂直帶過去。主體上下移動時用——放進水裡、從桌面拿起、"
            "沿著機身從上往下看。橫式素材出直式時上下沒有空間可移，"
            "本機會照實回報做不到。"
        ),
        needs_subject=True,
        min_seconds=2.5,
    ),
    Capability(
        name="push_in",
        when=(
            "收緊到一個細節上——鏡頭模組、螢幕上的數值、一個材質。"
            "本機依來源解析度決定收多緊，不會為了框滿而放大到糊。"
        ),
        needs_subject=True,
        min_seconds=2.5,
    ),
    Capability(
        name="pull_out",
        when="從細節退開，交代它在什麼脈絡裡，通常用在一段的收尾。",
        needs_subject=True,
        min_seconds=2.5,
    ),
)

# These name what a crop can do, not what a camera can do. A dolly, a crane or
# a true track all change viewpoint, which produces parallax -- foreground and
# background moving at different rates -- and no amount of moving a window
# over an already-shot frame produces that. Offering `dolly_zoom` would get it
# chosen for an effect the executor cannot deliver, which is the same failure
# as any other capability declared and not built.

MOVE_NAMES: tuple[str, ...] = tuple(move.name for move in CAMERA_MOVES)
MOVE_FLOORS: dict[str, float] = {
    move.name: move.min_seconds for move in CAMERA_MOVES
}

# The editorial question is richer than the five physical crop labels above.
# A source pan and a digital pan can look alike to the viewer while asking the
# executor to do opposite things; a comparison and a reveal can use the same
# two endpoints while carrying different timing.  Selection chooses one of
# these intentions first, then writes semantic looks.  Local code still owns
# coordinates, axes, easing and feasibility.
CAMERA_INTENTS: tuple[tuple[str, str], ...] = (
    ("hold", "固定數位裁切，讓一個已成立的構圖停住。"),
    (
        "use_source_motion",
        "素材自己的 authored／subject-follow 運鏡已完成揭示；數位框固定，讓原生運鏡演完。",
    ),
    (
        "follow_subject",
        "主體在可用片段內移動，數位框持續跟住它；只需一個主體落點。",
    ),
    (
        "reveal",
        "從一個落點走到另一個落點，後者是這顆要揭示的答案。",
    ),
    (
        "compare",
        "在兩個以上主體間移動，讓觀眾比較它們，而不是只抵達最後一個。",
    ),
    (
        "push_in",
        "同一主體由較鬆 framing 收到較緊 framing，走向細節。",
    ),
    (
        "pull_out",
        "同一主體由較緊 framing 退到較鬆 framing，交代脈絡。",
    ),
    (
        "multi_stop",
        "三個以上落點依序停住，逐一介紹一列物件或一串資訊。",
    ),
)

CAMERA_INTENT_NAMES: tuple[str, ...] = tuple(
    name for name, _ in CAMERA_INTENTS
)

# Where the subject sits when it does not fill the output ratio.
#
# A subject small in a clean frame is a composition, not a shortfall. Product
# work on a white sweep is mostly negative space, and the question it asks is
# where the subject sits in that space -- not how to get rid of it. Enlarging
# the picture to fill the frame answers a question nobody asked and softens
# the result to do it.
FRAMING_INTENTS: tuple[tuple[str, str], ...] = (
    (
        "thirds",
        "主體放在三分線上，留白在另一側。乾淨背景、產品陳列、"
        "帶情緒的空鏡都適合。這是大部分情況的預設。",
    ),
    (
        "centre",
        "主體置中。對稱構圖、正面對鏡、或主體本身就是畫面全部時使用。",
    ),
    (
        "fill",
        "主體盡量佔滿畫面。細節特寫、螢幕上的數值、材質質感這類"
        "要看清楚的鏡頭用這個；本機仍會守住放大上限，不會為了填滿而糊掉。",
    ),
)

INTENT_NAMES: tuple[str, ...] = tuple(name for name, _ in FRAMING_INTENTS)


def describe_for_prompt() -> str:
    """The vocabulary as the planner reads it.

    This used to be a menu of five named moves and it is now one shape,
    because the menu kept needing another entry -- settling at the ends, a
    stop on the way, a push that follows its subject -- and each one was a
    code change for something an editor would just do. What was missing was
    a primitive, not features.
    """

    lines = [
        "先選 `camera_intent`，再用 `looks` 說落點：",
    ]
    lines.extend(f"- `{name}`：{when}" for name, when in CAMERA_INTENTS)
    lines += [
        "",
        "意圖是剪輯判斷，looks 是它的語意路徑：hold／use_source_motion／"
        "follow_subject 用一個落點；reveal／compare 用兩個以上不同落點；"
        "push_in／pull_out 用同一主體兩次但 framing 一鬆一緊；multi_stop "
        "用三個以上落點。",
        "",
        "本機會量落點後決定實際沿水平或垂直方向移動。不要填座標，也不要"
        "把原素材的運鏡和數位裁切混成一件事。",
        "",
        "本機負責的是：每個落點實際在畫面的哪個位置、鏡頭走多快、"
        "兩端跟中途各停多久才算真的停下來、以及走不完的時候照實回報。"
        "這些都不要寫進計畫裡。",
        "",
        "你負責的是**看什麼**跟**看多久**。第二個是真的判斷：兩個字的 logo "
        "比一排三只手錶讀得快，而只有看過這顆畫面的人知道是哪一種。"
        "每個落點都要花時間，`seconds_needed` 要含得下全部落點加上"
        "中間的路程——落點太多而時間不夠，本機會照實記一筆，然後鏡頭"
        "會走不到最後那個落點。",
        "",
        "落點沒有填 `seconds` 的話，本機會給一個「還算得上停頓」的下限"
        f"（{SETTLE_SECONDS:g} 秒）。那是技術底線，不是建議長度。",
        "",
        "主體沒有填滿輸出比例時，它擺哪裡由 `framing` 決定：",
    ]
    for name, when in FRAMING_INTENTS:
        lines.append(f"- `{name}`：{when}")
    lines.append(
        "留白本身是合法的構圖，不是要被消滅的東西。本機不會為了填滿畫面"
        "而放大到糊，實際放大倍數會回報成數字。"
    )
    return "\n".join(lines)


# What this tool does not do, in the same place and for the same reason as
# what it does. A reviewer judging the cut against a brief has no way to know
# the difference between "this was done badly" and "this cannot be done here"
# -- so it reported a missing title card as a fault every round, which is a
# paid call spent on something no replan can ever fix, and a verdict of
# "revise" on a cut that was as asked.
CANNOT: tuple[tuple[str, str], ...] = (
    (
        "剪輯規劃階段自動建立字卡",
        "另有可編輯的 Web 字卡圖層與獨立輸出，但文字必須來自 brief "
        "核准清單或人工核准，shot replan 不會自行補字卡",
    ),
    ("轉場特效", "每一次都是硬切；長度與切點是唯一的節奏工具"),
    ("調色與濾鏡", "畫面只做裁切、縮放與響度處理"),
    ("速度變化", "沒有慢動作或加速"),
    ("畫面合成", "不疊圖、不分割畫面、不做子母畫面"),
)


def describe_limits_for_prompt() -> str:
    """The same list, for a prompt that is about to judge the result."""

    return "\n".join(
        f"- {what}：{why}" for what, why in CANNOT
    )
