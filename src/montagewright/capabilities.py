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

# The shortest rest that reads as the frame having stopped rather than
# slowed. Mirrors reframe.SETTLE_SECONDS, stated here because this is what
# the planner is told and the two must not drift.
SETTLE_SECONDS = 0.35


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

# Compatibility for legacy cached answers and the renderer dispatch table.
# New planning code must use CAMERA_INTENT_NAMES; unlike the removed
# CAMERA_MOVES/MOVE_FLOORS table this carries no timing authority.
MOVE_NAMES: tuple[str, ...] = ("hold", "pan", "tilt", "push_in", "pull_out")

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
    (
        "有數位運鏡時變速",
        "可以慢動作或加速（每顆 speed，0.25–4x），但變速只搭配固定數位框；"
        "有數位運鏡（跟拍、揭示、推拉、pan）的那顆暫不支援，會自動回原速",
    ),
    ("畫面合成", "不疊圖、不分割畫面、不做子母畫面"),
)


def describe_limits_for_prompt() -> str:
    """The same list, for a prompt that is about to judge the result."""

    return "\n".join(
        f"- {what}：{why}" for what, why in CANNOT
    )
