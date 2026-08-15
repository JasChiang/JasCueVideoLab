from montagewright.brief import initial_graphics_plan, parse_brief_markdown


def test_explicit_approved_copy_is_a_gemini_placeable_candidate():
    document = parse_brief_markdown('''請做一支安靜的產品短片。

```montagewright-approved-copy
{"version": 1, "items": [{"copy_id": "hero", "text": "展開新篇章", "allowed_kinds": ["opening_title"]}]}
```
''')

    projected = document.graphics_candidates()
    approved = next(one for one in projected if one.candidate_id == "approved.hero")

    assert approved.primary_text == "展開新篇章"
    assert approved.kind == "opening_title"
    assert approved.authority_fact_id == "hero"


def test_explicit_approved_copy_materialises_without_model_rewriting_it():
    document = parse_brief_markdown('''```montagewright-approved-copy
{"version": 1, "items": [{"copy_id": "hero", "text": "展開新篇章", "allowed_kinds": ["opening_title"]}]}
```''')
    selection = {
        "covered": [{
            "show_as_graphic": True,
            "graphic_candidate_id": "approved.hero",
            "graphic_design_family": "cinematic_title",
            "graphic_surface": "inherit",
            "graphic_motion": "inherit",
            "graphic_composition": "inherit",
            "graphic_shot_index": 0,
            "graphic_reason": "開場需要主標",
            "shot_indexes": [0],
        }],
    }

    plan = initial_graphics_plan(document, selection, shot_durations=[3.0])

    assert len(plan.cues) == 1
    assert plan.cues[0].primary_fact_id == "hero"
    assert plan.cues[0].status == "approved"
    assert plan.fact("hero").exact_text == "展開新篇章"


def test_plain_brief_candidate_remains_a_draft():
    document = parse_brief_markdown("Galaxy Z Fold8\n展開新篇章")
    candidate = document.graphics_candidates()[0]
    selection = {
        "covered": [{
            "show_as_graphic": True,
            "graphic_candidate_id": candidate.candidate_id,
            "graphic_design_family": "auto",
            "graphic_surface": "inherit",
            "graphic_motion": "inherit",
            "graphic_composition": "inherit",
            "graphic_shot_index": 0,
            "graphic_reason": "產品介紹",
            "shot_indexes": [0],
        }],
    }

    plan = initial_graphics_plan(document, selection, shot_durations=[3.0])

    assert plan.cues[0].status == "draft"
    assert plan.fact(plan.cues[0].primary_fact_id).approved is False


def test_markdown_bullet_constraints_are_instructions_not_one_copy_card():
    document = parse_brief_markdown(
        "# Direction\n\n"
        "- Keep the approved subject recognizable throughout the edit.\n"
        "- A partial entrance may be used as a transition, not a full endpoint.\n"
        "- Do not substitute a similar object.\n"
    )

    assert document.candidates == ()
    assert len(document.instructions) == 3
    assert all(note.kind == "editorial" for note in document.instructions)
    assert document.candidate_facts() == ()


def test_long_form_direction_is_not_misread_as_verbatim_screen_copy():
    brief = (
        "剪輯一支產品直式影片，沿用指定音樂並保持俐落但不倉促的節奏。"
        "影片也必須納入毛片中的教學與操作內容，讓觀眾看懂操作對象、手勢、"
        "介面反應與完成結果；不可只取漂亮但無法理解的零碎片段。"
        "操作流程若跨多個步驟，可用相鄰鏡頭或精簡蒙太奇呈現，但要保留因果順序。"
        "產品展示與教學內容交錯安排，避免全片都是定鏡；數位運鏡必須服務閱讀與動作，"
        "不可為動而動。若目標片長與內容完成度衝突，應優先保留完整的操作過程。"
        "每一段操作都應建立清楚的觀看方向，保留必要停頓，並在完成狀態可辨識後才切走。"
        "不要以無關的產品空鏡取代教學證據，也不要改變原始操作順序。"
    )
    assert len(brief) > 240

    document = parse_brief_markdown(brief)

    assert document.candidates == ()
    assert document.candidate_facts() == ()
    assert [note.kind for note in document.instructions] == ["editorial"]
    assert document.instructions[0].text == brief


def test_legacy_overlong_candidate_cannot_break_fact_serialisation():
    from dataclasses import replace

    document = parse_brief_markdown("Pixel 11\n操作教學")
    unsafe = replace(document.candidates[0], primary_text="x" * 241)
    legacy = replace(document, candidates=(unsafe,))

    assert legacy.candidate_facts() == ()
    assert legacy.candidates_json()["facts"] == []
