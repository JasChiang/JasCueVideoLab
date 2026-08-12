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
