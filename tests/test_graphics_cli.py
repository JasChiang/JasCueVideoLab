import json

from montagewright.cli import main
from montagewright.graphics import CopyFact, GraphicCue, GraphicsPlan


def _draft_output(tmp_path):
    output = tmp_path / "cut"
    work = output / "work"
    work.mkdir(parents=True)
    plan = GraphicsPlan(
        facts=[CopyFact(
            fact_id="draft", exact_text="需要人工核准",
            source_kind="model_draft", approved=False,
        )],
        cues=[GraphicCue(
            graphic_id="g00", kind="callout", primary_fact_id="draft",
            at_seconds=0.0, duration_seconds=2.0, status="draft",
        )],
    )
    (work / "graphics.json").write_text(
        json.dumps(plan.model_dump(mode="json")), encoding="utf-8"
    )
    return output


def test_cli_can_inspect_the_web_graphics_plan(tmp_path, capsys):
    output = _draft_output(tmp_path)

    assert main(["graphics", str(output), "inspect"]) == 0

    printed = capsys.readouterr().out
    assert "g00" in printed
    assert "需要人工核准" in printed


def test_cli_approval_is_an_explicit_human_copy_clone(tmp_path):
    output = _draft_output(tmp_path)

    assert main([
        "graphics", str(output), "approve", "--graphic-id", "g00",
    ]) == 0

    plan = GraphicsPlan.model_validate_json(
        (output / "work" / "graphics.json").read_text(encoding="utf-8")
    )
    cue = plan.cues[0]
    approved = plan.fact(cue.primary_fact_id)
    assert cue.status == "approved"
    assert approved.source_kind == "user"
    assert approved.approved_by == "human_review"
    assert approved.exact_text == "需要人工核准"
