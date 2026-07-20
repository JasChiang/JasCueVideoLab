from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .geometry import normalized_to_pixels
from .models import GroundingProposal
from .storage import read_json


PREDICTED_COLOR = "#ff3155"
REFERENCE_COLOR = "#00d084"


def _verdict(match: dict[str, Any]) -> tuple[str, str]:
    if not match.get("comparable"):
        reason = str(match.get("reason", "comparison unavailable"))
        if "label similarity" in reason:
            return "fail", "FAIL — selected the wrong target"
        return "unscored", f"UNSCORED — {reason}"
    iou = float(match["iou"])
    if iou >= 0.8:
        return "pass", "PASS"
    if iou >= 0.5:
        return "review", "REVIEW"
    return "fail", "FAIL"


def _find_proposal(run_dir: Path, entity_id: str, frame_time_ms: int) -> tuple[Path, GroundingProposal]:
    matches: list[tuple[int, Path, GroundingProposal]] = []
    for path in run_dir.glob("events/*/groundings/*/grounding.json"):
        proposal = GroundingProposal.model_validate(read_json(path))
        if proposal.entity_id != entity_id:
            continue
        matches.append((abs(proposal.frame_time_ms - frame_time_ms), path, proposal))
    if not matches:
        raise FileNotFoundError(f"no proposal for {entity_id} in {run_dir}")
    _, path, proposal = min(matches, key=lambda item: item[0])
    return path, proposal


def _draw_review_overlay(
    frame_path: Path,
    proposal: GroundingProposal,
    reference_box: list[int],
    reference_label: str,
    output_path: Path,
) -> None:
    with Image.open(frame_path).convert("RGB") as source:
        image = source.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(12, round(min(image.size) / 45)))
    line_width = max(3, round(min(image.size) / 180))

    reference_pixels = normalized_to_pixels(reference_box, *image.size)
    draw.rectangle(reference_pixels, outline=REFERENCE_COLOR, width=line_width)
    draw.text(
        (reference_pixels[0] + 5, reference_pixels[1] + 5),
        f"CODEX REF: {reference_label}",
        fill="#07110d",
        stroke_width=4,
        stroke_fill=REFERENCE_COLOR,
        font=font,
    )

    for candidate in proposal.candidates:
        predicted_pixels = normalized_to_pixels(candidate.box_2d, *image.size)
        draw.rectangle(predicted_pixels, outline=PREDICTED_COLOR, width=line_width)
        draw.text(
            (predicted_pixels[0] + 5, max(5, predicted_pixels[1] - 34)),
            f"GEMINI: {candidate.label} ({candidate.confidence:.2f})",
            fill="#180309",
            stroke_width=4,
            stroke_fill=PREDICTED_COLOR,
            font=font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def render_manual_review(
    artifact_root: Path,
    annotations_path: Path,
    output_dir: Path,
) -> Path:
    comparison = read_json(artifact_root / "comparison.json")
    annotations = read_json(annotations_path)
    references = {
        (box["entity_label"], box["frame_time_ms"]): box
        for box in annotations.get("boxes", [])
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    counts = {"pass": 0, "review": 0, "fail": 0, "unscored": 0}

    for run in comparison.get("human_annotation_comparison", []):
        run_id = run["run_id"]
        run_dir = artifact_root / run_id
        for index, match in enumerate(run.get("bbox_matches", []), start=1):
            reference_key = (match["reference_entity_label"], match["reference_frame_time_ms"])
            reference = references[reference_key]
            proposal_path, proposal = _find_proposal(
                run_dir,
                match["predicted_entity_id"],
                match["predicted_frame_time_ms"],
            )
            frame_path = proposal_path.parents[2] / "frame.png"
            image_name = f"{run_id}-{index:02d}-{proposal.entity_id}.png"
            _draw_review_overlay(
                frame_path,
                proposal,
                reference["box_2d"],
                reference["entity_label"],
                output_dir / image_name,
            )
            verdict_class, verdict = _verdict(match)
            counts[verdict_class] += 1
            if match.get("comparable"):
                metrics = (
                    f"IoU {match['iou']:.3f} · center distance "
                    f"{match['center_distance_normalized']:.1f}/1000"
                )
            else:
                metrics = html.escape(str(match.get("reason", "not comparable")))
            cards.append(
                "".join(
                    [
                        f'<article class="card {verdict_class}">',
                        f'<div class="card-head"><div><strong>{html.escape(run_id)}</strong>',
                        f'<span>{html.escape(reference["entity_label"])} @ '
                        f'{reference["frame_time_ms"] / 1000:.3f}s</span></div>',
                        f'<b>{html.escape(verdict)}</b></div>',
                        f'<a href="{html.escape(image_name)}"><img src="{html.escape(image_name)}" '
                        f'alt="{html.escape(run_id)} review overlay"></a>',
                        f'<p>{metrics}</p>',
                        f'<small>Gemini frame PTS: {proposal.frame_pts} · actual frame time: '
                        f'{proposal.frame_time_ms} ms · requested reference: '
                        f'{reference["frame_time_ms"]} ms</small>',
                        "</article>",
                    ]
                )
            )

    document = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>JasCueVideoLab Grounding reviewer-reference 審核</title>
<style>
:root{{--bg:#0b0f14;--panel:#151b23;--ink:#edf4ff;--muted:#9eacbd;--line:#2b3542}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1440px;margin:auto;padding:32px}} h1{{margin:0 0 8px}} .lead{{color:var(--muted);max-width:920px}}
.legend,.summary{{display:flex;gap:16px;flex-wrap:wrap;margin:18px 0}} .pill{{padding:8px 12px;border:1px solid var(--line);border-radius:999px}}
.reference{{color:{REFERENCE_COLOR}}}.predicted{{color:{PREDICTED_COLOR}}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}}
.card{{background:var(--panel);border:2px solid var(--line);border-radius:14px;overflow:hidden}} .card.pass{{border-color:#167a50}} .card.review{{border-color:#aa7b12}} .card.fail{{border-color:#a92d43}}
.card-head{{display:flex;justify-content:space-between;gap:16px;padding:14px 16px}} .card-head span{{display:block;color:var(--muted)}} .card-head b{{white-space:nowrap}}
.card img{{display:block;width:100%;height:auto;background:#000}} .card p,.card small{{display:block;margin:0;padding:10px 16px}} .card small{{padding-top:0;color:var(--muted)}}
code{{color:#d5e6ff}} @media(max-width:600px){{main{{padding:18px}}.grid{{grid-template-columns:1fr}}.card-head{{display:block}}}}
</style></head><body><main>
<h1>Grounding reviewer-reference 審核</h1>
<p class="lead">同一張原始影格疊加兩個框：<strong class="reference">綠色是 Codex 視覺檢查後手動輸入的參考框</strong>，<strong class="predicted">紅色是 Gemini 框</strong>。綠框未經獨立真人確認，不是 human ground truth；本頁也不使用模型 confidence 判定正確性。</p>
<div class="legend"><span class="pill">PASS: IoU ≥ 0.8</span><span class="pill">REVIEW: 0.5 ≤ IoU &lt; 0.8</span><span class="pill">FAIL: IoU &lt; 0.5 或選錯目標</span><span class="pill">UNSCORED: 沒有同時間 proposal</span></div>
<div class="summary"><span class="pill">PASS {counts['pass']}</span><span class="pill">REVIEW {counts['review']}</span><span class="pill">FAIL {counts['fail']}</span><span class="pill">UNSCORED {counts['unscored']}</span></div>
<p class="lead">注意：這是目前三個 Codex reviewer bbox × 三次 run 的審核矩陣，不是模型整體準確率。30 秒的 MacBook 參考框是寬鬆的整畫面近似值，應優先看 19 秒 iPhone 與 48 秒 MacBook。</p>
<section class="grid">{''.join(cards)}</section>
</main></body></html>"""
    output_path = output_dir / "index.html"
    output_path.write_text(document, encoding="utf-8")
    return output_path
