from __future__ import annotations

import html
import os
from pathlib import Path
from statistics import mean
from typing import Any

from .storage import read_json


def _group(
    *,
    label: str,
    summary_path: Path,
    output_dir: Path,
) -> tuple[str, dict[str, object]]:
    summary = read_json(summary_path)
    rows = [row for row in summary["runs"] if row.get("schema_valid")]
    ious = [float(row["reference_iou"]) for row in rows]
    passed = sum(iou >= 0.8 for iou in ious)
    cards: list[str] = []
    for row in rows:
        iou = float(row["reference_iou"])
        verdict = "PASS" if iou >= 0.8 else "FAIL"
        image_path = summary_path.parent / str(row["run"]) / "debug.png"
        relative_image = os.path.relpath(image_path, output_dir)
        cards.append(
            f'<article class="sample {verdict.lower()}">'
            f'<header><strong>{html.escape(str(row["run"]))}</strong><b>{verdict}</b></header>'
            f'<a href="{html.escape(relative_image)}"><img src="{html.escape(relative_image)}" '
            f'alt="{html.escape(label)} {html.escape(str(row["run"]))}"></a>'
            f'<p>IoU <strong>{iou:.3f}</strong> · center '
            f'{float(row["reference_center_distance"]):.1f}/1000 · confidence '
            f'{float(row["confidence"]):.2f}</p></article>'
        )
    stats: dict[str, object] = {
        "label": label,
        "runs": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "mean_iou": mean(ious),
        "min_iou": min(ious),
        "max_iou": max(ious),
        "target_description": summary["target_description"],
    }
    section = f"""<section class="group">
<div class="group-head"><div><h2>{html.escape(label)}</h2><p>{html.escape(str(summary['target_description']))}</p></div>
<div class="score"><strong>{passed}/{len(rows)} PASS</strong><span>mean IoU {stats['mean_iou']:.3f}</span></div></div>
<div class="samples">{''.join(cards)}</div></section>"""
    return section, stats


def render_grounding_ab_review(
    *,
    explicit_summary: Path,
    generic_summary: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    explicit_section, explicit = _group(
        label="A — 明確指定實例與層級",
        summary_path=explicit_summary,
        output_dir=output_dir,
    )
    generic_section, generic = _group(
        label="B — Content Map 原始泛化描述",
        summary_path=generic_summary,
        output_dir=output_dir,
    )
    conclusion = (
        f"明確描述通過 {explicit['passed']}/{explicit['runs']}；"
        f"泛化描述通過 {generic['passed']}/{generic['runs']}。"
        "錯框仍可能回報 0.98–1.00 confidence，因此 confidence 不可作為正確性 gate。"
    )
    document = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>JasCueVideoLab Grounding A/B</title>
<style>
:root{{--bg:#0b0f14;--panel:#151b23;--ink:#edf4ff;--muted:#9eacbd;--line:#2b3542;--pass:#00d084;--fail:#ff3155}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:30px}}
h1,h2,p{{margin-top:0}}.lead{{max-width:1000px;color:var(--muted)}}.conclusion{{padding:16px 18px;border-left:5px solid #4da3ff;background:var(--panel);font-size:18px}}
.group{{margin-top:34px}}.group-head{{display:flex;justify-content:space-between;gap:24px;align-items:start}}.group-head p{{color:var(--muted);max-width:900px}}.score{{display:grid;text-align:right;white-space:nowrap}}.score strong{{font-size:24px}}.score span{{color:var(--muted)}}
.samples{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}}.sample{{background:var(--panel);border:2px solid var(--line);border-radius:12px;overflow:hidden}}.sample.pass{{border-color:#167a50}}.sample.fail{{border-color:#a92d43}}.sample header{{display:flex;justify-content:space-between;padding:10px 12px}}.sample.pass header b{{color:var(--pass)}}.sample.fail header b{{color:var(--fail)}}.sample img{{width:100%;height:auto;display:block}}.sample p{{padding:10px 12px;margin:0;color:var(--muted)}}
@media(max-width:650px){{main{{padding:18px}}.group-head{{display:block}}.score{{text-align:left;margin-bottom:12px}}}}
</style></head><body><main><h1>同一影格 Grounding A/B</h1>
<p class="lead">固定 frame hash、PTS、時間、模型、thinking level 與 Structured Output schema，只改 target description。每組五次真實 Gemini API 呼叫；暫定 PASS 門檻為 Codex reviewer reference IoU ≥ 0.8。Reference 是 Codex 視覺檢查後手動輸入，未經獨立真人確認，不是 human ground truth。</p>
<p class="conclusion"><strong>結論：</strong>{html.escape(conclusion)}</p>
{explicit_section}{generic_section}
</main></body></html>"""
    output_path = output_dir / "index.html"
    output_path.write_text(document, encoding="utf-8")
    return output_path
