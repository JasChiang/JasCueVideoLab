from __future__ import annotations

import html
import json
import os
from pathlib import Path

from .models import ContentMap, DirectMomentMap, GroundingProposal, TemporalMap


def _relative(from_dir: Path, target: Path) -> str:
    # Preserve an artifact-local symlink instead of resolving it back to an
    # arbitrary source path outside the directory being served.
    return Path(os.path.relpath(target.absolute(), from_dir.resolve())).as_posix()


def render_timeline(
    *,
    content_map: ContentMap,
    video_path: Path,
    proposals: list[tuple[GroundingProposal, Path]],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    proposal_map: dict[str, list[tuple[GroundingProposal, Path]]] = {}
    for proposal, image_path in proposals:
        proposal_map.setdefault(proposal.event_id, []).append((proposal, image_path))
    event_cards = []
    for event in content_map.events:
        debug_items = []
        for proposal, image_path in proposal_map.get(event.event_id, []):
            relative_image = _relative(output_path.parent, image_path)
            candidates = ", ".join(
                f"{candidate.label} ({candidate.confidence:.2f})" for candidate in proposal.candidates
            ) or "不可見／無候選"
            debug_items.append(
                f'<figure><a href="{html.escape(relative_image)}" target="_blank">'
                f'<img src="{html.escape(relative_image)}" alt="Grounding debug"></a>'
                f'<figcaption>{html.escape(proposal.entity_id)} · PTS {proposal.frame_pts} · '
                f'{proposal.frame_time_ms} ms · {html.escape(candidates)}</figcaption></figure>'
            )
        keyframe = (
            f"{event.recommended_keyframe_ms} ms" if event.recommended_keyframe_ms is not None else "無可靠關鍵幀"
        )
        event_cards.append(
            f'''<article class="event" tabindex="0" role="button"
                 onclick="seekTo({event.start_ms})" onkeydown="if(event.key==='Enter')seekTo({event.start_ms})">
              <div class="event-time">{event.start_ms}–{event.end_ms} ms · {html.escape(event.boundary_precision.value)}</div>
              <h2>{html.escape(event.label)}</h2>
              <p>{html.escape(event.description)}</p>
              <p><strong>Keyframe:</strong> {html.escape(keyframe)} — {html.escape(event.keyframe_reason)}</p>
              <p><strong>Framing:</strong> {html.escape(event.framing_intent)}</p>
              <div class="debug-grid">{''.join(debug_items) or '<p class="muted">尚無 Grounding 結果</p>'}</div>
            </article>'''
        )
    video_url = _relative(output_path.parent, video_path)
    data = json.dumps(content_map.model_dump(mode="json"), ensure_ascii=False).replace("</", "<\\/")
    document = f'''<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>JasCueVideoLab Content Map</title>
<style>
:root{{--bg:#0b0e14;--panel:#141924;--text:#edf2f7;--muted:#9aa7b8;--accent:#5eead4}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:24px}} header{{display:grid;grid-template-columns:minmax(0,2fr) minmax(280px,1fr);gap:22px;align-items:start;position:sticky;top:0;background:#0b0e14ee;padding:12px 0;z-index:2}}
video{{width:100%;max-height:60vh;background:#000;border-radius:12px}} .summary{{padding:18px;background:var(--panel);border-radius:12px}}
.event{{margin:18px 0;padding:20px;background:var(--panel);border:1px solid #263043;border-radius:12px;cursor:pointer}}
.event:hover,.event:focus{{border-color:var(--accent);outline:none}} .event h2{{margin:.2rem 0}} .event-time,.muted{{color:var(--muted)}}
.debug-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}} figure{{margin:0}} img{{width:100%;border-radius:8px}} figcaption{{color:var(--muted);font-size:12px}}
@media(max-width:760px){{header{{grid-template-columns:1fr;position:static}}}}
</style></head><body><main>
<header><video id="player" controls preload="metadata" src="{html.escape(video_url)}"></video>
<section class="summary"><h1>Content Map</h1><p>{html.escape(content_map.summary)}</p>
<p>{len(content_map.events)} events · {len(content_map.entities)} entities · coarse semantic time</p>
<p class="muted">點選事件只會跳到語意搜尋起點；實際抽幀 PTS 顯示在 debug 圖下方。</p></section></header>
<section>{''.join(event_cards)}</section>
<script type="application/json" id="content-map">{data}</script>
<script>function seekTo(ms){{const p=document.getElementById('player');const target=ms/1000;const go=()=>{{const end=Number.isFinite(p.duration)?Math.max(0,p.duration-.001):target;p.currentTime=Math.min(target,end);p.play().catch(()=>{{}})}};if(p.readyState>=1)go();else p.addEventListener('loadedmetadata',go,{{once:true}});window.scrollTo({{top:0,behavior:'smooth'}})}}</script>
</main></body></html>'''
    output_path.write_text(document, encoding="utf-8")
    return output_path


def render_temporal_timeline(
    *,
    temporal_map: TemporalMap,
    video_path: Path,
    event_images: dict[str, Path],
    output_path: Path,
    sampling_interval_ms: int,
) -> Path:
    """Render a review page for the PTS-indexed storyboard fallback."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    for event in temporal_map.events:
        image_path = event_images.get(event.event_id)
        image_html = ""
        if image_path is not None:
            relative_image = _relative(output_path.parent, image_path)
            image_html = (
                f'<a href="{html.escape(relative_image)}" target="_blank">'
                f'<img src="{html.escape(relative_image)}" alt="{html.escape(event.label)}"></a>'
            )
        cards.append(
            f'''<article class="event" tabindex="0" role="button" onclick="seekTo({event.start_ms})"
            onkeydown="if(event.key==='Enter')seekTo({event.start_ms})">
            <div><span class="time">{event.start_ms}–{event.end_ms} ms</span>
            <h2>{html.escape(event.label)}</h2><p>{html.escape(event.observable_evidence)}</p>
            <p><strong>PTS keyframe:</strong> {event.recommended_keyframe_ms} ms · {html.escape(event.keyframe_reason)}</p></div>
            {image_html}</article>'''
        )
    video_url = _relative(output_path.parent, video_path)
    data = json.dumps(temporal_map.model_dump(mode="json"), ensure_ascii=False).replace("</", "<\\/")
    document = f'''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>PTS Storyboard Temporal Map</title>
<style>:root{{--bg:#0b0e14;--panel:#151b24;--text:#edf2f7;--muted:#9aa7b8;--accent:#5eead4}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:24px}}header{{position:sticky;top:0;background:#0b0e14f2;padding:10px 0 20px;z-index:2}}
video{{width:100%;max-height:52vh;background:#000;border-radius:12px}}.lead,.time{{color:var(--muted)}}
.event{{display:grid;grid-template-columns:1fr minmax(260px,42%);gap:20px;margin:18px 0;padding:20px;background:var(--panel);border:1px solid #293346;border-radius:12px;cursor:pointer}}
.event:hover,.event:focus{{border-color:var(--accent);outline:none}}h2{{margin:.25rem 0}}img{{display:block;width:100%;border-radius:9px}}
@media(max-width:760px){{header{{position:static}}.event{{grid-template-columns:1fr}}}}</style></head><body><main>
<header><video id="player" controls preload="metadata" src="{html.escape(video_url)}"></video>
<h1>PTS-indexed Storyboard Temporal Map</h1><p>{html.escape(temporal_map.summary)}</p>
<p class="lead">Gemini 只選既有 frame ID；所有毫秒值由 FFmpeg frame PTS 映射。取樣間隔約 {sampling_interval_ms} ms，仍是 coarse semantic time，不是剪輯點。</p></header>
<section>{''.join(cards)}</section><script type="application/json" id="temporal-map">{data}</script>
<script>function seekTo(ms){{const p=document.getElementById('player');const go=()=>{{p.currentTime=Math.min(ms/1000,Math.max(0,p.duration-.001));p.play().catch(()=>{{}})}};if(p.readyState>=1)go();else p.addEventListener('loadedmetadata',go,{{once:true}});window.scrollTo({{top:0,behavior:'smooth'}})}}</script>
</main></body></html>'''
    output_path.write_text(document, encoding="utf-8")
    return output_path


def render_direct_moment_timeline(
    *,
    moment_map: DirectMomentMap,
    video_path: Path,
    results: list[tuple[str, int, int, Path, GroundingProposal]],
    output_path: Path,
) -> Path:
    """Render direct MM:SS suggestions with exact extracted PTS and bbox overlays."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_id = {moment_id: (requested_ms, actual_ms, image, proposal) for moment_id, requested_ms, actual_ms, image, proposal in results}
    cards: list[str] = []
    for moment in moment_map.moments:
        requested_ms, actual_ms, image_path, proposal = by_id[moment.moment_id]
        relative_image = _relative(output_path.parent, image_path)
        candidates = ", ".join(
            f"{candidate.label} ({candidate.confidence:.2f})" for candidate in proposal.candidates
        ) or "不可見／無候選"
        cards.append(
            f'''<article class="event" tabindex="0" role="button" onclick="seekTo({requested_ms})"
            onkeydown="if(event.key==='Enter')seekTo({requested_ms})"><div>
            <span class="time">Gemini {html.escape(moment.timestamp_mmss)} · requested {requested_ms} ms · actual PTS {actual_ms} ms</span>
            <h2>{html.escape(moment.label)}</h2><p>{html.escape(moment.observable_evidence)}</p>
            <p><strong>Target:</strong> {html.escape(moment.grounding_target_description)}</p>
            <p><strong>Grounding:</strong> {html.escape(candidates)}</p></div>
            <a href="{html.escape(relative_image)}" target="_blank"><img src="{html.escape(relative_image)}" alt="Grounding debug"></a></article>'''
        )
    video_url = _relative(output_path.parent, video_path)
    data = json.dumps(moment_map.model_dump(mode="json"), ensure_ascii=False).replace("</", "<\\/")
    document = f'''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Direct MM:SS Moments + Grounding</title>
<style>:root{{--bg:#0b0e14;--panel:#151b24;--text:#edf2f7;--muted:#9aa7b8;--accent:#ff3155}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:24px}}header{{position:sticky;top:0;background:#0b0e14f2;padding:10px 0 20px;z-index:2}}
video{{width:100%;max-height:52vh;background:#000;border-radius:12px}}.lead,.time{{color:var(--muted)}}
.event{{display:grid;grid-template-columns:1fr minmax(300px,46%);gap:20px;margin:18px 0;padding:20px;background:var(--panel);border:1px solid #293346;border-radius:12px;cursor:pointer}}
.event:hover,.event:focus{{border-color:var(--accent);outline:none}}h2{{margin:.25rem 0}}img{{display:block;width:100%;border-radius:9px}}
@media(max-width:760px){{header{{position:static}}.event{{grid-template-columns:1fr}}}}</style></head><body><main>
<header><video id="player" controls preload="metadata" src="{html.escape(video_url)}"></video>
<h1>Direct Gemini MM:SS Moments + Grounding</h1><p>{html.escape(moment_map.summary)}</p>
<p class="lead">Gemini 直接提出 MM:SS；FFmpeg 另記錄實際抽到的 frame PTS。紅框是第二次 Gemini 單幀 Grounding，不是 tracking 或人工 ground truth。</p></header>
<section>{''.join(cards)}</section><script type="application/json" id="moment-map">{data}</script>
<script>function seekTo(ms){{const p=document.getElementById('player');const go=()=>{{p.currentTime=Math.min(ms/1000,Math.max(0,p.duration-.001));p.play().catch(()=>{{}})}};if(p.readyState>=1)go();else p.addEventListener('loadedmetadata',go,{{once:true}});window.scrollTo({{top:0,behavior:'smooth'}})}}</script>
</main></body></html>'''
    output_path.write_text(document, encoding="utf-8")
    return output_path
