from __future__ import annotations

import hashlib
import html
import json
import subprocess
import uuid
from fractions import Fraction
from math import ceil
from pathlib import Path
from time import monotonic
from typing import Any

from .billing import summarize_usage_and_list_price, summarize_usage_files
from .full_v1 import (
    DEFAULT_FILE_CACHE_ROOT,
    _shared_upload_dir,
    create_dense_event_catalog,
    mmss_to_ms,
)
from .gemini import MODEL_ID, GeminiLabClient
from .media import MediaCommandError, extract_frame, probe_video
from .models import (
    DenseFrame,
    DenseFrameCatalog,
    FullClipCard,
    TrimFrameEvidence,
    TrimHumanReview,
    TrimIntentDecision,
    TrimIntentProposal,
    VideoTrimIntentProposal,
)
from .shots import ShotManifest
from .storage import read_json, utc_now, write_json


TRIM_INTENT_CONTRACT_VERSION = "trim-intent-phase-selection-v2"


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _frame_evidence(frame: DenseFrame | None) -> TrimFrameEvidence | None:
    if frame is None:
        return None
    return TrimFrameEvidence(
        frame_id=frame.frame_id,
        requested_time_ms=frame.requested_time_ms,
        frame_time_ms=frame.frame_time_ms,
        frame_pts=frame.frame_pts,
        frame_hash=frame.frame_hash,
    )


def derive_trim_decision(
    proposal: TrimIntentProposal,
    catalog: DenseFrameCatalog,
    *,
    shot_id: str,
    shot_start_ms: int,
    shot_end_ms: int,
    proposal_path: Path,
    catalog_path: Path,
    handle_before_ms: int = 750,
    handle_after_ms: int = 1000,
) -> TrimIntentDecision:
    if proposal.source_asset_id != catalog.source_asset_id:
        raise ValueError("trim proposal source differs from dense catalog")
    if proposal.event_id != catalog.event_id:
        raise ValueError("trim proposal event differs from dense catalog")
    frames = {frame.frame_id: frame for frame in catalog.frames}
    positions = {frame.frame_id: index for index, frame in enumerate(catalog.frames)}
    if not proposal.usable:
        return TrimIntentDecision(
            source_asset_id=proposal.source_asset_id,
            event_id=proposal.event_id,
            shot_id=shot_id,
            usable=False,
            first_included_frame=None,
            last_included_frame=None,
            exclusive_out_frame=None,
            hold_start_frame=None,
            hold_end_frame=None,
            source_in_ms=None,
            source_out_ms=None,
            source_in_pts=None,
            source_out_pts=None,
            handle_in_ms=None,
            handle_out_ms=None,
            tail_intent=proposal.tail_intent,
            proposal_path=str(proposal_path.resolve()),
            catalog_path=str(catalog_path.resolve()),
        )

    recommended_in_frame_id = proposal.frame_id_for("recommended_in")
    recommended_out_frame_id = proposal.frame_id_for("recommended_out")
    assert recommended_in_frame_id is not None
    assert recommended_out_frame_id is not None
    first = frames[recommended_in_frame_id]
    exclusive = frames[recommended_out_frame_id]
    if positions[first.frame_id] >= positions[exclusive.frame_id]:
        raise ValueError("trim proposal in/out order is invalid")
    last = catalog.frames[positions[exclusive.frame_id] - 1]
    source_in_ms = first.frame_time_ms
    source_out_ms = exclusive.frame_time_ms
    if source_out_ms <= source_in_ms:
        raise ValueError("dense trim mapping produced an empty interval")
    return TrimIntentDecision(
        source_asset_id=proposal.source_asset_id,
        event_id=proposal.event_id,
        shot_id=shot_id,
        usable=True,
        first_included_frame=_frame_evidence(first),
        last_included_frame=_frame_evidence(last),
        exclusive_out_frame=_frame_evidence(exclusive),
        hold_start_frame=_frame_evidence(
            frames.get(proposal.frame_id_for("hold_start") or "")
        ),
        hold_end_frame=_frame_evidence(
            frames.get(proposal.frame_id_for("hold_end") or "")
        ),
        source_in_ms=source_in_ms,
        source_out_ms=source_out_ms,
        source_in_pts=first.frame_pts,
        source_out_pts=exclusive.frame_pts,
        handle_in_ms=max(shot_start_ms, source_in_ms - handle_before_ms),
        handle_out_ms=min(shot_end_ms, source_out_ms + handle_after_ms),
        tail_intent=proposal.tail_intent,
        proposal_path=str(proposal_path.resolve()),
        catalog_path=str(catalog_path.resolve()),
    )


def _render_review(
    *,
    source_path: Path,
    event: Any,
    proposal: TrimIntentProposal | VideoTrimIntentProposal,
    decision: TrimIntentDecision,
    catalog: DenseFrameCatalog | None,
    preview_path: Path | None,
    output_path: Path,
    evidence_paths: list[Path] | None = None,
) -> None:
    page_paths = (
        [Path(path) for path in catalog.contact_sheet_paths]
        if catalog is not None
        else list(evidence_paths or [])
    )
    pages = "".join(
        f'<img src="{html.escape(path.as_uri())}" alt="trim boundary evidence">'
        for path in page_paths
    )
    decision_json = html.escape(
        json.dumps(decision.model_dump(mode="json"), ensure_ascii=False, indent=2)
    )
    proposal_json = html.escape(
        json.dumps(proposal.model_dump(mode="json"), ensure_ascii=False, indent=2)
    )
    start_seconds = (decision.handle_in_ms or 0) / 1000
    preview = (
        f'<h2>Proposed trim preview</h2><video controls preload="metadata" '
        f'src="{html.escape(preview_path.as_uri())}"></video>'
        if preview_path is not None
        else "<h2>Proposed trim preview</h2><p>此事件沒有可預覽的 trim proposal。</p>"
    )
    document = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><title>Trim Intent Review</title>
<style>body{{font-family:-apple-system,sans-serif;background:#111;color:#eee;margin:24px}}
video,img{{max-width:100%;background:#000}} img{{margin:8px 0;border:1px solid #444}}
pre{{white-space:pre-wrap;background:#1d1d1d;padding:16px;border-radius:10px}}
.warning{{color:#ffd166}}</style></head><body>
<h1>Trim Intent Review</h1>
<p class="warning">AI proposal；尚未經人工核准，不是正式剪點。</p>
<h2>{html.escape(event.label)}</h2><p>{html.escape(event.description)}</p>
{preview}
<h2>Source with handles</h2>
<video controls preload="metadata" src="{html.escape(source_path.as_uri())}#t={start_seconds:.3f}"></video>
<h2>Dense frame evidence</h2>{pages}
<h2>Local PTS decision</h2><pre>{decision_json}</pre>
<h2>Gemini proposal</h2><pre>{proposal_json}</pre>
</body></html>"""
    output_path.write_text(document, encoding="utf-8")


def _render_trim_preview(
    source_path: Path,
    decision: TrimIntentDecision,
    output_path: Path,
) -> Path | None:
    if (
        not decision.usable
        or decision.source_in_ms is None
        or decision.source_out_ms is None
    ):
        return None
    duration_seconds = (decision.source_out_ms - decision.source_in_ms) / 1000
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{decision.source_in_ms / 1000:.3f}",
            "-i",
            str(source_path),
            "-t",
            f"{duration_seconds:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            "scale='min(1280,iw)':-2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )
    return output_path.resolve()


def run_trim_intent_event(
    clip_run_dir: Path,
    event_id: str,
    output_dir: Path,
    *,
    prompt_template: str,
    editorial_intent: str,
    sampling_fps: float = 4.0,
    temperature: float = 0.1,
) -> dict[str, Any]:
    started = monotonic()
    card_path = clip_run_dir / "gemini" / "clip-card" / "clip_card.json"
    source_record_path = clip_run_dir / "private-source.json"
    shots_path = clip_run_dir / "shots.json"
    card = FullClipCard.model_validate(read_json(card_path))
    event = next((item for item in card.events if item.event_id == event_id), None)
    if event is None:
        raise ValueError(f"unknown Clip Card event: {event_id}")
    source_path = Path(read_json(source_record_path)["path"])
    source_media = probe_video(source_path)
    if source_media.asset_id != card.source_asset_id:
        raise ValueError("private source identity differs from Clip Card")
    shots = ShotManifest.model_validate(read_json(shots_path))
    event_start_ms = mmss_to_ms(event.start_mmss)
    event_end_ms = mmss_to_ms(event.end_mmss)
    center_ms = (
        mmss_to_ms(event.recommended_keyframe_mmss)
        if event.recommended_keyframe_mmss is not None
        else (event_start_ms + event_end_ms) // 2
    )
    shot = next(
        (
            item
            for item in shots.shots
            if item.start_time_ms <= center_ms < item.end_time_ms
        ),
        None,
    )
    if shot is None:
        raise ValueError("event keyframe does not belong to a decoded shot")
    window_start_ms = max(event_start_ms, shot.start_time_ms)
    window_end_ms = min(event_end_ms, shot.end_time_ms, source_media.duration_ms)
    if window_end_ms - window_start_ms < 500:
        raise ValueError("event has no usable shot-local trim window")

    dense_dir = output_dir / "dense"
    dense_catalog_path = dense_dir / "dense-catalog.json"
    if dense_catalog_path.exists():
        catalog = DenseFrameCatalog.model_validate(read_json(dense_catalog_path))
        expected = (
            catalog.source_asset_id == card.source_asset_id
            and catalog.event_id == event.event_id
            and catalog.sampling_fps == sampling_fps
            and catalog.source_start_ms == window_start_ms
            and catalog.source_end_ms == window_end_ms
        )
        if not expected:
            raise ValueError("saved dense trim catalog differs from requested inputs")
    else:
        catalog = create_dense_event_catalog(
            source_path,
            card.source_asset_id,
            event,
            dense_dir,
            sampling_fps=sampling_fps,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
        )

    request_identity = {
        "contract_version": TRIM_INTENT_CONTRACT_VERSION,
        "model": MODEL_ID,
        "temperature": temperature,
        "source_asset_id": card.source_asset_id,
        "event": event.model_dump(mode="json"),
        "editorial_intent": editorial_intent,
        "sampling_fps": sampling_fps,
        "frame_hashes": [frame.frame_hash for frame in catalog.frames],
        "contact_sheet_hashes": catalog.contact_sheet_hashes,
        "prompt_sha256": hashlib.sha256(prompt_template.encode("utf-8")).hexdigest(),
        "schema_sha256": _canonical_hash(TrimIntentProposal.model_json_schema()),
    }
    variant_id = _canonical_hash(request_identity)[:16]
    variant_dir = output_dir / "gemini" / variant_id
    proposal_path = variant_dir / "trim_intent.json"
    reused = proposal_path.exists()
    execution_interaction: Path | None = None
    if reused:
        proposal = TrimIntentProposal.model_validate(read_json(proposal_path))
    else:
        attempt_number = len(list(variant_dir.glob("attempt-*"))) + 1
        attempt_dir = variant_dir / f"attempt-{attempt_number:02d}"
        write_json(variant_dir / "request-identity.json", request_identity)
        client = GeminiLabClient(temperature=temperature)
        execution_interaction = attempt_dir / "trim_intent.raw_interaction.json"
        try:
            proposal = client.analyze_trim_intent(
                event=event,
                catalog=catalog,
                prompt_template=prompt_template,
                editorial_intent=editorial_intent,
                run_id=f"trim-{uuid.uuid4().hex[:8]}",
                run_dir=attempt_dir,
            )
        except Exception as error:
            pricing = summarize_usage_and_list_price(output_dir / "gemini")
            execution_pricing = summarize_usage_files(
                [execution_interaction],
                relative_to=output_dir,
            )
            write_json(output_dir / "pricing.json", pricing)
            write_json(output_dir / "execution-pricing.json", execution_pricing)
            write_json(
                output_dir / "result.json",
                {
                    "status": "failed",
                    "variant_id": variant_id,
                    "gemini_reused": False,
                    "frame_count": len(catalog.frames),
                    "sampling_fps": sampling_fps,
                    "window_start_ms": window_start_ms,
                    "window_end_ms": window_end_ms,
                    "elapsed_seconds": round(monotonic() - started, 3),
                    "execution_pricing": execution_pricing,
                    "error": {"type": type(error).__name__, "message": str(error)},
                },
            )
            raise
        finally:
            client.close()
        write_json(proposal_path, proposal)

    decision = derive_trim_decision(
        proposal,
        catalog,
        shot_id=shot.shot_id,
        shot_start_ms=shot.start_time_ms,
        shot_end_ms=shot.end_time_ms,
        proposal_path=proposal_path,
        catalog_path=dense_catalog_path,
    )
    decision_path = output_dir / "trim-decision.json"
    write_json(decision_path, decision)
    write_json(output_dir / "active-variant.json", {"variant_id": variant_id})
    preview_path = _render_trim_preview(
        source_path,
        decision,
        output_dir / "trim-preview.mp4",
    )
    review_path = output_dir / "index.html"
    _render_review(
        source_path=source_path,
        event=event,
        proposal=proposal,
        decision=decision,
        catalog=catalog,
        preview_path=preview_path,
        output_path=review_path,
    )
    pricing = summarize_usage_and_list_price(output_dir / "gemini")
    execution_pricing = summarize_usage_files(
        [execution_interaction] if execution_interaction is not None else [],
        relative_to=output_dir,
    )
    write_json(output_dir / "pricing.json", pricing)
    write_json(output_dir / "execution-pricing.json", execution_pricing)
    result = {
        "decision_path": str(decision_path.resolve()),
        "review_path": str(review_path.resolve()),
        "preview_path": str(preview_path) if preview_path is not None else None,
        "dense_catalog_path": str(dense_catalog_path.resolve()),
        "variant_id": variant_id,
        "gemini_reused": reused,
        "frame_count": len(catalog.frames),
        "sampling_fps": sampling_fps,
        "window_start_ms": window_start_ms,
        "window_end_ms": window_end_ms,
        "elapsed_seconds": round(monotonic() - started, 3),
        "execution_pricing": execution_pricing,
    }
    write_json(output_dir / "result.json", result)
    return result


def run_video_trim_intent_event(
    clip_run_dir: Path,
    event_id: str,
    anchor_time_ms: int,
    output_dir: Path,
    *,
    prompt_template: str,
    editorial_intent: str,
    temperature: float = 0.1,
) -> dict[str, Any]:
    """Watch the complete proxy, then resolve its coarse MM:SS bounds to source PTS."""
    started = monotonic()
    card = FullClipCard.model_validate(
        read_json(clip_run_dir / "gemini" / "clip-card" / "clip_card.json")
    )
    event = next((item for item in card.events if item.event_id == event_id), None)
    if event is None:
        raise ValueError(f"unknown Clip Card event: {event_id}")
    source_path = Path(read_json(clip_run_dir / "private-source.json")["path"])
    proxy_path = clip_run_dir / "analysis-proxy.mp4"
    source_media = probe_video(source_path)
    proxy_media = probe_video(proxy_path)
    if source_media.asset_id != card.source_asset_id:
        raise ValueError("private source identity differs from Clip Card")
    if proxy_media.asset_id != card.proxy_asset_id:
        raise ValueError("analysis proxy identity differs from Clip Card")
    shots = ShotManifest.model_validate(read_json(clip_run_dir / "shots.json"))
    shot = next(
        (
            item
            for item in shots.shots
            if item.start_time_ms <= anchor_time_ms < item.end_time_ms
        ),
        None,
    )
    if shot is None:
        raise ValueError("selected catalog anchor does not belong to a decoded shot")
    event_start_ms = mmss_to_ms(event.start_mmss)
    event_end_ms = mmss_to_ms(event.end_mmss)
    allowed_start_ms = max(event_start_ms, shot.start_time_ms)
    allowed_end_ms = min(event_end_ms, shot.end_time_ms, source_media.duration_ms)
    allowed_start_second = (allowed_start_ms + 999) // 1000
    allowed_end_second = allowed_end_ms // 1000
    if allowed_end_second <= allowed_start_second:
        raise ValueError("selected event/shot has no whole-second trim interval")

    def as_mmss(second: int) -> str:
        return f"{second // 60:02d}:{second % 60:02d}"

    allowed_start_mmss = as_mmss(allowed_start_second)
    allowed_end_mmss = as_mmss(allowed_end_second)
    request_identity = {
        "contract_version": "video-trim-mmss-to-source-pts-v1",
        "model": MODEL_ID,
        "temperature": temperature,
        "source_asset_id": card.source_asset_id,
        "proxy_asset_id": card.proxy_asset_id,
        "event": event.model_dump(mode="json"),
        "anchor_time_ms": anchor_time_ms,
        "shot_id": shot.shot_id,
        "allowed_start_mmss": allowed_start_mmss,
        "allowed_end_mmss": allowed_end_mmss,
        "editorial_intent": editorial_intent,
        "prompt_sha256": hashlib.sha256(prompt_template.encode("utf-8")).hexdigest(),
        "schema_sha256": _canonical_hash(VideoTrimIntentProposal.model_json_schema()),
    }
    variant_id = _canonical_hash(request_identity)[:16]
    variant_dir = output_dir / "gemini-video" / variant_id
    proposal_path = variant_dir / "video_trim_intent.json"
    execution_interaction: Path | None = None
    reused = proposal_path.exists()
    raw_response_reparsed = False
    file_api_reused = False
    if reused:
        proposal = VideoTrimIntentProposal.model_validate(read_json(proposal_path))
    else:
        prior_raw_outputs = sorted(
            variant_dir.glob("attempt-*/video_trim_intent.raw_output.json"),
            reverse=True,
        )
        for raw_output_path in prior_raw_outputs:
            raw_output = read_json(raw_output_path)
            output_text = raw_output.get("output_text")
            if not isinstance(output_text, str):
                continue
            try:
                proposal = VideoTrimIntentProposal.model_validate_json(output_text)
            except Exception:
                continue
            if (
                proposal.source_asset_id != card.source_asset_id
                or proposal.event_id != event.event_id
            ):
                continue
            write_json(proposal_path, proposal)
            reused = True
            raw_response_reparsed = True
            break
    if not reused:
        client = GeminiLabClient(temperature=temperature)
        attempt_number = len(list(variant_dir.glob("attempt-*"))) + 1
        attempt_dir = variant_dir / f"attempt-{attempt_number:02d}"
        execution_interaction = attempt_dir / "video_trim_intent.raw_interaction.json"
        write_json(variant_dir / "request-identity.json", request_identity)
        try:
            upload_dir = _shared_upload_dir(
                proxy_media.sha256,
                file_cache_root=DEFAULT_FILE_CACHE_ROOT,
                legacy_search_root=Path(__file__).resolve().parents[2] / "artifacts",
            )
            uploaded, file_api_reused = client.ensure_video_upload(proxy_path, upload_dir)
            proposal = client.analyze_video_trim_intent(
                source_asset_id=card.source_asset_id,
                event=event,
                uploaded=uploaded,
                prompt_template=prompt_template,
                editorial_intent=editorial_intent,
                allowed_start_mmss=allowed_start_mmss,
                allowed_end_mmss=allowed_end_mmss,
                run_id=f"video-trim-{uuid.uuid4().hex[:8]}",
                run_dir=attempt_dir,
            )
        except Exception as error:
            pricing = summarize_usage_and_list_price(output_dir / "gemini-video")
            execution_pricing = summarize_usage_files(
                [execution_interaction],
                relative_to=output_dir,
            )
            write_json(output_dir / "pricing.json", pricing)
            write_json(output_dir / "execution-pricing.json", execution_pricing)
            write_json(
                output_dir / "result.json",
                {
                    "status": "failed",
                    "variant_id": variant_id,
                    "elapsed_seconds": round(monotonic() - started, 3),
                    "execution_pricing": execution_pricing,
                    "error": {"type": type(error).__name__, "message": str(error)},
                },
            )
            raise
        finally:
            client.close()
        write_json(proposal_path, proposal)

    boundary_path = output_dir / "video-trim-boundaries.json"
    evidence_paths: list[Path] = []
    if proposal.usable:
        assert proposal.recommended_in_mmss is not None
        assert proposal.recommended_out_mmss is not None
        requested_in_ms = mmss_to_ms(proposal.recommended_in_mmss)
        requested_out_ms = mmss_to_ms(proposal.recommended_out_mmss)
        if not (
            allowed_start_second * 1000
            <= requested_in_ms
            < requested_out_ms
            <= allowed_end_second * 1000
        ):
            raise ValueError("Gemini video trim bounds exceed the allowed event/shot interval")
        for label, value in (
            ("hold_start_mmss", proposal.hold_start_mmss),
            ("hold_end_mmss", proposal.hold_end_mmss),
            ("reset_start_mmss", proposal.reset_start_mmss),
        ):
            if value is None:
                continue
            value_ms = mmss_to_ms(value)
            if not (requested_in_ms <= value_ms <= requested_out_ms):
                raise ValueError(f"Gemini {label} falls outside recommended in/out")
        in_path = output_dir / "boundary-frames" / "DF000001-in.jpg"
        out_path = output_dir / "boundary-frames" / "DF000002-exclusive-out.jpg"
        in_frame = extract_frame(source_path, requested_in_ms, in_path)
        first_evidence = TrimFrameEvidence(
            frame_id="DF000001",
            requested_time_ms=requested_in_ms,
            frame_time_ms=in_frame.frame_time_ms,
            frame_pts=in_frame.frame_pts,
            frame_hash=in_frame.frame_hash,
        )
        out_evidence: TrimFrameEvidence | None = None
        out_boundary_ms = requested_out_ms
        try:
            out_frame = extract_frame(source_path, requested_out_ms, out_path)
        except MediaCommandError:
            if source_media.duration_ms - requested_out_ms > 100:
                raise
            time_base = Fraction(
                source_media.video.time_base.numerator,
                source_media.video.time_base.denominator,
            )
            start_pts = source_media.video.start_pts or 0
            out_boundary_pts = start_pts + ceil(
                Fraction(requested_out_ms, 1000) / time_base
            )
            evidence_paths = [in_path.resolve()]
            out_boundary_kind = "end_of_stream_time_boundary"
        else:
            if out_frame.frame_time_ms <= in_frame.frame_time_ms:
                raise ValueError("resolved video trim boundary frames are not chronological")
            out_evidence = TrimFrameEvidence(
                frame_id="DF000002",
                requested_time_ms=requested_out_ms,
                frame_time_ms=out_frame.frame_time_ms,
                frame_pts=out_frame.frame_pts,
                frame_hash=out_frame.frame_hash,
            )
            out_boundary_ms = out_frame.frame_time_ms
            out_boundary_pts = out_frame.frame_pts
            evidence_paths = [in_path.resolve(), out_path.resolve()]
            out_boundary_kind = "first_decoded_frame_at_or_after_request"
        write_json(
            boundary_path,
            {
                "time_semantics": "Gemini coarse MM:SS resolved to first decoded source frame at or after each request",
                "allowed_interval_mmss": [allowed_start_mmss, allowed_end_mmss],
                "requested_in_mmss": proposal.recommended_in_mmss,
                "requested_out_mmss": proposal.recommended_out_mmss,
                "first_included_frame": first_evidence.model_dump(mode="json"),
                "exclusive_out_kind": out_boundary_kind,
                "exclusive_out_time_ms": out_boundary_ms,
                "exclusive_out_pts": out_boundary_pts,
                "exclusive_out_frame": (
                    out_evidence.model_dump(mode="json")
                    if out_evidence is not None
                    else None
                ),
            },
        )
        decision = TrimIntentDecision(
            source_asset_id=card.source_asset_id,
            event_id=event.event_id,
            shot_id=shot.shot_id,
            usable=True,
            first_included_frame=first_evidence,
            last_included_frame=None,
            exclusive_out_frame=out_evidence,
            hold_start_frame=None,
            hold_end_frame=None,
            source_in_ms=in_frame.frame_time_ms,
            source_out_ms=out_boundary_ms,
            source_in_pts=in_frame.frame_pts,
            source_out_pts=out_boundary_pts,
            handle_in_ms=max(shot.start_time_ms, in_frame.frame_time_ms - 1000),
            handle_out_ms=min(shot.end_time_ms, out_boundary_ms + 1000),
            tail_intent=proposal.tail_intent,
            proposal_path=str(proposal_path.resolve()),
            catalog_path=str(boundary_path.resolve()),
        )
    else:
        write_json(
            boundary_path,
            {
                "time_semantics": "no usable Gemini video trim proposal",
                "allowed_interval_mmss": [allowed_start_mmss, allowed_end_mmss],
            },
        )
        decision = TrimIntentDecision(
            source_asset_id=card.source_asset_id,
            event_id=event.event_id,
            shot_id=shot.shot_id,
            usable=False,
            first_included_frame=None,
            last_included_frame=None,
            exclusive_out_frame=None,
            hold_start_frame=None,
            hold_end_frame=None,
            source_in_ms=None,
            source_out_ms=None,
            source_in_pts=None,
            source_out_pts=None,
            handle_in_ms=None,
            handle_out_ms=None,
            tail_intent=proposal.tail_intent,
            proposal_path=str(proposal_path.resolve()),
            catalog_path=str(boundary_path.resolve()),
        )
    decision_path = output_dir / "trim-decision.json"
    write_json(decision_path, decision)
    preview_path = _render_trim_preview(
        source_path,
        decision,
        output_dir / "trim-preview.mp4",
    )
    review_path = output_dir / "index.html"
    _render_review(
        source_path=source_path,
        event=event,
        proposal=proposal,
        decision=decision,
        catalog=None,
        preview_path=preview_path,
        evidence_paths=evidence_paths,
        output_path=review_path,
    )
    pricing = summarize_usage_and_list_price(output_dir / "gemini-video")
    execution_pricing = summarize_usage_files(
        [execution_interaction] if execution_interaction is not None else [],
        relative_to=output_dir,
    )
    write_json(output_dir / "pricing.json", pricing)
    write_json(output_dir / "execution-pricing.json", execution_pricing)
    result = {
        "status": "ok",
        "mode": "direct_video_mmss_to_source_pts",
        "decision_path": str(decision_path.resolve()),
        "review_path": str(review_path.resolve()),
        "preview_path": str(preview_path) if preview_path is not None else None,
        "variant_id": variant_id,
        "gemini_reused": reused,
        "raw_response_reparsed": raw_response_reparsed,
        "file_api_reused": file_api_reused,
        "elapsed_seconds": round(monotonic() - started, 3),
        "execution_pricing": execution_pricing,
    }
    write_json(output_dir / "result.json", result)
    return result


def review_trim_decision(
    decision_path: Path,
    output_path: Path,
    *,
    reviewer: str,
    decision: str,
    notes: str = "",
) -> TrimIntentDecision:
    if decision not in {"approved", "rejected"}:
        raise ValueError("human trim decision must be approved or rejected")
    current = TrimIntentDecision.model_validate(read_json(decision_path))
    if decision == "approved" and not current.usable:
        raise ValueError("an unusable trim proposal cannot be approved")
    reviewed = current.model_copy(
        update={
            "approval_status": decision,
            "requires_human_review": False,
            "human_review": TrimHumanReview(
                reviewer=reviewer,
                reviewed_at=utc_now(),
                decision=decision,
                notes=notes,
            ),
        }
    )
    reviewed = TrimIntentDecision.model_validate(reviewed.model_dump(mode="json"))
    write_json(output_path, reviewed)
    return reviewed
