#!/usr/bin/env python3
"""Verify a text claim from a Clip Card against exact original-video frames.

The original Clip Card is immutable. This script writes independent per-frame
Gemini image observations plus a local consensus record for human review.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from google import genai
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field

from jascue_video_lab.billing import summarize_usage_files
from jascue_video_lab.gemini import MODEL_ID, VISUAL_EVIDENCE_SYSTEM_INSTRUCTION
from jascue_video_lab.media import extract_frame, probe_video, sha256_file
from jascue_video_lab.schema import gemini_response_schema
from jascue_video_lab.storage import utc_now, write_json


class TextObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_hash: str
    frame_pts: int
    frame_time_ms: int
    observed_lines: list[str]
    relevant_text: str | None
    legibility: Literal["clear", "partial", "unreadable"]
    uncertain_characters: list[str]
    evidence_note: str


class CandidateAdjudication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_text: str
    decision: Literal["candidate", "other", "unreadable"]
    observed_discriminating_characters: str
    agreement_across_frames: Literal["all", "majority", "conflict"]
    evidence_note: str


def _raw_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=False)
    return value


def _normalized(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _crop_box(spec: str, width: int, height: int) -> tuple[int, int, int, int]:
    values = [float(value.strip()) for value in spec.split(",")]
    if len(values) != 4 or not all(0 <= value <= 1 for value in values):
        raise ValueError("--crop-relative must be x_min,y_min,x_max,y_max in 0..1")
    x1, y1, x2, y2 = values
    if not x1 < x2 or not y1 < y2:
        raise ValueError("--crop-relative must have increasing coordinates")
    return (
        round(x1 * width),
        round(y1 * height),
        round(x2 * width),
        round(y2 * height),
    )


def _contact_sheet(paths: list[Path], output: Path) -> None:
    opened = [Image.open(path).convert("RGB") for path in paths]
    try:
        target_width = max(image.width for image in opened)
        resized = [
            image.resize((target_width, round(image.height * target_width / image.width)))
            if image.width != target_width
            else image.copy()
            for image in opened
        ]
        label_height = 46
        sheet = Image.new(
            "RGB",
            (target_width, sum(image.height + label_height for image in resized)),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        y = 0
        for index, image in enumerate(resized, start=1):
            draw.text((12, y + 12), f"Exact-frame crop {index}", fill="black")
            y += label_height
            sheet.paste(image, (0, y))
            y += image.height
        output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output, quality=94)
    finally:
        for image in opened:
            image.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("clip_card", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--original-claim", required=True)
    parser.add_argument("--verification-target", required=True)
    parser.add_argument(
        "--candidates",
        default="",
        help="Pipe-separated candidate strings for a separate constrained adjudication",
    )
    parser.add_argument("--times-ms", default="2000,3000,4000")
    parser.add_argument(
        "--crop-relative",
        default="0.53,0.84,0.84,1.0",
        help="x_min,y_min,x_max,y_max in orientation-corrected frame coordinates",
    )
    parser.add_argument("--resolution", default="high", choices=["medium", "high", "ultra_high"])
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    media = probe_video(args.source)
    original_card = json.loads(args.clip_card.read_text(encoding="utf-8"))
    if original_card.get("source_asset_id") != media.asset_id:
        raise ValueError("Clip Card source_asset_id does not match source video")

    requested_times = [int(value) for value in args.times_ms.split(",")]
    frame_records: list[dict[str, Any]] = []
    crop_paths: list[Path] = []
    for index, requested_time_ms in enumerate(requested_times, start=1):
        frame_path = output_dir / "frames" / f"frame-{index:02d}.png"
        frame = extract_frame(args.source, requested_time_ms, frame_path)
        crop_path = output_dir / "crops" / f"crop-{index:02d}.png"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(frame_path) as image:
            crop = image.crop(_crop_box(args.crop_relative, image.width, image.height))
            crop.save(crop_path)
        crop_paths.append(crop_path)
        frame_records.append(
            {
                **frame.model_dump(mode="json"),
                "crop_path": str(crop_path.resolve()),
                "crop_hash": sha256_file(crop_path),
                "crop_relative_xyxy": [float(v) for v in args.crop_relative.split(",")],
            }
        )
    write_json(output_dir / "frames.json", frame_records)
    _contact_sheet(crop_paths, output_dir / "review-crops.jpg")

    client = genai.Client()
    observations: list[TextObservation] = []
    latencies: list[float] = []
    try:
        for index, (record, crop_path) in enumerate(zip(frame_records, crop_paths), start=1):
            prompt = f"""你是精確的畫面文字轉錄器。

只逐字轉錄這張裁切影像中可直接看見的文字。本次驗證目標是：{args.verification_target}

規則：
- 不得根據產品外觀、背景、已知產品名稱或常識補字。
- 不得從候選答案中猜測；本請求沒有提供候選型號。
- 保留原始英文字母、數字與空格。
- 看不清楚的字元使用 `?`，並列入 uncertain_characters。
- relevant_text 只放與驗證目標直接相關的完整文字；完全看不清楚時回傳 null。
- 不要判斷整支影片或產品是否真的屬於某型號，只回答這張影格寫了什麼。

不可變 metadata：
frame_hash={record['frame_hash']}
frame_pts={record['frame_pts']}
frame_time_ms={record['frame_time_ms']}
以上三個欄位必須原樣回傳。"""
            image_data = base64.b64encode(crop_path.read_bytes()).decode("ascii")
            api_request = {
                "model": MODEL_ID,
                "system_instruction": VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
                "store": False,
                "input": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "data": image_data,
                        "mime_type": "image/png",
                        "resolution": args.resolution,
                    },
                ],
                "generation_config": {"thinking_level": "low"},
                "response_format": {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": gemini_response_schema(TextObservation),
                },
            }
            request_record = {
                **api_request,
                "input": [
                    api_request["input"][0],
                    {
                        "type": "image",
                        "mime_type": "image/png",
                        "resolution": args.resolution,
                        "sha256": record["crop_hash"],
                    },
                ],
            }
            request_dir = output_dir / "gemini" / f"observation-{index:02d}"
            request_path = request_dir / "request.json"
            raw_interaction_path = request_dir / "raw_interaction.json"
            raw_output_path = request_dir / "raw_output.json"
            reusable = (
                request_path.exists()
                and raw_interaction_path.exists()
                and raw_output_path.exists()
                and json.loads(request_path.read_text(encoding="utf-8")) == request_record
            )
            if reusable:
                output_text = json.loads(raw_output_path.read_text(encoding="utf-8"))["output_text"]
                latencies.append(0.0)
            else:
                write_json(request_path, request_record)
                started = time.monotonic()
                interaction = client.interactions.create(**api_request)
                latencies.append(round(time.monotonic() - started, 3))
                write_json(raw_interaction_path, _raw_dump(interaction))
                write_json(raw_output_path, {"output_text": interaction.output_text})
                output_text = interaction.output_text
            observation = TextObservation.model_validate_json(output_text)
            expected = (record["frame_hash"], record["frame_pts"], record["frame_time_ms"])
            actual = (observation.frame_hash, observation.frame_pts, observation.frame_time_ms)
            if actual != expected:
                raise ValueError(f"Gemini changed immutable frame metadata: {actual} != {expected}")
            observations.append(observation)
            write_json(request_dir / "observation.json", observation)
            write_json(request_dir / "schema_validation.json", {"ok": True, "errors": []})

        adjudication: CandidateAdjudication | None = None
        candidates = [item.strip() for item in args.candidates.split("|") if item.strip()]
        if candidates:
            choices = candidates + ["other", "unreadable"]
            prompt = f"""你是獨立的畫面文字候選驗證器。以下三張圖片是同一展示牌在不同原始影片影格的裁切。

驗證目標：{args.verification_target}
允許答案：{json.dumps(choices, ensure_ascii=False)}

規則：
- 只比較圖片中的實際字形，不使用產品知識或常識。
- selected_text 必須逐字等於允許答案之一。
- 若沒有候選完全符合，選 other；看不清楚則選 unreadable。
- observed_discriminating_characters 說明區分候選的關鍵字元，例如型號中的兩位數字。
- 不要參考先前 Clip Card 或其他模型回答；本請求未提供那些答案。"""
            api_request = {
                "model": MODEL_ID,
                "system_instruction": VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
                "store": False,
                "input": [
                    {"type": "text", "text": prompt},
                    *[
                        {
                            "type": "image",
                            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                            "mime_type": "image/png",
                            "resolution": args.resolution,
                        }
                        for path in crop_paths
                    ],
                ],
                "generation_config": {"thinking_level": "low"},
                "response_format": {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": gemini_response_schema(CandidateAdjudication),
                },
            }
            request_record = {
                **api_request,
                "input": [
                    api_request["input"][0],
                    *[
                        {
                            "type": "image",
                            "mime_type": "image/png",
                            "resolution": args.resolution,
                            "sha256": record["crop_hash"],
                        }
                        for record in frame_records
                    ],
                ],
            }
            request_dir = output_dir / "gemini" / "candidate-adjudication"
            write_json(request_dir / "request.json", request_record)
            started = time.monotonic()
            interaction = client.interactions.create(**api_request)
            latencies.append(round(time.monotonic() - started, 3))
            write_json(request_dir / "raw_interaction.json", _raw_dump(interaction))
            write_json(request_dir / "raw_output.json", {"output_text": interaction.output_text})
            adjudication = CandidateAdjudication.model_validate_json(interaction.output_text)
            if adjudication.selected_text not in choices:
                raise ValueError("Gemini returned a candidate outside the allowed choices")
            write_json(request_dir / "adjudication.json", adjudication)
            write_json(request_dir / "schema_validation.json", {"ok": True, "errors": []})
    finally:
        client.close()

    normalized = [_normalized(item.relevant_text) for item in observations]
    counts = Counter(value for value in normalized if value)
    consensus_key, consensus_count = counts.most_common(1)[0] if counts else ("", 0)
    consensus_text = next(
        (item.relevant_text for item in observations if _normalized(item.relevant_text) == consensus_key),
        None,
    )
    original_matches = consensus_key == _normalized(args.original_claim)
    unanimous = bool(consensus_key) and consensus_count == len(observations)
    status = (
        "needs_human_review"
        if unanimous and not original_matches
        else "machine_verified"
        if unanimous
        else "uncertain"
    )
    constrained_text = adjudication.selected_text if adjudication else None
    machine_methods_agree = (
        adjudication is None or _normalized(constrained_text) == consensus_key
    )
    if not machine_methods_agree:
        status = "needs_human_review"
    billing = summarize_usage_files(
        list((output_dir / "gemini").rglob("raw_interaction.json")),
        relative_to=output_dir,
    )
    result = {
        "verification_version": 1,
        "source_asset_id": media.asset_id,
        "source_clip_id": args.clip_id,
        "original_clip_card_path": str(args.clip_card.resolve()),
        "original_clip_card_sha256": hashlib.sha256(args.clip_card.read_bytes()).hexdigest(),
        "original_claim": args.original_claim,
        "verification_target": args.verification_target,
        "status": status,
        "machine_consensus_text": consensus_text,
        "machine_consensus_count": consensus_count,
        "observation_count": len(observations),
        "original_claim_matches_consensus": original_matches,
        "observations": [item.model_dump(mode="json") for item in observations],
        "candidate_adjudication": (
            adjudication.model_dump(mode="json") if adjudication else None
        ),
        "machine_methods_agree": machine_methods_agree,
        "human_review": {"status": "pending", "reviewer": None, "decision": None},
        "recommended_action": (
            "show exact-frame crops and both claims to a human; do not overwrite or reject"
            if status == "needs_human_review"
            else "retain evidence and continue"
        ),
        "model": MODEL_ID,
        "image_resolution": args.resolution,
        "latency_seconds_by_request": latencies,
        "billing": billing,
        "generated_at": utc_now(),
    }
    write_json(output_dir / "clip_card.verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
