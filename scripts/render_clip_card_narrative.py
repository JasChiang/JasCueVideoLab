#!/usr/bin/env python3
"""Render a 16:9 review cut from a validated Clip Card narrative plan."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from jascue_video_lab.models import RushesCatalog
from jascue_video_lab.storage import read_json


def mmss_to_seconds(value: str) -> int:
    minutes, seconds = (int(part) for part in value.split(":"))
    return minutes * 60 + seconds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan_json", type=Path)
    parser.add_argument("catalog_json", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--exclude-feature", action="append", default=[])
    args = parser.parse_args()

    plan = read_json(args.plan_json)
    catalog = RushesCatalog.model_validate(read_json(args.catalog_json))
    clips = {f"sha256:{clip.sha256}": clip for clip in catalog.clips}
    chapters = [
        chapter
        for chapter in sorted(plan["chapters"], key=lambda item: item["order"])
        if chapter["feature_id"] not in set(args.exclude_feature)
    ]
    if not chapters:
        raise ValueError("no chapters remain after exclusions")

    segments_dir = args.output.parent / f"{args.output.stem}-segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    decisions: list[dict[str, object]] = []
    cursor = 0
    for index, chapter in enumerate(chapters):
        clip = clips.get(chapter["source_asset_id"])
        if clip is None:
            raise ValueError(f"unknown source asset: {chapter['source_asset_id']}")
        start = mmss_to_seconds(chapter["source_in_mmss"])
        end = mmss_to_seconds(chapter["source_out_mmss"])
        duration = end - start
        if duration <= 0:
            raise ValueError(f"empty chapter: {chapter['feature_id']}")
        fade_out = max(0.0, duration - 0.12)
        segment_path = segments_dir / f"{index:02d}-{chapter['feature_id']}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                str(start),
                "-i",
                clip.path,
                "-t",
                str(duration),
                "-vf",
                (
                    "scale=1920:1080:force_original_aspect_ratio=decrease,"
                    "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,"
                    "fps=30,setsar=1,format=yuv420p"
                ),
                "-af",
                (
                    "aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                    f"volume=0.58,afade=t=in:st=0:d=0.08,afade=t=out:st={fade_out}:d=0.12"
                ),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(segment_path),
            ],
            check=True,
        )
        decisions.append(
            {
                **chapter,
                "source_clip_id": clip.clip_id,
                "rendered_segment": str(segment_path),
                "output_start_seconds": cursor,
                "output_end_seconds": cursor + duration,
            }
        )
        cursor += duration
    args.output.parent.mkdir(parents=True, exist_ok=True)
    concat_list = segments_dir / "concat.txt"
    concat_list.write_text(
        "".join(
            f"file '{Path(item['rendered_segment']).resolve()}'\n" for item in decisions
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(args.output),
        ],
        check=True,
    )
    manifest = {
        "method": "gemini_full_clip_cards_to_reviewed_narrative_plan",
        "input_plan": str(args.plan_json),
        "excluded_features": args.exclude_feature,
        "exclusion_policy": "human evidence gate; never silently replace wrong-model footage",
        "duration_seconds": cursor,
        "chapters": decisions,
        "output": str(args.output),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
