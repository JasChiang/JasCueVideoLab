#!/usr/bin/env python3
"""Create a new picture edit from already-rendered, numbered A/V segments."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_json", type=Path)
    parser.add_argument("segments_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    sequence = json.loads(args.sequence_json.read_text(encoding="utf-8"))["sequence"]
    if not sequence:
        raise ValueError("sequence must not be empty")

    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    filters: list[str] = []
    concat_inputs: list[str] = []
    output_cursor = 0.0
    decisions: list[dict[str, object]] = []

    for input_index, item in enumerate(sequence):
        source_index = int(item["source_index"])
        start = float(item.get("start_seconds", 0.0))
        duration = float(item["duration_seconds"])
        if start < 0 or duration <= 0:
            raise ValueError(f"invalid trim at sequence item {input_index}")
        source = args.segments_dir / f"{source_index:02d}.mp4"
        if not source.exists():
            raise FileNotFoundError(source)
        command.extend(["-i", str(source)])
        filters.extend(
            [
                f"[{input_index}:v]trim=start={start}:duration={duration},"
                f"setpts=PTS-STARTPTS[v{input_index}]",
                f"[{input_index}:a]atrim=start={start}:duration={duration},"
                f"asetpts=PTS-STARTPTS[a{input_index}]",
            ]
        )
        concat_inputs.append(f"[v{input_index}][a{input_index}]")
        decisions.append(
            {
                **item,
                "source_segment": str(source),
                "output_start_seconds": round(output_cursor, 3),
                "output_end_seconds": round(output_cursor + duration, 3),
            }
        )
        output_cursor += duration

    filters.append(
        "".join(concat_inputs)
        + f"concat=n={len(sequence)}:v=1:a=1[outv][outa]"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[outv]",
            "-map",
            "[outa]",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(args.output),
        ]
    )
    subprocess.run(command, check=True)
    manifest = {
        "method": "resequence_existing_segments",
        "new_gemini_requests": 0,
        "estimated_new_cost_usd": 0.0,
        "duration_seconds": round(output_cursor, 3),
        "sequence": decisions,
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
