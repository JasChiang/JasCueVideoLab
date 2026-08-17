#!/usr/bin/env python3
"""Re-render a finished Web run from durable local artifacts only.

This intentionally has no model/client argument.  Selection, rhythm, source
windows and recorded crop paths remain authoritative.  A human-confirmed
endpoint override may rebuild one crop path with the current motion compiler;
it is labelled in stdout and never written back into the source run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from montagewright.executor import plan_render
from montagewright.pipeline import probe, read_crops
from montagewright.planning_release import resolve_preferred_camera_durations
from montagewright.reframe import build_look_path, retime_crop_path
from montagewright.renderer import render
from montagewright.schema import Clip, EDL, look_energy, reframe_of, subject_of


def _crop_centre_x(frame) -> float:
    return frame.crop.x + frame.crop.width / 2.0


def _has_legacy_push_rebound(path) -> bool:
    """Detect a size-changing route that sweeps past its final landing.

    Older runs expanded a ``sequential_read`` look before compiling the
    enclosing push/pull.  Their durable crop path can therefore travel to an
    edge and return to the actual push endpoint.  Small smoothing adjustments
    are ignored; this only catches a material reversal.
    """

    centres = [_crop_centre_x(one) for one in path.keyframes]
    if len(centres) < 3:
        return False
    meaningful = [
        later - earlier
        for earlier, later in zip(centres, centres[1:])
        if abs(later - earlier) >= 0.02
    ]
    if len(meaningful) < 2:
        return False
    reverses = any(
        earlier * later < 0.0
        for earlier, later in zip(meaningful, meaningful[1:])
    )
    excursion = max(centres) - min(centres)
    endpoint_travel = abs(centres[-1] - centres[0])
    return reverses and excursion > endpoint_travel + 0.04


def _endpoint_overrides(values: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for value in values:
        clip_id, raw_x = value.split("=", 1)
        x = float(raw_x)
        if not 0.0 <= x <= 1.0:
            raise ValueError(f"endpoint x must be in 0..1: {value}")
        result[clip_id] = x
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="web run directory containing run.json/out")
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--endpoint", action="append", default=[], metavar="KXX=X",
        help="human-confirmed final crop centre for one shot",
    )
    parser.add_argument(
        "--safe-one-shot",
        action="store_true",
        help=(
            "apply only semantics-preserving local duration resolutions; "
            "never calls a model"
        ),
    )
    parser.add_argument(
        "--mobile",
        action="store_true",
        help="also write a compact 960px-high H.264 preview",
    )
    args = parser.parse_args()

    run_root = args.run.resolve()
    output = run_root / "out"
    run = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    current = json.loads(
        (output / "work" / "current-timeline.json").read_text(encoding="utf-8")
    )
    selection = report["selection"]["shots"]
    aspect_name = str(report.get("direction", {}).get("aspect") or "9:16")
    width, height = (float(one) for one in aspect_name.split(":", 1))
    target_aspect = width / height

    source_root = Path(run["source"])
    source_paths = [one for one in source_root.iterdir() if one.is_file()]
    sources = {}

    def source_for(source_id: str):
        if source_id not in sources:
            path = next((one for one in source_paths if one.stem == source_id), None)
            if path is None:
                raise FileNotFoundError(f"source {source_id} is gone from {source_root}")
            sources[source_id] = probe(source_id, path)
        return sources[source_id]

    clips = []
    for index, entry in enumerate(current["shots"]):
        shot = entry.get("manual_plan") or selection[int(entry["selection_index"])]
        source_id = str(shot["source_id"])
        source_for(source_id)
        start = float(entry["in_seconds"])
        clips.append(Clip(
            clip_id=f"k{index:02d}",
            source_id=source_id,
            approx_in_seconds=start,
            approx_out_seconds=start + float(entry["seconds"]),
            in_looks_like=subject_of(shot),
            energy_intent=shot.get("energy", "medium"),
            audio_role=entry.get("audio_role", shot.get("audio_role", "discard")),
            audio_completion=entry.get(
                "audio_completion", shot.get("audio_completion", "none")
            ),
            picture_role=entry.get(
                "picture_role", shot.get("picture_role", "primary_action")
            ),
            coverage_claim_seconds=entry.get(
                "coverage_claim_seconds", shot.get("coverage_claim_seconds")
            ),
            usable_from_seconds=float(shot.get("usable_from_seconds") or 0.0),
            usable_to_seconds=float(shot.get("usable_to_seconds") or 0.0),
            reframe=reframe_of(shot),
        ))

    edl = EDL(
        project_id=f"offline-replay-{run['run_id']}",
        clips=clips,
        music_from_seconds=float(current.get("music_from_seconds") or 0.0),
        music_spans=current.get("music_spans") or [],
    )
    if args.safe_one_shot:
        command = list(run.get("command") or [])
        duration_mode = "preferred"
        if "--duration-mode" in command:
            try:
                duration_mode = str(command[command.index("--duration-mode") + 1])
            except IndexError:
                pass
        edl, notes = resolve_preferred_camera_durations(
            edl, duration_mode=duration_mode
        )
        for note in notes:
            print(f"camera duration resolution: {note}")
    recorded = read_crops(output / "work" / "crops.json")
    paths = {}
    for index, entry in enumerate(current["shots"]):
        clip_id = f"k{index:02d}"
        selection_index = int(entry["selection_index"])
        original = recorded.get(f"k{selection_index:02d}")
        if original is None:
            continue
        old_in = float(selection[selection_index].get("start_seconds") or 0.0)
        paths[clip_id] = retime_crop_path(
            original,
            old_in_seconds=old_in,
            new_in_seconds=float(entry["in_seconds"]),
            new_duration_seconds=(
                clips[index].approx_out_seconds - clips[index].approx_in_seconds
            ),
        )

    if args.safe_one_shot:
        for index, clip in enumerate(edl.clips):
            clip_id = f"k{index:02d}"
            path = paths.get(clip_id)
            reframe = clip.reframe
            if (
                path is None
                or not path.keyframes
                or reframe is None
                or reframe.editorial_intent not in {"push_in", "pull_out"}
                or not _has_legacy_push_rebound(path)
            ):
                continue
            source = sources[clip.source_id]
            duration = clip.approx_out_seconds - clip.approx_in_seconds
            first = path.keyframes[0].crop
            last = path.keyframes[-1].crop
            paths[clip_id] = build_look_path(
                [
                    (
                        min(0.35, duration * 0.15),
                        first.x + first.width / 2.0,
                        first.y + first.height / 2.0,
                        first.width,
                    ),
                    (
                        min(0.60, duration * 0.25),
                        last.x + last.width / 2.0,
                        last.y + last.height / 2.0,
                        last.width,
                    ),
                ],
                source_aspect=source.aspect_ratio,
                target_aspect=target_aspect,
                duration_seconds=duration,
                energy=look_energy(clip.energy_intent),
                source_width=source.width,
                source_height=source.height,
                output_width=int(current["output_size"][0]),
                output_height=int(current["output_size"][1]),
                clip_id=clip_id,
                monotonic_route=False,
            )
            print(
                f"{clip_id}: removed legacy push/pull rebound; "
                "preserved recorded start and final landing"
            )

    overrides = _endpoint_overrides(args.endpoint)
    for clip_id, endpoint_x in overrides.items():
        path = paths.get(clip_id)
        if path is None or not path.keyframes:
            raise ValueError(f"{clip_id} has no recorded crop path")
        index = int(clip_id[1:])
        clip = clips[index]
        source = sources[clip.source_id]
        duration = clip.approx_out_seconds - clip.approx_in_seconds
        first = path.keyframes[0].crop
        last = path.keyframes[-1].crop
        starts_x = first.x + first.width / 2.0
        ends_y = last.y + last.height / 2.0
        paths[clip_id] = build_look_path(
            [
                (min(0.35, duration * 0.15), starts_x,
                 first.y + first.height / 2.0, first.width),
                (min(0.60, duration * 0.25), endpoint_x, ends_y, last.width),
            ],
            source_aspect=source.aspect_ratio,
            target_aspect=target_aspect,
            duration_seconds=duration,
            energy=look_energy(clip.energy_intent),
            source_width=source.width,
            source_height=source.height,
            output_width=int(current["output_size"][0]),
            output_height=int(current["output_size"][1]),
            clip_id=clip_id,
            monotonic_route=True,
        )
        print(f"{clip_id}: applied human-confirmed endpoint x={endpoint_x:.3f}")

    plan = plan_render(
        edl,
        sources,
        target_aspect=target_aspect,
        crop_paths=paths,
        output_size=tuple(current["output_size"]),
        output_fps=int(current["output_fps"]),
    )
    music = None
    command = list(run.get("command") or [])
    if "--music" in command:
        candidate = Path(command[command.index("--music") + 1])
        if candidate.exists():
            music = candidate
    args.destination.mkdir(parents=True, exist_ok=True)
    result = render(
        plan,
        args.destination,
        music=music,
        keep_segments=True,
        keep_voice=False,
        under_speech=str(report.get("direction", {}).get("music_under_speech") or "duck"),
    )
    print(result.deliverable)
    if args.mobile:
        mobile = args.destination / "deliverable-mobile.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(result.deliverable),
            "-vf", "scale=-2:960",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "29",
            "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
            str(mobile),
        ], check=True)
        print(mobile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
