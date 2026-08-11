"""The cut as a timeline, so somebody can disagree with one shot.

A rendered file is the machine's answer and the whole of it. Everything this
pipeline gets wrong is the same shape -- the execution was faithful and the
plan was a poor call, a shot framed on the person who was not talking -- and
no amount of self-review fixes taste. Handing over a timeline moves the tool
from final render to first assembly, which is the thing it is actually good
at: watching five minutes, finding the thirteen sentences, aligning them,
working out the crop. Then a person opens it and changes one shot.

Two things already exist and nothing can use them. Every segment is rendered
with half a second of handle either side, for exactly this, and consumed by
nobody. And every shot carries why it was chosen, why it runs that long and
what degraded -- written into report.json, which is a debugging artifact, not
something an editor reads. Here the handles are just the source being longer
than the clip, and the reasons are markers sitting on the shots they explain.

Two flavours because the two applications disagree. FCPXML is what Final Cut
reads properly; Premiere has always been happier with the older xmeml, which
Resolve also takes.

One honest limit: the crop travels along an eased path evaluated per frame,
and an NLE interpolates between keyframes. The keyframes here land exactly
where they land in the render; the acceleration between them will not match.
Same in-point, same out-point, same framing at each key, a slightly different
feel in the middle -- and the rendered file is there when that matters.
"""

from __future__ import annotations

import html
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from montagewright.executor import (
    CropBox, RenderPlan, Segment, allocate_timeline_frames,
    seconds_to_frames,
)
from montagewright.schema import looks_of, move_of_shot

FPS = 30


def _frames(seconds: float, fps: int = FPS) -> int:
    return seconds_to_frames(seconds, fps)


def _native_rate(source) -> tuple[Fraction, int, str]:
    try:
        rate = Fraction(source.native_fps)
    except (ValueError, ZeroDivisionError):
        rate = Fraction(30, 1)
    timebase = max(1, round(float(rate)))
    ntsc = "TRUE" if rate.denominator == 1001 else "FALSE"
    return rate, timebase, ntsc


def _source_frames(seconds: float, source) -> int:
    rate, _, _ = _native_rate(source)
    return seconds_to_frames(seconds, Decimal(rate.numerator) / rate.denominator)


def _keys(
    segment: Segment, fps: int | None = None,
) -> list[tuple[float, CropBox]]:
    """The crop at each moment the path names, seconds from the shot's start."""

    if segment.crop_path is not None:
        if fps and not segment.crop_path.is_static:
            from montagewright.reframe import interpolate_crop_keyframes

            source = [
                {
                    "at": key.seconds,
                    "x": key.crop.x, "y": key.crop.y,
                    "w": key.crop.width, "h": key.crop.height,
                }
                for key in segment.crop_path.keyframes
            ]
            dense = []
            for frame in range(_frames(segment.duration_seconds, fps) + 1):
                at = frame / fps
                value = interpolate_crop_keyframes(source, at)
                if value is not None:
                    dense.append((at, CropBox(
                        value["x"], value["y"], value["w"], value["h"]
                    )))
            return dense
        return [
            (key.seconds, key.crop) for key in segment.crop_path.keyframes
        ]
    if segment.crop is not None:
        return [(0.0, segment.crop)]
    return []


def _placement(
    crop: CropBox, source_aspect: float, target_aspect: float
) -> tuple[float, float, float]:
    """Scale and offset that put this crop full-frame in the sequence.

    Worked in the sequence's own terms rather than the source's: an NLE fits
    the clip first, so the question is how much bigger than that fit the
    source has to be for the crop to fill the frame, and how far to slide it
    so the crop's centre lands on the frame's.
    """

    scale = 1.0 / max(crop.width, 1e-9)
    # How large the fitted source is, in frame widths, once scaled.
    fitted_height = (target_aspect / source_aspect) * scale
    centre_x = crop.x + crop.width / 2.0
    centre_y = crop.y + crop.height / 2.0
    return (
        scale,
        (0.5 - centre_x) * scale,
        (0.5 - centre_y) * fitted_height,
    )


def _notes(clip_id: str, index: int, report: dict[str, Any]) -> list[str]:
    """Why this shot is here, in the words the layers used at the time."""

    shots = report.get("selection", {}).get("shots", [])
    shot = shots[index] if index < len(shots) else {}
    rhythm = report.get("rhythm", {}).get(clip_id, {})
    verdict = report.get("shots", {}).get(clip_id, {})

    lines = []
    if shot.get("why"):
        lines.append(f"選片：{shot['why']}")
    if rhythm.get("why"):
        lines.append(f"長度：{rhythm['why']}")
    looks = looks_of(shot)
    if looks:
        # Names the move and then where it went, since a marker read in Final
        # Cut is the only account of the shot a person has there.
        where = " → ".join(f"{one.at}（{one.framing}）" for one in looks)
        lines.append(f"運鏡：{move_of_shot(shot)}　{where}")
    for step in report.get("degradations", []):
        if step.get("clip_id") == clip_id:
            lines.append(
                f"降級：{step.get('ladder')}"
                f"（{step.get('adjudication', 'unadjudicated')}）"
            )
    if verdict.get("note"):
        mark = "做到" if verdict.get("delivered") else "沒做到"
        lines.append(f"驗收（{mark}）：{verdict['note']}")
    return lines


def to_xmeml(
    plan: RenderPlan,
    report: dict[str, Any],
    *,
    name: str,
    width: int,
    height: int,
    fps: int | None = None,
    music: Path | None = None,
    graphics: Path | None = None,
) -> str:
    """FCP7 XML: what Premiere and Resolve open without complaint.

    Takes the bed for the same reason the FCPXML writer does, and for the
    moment ignores it: xmeml wants the track laid out as a second audio
    track with its own clipitems, which is a different shape from this
    file's single video track and has not been written. Better to accept
    the argument and say so here than to have two writers that cannot be
    called the same way.
    """

    fps = fps or plan.output_fps
    files: dict[str, str] = {}
    items: list[str] = []
    markers: list[str] = []
    boundaries = allocate_timeline_frames(
        [segment.duration_seconds for segment in plan.segments], fps
    )

    for index, (segment, (start_frame, end_frame)) in enumerate(
        zip(plan.segments, boundaries)
    ):
        source = segment.source
        _, native_timebase, native_ntsc = _native_rate(source)
        native_rate_xml = (
            f"<rate><timebase>{native_timebase}</timebase>"
            f"<ntsc>{native_ntsc}</ntsc></rate>"
        )
        file_id = f"file-{source.source_id}"
        if file_id not in files:
            files[file_id] = (
                f'<file id="{escape(file_id)}">'
                f"<name>{escape(source.path.name)}</name>"
                f"<pathurl>{escape(source.path.resolve().as_uri())}</pathurl>"
                f"{native_rate_xml}"
                f"<duration>{_source_frames(source.duration_seconds, source)}</duration>"
                f"<media><video><samplecharacteristics>"
                f"<width>{source.width}</width>"
                f"<height>{source.height}</height>"
                f"</samplecharacteristics></video><audio/></media></file>"
            )
            file_ref = files[file_id]
        else:
            file_ref = f'<file id="{escape(file_id)}"/>'

        motion = ""
        keys = _keys(segment, fps)
        if keys:
            aspect = source.aspect_ratio
            target = width / height
            scale_keys, centre_keys = [], []
            for seconds, crop in keys:
                scale, offset_x, offset_y = _placement(crop, aspect, target)
                at = _frames(seconds, fps)
                scale_keys.append(
                    f"<keyframe><when>{at}</when>"
                    f"<value>{scale * 100:.4f}</value></keyframe>"
                )
                centre_keys.append(
                    f"<keyframe><when>{at}</when><value>"
                    f"<horiz>{offset_x:.6f}</horiz>"
                    f"<vert>{-offset_y:.6f}</vert>"
                    f"</value></keyframe>"
                )
            motion = (
                "<filter><effect><name>Basic Motion</name>"
                "<effectid>basic</effectid>"
                "<effectcategory>motion</effectcategory>"
                "<effecttype>motion</effecttype><mediatype>video</mediatype>"
                "<parameter><parameterid>scale</parameterid>"
                "<name>Scale</name><valuemin>0</valuemin>"
                f"<valuemax>1000</valuemax>{''.join(scale_keys)}</parameter>"
                "<parameter><parameterid>center</parameterid>"
                f"<name>Center</name>{''.join(centre_keys)}</parameter>"
                "</effect></filter>"
            )

        items.append(
            f'<clipitem id="{escape(segment.clip_id)}">'
            f"<name>{escape(source.path.stem)}</name>"
            f"{native_rate_xml}"
            f"<start>{start_frame}</start><end>{end_frame}</end>"
            f"<in>{_source_frames(segment.in_seconds, source)}</in>"
            f"<out>{_source_frames(segment.out_seconds, source)}</out>"
            f"{file_ref}{motion}</clipitem>"
        )
        for note in _notes(segment.clip_id, index, report):
            markers.append(
                f"<marker><name>{escape(note[:60])}</name>"
                f"<comment>{escape(note)}</comment>"
                f"<in>{start_frame}</in><out>{end_frame}</out></marker>"
            )
    total_frames = boundaries[-1][1] if boundaries else 0
    graphics_track = ""
    if graphics is not None and Path(graphics).exists():
        graphic_path = Path(graphics)
        graphics_track = (
            '<track><clipitem id="graphics-overlay">'
            f'<name>{escape(graphic_path.stem)}</name>'
            f'<duration>{total_frames}</duration>'
            f'<rate><timebase>{fps}</timebase><ntsc>FALSE</ntsc></rate>'
            f'<start>0</start><end>{total_frames}</end>'
            f'<in>0</in><out>{total_frames}</out>'
            '<file id="graphics-overlay-file">'
            f'<name>{escape(graphic_path.name)}</name>'
            f'<pathurl>{escape(graphic_path.resolve().as_uri())}</pathurl>'
            f'<duration>{total_frames}</duration>'
            f'<rate><timebase>{fps}</timebase><ntsc>FALSE</ntsc></rate>'
            '<media><video><samplecharacteristics>'
            f'<width>{width}</width><height>{height}</height>'
            '<alphatype>straight</alphatype>'
            '</samplecharacteristics></video></media></file>'
            '</clipitem></track>'
        )
    audio_track = "<audio/>"
    if music is not None and Path(music).exists():
        music_path = Path(music)
        audio_track = (
            "<audio><track><clipitem id=\"music-bed\">"
            f"<name>{escape(music_path.stem)}</name>"
            f"<start>0</start><end>{total_frames}</end>"
            f"<in>0</in><out>{total_frames}</out>"
            '<file id="music-bed-file">'
            f"<name>{escape(music_path.name)}</name>"
            f"<pathurl>{escape(music_path.resolve().as_uri())}</pathurl>"
            f"<rate><timebase>{fps}</timebase><ntsc>FALSE</ntsc></rate>"
            f"<duration>{total_frames}</duration>"
            "<media><audio/></media></file></clipitem></track></audio>"
        )

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE xmeml>\n<xmeml version="5"><sequence>'
        f"<name>{escape(name)}</name><duration>{total_frames}</duration>"
        f"<rate><timebase>{fps}</timebase><ntsc>FALSE</ntsc></rate>"
        f"<media><video><format><samplecharacteristics>"
        f"<width>{width}</width><height>{height}</height>"
        f"</samplecharacteristics></format>"
        f"<track>{''.join(items)}</track>{graphics_track}</video>"
        f"{audio_track}</media>"
        f"{''.join(markers)}</sequence></xmeml>\n"
    )


def to_fcpxml(
    plan: RenderPlan,
    report: dict[str, Any],
    *,
    name: str,
    width: int,
    height: int,
    fps: int | None = None,
    music: Path | None = None,
    graphics: Path | None = None,
) -> str:
    """FCPXML: what Final Cut reads properly."""

    fps = fps or plan.output_fps

    def rational(seconds: float) -> str:
        return f"{_frames(seconds, fps) * 1001}/{fps * 1001}s"

    def rational_frames(frames: int) -> str:
        return f"{frames * 1001}/{fps * 1001}s"

    def source_rational(seconds: float, source) -> str:
        rate, _, _ = _native_rate(source)
        frames = _source_frames(seconds, source)
        return f"{frames * rate.denominator}/{rate.numerator}s"

    def source_time_with_output_offset(
        source_seconds: float, offset_seconds: float, source,
    ) -> str:
        rate, _, _ = _native_rate(source)
        base_frames = _source_frames(source_seconds, source)
        value = (
            Fraction(base_frames * rate.denominator, rate.numerator)
            + Fraction(_frames(offset_seconds, fps), fps)
        )
        return f"{value.numerator}/{value.denominator}s"

    assets: dict[str, str] = {}
    clips: list[str] = []
    boundaries = allocate_timeline_frames(
        [segment.duration_seconds for segment in plan.segments], fps
    )
    # One format per distinct source geometry, and the sequence's own. An
    # asset with no format leaves Final Cut to guess what shape the media
    # is, which it does by opening the file -- and guesses wrongly when the
    # file is not where the XML says.
    shapes: dict[tuple[int, int, str], str] = {}

    for index, (segment, (start_frame, end_frame)) in enumerate(
        zip(plan.segments, boundaries)
    ):
        source = segment.source
        asset_id = f"r{len(assets) + 2}"
        existing = next(
            (
                key
                for key, value in assets.items()
                if f'name="{html.escape(source.path.stem)}"' in value
            ),
            None,
        )
        if existing is None:
            native, _, _ = _native_rate(source)
            shape = (
                int(source.width), int(source.height),
                f"{native.numerator}/{native.denominator}",
            )
            if shape not in shapes:
                shapes[shape] = f"f{len(shapes) + 1}"
            assets[asset_id] = (
                f'<asset id="{asset_id}" name="{html.escape(source.path.stem)}" '
                f'start="0s" hasVideo="1" hasAudio="1" '
                f'format="{shapes[shape]}" audioSources="1" audioChannels="2" '
                f'duration="{source_rational(source.duration_seconds, source)}">'
                f'<media-rep kind="original-media" '
                f'src="{html.escape(source.path.resolve().as_uri())}"/>'
                f"</asset>"
            )
        else:
            asset_id = existing

        adjust = ""
        keys = _keys(segment, fps)
        if keys:
            target = width / height
            # A clip's own clock starts at its source in-point, not at
            # zero, and a keyframe's time is on that clock. Written from
            # zero, the animation began before the shot did and ended before
            # it ended -- so the head and tail of every move rendered with
            # no transform at all, which for a 16:9 source in a 9:16
            # sequence is the picture letterboxed in black.
            placed = [
                (source_time_with_output_offset(
                    segment.in_seconds, seconds, source
                ),
                 _placement(crop, source.aspect_ratio, target))
                for seconds, crop in keys
            ]
            if len(placed) == 1:
                # A still frame is two attributes. It was written as <param>
                # elements carrying a time, which is not a thing FCPXML has:
                # Final Cut refused the whole file with "no declaration for
                # attribute time of element param" and imported nothing.
                _, (scale, offset_x, offset_y) = placed[0]
                adjust = (
                    f'<adjust-transform '
                    f'position="{offset_x * width:.3f} {offset_y * height:.3f}" '
                    f'scale="{scale:.5f} {scale:.5f}"/>'
                )
            else:
                # A move is keyframes, and keyframes live inside a
                # keyframeAnimation inside the param they belong to.
                moves = []
                for name, pick in (
                    ("position",
                     lambda p: f"{p[1] * width:.3f} {p[2] * height:.3f}"),
                    ("scale", lambda p: f"{p[0]:.5f} {p[0]:.5f}"),
                ):
                    frames = "".join(
                        f'<keyframe time="{at}" value="{pick(where)}"/>'
                        for at, where in placed
                    )
                    moves.append(
                        f'<param name="{name}">'
                        f"<keyframeAnimation>{frames}</keyframeAnimation>"
                        f"</param>"
                    )
                adjust = (
                    "<adjust-transform>" + "".join(moves) + "</adjust-transform>"
                )

        notes = "".join(
            f'<marker start="{rational_frames(start_frame)}" '
            f'duration="{rational(0.1)}" '
            f'value="{html.escape(note[:180])}"/>'
            for note in _notes(segment.clip_id, index, report)
        )
        clips.append(
            f'<asset-clip name="{html.escape(source.path.stem)}" '
            f'ref="{asset_id}" offset="{rational_frames(start_frame)}" '
            f'start="{source_rational(segment.in_seconds, source)}" '
            f'duration="{rational_frames(end_frame - start_frame)}">'
            f"{adjust}{notes}</asset-clip>"
        )
    total_frames = boundaries[-1][1] if boundaries else 0

    # The bed, laid under the whole cut. It was never written at all: the
    # timeline carried the picture and left the music behind, so opening it
    # gave a silent film and no sign that there had been a track.
    bed = ""
    if music is not None and Path(music).exists():
        bed_id = f"r{len(assets) + 2}"
        assets[bed_id] = (
            f'<asset id="{bed_id}" name="{html.escape(Path(music).stem)}" '
            f'start="0s" hasAudio="1" audioSources="1" audioChannels="2" '
            f'duration="{rational_frames(total_frames)}">'
            f'<media-rep kind="original-media" '
            f'src="{html.escape(Path(music).resolve().as_uri())}"/>'
            f"</asset>"
        )
        bed = (
            f'<asset-clip name="{html.escape(Path(music).stem)}" '
            f'ref="{bed_id}" lane="-1" offset="0s" start="0s" '
            f'duration="{rational_frames(total_frames)}" audioRole="music"/>'
        )

    graphic_layer = ""
    if graphics is not None and Path(graphics).exists():
        graphic_id = f"r{len(assets) + 2}"
        graphic_path = Path(graphics)
        assets[graphic_id] = (
            f'<asset id="{graphic_id}" name="{html.escape(graphic_path.stem)}" '
            f'start="0s" hasVideo="1" format="r1" '
            f'duration="{rational_frames(total_frames)}">'
            f'<media-rep kind="original-media" '
            f'src="{html.escape(graphic_path.resolve().as_uri())}"/>'
            '</asset>'
        )
        graphic_layer = (
            f'<asset-clip name="{html.escape(graphic_path.stem)}" '
            f'ref="{graphic_id}" lane="1" offset="0s" start="0s" '
            f'duration="{rational_frames(total_frames)}" videoRole="titles"/>'
        )

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE fcpxml>\n<fcpxml version="1.9"><resources>'
        # No name. FFVideoFormat is the prefix Apple gives its built-in
        # presets -- FFVideoFormat1080p30 and the like -- so a bare
        # "FFVideoFormat" sends Final Cut looking for a preset that does not
        # exist, and it warns that the sequence's format is an unexpected
        # value. A custom size does not claim to be a preset; it states its
        # own dimensions and says what colour it is in.
        f'<format id="r1" width="{width}" height="{height}" '
        f'frameDuration="1001/{fps * 1001}s" '
        f'colorSpace="1-1-1 (Rec. 709)"/>'
        + "".join(
            f'<format id="{ident}" width="{shape[0]}" height="{shape[1]}" '
            f'frameDuration="{Fraction(shape[2]).denominator}/'
            f'{Fraction(shape[2]).numerator}s" '
            f'colorSpace="1-1-1 (Rec. 709)"/>'
            for shape, ident in shapes.items()
        )
        + f"{''.join(assets.values())}</resources>"
        f'<library><event name="{html.escape(name)}">'
        f'<project name="{html.escape(name)}"><sequence format="r1" '
        f'duration="{rational_frames(total_frames)}" tcStart="0s">'
        f"<spine>{''.join(clips)}{graphic_layer}{bed}</spine></sequence></project>"
        "</event></library></fcpxml>\n"
    )


def write_timelines(
    plan: RenderPlan,
    report: dict[str, Any],
    output_dir: Path,
    *,
    name: str,
    width: int,
    height: int,
) -> tuple[Path, Path]:
    """Both flavours, beside the render they describe."""

    premiere = output_dir / f"{name}.xml"
    finalcut = output_dir / f"{name}.fcpxml"
    graphics = output_dir / "graphics-overlay.mov"
    graphics = graphics if graphics.exists() else None
    premiere.write_text(
        to_xmeml(
            plan, report, name=name, width=width, height=height,
            graphics=graphics,
        ),
        encoding="utf-8",
    )
    finalcut.write_text(
        to_fcpxml(
            plan, report, name=name, width=width, height=height,
            graphics=graphics,
        ),
        encoding="utf-8",
    )
    return premiere, finalcut
