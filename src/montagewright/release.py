"""Local release gate: one writer, measured master, explicit approval."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import subprocess
from datetime import date
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from montagewright.job import EditJob


class OutputBusy(RuntimeError):
    pass


@dataclass
class OutputLease:
    path: Path
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        if payload.get("pid") == os.getpid():
            self.path.unlink(missing_ok=True)
        self.released = True

    def __del__(self) -> None:
        # CPython drops the command-local lease on every return/exception;
        # atexit remains the crash-safe fallback for normal interpreter exit.
        self.release()


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_output_lease(output: Path) -> OutputLease:
    output.mkdir(parents=True, exist_ok=True)
    path = output / ".montagewright.lock"
    payload = json.dumps({"pid": os.getpid()}, separators=(",", ":"))
    while True:
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                pid = int(existing.get("pid", 0))
            except (OSError, ValueError, TypeError):
                pid = 0
            if _alive(pid):
                raise OutputBusy(
                    f"output is already being written by process {pid}: {output}"
                )
            path.unlink(missing_ok=True)
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        lease = OutputLease(path)
        atexit.register(lease.release)
        return lease


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ReleaseManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["montagewright-release-v1"] = "montagewright-release-v1"
    status: Literal["draft", "released"]
    artifact: str
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ingest_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    blockers: tuple[str, ...] = ()
    approver: str | None = None
    approval_note: str | None = None


def technical_qc_faults(
    artifact: Path, job: EditJob, *, expected_duration: float | None = None,
) -> tuple[str, ...]:
    """Measure the encoded master against the delivery sheet."""

    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(artifact)],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        return (f"master cannot be probed: {completed.stderr.strip()}",)
    try:
        payload = json.loads(completed.stdout)
    except ValueError as error:
        return (f"master probe returned invalid JSON: {error}",)
    streams = payload.get("streams") or []
    picture = next((one for one in streams if one.get("codec_type") == "video"), None)
    audio = next((one for one in streams if one.get("codec_type") == "audio"), None)
    faults: list[str] = []
    if picture is None:
        faults.append("master has no video stream")
    else:
        if job.delivery.width is not None:
            wanted_size = (job.delivery.width, job.delivery.height)
        else:
            from montagewright.executor import delivery_size

            ratios = {"9:16": 9 / 16, "16:9": 16 / 9, "1:1": 1.0, "4:5": 4 / 5}
            wanted_size = delivery_size(ratios[job.delivery.aspect])
        actual_size = (int(picture.get("width") or 0), int(picture.get("height") or 0))
        if wanted_size is not None and actual_size != wanted_size:
            faults.append(f"master size {actual_size} does not match {wanted_size}")
        rate = str(picture.get("avg_frame_rate") or "")
        expected_rate = float(job.delivery.frame_rate)
        try:
            numerator, denominator = rate.split("/", 1)
            actual_rate = float(numerator) / float(denominator)
        except (ValueError, ZeroDivisionError):
            actual_rate = 0.0
        if abs(actual_rate - expected_rate) > 0.01:
            faults.append(
                f"master frame rate {rate or 'unknown'} does not match "
                f"{job.delivery.frame_rate}"
            )
        if picture.get("codec_name") != "h264":
            faults.append(f"master codec is {picture.get('codec_name')}, expected h264")
        if job.delivery.color in {"sdr_rec709", "normalize_to_sdr"}:
            transfer = str(picture.get("color_transfer") or "")
            primaries = str(picture.get("color_primaries") or "")
            # A few otherwise valid H.264/MP4 muxes expose the VUI colour on
            # decoded frames rather than on ffprobe's stream summary.  Read
            # one frame before declaring that the finished picture is
            # untagged; this is still measured output, never an assumption.
            if transfer not in {"bt709", "iec61966-2-1"} or primaries != "bt709":
                frame_probe = subprocess.run(
                    [
                        "ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_frames", "-read_intervals", "%+#1",
                        "-show_entries",
                        "frame=color_transfer,color_primaries", "-of", "json",
                        str(artifact),
                    ],
                    capture_output=True, text=True, check=False,
                )
                try:
                    frames = json.loads(frame_probe.stdout).get("frames") or []
                    first = frames[0] if frames else {}
                    transfer = str(first.get("color_transfer") or transfer)
                    primaries = str(first.get("color_primaries") or primaries)
                except (ValueError, TypeError):
                    pass
            if transfer not in {"bt709", "iec61966-2-1"} or primaries != "bt709":
                faults.append(
                    f"master colour tags are {primaries or 'unknown'}/"
                    f"{transfer or 'unknown'}, expected Rec.709"
                )
    if audio is None:
        faults.append("master has no audio stream")
    try:
        duration = float((payload.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        duration = 0.0
    if expected_duration is not None and abs(duration - expected_duration) > 0.08:
        faults.append(
            f"master duration {duration:.3f}s differs from resolved timeline "
            f"{expected_duration:.3f}s"
        )
    if job.delivery.duration_mode == "exact" and job.delivery.seconds > 0:
        tolerance = max(0.05, 1.0 / float(job.delivery.frame_rate))
        if abs(duration - job.delivery.seconds) > tolerance:
            faults.append(
                f"exact delivery is {duration:.3f}s, needs {job.delivery.seconds:.3f}s"
            )
    if job.delivery.duration_mode == "range":
        minimum = float(job.delivery.minimum_seconds or 0.0)
        maximum = float(job.delivery.maximum_seconds or 0.0)
        tolerance = max(0.05, 1.0 / float(job.delivery.frame_rate))
        if duration < minimum - tolerance or duration > maximum + tolerance:
            faults.append(
                f"ranged delivery is {duration:.3f}s, must be between "
                f"{minimum:.3f}s and {maximum:.3f}s"
            )

    # The renderer's loudnorm/limiter request is not proof of its output.
    # ebur128 measures the finished encoded stream, including subtitle copies.
    if audio is not None:
        measured = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", str(artifact),
             "-filter_complex", "ebur128=peak=true", "-f", "null", "-"],
            capture_output=True, text=True, check=False,
        )
        matches = re.findall(r"I:\s*(-?[0-9.]+) LUFS", measured.stderr)
        peaks = re.findall(r"Peak:\s*(-?[0-9.]+) dBFS", measured.stderr)
        if measured.returncode or not matches:
            faults.append("master loudness could not be measured")
        else:
            integrated = float(matches[-1])
            if abs(integrated - job.delivery.loudness_lufs) > 1.0:
                faults.append(
                    f"master loudness {integrated:.1f} LUFS differs from "
                    f"{job.delivery.loudness_lufs:.1f} LUFS"
                )
            if peaks and float(peaks[-1]) > -1.0:
                faults.append(f"master true peak {float(peaks[-1]):.1f} dBFS exceeds -1.0")
    return tuple(faults)


def finalize_release(
    output: Path,
    artifact: Path,
    job: EditJob,
    *,
    ingest_inventory_sha256: str,
    report_path: Path,
) -> ReleaseManifest:
    """Name a checked render draft or final and bind it to its source ingest."""

    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    blockers: list[str] = []
    artifact = artifact.resolve(strict=True)
    artifact_sha256 = _sha256(artifact)
    editorial_status = str(
        report.get("editorial_delivery_status")
        or report.get("delivery_status") or "needs_review"
    )
    if editorial_status != "ready":
        blockers.append("editorial report still requires review")
    if str(report.get("delivery_status") or "needs_review") == "needs_review":
        blockers.append("current artifact report still requires review")
    technical_qc_path = output / "work" / "technical-qc.json"
    if technical_qc_path.exists():
        try:
            technical_qc = json.loads(technical_qc_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            technical_qc = {"passed": False}
        if not technical_qc.get("passed"):
            blockers.append("current artifact failed technical QC")
    if report.get("plan_disagreements") and not job.release.approval_note:
        blockers.append("resolved timeline has release-gate disagreements")
    if not job.rights.acknowledged:
        blockers.append("asset rights have not been acknowledged")
    if job.rights.expires_on:
        try:
            if date.today() > date.fromisoformat(job.rights.expires_on):
                blockers.append(f"asset rights expired on {job.rights.expires_on}")
        except ValueError:
            blockers.append("rights.expires_on is not an ISO date")
    if job.rights.embargo_until:
        try:
            if date.today() < date.fromisoformat(job.rights.embargo_until):
                blockers.append(f"release is embargoed until {job.rights.embargo_until}")
        except ValueError:
            blockers.append("rights.embargo_until is not an ISO date")
    if job.rights.prohibited_visuals and not job.release.approval_note:
        blockers.append(
            "prohibited visuals require a producer approval note after picture review"
        )
    if job.release.producer_approval_required and not job.release.approver:
        blockers.append("producer approval has not been recorded")
    if job.release.producer_approval_required and (
        job.release.approved_artifact_sha256 != artifact_sha256
    ):
        blockers.append("producer approval is not bound to this artifact hash")
    if any(
        one.track == "picture"
        and one.kind in {"forbidden_presence", "exclusive_presence"}
        for one in job.obligations
    ) and not job.release.approval_note:
        blockers.append(
            "forbidden/exclusive picture rules require a producer note after "
            "reviewing the delivered crop"
        )
    if any(one.track == "graphic" for one in job.obligations):
        from montagewright.delivery_contract import graphic_obligation_faults
        from montagewright.graphics import GraphicsPlan

        graphics_path = output / "work" / "graphics.json"
        graphics = (
            GraphicsPlan.model_validate_json(graphics_path.read_text(encoding="utf-8"))
            if graphics_path.exists() else None
        )
        blockers.extend(graphic_obligation_faults(
            graphics, job.obligations,
            total=float(report.get("duration_seconds") or 0.0),
        ))
        if "graphics" not in artifact.stem:
            blockers.append("required graphics have not been burned into this artifact")

    released = not blockers
    destination = output / ("deliverable.mp4" if released else "draft-preview.mp4")
    if artifact != destination.resolve():
        staging = output / f".{destination.name}.staging"
        staging.write_bytes(artifact.read_bytes())
        staging.replace(destination)
        if not released and artifact.name.startswith("deliverable"):
            artifact.unlink(missing_ok=True)
    opposite = output / ("draft-preview.mp4" if released else "deliverable.mp4")
    if opposite != destination:
        opposite.unlink(missing_ok=True)
    manifest = ReleaseManifest(
        status="released" if released else "draft",
        artifact=str(destination), artifact_sha256=_sha256(destination),
        ingest_inventory_sha256=ingest_inventory_sha256,
        blockers=tuple(blockers), approver=job.release.approver,
        approval_note=job.release.approval_note,
    )
    path = output / "release-manifest.json"
    temporary = output / ".release-manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    report["editorial_delivery_status"] = editorial_status
    report["delivery_status"] = "ready" if released else "release_blocked"
    report["release"] = manifest.model_dump(mode="json")
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    temporary.replace(report_path)
    return manifest
