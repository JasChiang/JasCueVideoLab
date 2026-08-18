"""Single semantic authority for camera intent, routing and camera faults."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal


FaultSeverity = Literal["blocking_shot", "advisory", "note"]


@dataclass(frozen=True)
class Fault:
    shot_key: str
    kind: str
    severity: FaultSeverity
    message: str
    remedy: str = ""
    attempt_id: str | None = None


@dataclass(frozen=True)
class CameraRoutePolicy:
    expand_sequential_read: bool
    track_during_stops: bool
    continuous_read: bool
    monotonic_route: bool


def shot_key(shot: dict[str, Any]) -> str:
    """Content-address one commitment without depending on its list index."""

    identity = {
        "commitment_id": str(shot.get("commitment_id") or ""),
        "span_id": str(shot.get("span_id") or ""),
        "start_offset": shot.get(
            "start_offset_seconds", shot.get("start_seconds", 0.0)
        ),
    }
    encoded = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def camera_route_policy(reframe: Any | None) -> CameraRoutePolicy:
    """Compile editorial intent to routing behaviour without lossy labels."""

    looks = tuple(reframe.looks) if reframe is not None else ()
    intent = str(reframe.editorial_intent) if reframe is not None else "hold"
    route_intent = intent
    if (
        reframe is not None
        and intent == "hold"
        and reframe.camera_move in {"pan", "tilt"}
    ):
        route_intent = "reveal"  # compatibility for cached v1 Reframes
    sequential = bool(
        route_intent in {"reveal", "compare", "multi_stop"}
        and len(looks) == 1
        and looks[0].presentation_intent == "sequential_read"
    )
    return CameraRoutePolicy(
        expand_sequential_read=sequential,
        track_during_stops=intent in {"follow_subject", "push_in", "pull_out"},
        continuous_read=(
            sequential
            and not any(
                look.presentation_intent == "complete_hold" for look in looks
            )
        ),
        monotonic_route=sequential,
    )


def compile_camera(
    reframe: Any | None,
    *,
    path: Any | None = None,
    duration_seconds: float = 0.0,
    stable_key: str = "",
    attempt_id: str | None = None,
) -> "tuple[Fault, ...]":
    """Compile the camera faults this shot's geometry did not deliver."""

    from montagewright.reframe import camera_delivery_faults

    messages = camera_delivery_faults(
        reframe, path, duration_seconds=duration_seconds
    ) if path is not None else ()
    faults = tuple(
        Fault(
            shot_key=stable_key,
            kind="camera_delivery",
            severity=(
                "advisory" if "crop path ends at" in message
                else "blocking_shot"
            ),
            message=message,
            remedy="review the achieved crop or choose a feasible alternate",
            attempt_id=attempt_id,
        )
        for message in messages
    )
    return faults
