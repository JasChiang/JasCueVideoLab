from __future__ import annotations

import base64
import importlib.metadata
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

from google import genai

from .geometry import native_yxyx_to_canonical_xyxy
from .models import (
    ContentMap,
    DirectMomentMap,
    ExtractedFrame,
    GeminiNativeGroundingProposal,
    GroundingCandidate,
    GroundingProposal,
    IndexedStoryboardMap,
    MediaInfo,
    ModelProvenance,
    TargetCandidateMap,
    TemporalMap,
)
from .schema import gemini_response_schema
from .storage import append_error, utc_now, write_json


MODEL_ID = "gemini-3.5-flash"
API_NAME = "gemini_interactions"
SDK_NAME = "google-genai"


class GeminiContractError(RuntimeError):
    pass


def _provenance(run_id: str, interaction_id: str | None = None) -> ModelProvenance:
    return ModelProvenance(
        model_id=MODEL_ID,
        api=API_NAME,
        sdk=SDK_NAME,
        sdk_version=importlib.metadata.version("google-genai"),
        interaction_id=interaction_id,
        run_id=run_id,
        generated_at=utc_now(),
    )


def _raw_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=False)
    return value


def _is_file_api_not_found(error: BaseException) -> bool:
    values = [
        getattr(error, "code", None),
        getattr(error, "status_code", None),
        getattr(error, "status", None),
    ]
    text = " ".join(str(value) for value in values if value is not None).upper()
    return "404" in text or "NOT_FOUND" in text


class GeminiLabClient:
    def __init__(self, *, api_key: str | None = None, temperature: float = 0.2) -> None:
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for live Gemini calls")
        self.client = genai.Client(api_key=resolved_key)
        self.temperature = temperature

    def close(self) -> None:
        self.client.close()

    def upload_video(self, path: Path, artifact_dir: Path, timeout_seconds: int = 900) -> Any:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        try:
            if not path.exists():
                raise FileNotFoundError(path)
            # Do not resolve an ASCII artifact symlink back to a non-ASCII
            # source basename: the SDK puts the basename in an HTTP header.
            uploaded = self.client.files.upload(file=str(path.absolute()))
            write_json(artifact_dir / "file_upload_initial.json", _raw_dump(uploaded))
            deadline = time.monotonic() + timeout_seconds
            while not uploaded.state or uploaded.state.name == "PROCESSING":
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Gemini File API processing exceeded {timeout_seconds}s")
                time.sleep(5)
                uploaded = self.client.files.get(name=uploaded.name)
            write_json(artifact_dir / "file_upload_final.json", _raw_dump(uploaded))
            if uploaded.state.name != "ACTIVE":
                raise RuntimeError(f"Gemini File API ended in state {uploaded.state.name}")
            return uploaded
        except Exception as error:
            append_error(artifact_dir, "file_upload", error)
            raise

    def resume_video_upload(self, artifact_dir: Path, timeout_seconds: int = 900) -> Any:
        """Resume polling an upload recorded before the local process was interrupted."""
        artifact_dir.mkdir(parents=True, exist_ok=True)
        initial_path = artifact_dir / "file_upload_initial.json"
        try:
            if not initial_path.exists():
                raise FileNotFoundError(initial_path)
            initial = json.loads(initial_path.read_text(encoding="utf-8"))
            name = initial.get("name")
            if not isinstance(name, str) or not name:
                raise GeminiContractError("saved File API response has no file name")
            uploaded = self.client.files.get(name=name)
            deadline = time.monotonic() + timeout_seconds
            while not uploaded.state or uploaded.state.name == "PROCESSING":
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Gemini File API processing exceeded {timeout_seconds}s")
                time.sleep(5)
                uploaded = self.client.files.get(name=name)
            write_json(artifact_dir / "file_upload_final.json", _raw_dump(uploaded))
            if uploaded.state.name != "ACTIVE":
                raise RuntimeError(f"Gemini File API ended in state {uploaded.state.name}")
            return uploaded
        except Exception as error:
            append_error(artifact_dir, "file_upload_resume", error)
            raise

    def ensure_video_upload(
        self,
        path: Path,
        artifact_dir: Path,
        timeout_seconds: int = 900,
        *,
        force_reupload: bool = False,
    ) -> tuple[Any, bool]:
        """Reuse an ACTIVE saved File API object; reupload only after a confirmed 404."""
        initial_path = artifact_dir / "file_upload_initial.json"
        if initial_path.exists() and not force_reupload:
            try:
                uploaded = self.resume_video_upload(artifact_dir, timeout_seconds)
                write_json(
                    artifact_dir / "file_cache.json",
                    {"reused": True, "reason": "saved_file_api_object_is_active", "checked_at": utc_now()},
                )
                return uploaded, True
            except Exception as error:
                if not _is_file_api_not_found(error):
                    raise
                reason = "saved_file_api_object_expired_or_deleted"
        else:
            reason = "force_reupload" if force_reupload else "no_saved_file_api_object"

        if initial_path.exists():
            history_dir = artifact_dir / "history" / utc_now().replace(":", "-")
            for filename in ("file_upload_initial.json", "file_upload_final.json", "file_cache.json"):
                old_path = artifact_dir / filename
                if old_path.exists():
                    write_json(history_dir / filename, json.loads(old_path.read_text(encoding="utf-8")))
        uploaded = self.upload_video(path, artifact_dir, timeout_seconds)
        write_json(
            artifact_dir / "file_cache.json",
            {"reused": False, "reason": reason, "checked_at": utc_now()},
        )
        return uploaded, False

    def suggest_targets(
        self,
        *,
        media: MediaInfo,
        uploaded: Any,
        prompt_template: str,
        run_id: str,
        run_dir: Path,
    ) -> TargetCandidateMap:
        """Propose user-selectable targets without producing any boxes or tracking data."""
        provenance = _provenance(run_id)
        last_valid_mmss = max(0, (media.duration_ms - 1) // 1000)
        prompt = (
            prompt_template
            + "\n\n## 本次不可變輸入 metadata\n"
            + f"asset_id 必須原樣回傳：{media.asset_id}\n"
            + f"duration_ms 必須原樣回傳：{media.duration_ms}\n"
            + f"最後允許的整秒是 {last_valid_mmss // 60:02d}:{last_valid_mmss % 60:02d}\n"
            + "model_provenance 必須原樣回傳以下內容（interaction_id 先回傳 null）：\n"
            + provenance.model_dump_json()
        )
        request_record = {
            "model": MODEL_ID,
            "store": False,
            "input": [
                {"type": "video", "uri": uploaded.uri, "mime_type": uploaded.mime_type},
                {"type": "text", "text": prompt},
            ],
            "generation_config": {"temperature": self.temperature, "thinking_level": "low"},
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": gemini_response_schema(TargetCandidateMap),
            },
        }
        write_json(run_dir / "target_candidates.request.json", request_record)
        try:
            interaction = self.client.interactions.create(**request_record)
            write_json(run_dir / "target_candidates.raw_interaction.json", _raw_dump(interaction))
            write_json(
                run_dir / "target_candidates.raw_output.json",
                {"output_text": interaction.output_text},
            )
            parsed = TargetCandidateMap.model_validate_json(interaction.output_text)
            if parsed.asset_id != media.asset_id or parsed.duration_ms != media.duration_ms:
                raise GeminiContractError("Target Candidate Map echoed metadata incorrectly")
            final = parsed.model_copy(
                update={
                    "model_provenance": parsed.model_provenance.model_copy(
                        update={"interaction_id": interaction.id}
                    )
                }
            )
            write_json(run_dir / "target_candidates.json", final)
            write_json(run_dir / "target_candidates.schema_validation.json", {"ok": True, "errors": []})
            return final
        except Exception as error:
            write_json(
                run_dir / "target_candidates.schema_validation.json",
                {"ok": False, "errors": [{"type": type(error).__name__, "message": str(error)}]},
            )
            append_error(run_dir, "target_candidates", error)
            raise

    def analyze_video(
        self,
        *,
        media: MediaInfo,
        uploaded: Any,
        prompt_template: str,
        run_id: str,
        run_dir: Path,
        repair_attempts: int = 1,
    ) -> ContentMap:
        provenance = _provenance(run_id)
        base_prompt = (
            prompt_template
            + "\n\n## 本次不可變輸入 metadata\n"
            + f"asset_id 必須原樣回傳：{media.asset_id}\n"
            + f"duration_ms 必須原樣回傳：{media.duration_ms}\n"
            + "model_provenance 必須原樣回傳以下內容（interaction_id 先回傳 null）：\n"
            + provenance.model_dump_json()
        )
        previous_output: str | None = None
        previous_error: Exception | None = None
        attempt_results: list[dict[str, Any]] = []
        total_attempts = 1 + max(0, repair_attempts)

        for attempt_number in range(1, total_attempts + 1):
            prompt = base_prompt
            if attempt_number > 1:
                prompt += (
                    "\n\n## Contract 修正重試\n"
                    "前一次輸出未通過本機 Pydantic contract。請重新檢視同一支影片並重新產生完整 Content Map；"
                    "不要直接截斷超界時間、不要交換錯誤區間端點、不要補造影片結束後的事件。"
                    "若前一版聲稱的事件不在影片內，應刪除或依影片證據重新分段。\n"
                    f"不可超過的 duration_ms：{media.duration_ms}\n"
                    f"前一次驗證錯誤：{previous_error}\n"
                    "以下前次輸出僅供找錯，不是可信資料：\n"
                    + (previous_output or "<前次呼叫沒有可用 output_text>")
                )
            request_record = {
                "model": MODEL_ID,
                "store": False,
                "input": [
                    {"type": "video", "uri": uploaded.uri, "mime_type": uploaded.mime_type},
                    {"type": "text", "text": prompt},
                ],
                "generation_config": {"temperature": self.temperature, "thinking_level": "low"},
                "response_format": {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": gemini_response_schema(ContentMap),
                },
            }
            attempt_request = run_dir / f"content_map.attempt-{attempt_number:02d}.request.json"
            attempt_interaction = (
                run_dir / f"content_map.attempt-{attempt_number:02d}.raw_interaction.json"
            )
            attempt_output = run_dir / f"content_map.attempt-{attempt_number:02d}.raw_output.json"
            write_json(attempt_request, request_record)
            if attempt_number == 1:
                write_json(run_dir / "content_map.request.json", request_record)
            try:
                interaction = self.client.interactions.create(**request_record)
                previous_output = interaction.output_text
                raw_interaction = _raw_dump(interaction)
                raw_output = {"output_text": previous_output}
                write_json(attempt_interaction, raw_interaction)
                write_json(attempt_output, raw_output)
                parsed = ContentMap.model_validate_json(previous_output)
                if parsed.asset_id != media.asset_id or parsed.duration_ms != media.duration_ms:
                    raise GeminiContractError("Content Map echoed asset_id or duration_ms incorrectly")
                final = parsed.model_copy(
                    update={
                        "model_provenance": parsed.model_provenance.model_copy(
                            update={"interaction_id": interaction.id}
                        )
                    }
                )
                attempt_results.append({"attempt": attempt_number, "ok": True, "errors": []})
                write_json(run_dir / "content_map.request.json", request_record)
                write_json(run_dir / "content_map.raw_interaction.json", raw_interaction)
                write_json(run_dir / "content_map.raw_output.json", raw_output)
                write_json(run_dir / "content_map.json", final)
                write_json(
                    run_dir / "content_map.schema_validation.json",
                    {
                        "ok": True,
                        "recovered_by_repair": attempt_number > 1,
                        "successful_attempt": attempt_number,
                        "attempts": attempt_results,
                        "errors": [],
                    },
                )
                return final
            except Exception as error:
                previous_error = error
                detail = {"type": type(error).__name__, "message": str(error)}
                attempt_results.append({"attempt": attempt_number, "ok": False, "errors": [detail]})
                append_error(run_dir, f"content_map_attempt_{attempt_number:02d}", error)

        write_json(
            run_dir / "content_map.schema_validation.json",
            {
                "ok": False,
                "recovered_by_repair": False,
                "successful_attempt": None,
                "attempts": attempt_results,
                "errors": attempt_results[-1]["errors"] if attempt_results else [],
            },
        )
        if previous_error is None:
            raise GeminiContractError("Content Map failed without a recorded exception")
        raise previous_error

    def ground_frame(
        self,
        *,
        media: MediaInfo,
        frame: ExtractedFrame,
        event_id: str,
        event_description: str,
        entity_id: str,
        target_description: str,
        prompt_template: str,
        run_id: str,
        output_dir: Path,
    ) -> GroundingProposal:
        provenance = _provenance(run_id)
        replacements = {
            "target_description": target_description,
            "event_description": event_description,
            "entity_id": entity_id,
            "frame_time_ms": str(frame.frame_time_ms),
            "frame_pts": str(frame.frame_pts),
            "source_width": str(frame.width),
            "source_height": str(frame.height),
        }
        prompt = prompt_template
        for key, value in replacements.items():
            prompt = prompt.replace("{{" + key + "}}", value)
        prompt += (
            "\n\n## 本次不可變輸入 metadata\n"
            + f"asset_id: {media.asset_id}\n"
            + f"event_id: {event_id}\n"
            + f"entity_id: {entity_id}\n"
            + f"frame_pts: {frame.frame_pts}\n"
            + f"frame_time_ms: {frame.frame_time_ms}\n"
            + f"frame_hash: {frame.frame_hash}\n"
            + f"source_width: {frame.width}\n"
            + f"source_height: {frame.height}\n"
            + "上述欄位必須原樣回傳。model_provenance 必須原樣回傳以下內容（interaction_id 先回傳 null）：\n"
            + provenance.model_dump_json()
        )
        image_data = base64.b64encode(Path(frame.path).read_bytes()).decode("ascii")
        mime_type = mimetypes.guess_type(frame.path)[0] or "image/png"
        api_request = {
            "model": MODEL_ID,
            "store": False,
            "input": [
                {"type": "image", "data": image_data, "mime_type": mime_type},
                {"type": "text", "text": prompt},
            ],
            "generation_config": {"temperature": self.temperature, "thinking_level": "low"},
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": gemini_response_schema(GeminiNativeGroundingProposal),
            },
        }
        request_record = {
            **api_request,
            "input": [
                {"type": "image", "mime_type": mime_type, "sha256": frame.frame_hash},
                api_request["input"][1],
            ],
            "api_coordinate_order": "ymin,xmin,ymax,xmax",
            "canonical_coordinate_order": "xmin,ymin,xmax,ymax",
        }
        write_json(output_dir / "grounding.request.json", request_record)
        try:
            interaction = self.client.interactions.create(**api_request)
            write_json(output_dir / "grounding.raw_interaction.json", _raw_dump(interaction))
            write_json(output_dir / "grounding.raw_output.json", {"output_text": interaction.output_text})
            parsed = GeminiNativeGroundingProposal.model_validate_json(interaction.output_text)
            expected = {
                "asset_id": media.asset_id,
                "event_id": event_id,
                "entity_id": entity_id,
                "frame_pts": frame.frame_pts,
                "frame_time_ms": frame.frame_time_ms,
                "frame_hash": frame.frame_hash,
                "source_width": frame.width,
                "source_height": frame.height,
            }
            mismatches = {
                key: {"expected": value, "actual": getattr(parsed, key)}
                for key, value in expected.items()
                if getattr(parsed, key) != value
            }
            if mismatches:
                raise GeminiContractError(f"Grounding metadata mismatch: {mismatches}")
            native_final = parsed.model_copy(
                update={
                    "model_provenance": parsed.model_provenance.model_copy(
                        update={"interaction_id": interaction.id}
                    )
                }
            )
            write_json(output_dir / "grounding.native.json", native_final)
            final = GroundingProposal(
                asset_id=native_final.asset_id,
                event_id=native_final.event_id,
                entity_id=native_final.entity_id,
                frame_pts=native_final.frame_pts,
                frame_time_ms=native_final.frame_time_ms,
                frame_hash=native_final.frame_hash,
                source_width=native_final.source_width,
                source_height=native_final.source_height,
                visible=native_final.visible,
                occlusion=native_final.occlusion,
                visibility_reason=native_final.visibility_reason,
                candidates=[
                    GroundingCandidate(
                        box_2d=native_yxyx_to_canonical_xyxy(candidate.box_2d_yxyx),
                        label=candidate.label,
                        confidence=candidate.confidence,
                        disambiguation_reason=candidate.disambiguation_reason,
                    )
                    for candidate in native_final.candidates
                ],
                model_provenance=native_final.model_provenance,
            )
            write_json(output_dir / "grounding.json", final)
            write_json(
                output_dir / "grounding.coordinate_transform.json",
                {
                    "api_field": "box_2d_yxyx",
                    "api_order": ["y_min", "x_min", "y_max", "x_max"],
                    "canonical_field": "box_2d",
                    "canonical_order": ["x_min", "y_min", "x_max", "y_max"],
                    "method": "deterministic axis reorder; no heuristic inference",
                },
            )
            write_json(
                output_dir / "grounding.schema_validation.json",
                {"ok": True, "api_native_schema": True, "canonical_schema": True, "errors": []},
            )
            return final
        except Exception as error:
            write_json(
                output_dir / "grounding.schema_validation.json",
                {"ok": False, "errors": [{"type": type(error).__name__, "message": str(error)}]},
            )
            append_error(output_dir, "grounding", error)
            raise

    def analyze_temporal_video(
        self,
        *,
        media: MediaInfo,
        uploaded: Any,
        prompt_template: str,
        run_id: str,
        run_dir: Path,
    ) -> TemporalMap:
        """Run a deliberately small timing-only pass for prompt-complexity A/B testing."""
        provenance = _provenance(run_id)
        prompt = (
            prompt_template
            + "\n\n## 本次不可變輸入 metadata\n"
            + f"asset_id 必須原樣回傳：{media.asset_id}\n"
            + f"duration_ms 必須原樣回傳：{media.duration_ms}\n"
            + "model_provenance 必須原樣回傳以下內容（interaction_id 先回傳 null）：\n"
            + provenance.model_dump_json()
        )
        request_record = {
            "model": MODEL_ID,
            "store": False,
            "input": [
                {"type": "video", "uri": uploaded.uri, "mime_type": uploaded.mime_type},
                {"type": "text", "text": prompt},
            ],
            "generation_config": {"temperature": self.temperature, "thinking_level": "low"},
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": gemini_response_schema(TemporalMap),
            },
        }
        write_json(run_dir / "temporal_map.request.json", request_record)
        try:
            interaction = self.client.interactions.create(**request_record)
            raw_interaction = _raw_dump(interaction)
            raw_output = {"output_text": interaction.output_text}
            write_json(run_dir / "temporal_map.raw_interaction.json", raw_interaction)
            write_json(run_dir / "temporal_map.raw_output.json", raw_output)
            parsed = TemporalMap.model_validate_json(interaction.output_text)
            if parsed.asset_id != media.asset_id or parsed.duration_ms != media.duration_ms:
                raise GeminiContractError("Temporal Map echoed asset_id or duration_ms incorrectly")
            final = parsed.model_copy(
                update={
                    "model_provenance": parsed.model_provenance.model_copy(
                        update={"interaction_id": interaction.id}
                    )
                }
            )
            write_json(run_dir / "temporal_map.json", final)
            write_json(run_dir / "temporal_map.schema_validation.json", {"ok": True, "errors": []})
            return final
        except Exception as error:
            write_json(
                run_dir / "temporal_map.schema_validation.json",
                {"ok": False, "errors": [{"type": type(error).__name__, "message": str(error)}]},
            )
            append_error(run_dir, "temporal_map", error)
            raise

    def analyze_indexed_storyboard(
        self,
        *,
        media: MediaInfo,
        frames: list[dict[str, Any]],
        prompt_template: str,
        run_id: str,
        run_dir: Path,
    ) -> IndexedStoryboardMap:
        """Let Gemini select immutable frame IDs instead of generating timestamps."""
        provenance = _provenance(run_id)
        prompt = (
            prompt_template
            + "\n\n## 本次不可變輸入 metadata\n"
            + f"asset_id 必須原樣回傳：{media.asset_id}\n"
            + f"duration_ms 必須原樣回傳：{media.duration_ms}\n"
            + "model_provenance 必須原樣回傳以下內容（interaction_id 先回傳 null）：\n"
            + provenance.model_dump_json()
        )
        api_input: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        recorded_input: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        ordered_ids: list[str] = []
        for frame in frames:
            frame_id = str(frame["frame_id"])
            ordered_ids.append(frame_id)
            label = (
                f"FRAME_ID={frame_id}; exact_frame_pts={frame['frame_pts']}; "
                f"exact_frame_time_ms={frame['frame_time_ms']}"
            )
            data = base64.b64encode(Path(frame["image_path"]).read_bytes()).decode("ascii")
            mime_type = mimetypes.guess_type(str(frame["image_path"]))[0] or "image/jpeg"
            api_input.extend(
                [
                    {"type": "text", "text": label},
                    {"type": "image", "data": data, "mime_type": mime_type},
                ]
            )
            recorded_input.extend(
                [
                    {"type": "text", "text": label},
                    {
                        "type": "image",
                        "mime_type": mime_type,
                        "sha256": frame["image_hash"],
                    },
                ]
            )
        api_request = {
            "model": MODEL_ID,
            "store": False,
            "input": api_input,
            "generation_config": {"temperature": self.temperature, "thinking_level": "low"},
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": gemini_response_schema(IndexedStoryboardMap),
            },
        }
        write_json(
            run_dir / "indexed_storyboard.request.json",
            {**api_request, "input": recorded_input, "frame_ids_in_order": ordered_ids},
        )
        try:
            interaction = self.client.interactions.create(**api_request)
            write_json(run_dir / "indexed_storyboard.raw_interaction.json", _raw_dump(interaction))
            write_json(
                run_dir / "indexed_storyboard.raw_output.json",
                {"output_text": interaction.output_text},
            )
            parsed = IndexedStoryboardMap.model_validate_json(interaction.output_text)
            if parsed.asset_id != media.asset_id or parsed.duration_ms != media.duration_ms:
                raise GeminiContractError("Indexed Storyboard echoed metadata incorrectly")
            positions = {frame_id: index for index, frame_id in enumerate(ordered_ids)}
            previous_last = -1
            for event in parsed.events:
                selected = [
                    event.first_frame_id,
                    event.recommended_frame_id,
                    event.last_frame_id,
                ]
                unknown = [frame_id for frame_id in selected if frame_id not in positions]
                if unknown:
                    raise GeminiContractError(f"unknown storyboard frame IDs: {unknown}")
                first, recommended, last = (positions[frame_id] for frame_id in selected)
                if not first <= recommended <= last:
                    raise GeminiContractError(
                        f"event {event.event_id} frame IDs are not first <= recommended <= last"
                    )
                if first <= previous_last:
                    raise GeminiContractError(f"event {event.event_id} overlaps or is out of order")
                previous_last = last
            final = parsed.model_copy(
                update={
                    "model_provenance": parsed.model_provenance.model_copy(
                        update={"interaction_id": interaction.id}
                    )
                }
            )
            write_json(run_dir / "indexed_storyboard.json", final)
            write_json(run_dir / "indexed_storyboard.schema_validation.json", {"ok": True, "errors": []})
            return final
        except Exception as error:
            write_json(
                run_dir / "indexed_storyboard.schema_validation.json",
                {"ok": False, "errors": [{"type": type(error).__name__, "message": str(error)}]},
            )
            append_error(run_dir, "indexed_storyboard", error)
            raise

    def analyze_direct_moments(
        self,
        *,
        media: MediaInfo,
        uploaded: Any,
        prompt_template: str,
        run_id: str,
        run_dir: Path,
        locked_target_id: str | None = None,
        locked_target_description: str | None = None,
    ) -> DirectMomentMap:
        """Ask directly for a few MM:SS screenshot moments, without event boundaries."""
        provenance = _provenance(run_id)
        last_valid_mmss = max(0, (media.duration_ms - 1) // 1000)
        prompt = (
            prompt_template
            + "\n\n## 本次不可變輸入 metadata\n"
            + f"asset_id 必須原樣回傳：{media.asset_id}\n"
            + f"duration_ms 必須原樣回傳：{media.duration_ms}\n"
            + f"最後允許的整秒是 {last_valid_mmss // 60:02d}:{last_valid_mmss % 60:02d}\n"
            + "model_provenance 必須原樣回傳以下內容（interaction_id 先回傳 null）：\n"
            + provenance.model_dump_json()
        )
        if locked_target_id is not None or locked_target_description is not None:
            if not locked_target_id or not locked_target_description:
                raise ValueError("locked target id and description must be provided together")
            prompt += (
                "\n\n## 使用者指定的不可變 Grounding target\n"
                + f"grounding_target_id 必須逐字回傳：{locked_target_id}\n"
                + "grounding_target_description 必須逐字回傳："
                + locked_target_description
                + "\n不得改選背板、相似物件或其他展示品。"
            )
        request_record = {
            "model": MODEL_ID,
            "store": False,
            "input": [
                {"type": "video", "uri": uploaded.uri, "mime_type": uploaded.mime_type},
                {"type": "text", "text": prompt},
            ],
            "generation_config": {"temperature": self.temperature, "thinking_level": "low"},
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": gemini_response_schema(DirectMomentMap),
            },
        }
        write_json(run_dir / "direct_moments.request.json", request_record)
        try:
            interaction = self.client.interactions.create(**request_record)
            write_json(run_dir / "direct_moments.raw_interaction.json", _raw_dump(interaction))
            write_json(
                run_dir / "direct_moments.raw_output.json",
                {"output_text": interaction.output_text},
            )
            parsed = DirectMomentMap.model_validate_json(interaction.output_text)
            if parsed.asset_id != media.asset_id or parsed.duration_ms != media.duration_ms:
                raise GeminiContractError("Direct Moment Map echoed metadata incorrectly")
            if locked_target_id is not None:
                target_mismatches = [
                    moment.moment_id
                    for moment in parsed.moments
                    if moment.grounding_target_id != locked_target_id
                    or moment.grounding_target_description != locked_target_description
                ]
                if target_mismatches:
                    raise GeminiContractError(
                        f"Direct Moment Map changed locked target in moments: {target_mismatches}"
                    )
            final = parsed.model_copy(
                update={
                    "model_provenance": parsed.model_provenance.model_copy(
                        update={"interaction_id": interaction.id}
                    )
                }
            )
            write_json(run_dir / "direct_moments.json", final)
            write_json(run_dir / "direct_moments.schema_validation.json", {"ok": True, "errors": []})
            return final
        except Exception as error:
            write_json(
                run_dir / "direct_moments.schema_validation.json",
                {"ok": False, "errors": [{"type": type(error).__name__, "message": str(error)}]},
            )
            append_error(run_dir, "direct_moments", error)
            raise
