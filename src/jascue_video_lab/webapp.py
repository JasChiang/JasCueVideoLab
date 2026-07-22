from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field, model_validator

from .blind_review import (
    BlindReviewService,
    QueryProposalOptions,
    ReviewVerdict,
)
from .media import MediaCommandError
from .models import (
    EvidenceApprovalSource,
    PredicateRequiredAt,
    StrictModel,
    TargetIdentityScope,
)


WEB_ROOT = Path(__file__).resolve().parent / "web"
MAX_UPLOAD_BYTES = int(os.environ.get("JCVL_MAX_UPLOAD_BYTES", str(2 * 1024**3)))


class CandidateRequest(StrictModel):
    runs: int = Field(default=1, ge=1, le=5)


class TargetRequest(StrictModel):
    candidate_id: str | None = None
    target_id: str | None = None
    target_description: str | None = None
    editorial_goal: str | None = None
    identity_scope: TargetIdentityScope = TargetIdentityScope.WHOLE_INSTANCE
    parent_target_id: str | None = None
    parent_target_description: str | None = None
    parent_identity_cues: tuple[str, ...] = ()
    identity_cues: tuple[str, ...] = ()
    context_cues: tuple[str, ...] = ()
    stable_exclusions: tuple[str, ...] = ()
    observable_predicate: str | None = None
    predicate_required_at: PredicateRequiredAt = PredicateRequiredAt.CANDIDATE
    predicate_precondition: str | None = None
    predicate_apex: str | None = None
    predicate_postcondition: str | None = None
    predicate_required_evidence: tuple[str, ...] = ()
    predicate_disqualifying_conditions: tuple[str, ...] = ()
    framing_required_target_ids: tuple[str, ...] = ()
    framing_preferred_target_ids: tuple[str, ...] = ()
    framing_sacrificable_target_ids: tuple[str, ...] = ()
    framing_overlay_keepout_target_ids: tuple[str, ...] = ()
    framing_intent: str | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> "TargetRequest":
        has_candidate = bool(self.candidate_id)
        has_manual = bool(self.target_id) or bool(self.target_description)
        if has_candidate == has_manual:
            raise ValueError("provide exactly one candidate or one manual target")
        if has_manual and (not self.target_id or not self.target_description):
            raise ValueError("manual target_id and target_description are required together")
        return self

    def query_options(self) -> QueryProposalOptions:
        fields = QueryProposalOptions.model_fields
        return QueryProposalOptions(
            **{
                name: getattr(self, name)
                for name in fields
            }
        )


class QueryApprovalRequest(StrictModel):
    approved_by: str = Field(min_length=1)
    proposal_id: str = Field(min_length=1)
    proposal_definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approval_source: EvidenceApprovalSource = EvidenceApprovalSource.HUMAN_REVIEW
    query_id: str | None = None
    source_reference: str | None = None
    policy_reference: str | None = None

    @model_validator(mode="after")
    def validate_authority(self) -> "QueryApprovalRequest":
        if self.approval_source != EvidenceApprovalSource.HUMAN_REVIEW:
            raise ValueError(
                "the interactive review endpoint only records human_review approval"
            )
        if self.policy_reference is not None:
            raise ValueError("interactive human approval cannot claim an auto policy")
        return self


class MomentRequest(StrictModel):
    runs: int = Field(default=1, ge=1, le=5)


class GroundRequest(StrictModel):
    moment_id: str = Field(min_length=1)
    mode: Literal["exact_frame", "direct_video"] = "exact_frame"


class ReviewRequest(StrictModel):
    verdict: ReviewVerdict
    notes: str = ""
    reviewer_name: str | None = None
    corrected_box_2d: tuple[
        Annotated[int, Field(ge=0, le=1000)],
        Annotated[int, Field(ge=0, le=1000)],
        Annotated[int, Field(ge=0, le=1000)],
        Annotated[int, Field(ge=0, le=1000)],
    ] | None = None

    @model_validator(mode="after")
    def validate_box(self) -> "ReviewRequest":
        if self.corrected_box_2d is not None:
            x_min, y_min, x_max, y_max = self.corrected_box_2d
            if x_min >= x_max or y_min >= y_max:
                raise ValueError("corrected box must satisfy xmin < xmax and ymin < ymax")
        return self


def create_app(service: BlindReviewService | None = None) -> FastAPI:
    workflow = service or BlindReviewService()
    app = FastAPI(
        title="JasCueVideoLab Blind Review",
        description="Local, human-first Gemini video Grounding validation",
        version="0.1.0",
    )
    app.state.workflow = workflow
    app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")

    @app.exception_handler(FileNotFoundError)
    async def not_found_handler(_request, error: FileNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(error)})

    @app.exception_handler(PermissionError)
    async def permission_handler(_request, error: PermissionError):
        return JSONResponse(status_code=403, content={"detail": str(error)})

    @app.exception_handler(ValueError)
    async def validation_handler(_request, error: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.exception_handler(MediaCommandError)
    async def media_handler(_request, error: MediaCommandError):
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((WEB_ROOT / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {
            "ok": True,
            "local_only": True,
            "max_upload_bytes": MAX_UPLOAD_BYTES,
        }

    @app.post("/api/sessions")
    async def create_session(
        file: UploadFile = File(...),
        use_analysis_proxy: bool = Form(True),
    ) -> dict[str, object]:
        filename = file.filename or "uploaded-video.mp4"
        incoming_dir = workflow.data_root / ".incoming"
        incoming_dir.mkdir(parents=True, exist_ok=True)
        incoming = incoming_dir / f"{uuid.uuid4().hex}.upload"
        total = 0
        try:
            with incoming.open("wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"upload exceeds {MAX_UPLOAD_BYTES} bytes",
                        )
                    handle.write(chunk)
            if total == 0:
                raise HTTPException(status_code=422, detail="empty upload")
            return await run_in_threadpool(
                workflow.create_session_from_file,
                incoming,
                original_filename=filename,
                use_analysis_proxy=use_analysis_proxy,
                move_source=True,
            )
        finally:
            await file.close()
            if incoming.exists():
                incoming.unlink()

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, object]:
        return workflow.session_view(session_id)

    @app.get("/api/sessions/{session_id}/video")
    def session_video(session_id: str) -> FileResponse:
        session = workflow._session(session_id)
        return FileResponse(Path(session["source_path"]), media_type="video/mp4")

    @app.post("/api/sessions/{session_id}/candidates")
    def candidates(session_id: str, body: CandidateRequest) -> dict[str, object]:
        return workflow.suggest_targets(session_id, runs=body.runs)

    @app.post("/api/sessions/{session_id}/target")
    def select_target(session_id: str, body: TargetRequest) -> dict[str, object]:
        selection = workflow.select_target(
            session_id,
            candidate_id=body.candidate_id,
            target_id=body.target_id,
            target_description=body.target_description,
            query_options=body.query_options(),
        )
        return selection.model_dump(mode="json")

    @app.post("/api/sessions/{session_id}/query/approve")
    def approve_query(
        session_id: str, body: QueryApprovalRequest
    ) -> dict[str, object]:
        return workflow.approve_query_proposal(
            session_id,
            approved_by=body.approved_by,
            expected_proposal_id=body.proposal_id,
            expected_proposal_definition_sha256=(
                body.proposal_definition_sha256
            ),
            approval_source=body.approval_source,
            query_id=body.query_id,
            source_reference=body.source_reference,
            policy_reference=body.policy_reference,
        )

    @app.post("/api/sessions/{session_id}/moments")
    def moments(session_id: str, body: MomentRequest) -> dict[str, object]:
        return workflow.analyze_moments(session_id, runs=body.runs)

    @app.post("/api/sessions/{session_id}/ground")
    def ground(session_id: str, body: GroundRequest) -> dict[str, object]:
        return workflow.ground_moment(
            session_id,
            moment_id=body.moment_id,
            mode=body.mode,
        )

    def _review_dir(session_id: str, review_id: str) -> Path:
        session = workflow._session(session_id)
        if review_id not in session["reviews"]:
            raise FileNotFoundError(f"unknown review {review_id}")
        return workflow._session_dir(session_id) / "reviews" / review_id

    @app.get("/api/sessions/{session_id}/reviews/{review_id}/blind-image")
    def blind_image(session_id: str, review_id: str) -> FileResponse:
        return FileResponse(_review_dir(session_id, review_id) / "blind.png", media_type="image/png")

    @app.get("/api/sessions/{session_id}/reviews/{review_id}/revealed-image")
    def revealed_image(session_id: str, review_id: str) -> FileResponse:
        workflow.reveal_review(session_id, review_id)
        return FileResponse(
            _review_dir(session_id, review_id) / "revealed.png", media_type="image/png"
        )

    @app.post("/api/sessions/{session_id}/reviews/{review_id}")
    def submit_review(
        session_id: str, review_id: str, body: ReviewRequest
    ) -> dict[str, object]:
        return workflow.submit_review(
            session_id,
            review_id,
            verdict=body.verdict,
            notes=body.notes,
            reviewer_name=body.reviewer_name,
            corrected_box_2d=body.corrected_box_2d,
        )

    @app.get("/api/sessions/{session_id}/reviews/{review_id}/reveal")
    def reveal(session_id: str, review_id: str) -> dict[str, object]:
        return workflow.reveal_review(session_id, review_id)

    @app.get("/api/sessions/{session_id}/export")
    def export(session_id: str) -> JSONResponse:
        return JSONResponse(
            workflow.export_session(session_id),
            headers={
                "Content-Disposition": f'attachment; filename="jascue-blind-review-{session_id}.json"'
            },
        )

    return app


app = create_app()
