#!/usr/bin/env python3
"""High-recall Gemini retrieval before the geometry-aware feature planner."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import uuid
from pathlib import Path

from google import genai
from google.genai import types

from jascue_video_lab.billing import summarize_usage_files
from jascue_video_lab.clip_card_retrieval import (
    FeatureShortlistPlan,
    compact_retrieval_card,
    validate_feature_shortlist,
)
from jascue_video_lab.gemini import MODEL_ID, _raw_dump
from jascue_video_lab.models import (
    FeatureEditBrief,
    FullClipCard,
    ModelProvenance,
    RushesCatalog,
)
from jascue_video_lab.schema import gemini_response_schema
from jascue_video_lab.storage import append_error, read_json, utc_now, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog_json", type=Path)
    parser.add_argument("brief_json", type=Path)
    parser.add_argument("prepared_library", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--thinking-level", choices=["low", "high"], default="low")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
    catalog = RushesCatalog.model_validate(read_json(args.catalog_json))
    brief = FeatureEditBrief.model_validate(read_json(args.brief_json))
    cards: dict[str, FullClipCard] = {}
    for clip in catalog.clips:
        card_path = (
            args.prepared_library
            / "clips"
            / clip.sha256[:16]
            / "gemini"
            / "clip-card"
            / "clip_card.json"
        )
        card = FullClipCard.model_validate(read_json(card_path))
        cards[card.source_asset_id] = card

    provenance = ModelProvenance(
        model_id=MODEL_ID,
        api="gemini_interactions",
        sdk="google-genai",
        sdk_version=importlib.metadata.version("google-genai"),
        run_id=f"feature-shortlist-{uuid.uuid4().hex[:12]}",
        generated_at=utc_now(),
        interaction_id=None,
    )
    evidence = [compact_retrieval_card(card) for card in cards.values()]
    prompt = f"""
你是 evidence-bound 的影片素材召回器。請先為每個 brief chapter 從完整 Clip Card
library 找出值得進入精細選片的 event 候選。本階段只做高召回 retrieval，不決定
frame、bbox、crop、剪點或最終排名。

規則：
1. brief 是使用者允許的敘事 claim，不是畫面證據。只能依 observable_evidence
   判斷候選；不得使用品牌／產品常識補足畫面。
2. chapters 必須依 brief 順序恰好回傳一次。
3. supported 回傳 2–8 個不同 asset/event；partial 回傳 1–8 個；
   not_found 回傳空 candidates。真實證據不足時不得為了數量虛構候選。
4. source_asset_id 與 event_id 只能逐字引用下方 library。
5. retrieval_reason 簡要說明畫面為何可能符合 brief，並保留衝突與風險。
6. 不輸出 frame ID、時間、座標、模型規格或未觀察到的功能。

contract_version 必須原樣回傳：clip-card-feature-shortlist-v1
project_id 必須原樣回傳：{brief.project_id}
catalog_id 必須原樣回傳：{catalog.catalog_id}
model_provenance 必須原樣回傳：
{provenance.model_dump_json(indent=2)}

## 使用者 brief
{brief.model_dump_json(indent=2)}

## 完整精簡 Clip Card library
{json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))}
""".strip()
    request = {
        "model": MODEL_ID,
        "system_instruction": (
            "Use only supplied Clip Card evidence. This is retrieval, not final "
            "selection or geometry. Never replace visible evidence with model memory."
        ),
        "store": False,
        "input": [{"type": "text", "text": prompt}],
        "generation_config": {
            "thinking_level": args.thinking_level,
            "max_output_tokens": 12_000,
        },
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": gemini_response_schema(FeatureShortlistPlan),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "feature-shortlist.request.json", request)
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1)
        ),
    )
    try:
        try:
            interaction = client.interactions.create(**request)
        except Exception as error:
            append_error(args.output_dir, "feature_shortlist", error)
            write_json(
                args.output_dir / "feature-shortlist.schema-validation.json",
                {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "request_sent": True,
                    "raw_interaction_saved": False,
                },
            )
            raise
    finally:
        client.close()
    raw_path = args.output_dir / "feature-shortlist.raw_interaction.json"
    write_json(raw_path, _raw_dump(interaction))
    write_json(
        args.output_dir / "feature-shortlist.raw_output.json",
        {"output_text": interaction.output_text},
    )
    plan = FeatureShortlistPlan.model_validate_json(interaction.output_text)
    validate_feature_shortlist(
        plan,
        brief=brief,
        catalog=catalog,
        cards=cards,
    )
    final = plan.model_copy(
        update={
            "model_provenance": plan.model_provenance.model_copy(
                update={"interaction_id": getattr(interaction, "id", None) or ""}
            )
        }
    )
    plan_path = args.output_dir / "feature-shortlist.json"
    write_json(plan_path, final)
    write_json(
        args.output_dir / "feature-shortlist.schema-validation.json",
        {"ok": True, "chapter_count": len(final.chapters)},
    )
    pricing = summarize_usage_files([raw_path], relative_to=args.output_dir)
    write_json(args.output_dir / "pricing.json", pricing)
    print(
        json.dumps(
            {
                "shortlist_path": str(plan_path.resolve()),
                "chapter_count": len(final.chapters),
                "candidate_count": sum(
                    len(chapter.candidates) for chapter in final.chapters
                ),
                "pricing": pricing,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
