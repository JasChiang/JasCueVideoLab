from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


_RETIRED_MESSAGE = (
    "Gemini polygon-to-SAM A/B execution is retired on this branch. "
    "The primary and supported path is a reviewed bbox seed passed to SAM; "
    "historical polygon artifacts remain read-only in the report."
)


def run_segmentation_seed_ab(
    *,
    case_dir: Path,
    checkpoint_path: Path,
    target_description: str,
    event_description: str,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    """Reject the retired Gemini-polygon experiment without touching artifacts."""
    del (
        case_dir,
        checkpoint_path,
        target_description,
        event_description,
        output_dir,
        run_id,
    )
    raise RuntimeError(_RETIRED_MESSAGE)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retired Gemini polygon-to-SAM experiment (read-only artifacts only)"
    )
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--run-id", required=True)
    parser.parse_args()
    parser.error(_RETIRED_MESSAGE)


if __name__ == "__main__":
    main()
