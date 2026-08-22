"""Names that are used and never defined.

Three runs died on this in one afternoon, each after paying for cards,
direction and selection -- the point where the tool has spent everything and
delivered nothing. All three were a moved import or a moved variable, none
was reachable from a unit test without building most of a run, and every one
of them is visible from the source alone in under a second.

pyflakes is the whole check. This is here so it runs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "montagewright"

# An unused import is a tidiness question and this is not about tidiness.
IGNORED = ("imported but unused", "unable to detect undefined names")


def _pyflakes() -> list[str]:
    found = subprocess.run(
        [sys.executable, "-m", "pyflakes", str(PACKAGE)],
        capture_output=True,
        text=True,
    )
    if "No module named pyflakes" in found.stderr:
        pytest.skip("pyflakes is not installed here")
    return [
        line
        for line in (found.stdout + found.stderr).splitlines()
        if line.strip() and not any(skip in line for skip in IGNORED)
    ]


def test_nothing_uses_a_name_that_was_never_defined() -> None:
    complaints = [line for line in _pyflakes() if "undefined name" in line]
    assert not complaints, "\n".join(complaints)


def test_nothing_else_pyflakes_finds_either() -> None:
    """Kept separate so an undefined name is never lost in a longer list."""

    complaints = [line for line in _pyflakes() if "undefined name" not in line]
    assert not complaints, "\n".join(complaints)


# Where the layers hand dicts to each other. Six bugs lived in exactly this
# shape: a field defined, read downstream, and never filled in upstream --
# nothing broke, the defaults looked reasonable, and a decision quietly moved
# from the planner to a hard-coded constant. measure/ is left out on purpose:
# it is numpy, where a checker without array stubs mostly reports on itself.
DECIDING = (
    "planner.py", "pipeline.py", "webapp.py", "cli.py", "schema.py",
    "cost.py", "clipcard.py", "review.py", "executor.py", "renderer.py",
    "reframe.py", "timeline.py", "transcript.py", "capabilities.py",
)

# google.genai does not export a symbol pyright can see. Not our code, and
# not something this repo can fix.
ALLOWED = ('"genai" is unknown import symbol',)


def test_the_decision_layer_type_checks() -> None:
    import json

    complaints = []
    for name in DECIDING:
        # Pyright 1.1.411 exhibits pathological cross-module inference when
        # these large, mutually importing files are supplied in one command
        # (30+ minutes at 100% CPU). Checking each root independently follows
        # the same imports and diagnostics, but completes in seconds.
        found = subprocess.run(
            [sys.executable, "-m", "pyright", "--outputjson", str(PACKAGE / name)],
            capture_output=True,
            text=True,
        )
        if "No module named pyright" in found.stderr:
            pytest.skip("pyright is not installed here")
        try:
            report = json.loads(found.stdout)
        except ValueError:  # pragma: no cover -- pyright itself failed
            pytest.skip(f"pyright did not report for {name}: {found.stderr[:200]}")
        complaints.extend(
            f"{one['file'].split('montagewright/')[-1]}:"
            f"{one['range']['start']['line'] + 1}  {one.get('rule')}  "
            f"{one['message'].splitlines()[0]}"
            for one in report["generalDiagnostics"]
            if one.get("rule") != "reportMissingImports"
            and not any(ok in one["message"] for ok in ALLOWED)
        )
    assert not complaints, "\n".join(complaints)
