"""Small, dependency-free project environment loading.

Montagewright is commonly launched from its Web UI, where there is no shell
prompt at which to ``source .env``.  Read the conventional project file, but
never replace a value the parent process supplied explicitly.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path


_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _value(text: str) -> str:
    value = text.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value[1:-1]
        return parsed if isinstance(parsed, str) else str(parsed)
    # An unquoted inline comment begins after whitespace.  A # inside a key
    # remains data, as expected for tokens and URLs.
    return re.split(r"\s+#", value, maxsplit=1)[0].rstrip()


def load_project_env(path: Path | None = None) -> Path | None:
    """Load ``.env`` from the working directory without overriding the shell."""

    source = path or (Path.cwd() / ".env")
    if not source.is_file():
        return None
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if _NAME.fullmatch(name):
            os.environ.setdefault(name, _value(value))
    return source
