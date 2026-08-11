import os
from pathlib import Path

from montagewright.environment import load_project_env


def test_project_env_loads_quotes_comments_and_export(tmp_path, monkeypatch):
    source = tmp_path / ".env"
    source.write_text(
        "# local credentials\n"
        "GEMINI_API_KEY='from file'\n"
        "export GOOGLE_API_KEY=other # comment\n"
        "TOKEN_WITH_HASH=abc#123\n",
        encoding="utf-8",
    )
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "TOKEN_WITH_HASH"):
        monkeypatch.delenv(name, raising=False)

    assert load_project_env(source) == source
    assert os.environ["GEMINI_API_KEY"] == "from file"
    assert os.environ["GOOGLE_API_KEY"] == "other"
    assert os.environ["TOKEN_WITH_HASH"] == "abc#123"


def test_project_env_does_not_override_parent_environment(tmp_path, monkeypatch):
    source = tmp_path / ".env"
    source.write_text("GEMINI_API_KEY=file-value\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "shell-value")

    load_project_env(source)

    assert os.environ["GEMINI_API_KEY"] == "shell-value"
