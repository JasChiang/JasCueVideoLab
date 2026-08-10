"""Keep uploaded media addressable across calls and across runs.

The File API holds a file for 48 hours. Uploading the same 74 proxies again
for the second planning call, and again for every review round after that,
costs minutes of wall time and achieves nothing -- the bytes are already
there. What changes between calls is the question, not the material.

The cache is keyed by content hash, so a re-encoded proxy is a different entry
and an unchanged one is a hit however many runs have passed. Entries are
checked against the service before use, because a cache that lies about what
is still live is worse than no cache: the call fails deep inside a paid
request rather than at the point of upload.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The service expires files at 48 hours. Treating anything past 46 as gone
# leaves room for a long call to finish rather than losing its inputs midway.
LIFETIME_SECONDS = 46 * 3600


def default_cache_path() -> Path:
    """Where the cache lives when nobody says otherwise.

    Keyed by content, so it belongs to the material rather than to a run.
    Storing it under the output directory, which is what this did first, meant
    a second cut of the same footage into a new folder re-uploaded all
    seventy-five files -- roughly eight minutes spent proving the bytes had not
    changed.
    """

    root = Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    ) / "montagewright"
    return root / "uploads.json"


def content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class UploadCache:
    """asset hash -> File API URI, with the service as the final word."""

    path: Path
    entries: dict[str, dict[str, Any]]

    @classmethod
    def load(cls, path: Path) -> "UploadCache":
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            entries = {}
        return cls(path=path, entries=entries)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.entries, indent=1), encoding="utf-8"
        )

    def _live(self, entry: dict[str, Any], client: Any) -> bool:
        if time.time() - float(entry.get("uploaded_at", 0)) > LIFETIME_SECONDS:
            return False
        try:
            remote = client.files.get(name=entry["name"])
        except Exception:
            return False
        state = getattr(remote.state, "name", str(remote.state))
        return state == "ACTIVE"

    def uri_for(
        self, path: Path, client: Any, *, mime_type: str
    ) -> tuple[str, bool]:
        """Return a live URI for this file, uploading only if there is none."""

        key = content_hash(path)
        entry = self.entries.get(key)
        if entry and self._live(entry, client):
            return entry["uri"], True

        uploaded = upload_now(path, client)

        self.entries[key] = {
            "uri": uploaded.uri,
            "name": uploaded.name,
            "mime_type": mime_type,
            "source": str(path),
            "uploaded_at": time.time(),
        }
        self.save()
        return uploaded.uri, False


@contextmanager
def _ascii_named(path: Path):
    """Give the uploader a name it can put in a header, and clean up after.

    Hardlinked rather than copied: these are proxies and previews, and
    copying a folder of them to rename it is minutes of disk for nothing.
    """

    try:
        path.name.encode("ascii")
    except UnicodeEncodeError:
        pass
    else:
        yield path
        return

    safe = Path(tempfile.mkdtemp(prefix="montagewright-upload-"))
    linked = safe / f"{content_hash(path)[:16]}{path.suffix}"
    try:
        os.link(path, linked)
    except OSError:
        shutil.copyfile(path, linked)
    try:
        yield linked
    finally:
        shutil.rmtree(safe, ignore_errors=True)


def default_library() -> Path:
    """Where what was learned about the material lives, across every run.

    Redirectable, because the web app had three hand-written copies of this
    path and a test has nowhere to put a fixture that the code will look in.
    """

    return Path(
        os.environ.get(
            "MONTAGEWRIGHT_LIBRARY",
            Path.home() / ".cache" / "montagewright" / "library",
        )
    )


def upload_now(path: Path, client: Any) -> Any:
    """Upload and wait until the file can actually be used.

    An upload comes back before the service has finished with it, and using
    the URI in that window fails with "not in an ACTIVE state". The cached
    path always waited; the uncached path in five other modules did not, so
    it worked on small files and failed on the first long one. One function,
    because it is one fact about the API.
    """

    # The upload puts the filename in a header, and a header is latin-1. A
    # Chinese name -- which is most of the material this is pointed at -- came
    # back as UnicodeEncodeError for every clip in the folder. The bytes are
    # what is being sent; the name is not part of them.
    from montagewright.cost import BudgetSpent
    from montagewright.planner import _is_spend_cap, _provider_budget_message

    path = Path(path)
    try:
        with _ascii_named(path) as sendable:
            uploaded = client.files.upload(file=str(sendable))
    except Exception as error:
        # The same 429, on the other API surface. `ask` has translated this
        # since the first time it happened, and uploads went straight past
        # that: a run with a finished film, a report and every card paid for
        # died on an upload with a raw traceback, recorded as a crash rather
        # than as having run out of money. Which of the two it was decides
        # whether the answer is to debug or to top up.
        raw = " ".join(str(error).split())
        if "ACCESS_TOKEN_TYPE_UNSUPPORTED" in raw:
            raise RuntimeError(
                "This credential type cannot call the Gemini Developer Files "
                "API. MontageWright is currently using the Developer API, so "
                "use a standard Gemini API key (not a service-account-backed "
                "Google Cloud/Enterprise API key), or migrate the media layer "
                "to Google Cloud Storage and the Enterprise backend. Provider "
                f"detail: {raw[:600]}"
            ) from error
        if _is_spend_cap(error):
            # Keep ai.studio/spend in this source because project-cap recovery
            # remains one of the branches, but do not call an empty Prepay
            # balance a monthly cap.  _provider_budget_message preserves the
            # provider detail and gives the matching remedy.
            provider_budget = _provider_budget_message(error)
            raise BudgetSpent(provider_budget or (
                "Gemini billing rejected this request; review AI Studio "
                "Billing or ai.studio/spend, then resume."
            )) from error
        raise
    while getattr(uploaded.state, "name", str(uploaded.state)) == "PROCESSING":
        time.sleep(2.0)
        uploaded = client.files.get(name=uploaded.name)
    state = getattr(uploaded.state, "name", str(uploaded.state))
    if state != "ACTIVE":
        raise RuntimeError(f"{Path(path).name} ended upload in state {state}")
    return uploaded
