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

import base64
import hashlib
import json
import os
import re
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
# File processing normally takes seconds.  A provider-side job can also stay
# PROCESSING forever, and without a deadline one bad proxy pins the whole card
# library behind it.  Five minutes is deliberately generous while still
# making the failure local to one asset rather than to the run.
UPLOAD_PROCESSING_TIMEOUT_SECONDS = 5 * 60.0
UPLOAD_POLL_SECONDS = 2.0
FILE_STATUS_ATTEMPTS = 3
FILE_STATUS_BACKOFF_SECONDS = 0.5


class UploadProcessingTimeout(RuntimeError):
    """The provider accepted an upload but never made it usable."""


def _provider_status_code(error: Exception) -> int | None:
    """Read an HTTP status without depending on one SDK exception class."""

    for name in ("status_code", "code"):
        value = getattr(error, name, None)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    match = re.search(r"(?<!\d)([45]\d\d)(?!\d)", str(error))
    return int(match.group(1)) if match else None


def _get_remote_file(
    client: Any,
    name: str,
    *,
    sleep: Any = time.sleep,
) -> Any:
    """Retry status reads only; never turn a transient 5xx into an upload."""

    for attempt in range(FILE_STATUS_ATTEMPTS):
        try:
            return client.files.get(name=name)
        except Exception as error:
            status = _provider_status_code(error)
            if status not in {500, 502, 503, 504} or attempt == FILE_STATUS_ATTEMPTS - 1:
                raise
            sleep(FILE_STATUS_BACKOFF_SECONDS * (2**attempt))
    raise AssertionError("file status retry loop did not return")


def _wait_until_active(
    uploaded: Any,
    path: Path,
    client: Any,
    *,
    processing_timeout_seconds: float,
    poll_seconds: float,
    clock: Any,
    sleep: Any,
) -> Any:
    """Continue polling one known remote object; never upload a replacement."""

    if processing_timeout_seconds <= 0:
        raise ValueError("processing_timeout_seconds must be greater than zero")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be greater than zero")
    started = float(clock())
    state = getattr(uploaded.state, "name", str(uploaded.state))
    while state == "PROCESSING":
        elapsed = float(clock()) - started
        remaining = processing_timeout_seconds - elapsed
        if remaining <= 0:
            raise UploadProcessingTimeout(
                f"{Path(path).name} was still PROCESSING after "
                f"{processing_timeout_seconds:g}s"
            )
        sleep(min(poll_seconds, remaining))
        uploaded = _get_remote_file(client, uploaded.name, sleep=sleep)
        state = getattr(uploaded.state, "name", str(uploaded.state))
    if state != "ACTIVE":
        raise RuntimeError(f"{Path(path).name} ended upload in state {state}")
    return uploaded


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


def _remote_name_for(sha256: str) -> str:
    """A stable File resource name lets a lost upload response be recovered."""

    return f"files/mw-{sha256[:37]}"


def _remote_matches(remote: Any, *, sha256: str, size_bytes: int) -> bool:
    """ACTIVE means processed; hash and size prove which complete bytes."""

    remote_hash = getattr(remote, "sha256_hash", None)
    remote_size = getattr(remote, "size_bytes", None)
    try:
        decoded = base64.b64decode(
            str(remote_hash or ""), validate=True
        )
        # The Files resource documents sha256Hash as base64-encoded bytes.
        # In the live Developer API it can instead be base64 of the
        # 64-character hexadecimal digest.  Normalize both representations;
        # SHA-256 itself is unchanged.
        if len(decoded) == hashlib.sha256().digest_size:
            remote_hex = decoded.hex()
        else:
            remote_hex = decoded.decode("ascii").lower()
            if not re.fullmatch(r"[0-9a-f]{64}", remote_hex):
                return False
        return remote_hex == sha256.lower() and int(remote_size) == int(
            size_bytes
        )
    except (UnicodeDecodeError, TypeError, ValueError):
        return False


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
        # Web and CLI runs may finish uploads concurrently. Publishing bytes
        # directly exposes a half-written JSON document after interruption or
        # to a reader arriving mid-write. Stage a complete, flushed generation
        # beside the cache and switch it atomically.
        fd, raw = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        staged = Path(raw)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self.entries, stream, indent=1)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(staged, self.path)
        finally:
            staged.unlink(missing_ok=True)

    def _live(
        self,
        entry: dict[str, Any],
        client: Any,
        *,
        expected_sha256: str,
        expected_size: int,
    ) -> bool:
        if time.time() - float(entry.get("uploaded_at", 0)) > LIFETIME_SECONDS:
            return False
        try:
            remote = _get_remote_file(client, entry["name"])
        except Exception as error:
            if _provider_status_code(error) == 404:
                return False
            # Authentication failures and transient provider failures are not
            # evidence that the bytes disappeared. Propagate rather than
            # silently uploading a duplicate remote object.
            raise
        state = getattr(remote.state, "name", str(remote.state))
        if state == "PROCESSING":
            remote = _wait_until_active(
                remote,
                Path(str(entry.get("source") or entry["name"])),
                client,
                processing_timeout_seconds=UPLOAD_PROCESSING_TIMEOUT_SECONDS,
                poll_seconds=UPLOAD_POLL_SECONDS,
                clock=time.monotonic,
                sleep=time.sleep,
            )
            state = getattr(remote.state, "name", str(remote.state))
        if state != "ACTIVE":
            return False
        if not _remote_matches(
            remote, sha256=expected_sha256, size_bytes=expected_size
        ):
            raise RuntimeError(
                f"cached Gemini File {entry['name']} is ACTIVE but its "
                "remote hash/size does not match the local file"
            )
        return True

    def uri_for(
        self, path: Path, client: Any, *, mime_type: str
    ) -> tuple[str, bool]:
        """Return a live URI for this file, uploading only if there is none."""

        key = content_hash(path)
        size = path.stat().st_size
        entry = self.entries.get(key)
        if entry and self._live(
            entry, client, expected_sha256=key, expected_size=size
        ):
            return entry["uri"], True

        # A new local file is not evidence that a remote object exists.  Do
        # not speculatively GET its content-derived name: the Files API uses
        # PERMISSION_DENIED for a name that is absent *or* not owned, so that
        # probe cannot distinguish a normal first upload from an auth error.
        # The stable name exists only to recover after this upload was
        # actually attempted and its response was lost.
        remote_name = _remote_name_for(key)

        def remember_pending(remote: Any) -> None:
            self.entries[key] = {
                "uri": remote.uri,
                "name": remote.name,
                "mime_type": mime_type,
                "source": str(path),
                "uploaded_at": time.time(),
                "state": getattr(remote.state, "name", str(remote.state)),
            }
            self.save()

        try:
            uploaded = upload_now(
                path, client, on_uploaded=remember_pending,
                remote_name=remote_name,
            )
        except Exception as upload_error:
            # Only an upload that was genuinely dispatched earns a recovery
            # lookup.  A 5xx may mean the bytes committed but the response was
            # lost; 409 means the stable name already committed.  Never issue
            # a second upload here.  If recovery itself is inconclusive, keep
            # the original provider error and let the run stop.
            if _provider_status_code(upload_error) not in {
                409, 500, 502, 503, 504,
            }:
                raise
            try:
                uploaded = _get_remote_file(client, remote_name)
                if getattr(
                    uploaded.state, "name", str(uploaded.state)
                ) == "PROCESSING":
                    uploaded = _wait_until_active(
                        uploaded,
                        path,
                        client,
                        processing_timeout_seconds=(
                            UPLOAD_PROCESSING_TIMEOUT_SECONDS
                        ),
                        poll_seconds=UPLOAD_POLL_SECONDS,
                        clock=time.monotonic,
                        sleep=time.sleep,
                    )
            except Exception:
                raise upload_error
        verified = _get_remote_file(client, uploaded.name)
        verified_state = getattr(
            verified.state, "name", str(verified.state)
        )
        if verified_state != "ACTIVE":
            self.entries.pop(key, None)
            self.save()
            raise RuntimeError(
                f"{path.name} upload recovery ended in state "
                f"{verified_state}; refusing to cache it as ACTIVE"
            )
        if not _remote_matches(verified, sha256=key, size_bytes=size):
            self.entries.pop(key, None)
            self.save()
            raise RuntimeError(
                f"{path.name} upload became ACTIVE but its remote hash/size "
                "does not match the local file"
            )

        self.entries[key] = {
            "uri": uploaded.uri,
            "name": uploaded.name,
            "mime_type": mime_type,
            "source": str(path),
            "uploaded_at": time.time(),
            "state": "ACTIVE",
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

    cache_home = Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    )
    return Path(
        os.environ.get(
            "MONTAGEWRIGHT_LIBRARY",
            cache_home / "montagewright" / "library",
        )
    )


def upload_now(
    path: Path,
    client: Any,
    *,
    processing_timeout_seconds: float = UPLOAD_PROCESSING_TIMEOUT_SECONDS,
    poll_seconds: float = UPLOAD_POLL_SECONDS,
    clock: Any = time.monotonic,
    sleep: Any = time.sleep,
    on_uploaded: Any | None = None,
    remote_name: str | None = None,
) -> Any:
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
            uploaded = client.files.upload(
                file=str(sendable),
                config=({"name": remote_name} if remote_name else None),
            )
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
    if on_uploaded is not None:
        on_uploaded(uploaded)
    return _wait_until_active(
        uploaded,
        path,
        client,
        processing_timeout_seconds=processing_timeout_seconds,
        poll_seconds=poll_seconds,
        clock=clock,
        sleep=sleep,
    )
