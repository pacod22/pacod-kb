"""Server-side UI sessions: a person's pacod-id tokens, encrypted, in the UI's SQLite file.

The signed session cookie holds only a random secret. The row is keyed by that secret's
SHA-256 and its tokens are Fernet-encrypted under a key derived (HKDF) from the secret and
UI_SESSION_SECRET, so a copy of the database alone yields no token, and changing
UI_SESSION_SECRET signs everyone out.

Rows live in ``ui_sessions`` next to the old ``ui_users`` table (left in place, unused) in
the file named by UI_DB_URL or UI_DB_PATH. A refresh updates a row only if it still has the
version it was read at (compare-and-swap), so two refreshes of one session cannot both win.
Synchronous (SQLAlchemy); async callers use ``run_in_threadpool``.

Must not import ``pipeline``: Dockerfile.ui ships only ``web/``.
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from sqlalchemy import Integer, String, Text, create_engine, delete, select, update
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class _Base(DeclarativeBase):
    pass


class UiSession(_Base):
    __tablename__ = "ui_sessions"

    id_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    sub: Mapped[str] = mapped_column(String(128), index=True)
    sid: Mapped[str] = mapped_column(String(128), index=True)
    email: Mapped[str] = mapped_column(String(320), default="")
    name: Mapped[str] = mapped_column(String(200), default="")
    access_enc: Mapped[str] = mapped_column(Text)
    refresh_enc: Mapped[str] = mapped_column(Text)
    access_expires_at: Mapped[int] = mapped_column(Integer)
    expires_at: Mapped[int] = mapped_column(Integer, index=True)
    created_at: Mapped[str] = mapped_column(String(40))
    last_seen_at: Mapped[str] = mapped_column(String(40))
    version: Mapped[int] = mapped_column(Integer, default=1)


@dataclass(frozen=True)
class SessionRecord:
    id_hash: str
    sub: str
    sid: str
    email: str
    name: str
    access_token: str
    refresh_token: str
    access_expires_at: int
    expires_at: int
    version: int


def db_url() -> str:
    """UI_DB_URL, else SQLite at UI_DB_PATH (the kb_ui_data volume in Docker)."""
    override = os.environ.get("UI_DB_URL", "").strip()
    if override:
        return override
    path = Path(os.environ.get("UI_DB_PATH", "/data/kb-ui/users.db"))
    return f"sqlite:///{path}"


def _id_hash(cookie_secret: str) -> str:
    return hashlib.sha256(cookie_secret.encode()).hexdigest()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class SessionStore:
    def __init__(self, url: str, server_secret: str, lifetime_secs: int) -> None:
        if url.startswith("sqlite:///"):
            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(url, echo=False)
        self._factory = sessionmaker(self._engine, expire_on_commit=False)
        self._salt = hashlib.sha256(server_secret.encode()).digest()
        self.lifetime_secs = lifetime_secs

    def init(self) -> None:
        _Base.metadata.create_all(self._engine)

    def close(self) -> None:
        self._engine.dispose()

    def _fernet(self, cookie_secret: str) -> Fernet:
        key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._salt,
            info=b"kb-ui session tokens v1",
        ).derive(cookie_secret.encode())
        return Fernet(base64.urlsafe_b64encode(key))

    def create(
        self,
        *,
        sub: str,
        sid: str,
        email: str,
        name: str,
        access: str,
        refresh: str,
        access_expires_at: int,
    ) -> str:
        """Store a new session; returns the cookie secret (shown to nobody but the cookie)."""
        cookie_secret = secrets.token_urlsafe(32)
        f = self._fernet(cookie_secret)
        now = _now_iso()
        with self._factory.begin() as s:
            s.add(
                UiSession(
                    id_hash=_id_hash(cookie_secret),
                    sub=sub,
                    sid=sid,
                    email=email[:320],
                    name=name[:200],
                    access_enc=f.encrypt(access.encode()).decode(),
                    refresh_enc=f.encrypt(refresh.encode()).decode(),
                    access_expires_at=access_expires_at,
                    expires_at=int(time.time()) + self.lifetime_secs,
                    created_at=now,
                    last_seen_at=now,
                    version=1,
                )
            )
        return cookie_secret

    def load(self, cookie_secret: str) -> SessionRecord | None:
        """The session, or None when it is missing, expired or cannot be decrypted."""
        with self._factory() as s:
            row = s.get(UiSession, _id_hash(cookie_secret))
        if row is None:
            return None
        if row.expires_at <= int(time.time()):
            self.delete(cookie_secret)
            return None
        f = self._fernet(cookie_secret)
        try:
            access = f.decrypt(row.access_enc.encode()).decode()
            refresh = f.decrypt(row.refresh_enc.encode()).decode()
        except InvalidToken:
            return None  # another UI_SESSION_SECRET: the session is unreadable, sign in again
        return SessionRecord(
            id_hash=row.id_hash,
            sub=row.sub,
            sid=row.sid,
            email=row.email,
            name=row.name,
            access_token=access,
            refresh_token=refresh,
            access_expires_at=row.access_expires_at,
            expires_at=row.expires_at,
            version=row.version,
        )

    def update_tokens(
        self,
        cookie_secret: str,
        *,
        expected_version: int,
        access: str,
        refresh: str,
        access_expires_at: int,
        email: str,
        name: str,
    ) -> bool:
        """Replace the tokens if the row still has *expected_version*; False otherwise."""
        f = self._fernet(cookie_secret)
        with self._factory.begin() as s:
            result = s.execute(
                update(UiSession)
                .where(
                    UiSession.id_hash == _id_hash(cookie_secret),
                    UiSession.version == expected_version,
                )
                .values(
                    access_enc=f.encrypt(access.encode()).decode(),
                    refresh_enc=f.encrypt(refresh.encode()).decode(),
                    access_expires_at=access_expires_at,
                    email=email[:320],
                    name=name[:200],
                    last_seen_at=_now_iso(),
                    version=expected_version + 1,
                )
            )
            return bool(result.rowcount)  # type: ignore[attr-defined]

    def delete(self, cookie_secret: str) -> None:
        with self._factory.begin() as s:
            s.execute(delete(UiSession).where(UiSession.id_hash == _id_hash(cookie_secret)))

    def delete_by_sid(self, sid: str) -> int:
        with self._factory.begin() as s:
            return int(s.execute(delete(UiSession).where(UiSession.sid == sid)).rowcount)  # type: ignore[attr-defined]

    def delete_by_sub(self, sub: str) -> int:
        with self._factory.begin() as s:
            return int(s.execute(delete(UiSession).where(UiSession.sub == sub)).rowcount)  # type: ignore[attr-defined]

    def purge_expired(self) -> int:
        with self._factory.begin() as s:
            stale = UiSession.expires_at <= int(time.time())
            return int(s.execute(delete(UiSession).where(stale)).rowcount)  # type: ignore[attr-defined]

    def session_ids(self) -> set[str]:
        with self._factory() as s:
            return set(s.execute(select(UiSession.id_hash)).scalars())
