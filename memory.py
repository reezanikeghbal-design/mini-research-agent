"""Lightweight JSON conversation memory.

One JSON file per conversation: ``<data_dir>/<conversation_id>.json``.

Suitable for a local demo or a single-user tool. It is NOT a multi-user
database: there is no locking, so two processes writing the same conversation
at the same time can lose an update.
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from errors import InvalidConversationIdError, MemoryWriteError
from models import ChatMessage, ConversationFile, ConversationRecord

logger = logging.getLogger("mra.memory")

_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ConversationMemory:
    """Save and retrieve conversation history from local JSON files."""

    def __init__(self, data_dir: Path, max_records: int = 200) -> None:
        """
        Args:
            data_dir: directory that holds the conversation files.
            max_records: oldest records beyond this count are dropped on save,
                so a file (and what we load) cannot grow without bound.
        """
        self._data_dir = Path(data_dir)
        self._max_records = max_records

    # --- identifiers ---------------------------------------------------------
    @staticmethod
    def new_conversation_id() -> str:
        """Create a new unique conversation identifier."""
        return uuid.uuid4().hex

    @staticmethod
    def validate_conversation_id(conversation_id: str) -> str:
        """Return the id if it is safe to use as a file name, else raise.

        Only letters, digits, ``-`` and ``_`` (max 64 chars) are allowed, which
        rules out path traversal such as ``../../etc/passwd``.
        """
        if not isinstance(conversation_id, str) or not _ID_PATTERN.match(conversation_id):
            raise InvalidConversationIdError(
                "Conversation id may only contain letters, digits, '-' and '_' "
                "(max 64 characters)."
            )
        return conversation_id

    # --- public API ----------------------------------------------------------
    def add_record(self, record: ConversationRecord) -> bool:
        """Append a record to its conversation file.

        Idempotent: a record whose ``record_id`` is already stored is skipped,
        so retrying a save never creates duplicates.

        Returns:
            True if the record was added, False if it already existed.

        Raises:
            MemoryWriteError: the file could not be written.
        """
        path = self._path(record.conversation_id)
        conversation = self._load(path) or ConversationFile(
            conversation_id=record.conversation_id
        )
        if any(existing.record_id == record.record_id for existing in conversation.records):
            logger.info("event=memory_duplicate_skipped record_id=%s", record.record_id)
            return False

        conversation.records.append(record)
        if len(conversation.records) > self._max_records:
            conversation.records = conversation.records[-self._max_records :]
        self._atomic_write(path, conversation.model_dump_json(indent=2))
        logger.info(
            "event=memory_saved conversation_id=%s records=%d",
            record.conversation_id,
            len(conversation.records),
        )
        return True

    def get_records(self, conversation_id: str) -> list[ConversationRecord]:
        """All stored records of a conversation (empty if none/corrupt)."""
        conversation = self._load(self._path(conversation_id))
        return list(conversation.records) if conversation else []

    def get_recent_messages(self, conversation_id: str, limit: int) -> list[ChatMessage]:
        """Return at most ``limit`` most recent messages, oldest first.

        Each record yields a user message and an assistant message. Failed
        runs are never stored, so history only contains answered questions.
        """
        if limit <= 0:
            return []
        records = self.get_records(conversation_id)[-limit:]
        messages: list[ChatMessage] = []
        for record in records:
            messages.append(ChatMessage(role="user", content=record.question))
            messages.append(ChatMessage(role="assistant", content=record.answer))
        return messages[-limit:]

    # --- internals -----------------------------------------------------------
    def _path(self, conversation_id: str) -> Path:
        self.validate_conversation_id(conversation_id)
        return self._data_dir / f"{conversation_id}.json"

    def _load(self, path: Path) -> ConversationFile | None:
        """Load a conversation file. Missing -> None. Corrupt -> quarantine, None."""
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("event=memory_unreadable path=%s error=%s", path.name, type(exc).__name__)
            self._quarantine(path)
            return None

        try:
            return ConversationFile.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("event=memory_corrupted path=%s error=%s", path.name, type(exc).__name__)
            self._quarantine(path)
            return None

    def _quarantine(self, path: Path) -> None:
        """Move an unreadable file aside so it is preserved but not reused."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        target = path.with_name(f"{path.name}.corrupt-{stamp}")
        try:
            path.rename(target)
            logger.warning("event=memory_quarantined moved_to=%s", target.name)
        except OSError as exc:
            logger.error("event=memory_quarantine_failed error=%s", type(exc).__name__)

    def _atomic_write(self, path: Path, payload: str) -> None:
        """Write via a temp file + ``os.replace`` so readers never see half a file."""
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            with tmp.open("w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except OSError as exc:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            logger.error("event=memory_write_failed error=%s", type(exc).__name__)
            raise MemoryWriteError(
                f"Could not save conversation history ({type(exc).__name__})."
            ) from exc
