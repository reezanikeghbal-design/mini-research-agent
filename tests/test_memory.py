"""Tests for JSON conversation memory (all filesystem work in tmp dirs)."""

import json
import os

import pytest

from errors import InvalidConversationIdError, MemoryWriteError
from memory import ConversationMemory
from models import ConversationRecord

CID = "conv-1"


def record(n: int, cid: str = CID, **kwargs) -> ConversationRecord:
    return ConversationRecord(
        record_id=f"rec-{n}",
        conversation_id=cid,
        question=f"Question {n}?",
        answer=f"Answer {n}.",
        status="completed",
        **kwargs,
    )


def test_save_and_retrieve_roundtrip(tmp_path):
    memory = ConversationMemory(tmp_path)
    memory.add_record(record(1, tool_metadata={"queries": ["q"]}))

    fresh = ConversationMemory(tmp_path)  # new instance proves it was persisted
    loaded = fresh.get_records(CID)
    assert len(loaded) == 1
    assert loaded[0].question == "Question 1?"
    assert loaded[0].tool_metadata == {"queries": ["q"]}
    assert loaded[0].created_at  # timestamp saved
    assert [m.role for m in fresh.get_recent_messages(CID, 10)] == ["user", "assistant"]


def test_missing_file_returns_empty_history(tmp_path):
    memory = ConversationMemory(tmp_path / "does-not-exist")

    assert memory.get_recent_messages("never-used", 10) == []
    assert memory.get_records("never-used") == []


def test_empty_conversation_and_zero_limit(tmp_path):
    memory = ConversationMemory(tmp_path)
    memory.add_record(record(1))

    assert memory.get_recent_messages(CID, 0) == []
    assert memory.get_recent_messages("other-conversation", 5) == []


def test_recent_messages_are_limited_and_ordered(tmp_path):
    memory = ConversationMemory(tmp_path)
    for n in range(1, 6):
        memory.add_record(record(n))

    messages = memory.get_recent_messages(CID, 3)

    assert [m.content for m in messages] == ["Answer 4.", "Question 5?", "Answer 5."]


def test_corrupted_json_is_quarantined_and_memory_keeps_working(tmp_path):
    path = tmp_path / f"{CID}.json"
    path.write_text('{"conversation_id": "conv-1", "records": [BROKEN', encoding="utf-8")
    memory = ConversationMemory(tmp_path)

    assert memory.get_recent_messages(CID, 10) == []  # no crash
    assert not path.exists()
    quarantined = list(tmp_path.glob(f"{CID}.json.corrupt-*"))
    assert len(quarantined) == 1  # evidence preserved for debugging

    memory.add_record(record(1))  # and new data can be saved
    assert len(memory.get_records(CID)) == 1


def test_valid_json_with_wrong_structure_counts_as_corrupt(tmp_path):
    (tmp_path / f"{CID}.json").write_text(json.dumps({"records": "nope"}), encoding="utf-8")
    memory = ConversationMemory(tmp_path)

    assert memory.get_records(CID) == []
    assert list(tmp_path.glob("*.corrupt-*"))


def test_saving_the_same_record_twice_does_not_duplicate(tmp_path):
    memory = ConversationMemory(tmp_path)

    assert memory.add_record(record(1)) is True
    assert memory.add_record(record(1)) is False  # a retried save
    assert len(memory.get_records(CID)) == 1


def test_max_records_trims_oldest(tmp_path):
    memory = ConversationMemory(tmp_path, max_records=3)
    for n in range(1, 6):
        memory.add_record(record(n))

    assert [r.record_id for r in memory.get_records(CID)] == ["rec-3", "rec-4", "rec-5"]


def test_conversations_are_isolated(tmp_path):
    memory = ConversationMemory(tmp_path)
    memory.add_record(record(1, cid="a"))
    memory.add_record(record(2, cid="b"))

    assert [r.record_id for r in memory.get_records("a")] == ["rec-1"]
    assert [r.record_id for r in memory.get_records("b")] == ["rec-2"]


@pytest.mark.parametrize("bad_id", ["../evil", "a/b", "", "x" * 65, "with space", "..", "a.json"])
def test_unsafe_conversation_ids_are_rejected(tmp_path, bad_id):
    memory = ConversationMemory(tmp_path)

    with pytest.raises(InvalidConversationIdError):
        memory.get_records(bad_id)


def test_new_conversation_ids_are_unique_and_valid():
    ids = {ConversationMemory.new_conversation_id() for _ in range(50)}

    assert len(ids) == 50
    for conversation_id in ids:
        ConversationMemory.validate_conversation_id(conversation_id)


def test_failed_write_keeps_old_data_and_leaves_no_temp_files(tmp_path, monkeypatch):
    memory = ConversationMemory(tmp_path)
    memory.add_record(record(1))

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr("memory.os.replace", boom)
    with pytest.raises(MemoryWriteError):
        memory.add_record(record(2))
    monkeypatch.undo()

    assert [r.record_id for r in memory.get_records(CID)] == ["rec-1"]  # original intact
    assert [p.name for p in tmp_path.iterdir()] == [f"{CID}.json"]  # no .tmp leftovers


def test_unwritable_directory_raises_memory_write_error(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory")
    memory = ConversationMemory(blocker / "data")

    with pytest.raises(MemoryWriteError):
        memory.add_record(record(1))


def test_written_file_is_valid_json(tmp_path):
    ConversationMemory(tmp_path).add_record(record(1))

    data = json.loads((tmp_path / f"{CID}.json").read_text(encoding="utf-8"))
    assert data["conversation_id"] == CID
    assert data["records"][0]["answer"] == "Answer 1."
    assert os.listdir(tmp_path) == [f"{CID}.json"]
