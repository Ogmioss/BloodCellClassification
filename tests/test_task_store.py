"""Tests for TaskStore persistence layer."""

import json

import pytest

from src.api.task_store import TaskResult, TaskStatus, TaskStore


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "tasks.json"


@pytest.fixture
def store(store_path):
    return TaskStore(file_path=str(store_path))


def _make_task(task_id: str = "test-1", task_type: str = "train") -> TaskResult:
    return TaskResult(
        task_id=task_id,
        task_type=task_type,
        status=TaskStatus.PENDING,
        created_at="2026-01-01T00:00:00",
    )


class TestTaskStoreCRUD:
    def test_put_and_get(self, store):
        task = _make_task()
        store.put(task.task_id, task)
        retrieved = store.get(task.task_id)
        assert retrieved is not None
        assert retrieved.task_id == "test-1"

    def test_get_missing_returns_none(self, store):
        assert store.get("nonexistent") is None

    def test_delete(self, store):
        task = _make_task()
        store.put(task.task_id, task)
        assert store.delete(task.task_id) is True
        assert store.get(task.task_id) is None

    def test_delete_missing_returns_false(self, store):
        assert store.delete("nonexistent") is False

    def test_list_all(self, store):
        store.put("t1", _make_task("t1", "train"))
        store.put("t2", _make_task("t2", "evaluate"))
        assert len(store.list_all()) == 2

    def test_list_all_filter_by_status(self, store):
        t1 = _make_task("t1")
        t2 = _make_task("t2")
        t2.status = TaskStatus.COMPLETED
        store.put("t1", t1)
        store.put("t2", t2)
        assert len(store.list_all(status=TaskStatus.PENDING)) == 1

    def test_list_all_filter_by_task_type(self, store):
        store.put("t1", _make_task("t1", "train"))
        store.put("t2", _make_task("t2", "evaluate"))
        assert len(store.list_all(task_type="train")) == 1


class TestTaskStorePersistence:
    def test_survives_reload(self, store_path):
        store1 = TaskStore(file_path=str(store_path))
        store1.put("t1", _make_task("t1"))
        del store1

        store2 = TaskStore(file_path=str(store_path))
        assert store2.get("t1") is not None

    def test_update_task_persists(self, store_path):
        store1 = TaskStore(file_path=str(store_path))
        store1.put("t1", _make_task("t1"))
        store1.update_task("t1", status=TaskStatus.RUNNING, started_at="2026-01-01T00:01:00")
        del store1

        store2 = TaskStore(file_path=str(store_path))
        task = store2.get("t1")
        assert task.status == TaskStatus.RUNNING
        assert task.started_at == "2026-01-01T00:01:00"

    def test_empty_file_handled_gracefully(self, store_path):
        store_path.write_text("")
        store = TaskStore(file_path=str(store_path))
        assert len(store) == 0

    def test_corrupt_file_handled_gracefully(self, store_path):
        store_path.write_text("{invalid json")
        store = TaskStore(file_path=str(store_path))
        assert len(store) == 0

    def test_atomic_write_produces_valid_json(self, store_path):
        store = TaskStore(file_path=str(store_path))
        store.put("t1", _make_task("t1"))
        raw = store_path.read_text()
        data = json.loads(raw)
        assert "t1" in data


class TestTaskStoreUpdateTask:
    def test_update_existing_task(self, store):
        store.put("t1", _make_task("t1"))
        store.update_task("t1", status=TaskStatus.FAILED, error="boom")
        task = store.get("t1")
        assert task.status == TaskStatus.FAILED
        assert task.error == "boom"

    def test_update_nonexistent_task_is_noop(self, store):
        store.update_task("ghost", status=TaskStatus.RUNNING)


class TestGPUMutex:
    """Tests for GPU task exclusivity."""

    def test_no_gpu_task_returns_false(self, store):
        assert store.has_running_gpu_task() is False

    def test_pending_train_blocks(self, store):
        store.put("t1", _make_task("t1", "train"))
        assert store.has_running_gpu_task() is True

    def test_running_evaluate_blocks(self, store):
        task = _make_task("t1", "evaluate")
        task.status = TaskStatus.RUNNING
        store.put("t1", task)
        assert store.has_running_gpu_task() is True

    def test_completed_train_does_not_block(self, store):
        task = _make_task("t1", "train")
        task.status = TaskStatus.COMPLETED
        store.put("t1", task)
        assert store.has_running_gpu_task() is False

    def test_failed_evaluate_does_not_block(self, store):
        task = _make_task("t1", "evaluate")
        task.status = TaskStatus.FAILED
        store.put("t1", task)
        assert store.has_running_gpu_task() is False

    def test_cpu_task_does_not_block(self, store):
        task = _make_task("t1", "batch_inference")
        store.put("t1", task)
        assert store.has_running_gpu_task() is False

    def test_validate_data_does_not_block(self, store):
        task = _make_task("t1", "validate_data")
        store.put("t1", task)
        assert store.has_running_gpu_task() is False
