"""
Persistent task storage backed by a JSON file.

Thread-safe via threading.Lock. Reads reload from disk to support
multi-worker uvicorn deployments (eventual consistency).

Atomic writes via temp file + rename to prevent corruption on crash.
"""

import json
import logging
import os
import threading
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_TASK_STORE_PATH = "./data/tasks.json"


# ============================================================
# Task models (moved here to avoid circular imports)
# ============================================================

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskResult(BaseModel):
    task_id: str
    task_type: str
    status: TaskStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None


# ============================================================
# Persistent store
# ============================================================

class TaskStore:
    """Thread-safe, file-backed task storage."""

    def __init__(self, file_path: Optional[str] = None):
        self._file_path = Path(
            file_path or os.getenv("TASK_STORE_PATH", DEFAULT_TASK_STORE_PATH)
        )
        self._lock = threading.Lock()
        self._tasks: dict[str, TaskResult] = {}
        self._load()

    def _load(self) -> None:
        """Load tasks from disk."""
        if not self._file_path.exists():
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            self._tasks = {}
            return

        try:
            raw = self._file_path.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else {}
            self._tasks = {
                task_id: TaskResult(**task_data)
                for task_id, task_data in data.items()
            }
            logger.info("Loaded %d tasks from %s", len(self._tasks), self._file_path)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(
                "Failed to load tasks from %s: %s. Starting fresh.",
                self._file_path, e,
            )
            self._tasks = {}

    def _save(self) -> None:
        """Write current state to disk. Must be called under self._lock."""
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            task_id: task.model_dump(mode="json")
            for task_id, task in self._tasks.items()
        }
        tmp_path = self._file_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_path.rename(self._file_path)

    def put(self, task_id: str, task: TaskResult) -> None:
        """Add or update a task and persist."""
        with self._lock:
            self._tasks[task_id] = task
            self._save()

    def get(self, task_id: str) -> Optional[TaskResult]:
        """Get a task by ID. Reloads from disk for multi-worker freshness."""
        with self._lock:
            self._load()
            return self._tasks.get(task_id)

    def delete(self, task_id: str) -> bool:
        """Delete a task. Returns True if deleted, False if not found."""
        with self._lock:
            if task_id not in self._tasks:
                return False
            del self._tasks[task_id]
            self._save()
            return True

    def list_all(
        self,
        status: Optional[TaskStatus] = None,
        task_type: Optional[str] = None,
    ) -> list[TaskResult]:
        """List tasks with optional filters, sorted by created_at desc."""
        with self._lock:
            self._load()
            tasks = list(self._tasks.values())

        if status:
            tasks = [t for t in tasks if t.status == status]
        if task_type:
            tasks = [t for t in tasks if t.task_type == task_type]

        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks

    def update_task(self, task_id: str, **fields) -> None:
        """Update specific fields on a task and persist."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                logger.warning("update_task: task %s not found", task_id)
                return
            for key, value in fields.items():
                setattr(task, key, value)
            self._save()

    # GPU task types that require exclusive access
    GPU_TASK_TYPES = {"train", "evaluate"}

    def has_running_gpu_task(self) -> bool:
        """Check if a GPU task (train or evaluate) is PENDING or RUNNING."""
        with self._lock:
            self._load()
            return any(
                t.task_type in self.GPU_TASK_TYPES
                and t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
                for t in self._tasks.values()
            )

    def __contains__(self, task_id: str) -> bool:
        with self._lock:
            return task_id in self._tasks

    def __len__(self) -> int:
        with self._lock:
            return len(self._tasks)
