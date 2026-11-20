"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    @staticmethod
    def _key(user_id: str, request_id: str | None) -> str:
        return request_id or user_id

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Lưu input + thời điểm bắt đầu, khoá theo request_id (hoặc user_id)."""
        key = self._key(user_id, request_id)
        self._open[key] = {
            "request_id": request_id or uuid.uuid4().hex[:12],
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
            "_t0": time.perf_counter(),
        }

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Lưu output, quyết định của layer, latency; append vào self.logs."""
        key = self._key(user_id, request_id)
        start = self._open.pop(key, None) or {
            "request_id": request_id or uuid.uuid4().hex[:12],
            "user_id": user_id,
            "input": None,
            "started_at": utc_now_iso(),
            "_t0": time.perf_counter(),
        }
        latency_ms = (time.perf_counter() - start.pop("_t0")) * 1000
        self.logs.append(
            {
                **start,
                "output": text,
                "blocked": blocked,
                "layer": layer,
                "finished_at": utc_now_iso(),
                "latency_ms": round(latency_ms, 2),
            }
        )

    def export_json(self, filepath: str | None = None):
        """Ghi logs ra đĩa (JSON array), mặc định <repo>/outputs/audit_log.json."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
