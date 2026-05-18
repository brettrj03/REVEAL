"""Custom persistence helpers for pydantic-graph."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic_core import ValidationError
from pydantic_graph.persistence.file import FileStatePersistence

logger = logging.getLogger(__name__)


class SafeFileStatePersistence(FileStatePersistence):
    """FileStatePersistence that attempts to repair known snapshot issues."""

    def _load_sync(self):  # type: ignore[override]
        try:
            return super()._load_sync()
        except ValidationError as exc:
            if self._repair_snapshot_file(exc):
                return super()._load_sync()
            raise

    def _repair_snapshot_file(self, error: ValidationError) -> bool:
        """Attempt to repair corrupt snapshot entries (missing node_id)."""
        try:
            content = self.json_file.read_text()
            data = json.loads(content) if content else []
        except Exception:
            return False

        if not isinstance(data, list):
            return False

        repaired = False
        for entry in data:
            if not isinstance(entry, dict):
                continue
            if entry.get('kind') != 'node':
                continue
            node_info = entry.get('node')
            node_id = None
            if isinstance(node_info, dict):
                node_id = node_info.get('node_id')
            if node_id:
                continue
            snapshot_id = entry.get('id')
            if isinstance(snapshot_id, str) and ':' in snapshot_id:
                node_id = snapshot_id.split(':', 1)[0]
            if not node_id:
                continue
            entry['node'] = {'node_id': node_id}
            repaired = True

        if repaired:
            logger.warning(
                "Repaired snapshot file %s after validation error %s", self.json_file, error
            )
            self.json_file.write_text(json.dumps(data, indent=2))
        return repaired

