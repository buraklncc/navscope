"""Small JSON cache helpers for TEFAS responses."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


class JsonFileCache:
    """Polite local file cache for repeatable provider calls."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir

    def get(self, key: str) -> Any | None:
        path = self._path_for_key(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, key: str, value: Any) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._path_for_key(key)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return path

    def cleanup(self, retention_days: int = 3, *, now: datetime | None = None) -> int:
        """Delete bounded, expired JSON entries from this cache directory only."""

        if not self.cache_dir.exists():
            return 0
        retention = max(1, int(retention_days))
        cutoff = (now or datetime.now()) - timedelta(days=retention)
        cache_root = self.cache_dir.resolve()
        removed = 0
        for path in self.cache_dir.glob("*.json"):
            resolved = path.resolve()
            if cache_root not in resolved.parents:
                continue
            modified = datetime.fromtimestamp(path.stat().st_mtime)
            if modified < cutoff:
                path.unlink(missing_ok=True)
                removed += 1
        return removed

    def _path_for_key(self, key: str) -> Path:
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("_")
        return self.cache_dir / f"{safe_key}.json"


def tefas_price_cache_key(fund_code: str, lookback_months: int, fetch_date: date) -> str:
    """Build a daily TEFAS price cache key."""

    return f"price_{fund_code.upper()}_{lookback_months}m_{fetch_date.isoformat()}"

