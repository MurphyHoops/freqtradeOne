# -*- coding: utf-8 -*-
"""全局状态持久化与恢复模块。

StateStore 将 GlobalState、EquityProvider 以及 ReservationAgent 的快照合并
写入磁盘，支持策略在崩溃或重启后恢复运行。
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Optional

try:
    from freqtrade.exceptions import OperationalException  # type: ignore
except Exception:  # pragma: no cover - fallback when freqtrade is absent
    class OperationalException(RuntimeError):
        ...


class BaseStateStore(ABC):
    """Abstract state store contract."""

    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_if_exists(self) -> None:
        raise NotImplementedError


class JsonStateStore(BaseStateStore):
    """Serialize GlobalState + EquityProvider + ReservationAgent snapshots to disk."""

    def __init__(self, filepath: str, state, eq_provider, reservation_agent) -> None:
        self.filepath = filepath
        self.state = state
        self.eq = eq_provider
        self.reservation = reservation_agent
        self._lock_path = f"{self.filepath}.lock"

    def _acquire_lock(self, timeout: float = 0.5) -> Optional[int]:
        start = time.monotonic()
        while True:
            try:
                return os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            except FileExistsError:
                if (time.monotonic() - start) >= timeout:
                    return None
                time.sleep(0.01)

    def _release_lock(self, fd: Optional[int]) -> None:
        if fd is None:
            return
        try:
            os.close(fd)
        except Exception:
            pass
        try:
            os.remove(self._lock_path)
        except Exception:
            pass

    def save(self) -> None:
        """Write state atomically with a best-effort lock."""

        lock_fd = self._acquire_lock()
        try:
            snapshot = {
                "global_state": self.state.to_snapshot(),
                "equity_provider": self.eq.to_snapshot(),
                "reservations": self.reservation.to_snapshot(),
            }
            tmp = self.filepath + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(snapshot, handle, default=str)
            os.replace(tmp, self.filepath)
        except Exception as exc:
            print(f"[CRITICAL] State save failed: {exc}")
        finally:
            self._release_lock(lock_fd)

    def load_if_exists(self) -> None:
        """Restore state from disk if present."""

        if not os.path.isfile(self.filepath):
            return
        try:
            with open(self.filepath, "r", encoding="utf-8") as handle:
                snapshot = json.load(handle)
            self.state.restore_snapshot(snapshot.get("global_state", {}))
            self.eq.restore_snapshot(snapshot.get("equity_provider", {}))
            self.reservation.restore_snapshot(snapshot.get("reservations", {}))
            self.state.reset_cycle_after_restore()
        except FileNotFoundError:
            return
        except (json.JSONDecodeError, IOError, PermissionError) as exc:
            raise OperationalException(
                "State file corrupted or unreadable. Manual intervention required to prevent debt reset."
            ) from exc
        except Exception as exc:
            raise OperationalException(
                "State file corrupted or unreadable. Manual intervention required to prevent debt reset."
            ) from exc


class NoopStateStore(BaseStateStore):
    def save(self) -> None:  # pragma: no cover - trivial
        return

    def load_if_exists(self) -> None:  # pragma: no cover - trivial
        return


StateStore = JsonStateStore
