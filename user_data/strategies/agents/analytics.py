from __future__ import annotations

import csv
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class AnalyticsAgent:
    """负责记录策略运行期间的关键事件并输出 JSONL/CSV 监控数据。"""

    def __init__(self, log_dir: Path) -> None:
        """初始化日志目录并准备基础文件句柄。"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.date_tag = datetime.utcnow().strftime("%Y%m%d")
        self.jsonl_path = self.log_dir / f"v29_analytics.{self.date_tag}.jsonl"
        self.csv_path = self.log_dir / "v29_analytics.csv"
        self._csv_headers = [
            "ts",
            "bar_tick",
            "pnl",
            "debt_pool",
            "cap_used_pct",
            "fast_alloc_size",
            "slow_alloc_size",
            "reservations",
            "cycle_cleared",
        ]
        self._csv_header_written = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        self._lock = threading.Lock()

    def _write_jsonl(self, payload: dict[str, Any]) -> None:
        """将单条事件以 JSONL 形式追加写入。"""
        record = dict(payload)
        record.setdefault("ts", datetime.utcnow().isoformat())
        with self._lock:
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _write_finalize_csv(self, row: dict[str, Any]) -> None:
        """将 finalize 指标写入 CSV，便于外部统计。"""
        with self._lock:
            write_header = not self._csv_header_written
            with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._csv_headers)
                if write_header:
                    writer.writeheader()
                    self._csv_header_written = True
                writer.writerow(row)

    def log_finalize(
        self,
        bar_tick: int,
        pnl: float,
        debt_pool: float,
        fast_alloc_size: float,
        slow_alloc_size: float,
        cap_used_pct: float,
        reservations: int,
        cycle_cleared: bool,
    ) -> None:
        """记录 finalize 阶段的资金、风险与预约状态。"""
        payload = {
            "event": "finalize",
            "bar_tick": bar_tick,
            "pnl": pnl,
            "debt_pool": debt_pool,
            "fast_alloc_size": fast_alloc_size,
            "slow_alloc_size": slow_alloc_size,
            "cap_used_pct": cap_used_pct,
            "reservations": reservations,
            "cycle_cleared": cycle_cleared,
        }
        self._write_jsonl(payload)
        csv_row = {
            "ts": datetime.utcnow().isoformat(),
            "bar_tick": bar_tick,
            "pnl": pnl,
            "debt_pool": debt_pool,
            "cap_used_pct": cap_used_pct,
            "fast_alloc_size": fast_alloc_size,
            "slow_alloc_size": slow_alloc_size,
            "reservations": reservations,
            "cycle_cleared": int(bool(cycle_cleared)),
        }
        self._write_finalize_csv(csv_row)

    def log_reservation(
        self,
        event: str,
        reservation_id: str,
        pair: str,
        bucket: str,
        risk: float,
    ) -> None:
        """记录预约的创建、释放或过期动作。"""
        payload = {
            "event": "reservation",
            "action": event,
            "reservation_id": reservation_id,
            "pair": pair,
            "bucket": bucket,
            "risk": risk,
        }
        self._write_jsonl(payload)

    def log_exit(self, pair: str, trade_id: str, reason: str) -> None:
        """记录单笔交易的退出原因。"""
        payload = {
            "event": "exit",
            "pair": pair,
            "trade_id": trade_id,
            "reason": reason,
        }
        self._write_jsonl(payload)

    def log_invariant(self, report: dict[str, Any]) -> None:
        """记录不变式校验结果，便于追踪风险异常。"""
        payload = {
            "event": "invariant",
            "report": report,
        }
        self._write_jsonl(payload)

    def log_debug(
        self,
        tag: str,
        message: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        """输出调试信息，帮助定位临时问题。"""
        record = {
            "event": "debug",
            "tag": tag,
            "message": message,
        }
        if payload:
            record.update(payload)
        self._write_jsonl(record)
