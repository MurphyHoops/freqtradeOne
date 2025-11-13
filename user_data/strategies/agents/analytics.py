"""策略运行过程的事件采集与持久化工具。

`AnalyticsAgent` 用于在 TaxBrainV29 流程中记录 finalize、预约、退出、风控校验
等关键节点的上下文信息，并同时输出 JSONL 与 CSV 文件，方便后续的回放分析与
实时监控。

示例：
    >>> from pathlib import Path
    >>> from user_data.strategies.agents.analytics import AnalyticsAgent
    >>> agent = AnalyticsAgent(Path('user_data/logs'))
    >>> agent.log_exit('BTC/USDT', 'trade-1', 'manual_close')
"""

from __future__ import annotations

import csv
import json
import threading
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional


class AnalyticsAgent:
    """统一的策略事件落盘工具，负责写入 JSONL 及聚合 CSV。"""

    def __init__(self, log_dir: Path) -> None:
        """初始化日志目录、CSV 表头状态与线程锁。

        Args:
            log_dir: 希望输出日志的根目录。
        """

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.date_tag = datetime.now(UTC).strftime("%Y%m%d")
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
            "tier_summary",
        ]
        self._csv_header_written = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        self._lock = threading.Lock()

    def _write_jsonl(self, payload: dict[str, Any]) -> None:
        """向 JSONL 文件追加一条事件记录。"""

        record = dict(payload)
        record.setdefault("ts", datetime.now(UTC).isoformat())
        with self._lock:
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_finalize_csv(self, row: dict[str, Any]) -> None:
        """以统一表头将 finalize 数据写入 CSV。"""

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
        tier_summary: Optional[dict[str, Any]] = None,
    ) -> None:
        """记录 finalize 阶段的组合风险与拨款状态。"""

        payload: dict[str, Any] = {
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
        if tier_summary:
            payload["tier_summary"] = tier_summary
        self._write_jsonl(payload)
        csv_row = {
            "ts": datetime.now(UTC).isoformat(),
            "bar_tick": bar_tick,
            "pnl": pnl,
            "debt_pool": debt_pool,
            "cap_used_pct": cap_used_pct,
            "fast_alloc_size": fast_alloc_size,
            "slow_alloc_size": slow_alloc_size,
            "reservations": reservations,
            "cycle_cleared": int(bool(cycle_cleared)),
            "tier_summary": json.dumps(tier_summary, ensure_ascii=False) if tier_summary else "",
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
        """记录预约生命周期事件（创建 / 释放 / 遗弃）。"""

        payload = {
            "event": "reservation",
            "action": event,
            "reservation_id": reservation_id,
            "pair": pair,
            "bucket": bucket,
            "risk": risk,
        }
        self._write_jsonl(payload)

    def log_exit(self, pair: str, trade_id: str, reason: str, **details) -> None:
        """��¼���ʽ����˳���ԭ�򲢷����ؼ�������."""

        payload: dict[str, Any] = {
            "event": "exit",
            "pair": pair,
            "trade_id": trade_id,
            "reason": reason,
        }
        if details:
            payload.update({k: v for k, v in details.items() if v is not None})
        self._write_jsonl(payload)

    def log_exit_tag_series(self, pair: str, tags) -> None:
        """Aggregate vectorized exit tags for observability."""

        if tags is None:
            return
        values: list[str] = []
        try:
            iterable = tags.dropna()
        except Exception:
            iterable = tags
        for item in getattr(iterable, "tolist", lambda: list(iterable))():
            if not item:
                continue
            try:
                token = str(item)
            except Exception:
                continue
            if not token:
                continue
            values.append(token)
        if not values:
            return
        counts = Counter(values)
        self._write_jsonl(
            {
                "event": "exit_tag_agg",
                "pair": pair,
                "counts": dict(counts),
            }
        )

    def log_invariant(self, report: dict[str, Any]) -> None:
        """记录风险不变式检查的结果。"""

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
        """输出调试事件，便于排查临时问题。"""

        record = {
            "event": "debug",
            "tag": tag,
            "message": message,
        }
        if payload:
            record.update(payload)
        self._write_jsonl(record)
