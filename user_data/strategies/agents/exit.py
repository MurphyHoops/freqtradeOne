# agents/exit.py

from __future__ import annotations
from typing import Optional

try:
    from exits.router import EXIT_ROUTER, ImmediateContext
except Exception:  # pragma: no cover
    EXIT_ROUTER = None  # type: ignore
    ImmediateContext = None  # type: ignore


class ExitPolicyV29:
    def __init__(self, state, eq_provider, cfg, dp=None) -> None:
        self.state = state
        self.eq_provider = eq_provider
        self.cfg = cfg
        self.dp = dp  # <--- 保存 dp，供 decide() 注入 Router Context

    def decide(self, pair: str, trade_id: str, current_profit_pct: Optional[float]) -> Optional[str]:
        """先跑 ImmediateExit 规则，没命中再落回旧逻辑。"""
        if EXIT_ROUTER is not None and hasattr(self.state, "get_trade_by_id"):
            try:
                trade = self.state.get_trade_by_id(trade_id)
                ctx = ImmediateContext(
                    pair=pair,
                    trade=trade,
                    now=getattr(self.state, "now", None),
                    profit=(current_profit_pct or 0.0),
                    dp=self.dp,           # <--- 这里原来用到了 self.dp
                    cfg=self.cfg,
                    state=self.state,
                )
                reason = EXIT_ROUTER.close_now_reason(ctx)
                if reason:
                    return reason
            except Exception:
                pass

        # 旧逻辑（保留你原有实现，先返回 None 也可）
        return None
