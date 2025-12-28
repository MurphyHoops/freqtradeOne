"""TaxBrainV29 ����·��������Э���㡣

"""

from __future__ import annotations
from typing import Optional, Any, Dict
import talib.abstract as ta
import copy
import json
import sys
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
import types
from typing import Any, Dict, Optional, Set, List, Tuple
import pandas as pd
from pandas import Timestamp
import math

ROOT_PATH = Path(__file__).resolve().parents[2]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))
MODULE_ALIAS = 'user_data.strategies.TaxBrainV29'
if MODULE_ALIAS not in sys.modules:
    sys.modules[MODULE_ALIAS] = sys.modules.get(__name__)
module_obj = sys.modules.get(__name__)
if module_obj is None:
    module_obj = types.ModuleType(__name__)
    sys.modules[__name__] = module_obj
module_obj.__dict__.update(globals())
try:
    from user_data.strategies.agents.exits.facade import ExitFacade
    from user_data.strategies.agents.exits.router import EXIT_ROUTER, SLContext, TPContext
    from user_data.strategies.agents.exits.exit import ExitTags
except Exception:  # pragma: no cover
    ExitFacade = None  # type: ignore
    EXIT_ROUTER = None  # type: ignore
    SLContext = None    # type: ignore
    TPContext = None    # type: ignore
    ExitTags = None     # type: ignore
try:
    from freqtrade.enums import RunMode

except Exception:
    RunMode = None
try:
    from freqtrade.strategy import (

        IStrategy,
        Trade,
        Order,
        PairLocks,
        informative,  # @informative decorator

        # Hyperopt Parameters

        BooleanParameter,
        CategoricalParameter,
        DecimalParameter,
        IntParameter,
        RealParameter,

        # timeframe helpers

        timeframe_to_minutes,
        timeframe_to_next_date,
        timeframe_to_prev_date,

        # Strategy helper functions

        merge_informative_pair,
        stoploss_from_absolute,
        stoploss_from_open,
    )
except Exception:  # pragma: no cover

    class IStrategy:  # type: ignore

        def __init__(self, *args, **kwargs):
            if not hasattr(self, "dp"):
                self.dp = types.SimpleNamespace()

    class Trade:  # type: ignore
        pass

    class Order:  # type: ignore
        pass

    class PairLocks:  # type: ignore
        pass

    class BooleanParameter:  # type: ignore
        pass

    class CategoricalParameter:  # type: ignore
        pass

    class DecimalParameter:  # type: ignore
        pass

    class IntParameter:  # type: ignore
        pass

    class RealParameter:  # type: ignore
        pass

    def informative(timeframe: str):  # type: ignore

        def decorator(func):
            return func
        return decorator

    def timeframe_to_minutes(timeframe: str):  # type: ignore
        return 0

    def timeframe_to_next_date(timeframe: str, date):  # type: ignore
        return date

    def timeframe_to_prev_date(timeframe: str, date):  # type: ignore
        return date

    def merge_informative_pair(*args, **kwargs):  # type: ignore
        return None

    def stoploss_from_absolute(*args, **kwargs):  # type: ignore
        return 0.0

    def stoploss_from_open(*args, **kwargs):  # type: ignore
        return 0.0

# TaxBrainV29.py 顶部（和其它 import 放一起）

from user_data.strategies.agents.exits import rules_threshold  # 注册 SL/TP + 即时退出规则
from user_data.strategies.config.v29_config import V29Config, apply_overrides
from user_data.strategies.agents.portfolio.treasury import TreasuryAgent
from user_data.strategies.agents.portfolio.tier import TierAgent, TierManager
from user_data.strategies.agents.portfolio.sizer import SizerAgent
from user_data.strategies.agents.signals.builder import (

    collect_factor_requirements,
    collect_indicator_requirements,
)
from user_data.strategies.agents.signals import builder, indicators, schemas
from user_data.strategies.agents.portfolio.risk import RiskAgent
from user_data.strategies.agents.portfolio.reservation import ReservationAgent
from user_data.strategies.agents.portfolio.persist import StateStore
from user_data.strategies.agents.exits.exit import ExitPolicyV29
from user_data.strategies.agents.portfolio.execution import ExecutionAgent
from user_data.strategies.agents.portfolio.cycle import CycleAgent
from user_data.strategies.agents.portfolio.analytics import AnalyticsAgent
from user_data.strategies.agents.portfolio.global_backend import (
    GlobalRiskBackend,
    LocalGlobalBackend,
    RedisGlobalBackend,
)
from user_data.strategies.agents.market.sensor import MarketSensor

# safe, optional import – file lives next to your strategy agents

try:
    from . import evaluate_sl_override

except Exception:  # pragma: no cover

    # type: ignore

    def evaluate_sl_override(ctx: Dict[str, Any], base_sl_pct: float | None) -> Optional[float]:
        return None

# 让所有内置入场信号完成“注册”

from user_data.strategies.agents.signals import builtin_signals as _signals  # noqa: F401

class _NoopStateStore:
    """No-op persistence adapter used during backtests/hyperopt to keep state ephemeral."""
    def save(self) -> None:  # pragma: no cover - trivial
        return
    def load_if_exists(self) -> None:  # pragma: no cover - trivial
        return

@dataclass
class ActiveTradeMeta:
    """ActiveTradeMeta 用于记录在市订单在 TaxBrainV29 流程中的关键元数据，方便执行、退出与风控代理在不同 hook 间共享统一视图。
    Attributes:
        sl_pct: 建仓时设定的止损百分比（通常为负值，代表最大亏损距离）。
        tp_pct: 建仓时设定的止盈百分比。
        direction: 信号方向，取值为 'long' 或 'short'。
        entry_bar_tick: 建仓发生时的全局 bar_tick 序号，用于与 CycleAgent 对齐。
        entry_price: 建仓价格，供执行/退出时计算盈亏。
        bucket: 极坐标场标记，long/short。
        icu_bars_left: ICU 强制回收剩余 bar 数，None 表示当前不受 ICU 约束。
    """
    sl_pct: float
    tp_pct: float
    direction: str
    entry_bar_tick: int
    entry_price: float
    bucket: str
    icu_bars_left: Optional[int]
    exit_profile: Optional[str] = None
    recipe: Optional[str] = None
    plan_timeframe: Optional[str] = None
    plan_atr_pct: Optional[float] = None
    tier_name: Optional[str] = None

@dataclass
class PairState:
    """PairState 汇聚单个交易对的即时状态，用于信号评估、财政拨款与风险检查之间的信息同步。

    Attributes:
        closs: 连续亏损计数，驱动 TierPolicy 的升降级与冷却时长。
        local_loss: 累计局部亏损金额，用于疼痛加权财政拨款。
        cooldown_bars_left: 当前剩余冷却 bar 数，>0 时禁止开新仓。
        last_dir: 最近一次信号方向，None 表示暂无有效信号。
        last_score: 最近一次信号期望收益（edge）数值。
        last_squad: 最近一次信号所属 squad（MRL/PBL/TRS 等）。
        last_sl_pct: 最近一次信号建议的止损百分比。
        last_tp_pct: 最近一次信号建议的止盈百分比。
        last_atr_pct: 最近一次记录的 ATR 百分比，供早锁盈估算使用。
        active_trades: 当前在市的交易字典，键为 trade_id，值为 ActiveTradeMeta。
    """
    closs: int = 0
    local_loss: float = 0.0
    cooldown_bars_left: int = 0
    last_dir: Optional[str] = None
    last_score: float = 0.0
    last_kind: Optional[str] = None
    last_squad: Optional[str] = None
    last_sl_pct: float = 0.0
    last_tp_pct: float = 0.0
    last_atr_pct: float = 0.0
    last_exit_profile: Optional[str] = None
    last_recipe: Optional[str] = None
    active_trades: Dict[str, ActiveTradeMeta] = field(default_factory=dict)


@dataclass(frozen=True)
class GateResult:
    """Normalized gatekeeping outcome used by custom_stake_amount."""

    allowed: bool
    thresholds: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    debt: Optional[float] = None
    closs: Optional[int] = None

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any], default_closs: Optional[int]) -> "GateResult":
        thresholds = payload.get("thresholds") if isinstance(payload, dict) else {}
        return cls(
            allowed=bool(payload.get("allowed", True)) if isinstance(payload, dict) else True,
            thresholds=thresholds if isinstance(thresholds, dict) else {},
            reason=str(payload.get("reason", "")) if isinstance(payload, dict) else "",
            debt=payload.get("debt") if isinstance(payload, dict) else None,
            closs=payload.get("closs", default_closs) if isinstance(payload, dict) else default_closs,
        )

@dataclass
class TreasuryState:
    """TreasuryState 持久化极坐标场的拨款参数以及财政周期起点。"""

    k_long: float = 0.0
    k_short: float = 0.0
    theta: float = 0.0
    final_r: float = 0.0
    available: float = 0.0
    bias: float = 0.0
    volatility: float = 1.0
    cycle_start_tick: int = 0
    cycle_start_equity: float = 0.0

    def to_snapshot(self) -> Dict[str, Any]:
        """导出财政子模块的可序列化快照数据，供 StateStore 写入磁盘。"""

        return {
            "k_long": self.k_long,
            "k_short": self.k_short,
            "theta": self.theta,
            "final_r": self.final_r,
            "available": self.available,
            "bias": self.bias,
            "volatility": self.volatility,
            "cycle_start_tick": self.cycle_start_tick,
            "cycle_start_equity": self.cycle_start_equity,
        }

    def restore_snapshot(self, payload: Dict[str, Any]) -> None:
        """从 StateStore 读取的字典中恢复财政拨款状态。"""

        self.k_long = float(payload.get("k_long", 0.0))
        self.k_short = float(payload.get("k_short", 0.0))
        self.theta = float(payload.get("theta", 0.0))
        self.final_r = float(payload.get("final_r", 0.0))
        self.available = float(payload.get("available", 0.0))
        self.bias = float(payload.get("bias", 0.0))
        self.volatility = float(payload.get("volatility", 1.0))
        self.cycle_start_tick = int(payload.get("cycle_start_tick", 0))
        self.cycle_start_equity = float(payload.get("cycle_start_equity", 0.0))

class GlobalState:
    """GlobalState 负责维护 TaxBrainV29 的组合级状态，包括债务池、各交易对风险、财政拨款以及 bar_tick 进度。
    该容器为多个代理提供读写接口，并承担快照持久化与恢复的职责。
    """

    def __init__(self, cfg: V29Config, backend: Optional[GlobalRiskBackend] = None) -> None:
        """初始化全局状态容器。
        Args:
            cfg: 当前加载的 V29Config，用于读取 CAP、税率等参数。
        """
        self.cfg = cfg
        self.backend = backend
        self.per_pair: Dict[str, PairState] = {}
        self.debt_pool: float = 0.0
        self.trade_risk_ledger: Dict[str, float] = {}
        self.pair_risk_open: Dict[str, float] = {}
        self.treasury = TreasuryState()
        self.bar_tick: int = 0
        self.current_cycle_ts: Optional[float] = None
        self.last_finalized_bar_ts: Optional[float] = None
        self.reported_pairs_for_current_cycle: set[str] = set()
        self.last_finalize_walltime: float = time.time()

        # === Nominal position ledgers (amount * price, leverage-agnostic) ===
        self.trade_stake_ledger: Dict[str, float] = {}
        self.pair_stake_open: Dict[str, float] = {}

    def _canonical_pair(self, pair: str | None) -> str:
        if not pair:
            return ""
        return str(pair).split(":")[0]

    def get_pair_state(self, pair: str) -> PairState:
        """获取指定交易对的 PairState，如不存在则初始化默认实例。
        Args:
            pair: 交易对名称，例如 'BTC/USDT'。
        Returns:
            PairState: 可供调用方修改的状态对象。
        """
        key = self._canonical_pair(pair)
        if key not in self.per_pair:
            self.per_pair[key] = PairState()
        return self.per_pair[key]

    def get_total_open_risk(self) -> float:
        """汇总所有交易对当前在市风险，用于组合 CAP 控制。
        Returns:
            float: 当前组合层面的风险敞口总和。
        """
        return sum(self.pair_risk_open.values())

    def get_dynamic_portfolio_cap_pct(self, equity: float) -> float:
        """根据当前权益与债务压力计算组合 VaR 上限占比。
        Args:
            equity: 当前账户权益。
        Returns:
            float: 调整后的组合 CAP 百分比；债务率过高或 equity<=0 时会自动折减。
        """
        risk_cfg = getattr(self.cfg, "risk", self.cfg)
        base = risk_cfg.portfolio_cap_pct_base
        if equity <= 0:
            return base * 0.5
        if (self.debt_pool / equity) > risk_cfg.drawdown_threshold_pct:
            return base * 0.5
        return base

    def per_pair_cap_room(self, pair: str, equity: float, tier_pol, reserved: float) -> float:
        """计算指定交易对在单票 CAP 约束下剩余可分配的风险空间。
        Args:
            pair: 交易对名称。
            equity: 当前账户权益。
            tier_pol: 对应 CLOSS 等级的 TierPolicy。
            reserved: 当前交易对已预约的风险额度。
        Returns:
            float: 在 CAP 范围内仍可追加的风险额度。
        """
        key = self._canonical_pair(pair)
        cap = tier_pol.per_pair_risk_cap_pct * equity
        used = self.pair_risk_open.get(key, 0.0) + reserved
        return max(0.0, cap - used)

    def record_trade_open(
        self,
        pair: str,
        trade_id: str,
        real_risk: float,
        sl_pct: float,
        tp_pct: float,
        direction: str,
        bucket: str,
        entry_price: float,
        tier_pol,
        exit_profile: Optional[str] = None,
        recipe: Optional[str] = None,
        plan_timeframe: Optional[str] = None,
        plan_atr_pct: Optional[float] = None,
        tier_name: Optional[str] = None,
        stake_nominal: Optional[float] = None,  # nominal position size (amount * price)
    ) -> None:
        """记录新建仓的风险占用，并创建 ActiveTradeMeta 以便执行与退出逻辑后续引用。
        Args:
            pair: 交易对名称。
            trade_id: Freqtrade 分配的交易标识。
            real_risk: 根据止损距离计算出的真实风险敞口。
            sl_pct: 建仓时的止损百分比。
            tp_pct: 建仓时的止盈百分比。
            direction: 方向标识 'long' 或 'short'。
            bucket: 当前订单使用的拨款桶（long/short）。
            entry_price: 建仓价格。
            tier_pol: 所属 TierPolicy，用于初始化 ICU 倒计时与冷却时长。
        """
        pst = self.get_pair_state(pair)
        pair_key = self._canonical_pair(pair)
        tier_name = tier_name or (getattr(tier_pol, "name", None) if tier_pol else None)
        # 记录的风险应为含杠杆的实际损失额
        self.trade_risk_ledger[trade_id] = float(real_risk)
        self.pair_risk_open[pair_key] = self.pair_risk_open.get(
            pair_key, 0.0) + float(real_risk)
        
        # === Derive nominal stake from risk/sl_pct and accumulate per-pair exposure ===
        if stake_nominal is None:
            if sl_pct and sl_pct > 0:
                stake_nominal = real_risk / sl_pct
            else:
                stake_nominal = 0.0

        stake_nominal = float(max(stake_nominal or 0.0, 0.0))
        self.trade_stake_ledger[trade_id] = stake_nominal
        self.pair_stake_open[pair_key] = self.pair_stake_open.get(pair_key, 0.0) + stake_nominal

        icu_left = tier_pol.icu_force_exit_bars if tier_pol.icu_force_exit_bars > 0 else None
        pst.active_trades[trade_id] = ActiveTradeMeta(
            sl_pct=float(sl_pct),
            tp_pct=float(tp_pct),
            direction=str(direction),
            entry_bar_tick=self.bar_tick,
            entry_price=float(entry_price),
            bucket=str(bucket),
            icu_bars_left=icu_left,
            exit_profile=exit_profile,
            recipe=recipe,
            plan_timeframe=plan_timeframe,
            plan_atr_pct=float(plan_atr_pct) if plan_atr_pct is not None else None,
            tier_name=tier_name,
        )
        pst.cooldown_bars_left = max(
            pst.cooldown_bars_left, tier_pol.cooldown_bars)

    def record_trade_close(self, pair: str, trade_id: str, profit_abs: float, tier_mgr) -> None:
            """
            在平仓时回收风险、更新债务池。
            逻辑修正：
            1. Max Tier 之前的亏损：只增加 pst.local_loss，不增加 self.debt_pool。
            2. Max Tier 及之后的亏损：增加 self.debt_pool，并清空 pst.local_loss。
            3. 盈利：优先偿还 pst.local_loss，剩余的(excess)用于偿还 self.debt_pool。
            """
            pst = self.get_pair_state(pair)
            pair_key = self._canonical_pair(pair)

            # 1) 回收风险/保证金账本 (保持原样)
            was_risk = self.trade_risk_ledger.pop(trade_id, 0.0)
            self.pair_risk_open[pair_key] = max(0.0, self.pair_risk_open.get(pair_key, 0.0) - was_risk)
            if self.pair_risk_open.get(pair_key, 0.0) <= 1e-12:
                self.pair_risk_open[pair_key] = 0.0

            was_stake = self.trade_stake_ledger.pop(trade_id, 0.0)
            self.pair_stake_open[pair_key] = max(0.0, self.pair_stake_open.get(pair_key, 0.0) - was_stake)
            if self.pair_stake_open.get(pair_key, 0.0) <= 1e-12:
                self.pair_stake_open[pair_key] = 0.0

            # 2) 从活跃订单中移除 (保持原样)
            pst.active_trades.pop(trade_id, None)
            prev_closs = pst.closs

            # 3) 获取最大亏损等级
            routing_map = getattr(tier_mgr, "_routing_map", None) or {}
            max_closs = max(routing_map.keys()) if routing_map else 3

            # A) 盈利处理逻辑 (Profit Handling)
            if profit_abs >= 0:
                excess_profit = pst.local_loss - profit_abs
                pst.local_loss = 0
                self.debt_pool =max(0, self.debt_pool + excess_profit)

                if self.backend:
                    if excess_profit>0:
                        self.backend.add_loss(excess_profit)
                    else:
                        self.backend.repay_loss(0 - excess_profit)
                        
                # 4. 重置连续亏损计数和冷却
                pst.closs = 0
                pol = tier_mgr.get(pst.closs)
                pst.cooldown_bars_left = max(pst.cooldown_bars_left, pol.cooldown_bars_after_win)
                return

            # B) 亏损处理逻辑 (Loss Handling)
            loss = abs(profit_abs)

            # 情况 1: 达到了 Max Tier (熔断/清算)
            # 只有在这里，亏损才会进入中央债务 (Central Debt)
            if prev_closs >= max_closs:
                # 增加中央债务
                self.debt_pool += loss + pst.local_loss # 本地债务给中央
                if self.backend and loss > 0:
                    self.backend.add_loss(loss + pst.local_loss)
                
                # Max Tier 触发后，本地债务清空 (Reset Local)
                # 这意味着之前的累积局部亏损被"原谅"或视为沉没成本，重新开始
                pst.local_loss = 0.0 
                
                # 重置 closs 为 0 (或者你可以选择保持在 max，取决于你想让它继续冷却多久)
                # 这里通常重置为 0 让它重新开始，或者保留高 closs 让它进入很长的冷却
                pst.closs = 0 
                
                pol = tier_mgr.get(pst.closs)
                pst.cooldown_bars_left = max(pst.cooldown_bars_left, pol.cooldown_bars)
                return

            # 情况 2: 普通亏损 (Before Max Tier)
            pst.local_loss += loss

            # 升级 closs
            pst.closs = prev_closs + 1
            pol = tier_mgr.get(pst.closs)
            pst.cooldown_bars_left = max(pst.cooldown_bars_left, pol.cooldown_bars)

    def to_snapshot(self) -> Dict[str, Any]:
        """序列化全局状态为字典，便于持久化或调试。
        Returns:
            Dict[str, Any]: 包含债务池、各交易对状态与财政信息的快照。
        """
        per_pair_snap: Dict[str, Any] = {}
        for pair, pst in self.per_pair.items():
            per_pair_snap[pair] = {
                "closs": pst.closs,
                "local_loss": pst.local_loss,
                "cooldown_bars_left": pst.cooldown_bars_left,
                "last_dir": pst.last_dir,
                "last_score": pst.last_score,
                "last_kind": pst.last_kind,
                "last_squad": pst.last_squad,
                "last_sl_pct": pst.last_sl_pct,
                "last_tp_pct": pst.last_tp_pct,
                "last_atr_pct": pst.last_atr_pct,
                "last_exit_profile": pst.last_exit_profile,
                "last_recipe": pst.last_recipe,
                "active_trades": {
                    tid: {
                        "sl_pct": meta.sl_pct,
                        "tp_pct": meta.tp_pct,
                        "direction": meta.direction,
                        "entry_bar_tick": meta.entry_bar_tick,
                        "entry_price": meta.entry_price,
                        "bucket": meta.bucket,
                        "icu_bars_left": meta.icu_bars_left,
                        "exit_profile": meta.exit_profile,
                        "recipe": meta.recipe,
                    }
                    for tid, meta in pst.active_trades.items()
                },
            }
        return {
            "debt_pool": self.debt_pool,
            "per_pair": per_pair_snap,
            "trade_risk_ledger": self.trade_risk_ledger,
            "pair_risk_open": self.pair_risk_open,
            "treasury": self.treasury.to_snapshot(),
            "bar_tick": self.bar_tick,
            "current_cycle_ts": self.current_cycle_ts,
            "last_finalized_bar_ts": self.last_finalized_bar_ts,
            "last_finalize_walltime": self.last_finalize_walltime,
        }

    def restore_snapshot(self, payload: Dict[str, Any]) -> None:
        """从 `to_snapshot` 生成的字典恢复全局状态，用于断点续跑或崩溃恢复。
        Args:
            payload: 先前保存的状态字典。
        """
        self.debt_pool = float(payload.get("debt_pool", 0.0))
        self.trade_risk_ledger = {k: float(v) for k, v in payload.get(
            "trade_risk_ledger", {}).items()}
        self.pair_risk_open = {k: float(v) for k, v in payload.get(
            "pair_risk_open", {}).items()}
        self.bar_tick = int(payload.get("bar_tick", 0))
        self.current_cycle_ts = payload.get("current_cycle_ts")
        self.last_finalized_bar_ts = payload.get("last_finalized_bar_ts")
        self.last_finalize_walltime = float(
            payload.get("last_finalize_walltime", time.time()))
        self.per_pair = {}
        for pair, pst_payload in payload.get("per_pair", {}).items():
            pst = PairState(
                closs=int(pst_payload.get("closs", 0)),
                local_loss=float(pst_payload.get("local_loss", 0.0)),
                cooldown_bars_left=int(
                    pst_payload.get("cooldown_bars_left", 0)),
                last_dir=pst_payload.get("last_dir"),
                last_score=float(pst_payload.get("last_score", 0.0)),
                last_kind=pst_payload.get("last_kind"),
                last_squad=pst_payload.get("last_squad"),
                last_sl_pct=float(pst_payload.get("last_sl_pct", 0.0)),
                last_tp_pct=float(pst_payload.get("last_tp_pct", 0.0)),
                last_atr_pct=float(pst_payload.get("last_atr_pct", 0.0)),
                last_exit_profile=pst_payload.get("last_exit_profile"),
                last_recipe=pst_payload.get("last_recipe"),
            )
            for tid, meta_payload in pst_payload.get("active_trades", {}).items():
                pst.active_trades[tid] = ActiveTradeMeta(
                    sl_pct=float(meta_payload.get("sl_pct", 0.0)),
                    tp_pct=float(meta_payload.get("tp_pct", 0.0)),
                    direction=str(meta_payload.get("direction", "")),
                    entry_bar_tick=int(meta_payload.get("entry_bar_tick", 0)),
                    entry_price=float(meta_payload.get("entry_price", 0.0)),
                    bucket=str(meta_payload.get("bucket", "long")),
                    icu_bars_left=(
                        int(meta_payload["icu_bars_left"])
                        if meta_payload.get("icu_bars_left") is not None
                        else None
                    ),
                    exit_profile=meta_payload.get("exit_profile"),
                    recipe=meta_payload.get("recipe"),
                )
            self.per_pair[pair] = pst
        self.treasury.restore_snapshot(payload.get("treasury", {}))

    def reset_cycle_after_restore(self) -> None:
        """在完成 StateStore 恢复后重置周期追踪计数，使后续 finalize 流程重新对齐当前时间。"""
        self.current_cycle_ts = None
        self.reported_pairs_for_current_cycle = set()
        self.last_finalize_walltime = time.time()

class EquityProvider:
    """EquityProvider 封装权益读写接口，供 CycleAgent、SizerAgent 等组件查询或更新账户权益。"""

    def __init__(self, initial_equity: float) -> None:
        """初始化 EquityProvider。
        Args:
            initial_equity: Dry-run 或实盘环境下的初始权益。
        """
        self.equity_current = float(initial_equity)

    def to_snapshot(self) -> Dict[str, float]:
        """导出权益数值快照，便于持久化。
        Returns:
            Dict[str, float]: 仅包含当前权益的字典。
        """
        return {"equity_current": self.equity_current}

    def restore_snapshot(self, payload: Dict[str, Any]) -> None:
        """从快照字典恢复权益数值。
        Args:
            payload: 持久化层读取的权益字典。
        """
        self.equity_current = float(payload.get(
            "equity_current", self.equity_current))

    def get_equity(self) -> float:
        """返回当前记录的权益数值。
        Returns:
            float: 最新权益。
        """
        return self.equity_current

    def on_trade_closed_update(self, profit_abs: float) -> None:
        """在交易结束后，根据绝对盈亏增减权益。
        Args:
            profit_abs: 交易的绝对盈利（亏损为负值）。
        """
        self.equity_current += float(profit_abs)

class TaxBrainV29(IStrategy):
    """

    """
    timeframe = V29Config().system.timeframe
    can_short = True
    startup_candle_count = V29Config().system.startup_candle_count
    # minimal_roi = {"0": 0.0003    }
    # stoploss = -0.2
    use_custom_roi = False
    use_custom_stoploss = False
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_buy_signals = False
    ignore_sell_signals = True
    _indicator_requirements_map: Dict[Optional[str], Set[str]] = {}

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise strategy state, analytics, and persistence layer."""
        base_cfg = V29Config()
        self.cfg = apply_overrides(base_cfg, config.get("strategy_params", {}))
        self.timeframe = self.cfg.system.timeframe

        self.startup_candle_count = self.cfg.system.startup_candle_count
        self.stoploss = self.cfg.trading.sizing.enforce_leverage*-0.2
        self.minimal_roi = {"0": 0.50*self.cfg.trading.sizing.enforce_leverage}
        try:
            self.__class__.timeframe = self.cfg.system.timeframe
            self.__class__.startup_candle_count = self.cfg.system.startup_candle_count
        except Exception:
            pass
        extra_signal_factors = getattr(self.cfg, "extra_signal_factors", None)
        self._factor_requirements = collect_factor_requirements(
            extra_signal_factors)
        self._indicator_requirements = collect_indicator_requirements(
            extra_signal_factors)
        self.__class__._indicator_requirements_map = self._indicator_requirements
        config_timeframes = tuple(
            getattr(self.cfg, "informative_timeframes", ()))
        inferred_timeframes = tuple(
            tf for tf in self._factor_requirements.keys() if tf)
        self._informative_timeframes = self._derive_informative_timeframes(
            config_timeframes, inferred_timeframes
        )
        self._informative_last: Dict[str, Dict[str, pd.Series]] = {}
        self._informative_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._informative_gc_last_ts: float = 0.0
        self._informative_gc_interval_sec: int = 900  # sweep every 15 minutes to curb memory growth
        self._register_informative_methods()
        super().__init__(config)
        self.tier_mgr = TierManager(self.cfg)
        self.tier_agent = TierAgent()
        user_data_dir = Path(config.get("user_data_dir", "."))
        initial_equity = float(config.get(
            "dry_run_wallet", self.cfg.system.dry_run_wallet_fallback))
        self.eq_provider = EquityProvider(initial_equity)

        runmode_cfg = config.get("runmode") if isinstance(config, dict) else None
        runmode_token = str(getattr(runmode_cfg, "value", runmode_cfg or "")).lower()
        force_local_backend = False
        if RunMode is not None and runmode_cfg in (getattr(RunMode, "HYPEROPT", None), getattr(RunMode, "BACKTEST", None)):
            force_local_backend = True
        elif runmode_token:
            force_local_backend = any(key in runmode_token for key in ("hyperopt", "backtest"))

        backend_mode = str(
            getattr(getattr(self.cfg, "system", None), "global_backend_mode", getattr(self.cfg, "global_backend_mode", "local"))
        ).lower()
        if force_local_backend:
            if backend_mode != "local":
                try:
                    self.logger.info("Forcing LocalGlobalBackend for safety in Backtest/Hyperopt mode")
                except Exception:
                    print("Forcing LocalGlobalBackend for safety in Backtest/Hyperopt mode")
            backend_mode = "local"
        if backend_mode == "redis":
            self.global_backend = RedisGlobalBackend(
                host=self.cfg.system.redis_host,
                port=self.cfg.system.redis_port,
                db=self.cfg.system.redis_db,
                password=None,
                namespace=self.cfg.system.redis_namespace,
            )
        elif backend_mode == "local":
            self.global_backend = LocalGlobalBackend()
        else:
            raise ValueError(f"Unknown global_backend_mode: {self.cfg.system.global_backend_mode}")

        sensor_weights = dict(getattr(getattr(self.cfg, "sensor", None), "weights", {}) or {})
        sensor_entropy = float(
            getattr(getattr(self.cfg, "sensor", None), "entropy_factor", getattr(getattr(self.cfg, "risk", None), "entropy_factor", 0.4))
        )
        self.market_sensor = MarketSensor(self.global_backend, weights=sensor_weights, entropy_factor=sensor_entropy)

        self.state = GlobalState(self.cfg, backend=self.global_backend)
        self.analytics = AnalyticsAgent(user_data_dir / "logs")
        self.reservation = ReservationAgent(self.cfg, analytics=self.analytics, backend=self.global_backend)
        self.treasury_agent = TreasuryAgent(self.cfg, self.tier_mgr, backend=self.global_backend)
        self.risk_agent = RiskAgent(self.cfg, self.reservation, self.tier_mgr, backend=self.global_backend)
        self.exit_facade = ExitFacade(self.cfg, self.tier_mgr) if ExitFacade else None
        if self.exit_facade:
            self.exit_facade.attach_strategy(self)
            self.exit_facade.set_dataprovider(getattr(self, "dp", None))
        self.exit_policy = ExitPolicyV29(self.state, self.eq_provider, self.cfg, dp=getattr(self, "dp", None))
        self.exit_policy.set_strategy(self)
        state_file = (user_data_dir / "taxbrain_v29_state.json").resolve()
        # ���봰�岻�� bot_start() �� backtest/hyperopt ҲҪ����������־û�
        self._runmode_token: str = self._compute_runmode_token()
        self._persist_enabled: bool = not self._is_backtest_like_runmode()
        self.persist = StateStore(
            filepath=str(state_file),
            state=self.state,
            eq_provider=self.eq_provider,
            reservation_agent=self.reservation,
        )
        if not self._persist_enabled:
            # backtest/hyperopt ��չʾ���ڴ�״̬��ֹ���� closs �����ݣ�����ְ�� JSON �ļ�
            self.persist = _NoopStateStore()
        try:
            self._tier_debug(
                f"init runmode_token={self._runmode_token or 'unknown'} backtest_like={self._is_backtest_like_runmode()} persist_enabled={self._persist_enabled}"
            )
        except Exception:
            pass
        self.cycle_agent = CycleAgent(
            cfg=self.cfg,
            state=self.state,
            reservation=self.reservation,
            treasury=self.treasury_agent,
            risk=self.risk_agent,
            analytics=self.analytics,
            persist=self.persist,
            tier_mgr=self.tier_mgr,
            backend=self.global_backend,
        )
        self.sizer = SizerAgent(
            self.state, self.reservation, self.eq_provider, self.cfg, self.tier_mgr, backend=self.global_backend)
        self.execution = ExecutionAgent(
            self.state, self.reservation, self.eq_provider, self.cfg)
        self._last_signal: Dict[str, Optional[schemas.Candidate]] = {}
        self._pending_entry_meta: Dict[str, Dict[str, Any]] = {}
        self._tf_sec = self._tf_to_sec(self.cfg.system.timeframe)
        self._candidate_pool_limit = int(
            max(1, getattr(self.cfg, "candidate_pool_max_per_side", 4))
        )

    def set_dataprovider(self, dp):
        """
        Freqtrade 在策略实例创建后，会回调此方法注入 DataProvider。
        这里顺便把 dp 传给 exit_policy，供 Router 的 Immediate/Vector/Threshold 规则使用。
        """

        # 调用父类实现（如果父类没有该方法也不会出错）
        try:
            super().set_dataprovider(dp)
        except AttributeError:
            pass

        # 桥接给策略本身与退出策略
        self.dp = dp
        if hasattr(self, "exit_policy") and self.exit_policy is not None:
            self.exit_policy.dp = dp
        if hasattr(self, "sizer"):
            self.sizer.set_dataprovider(dp)
        if self.exit_facade:
            self.exit_facade.set_dataprovider(dp)

    @staticmethod
    def _tf_to_sec(tf: str) -> int:
        """将 Freqtrade 的 timeframe 字符串转换为秒数，用于 CycleAgent 的节奏控制。
        Args:
            tf: 形如 '5m'、'1h' 的时间粒度。
        Returns:
            int: 对应的秒数，未识别时回退到 300 秒。
        """
        if tf.endswith("m"):
            return int(tf[:-1]) * 60
        if tf.endswith("h"):
            return int(tf[:-1]) * 3600
        if tf.endswith("d"):
            return int(tf[:-1]) * 86400
        return 300

    def _tier_debug(self, message: str) -> None:
        """Append lightweight tier debug info into user_data/logs/tier_debug.log."""
        try:
            user_dir = Path(self.config.get("user_data_dir", "user_data"))
            log_path = user_dir / "logs" / "tier_debug.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            ts = datetime.utcnow().isoformat()
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(f"{ts} {message}\n")
        except Exception:
            pass

    def _compute_runmode_token(self) -> str:
        cfg_mode = self.config.get("runmode") if hasattr(
            self, "config") else None
        if cfg_mode is not None:
            return str(getattr(cfg_mode, "value", cfg_mode)).lower()
        strategy_mode = getattr(self, "runmode", None)
        if strategy_mode is not None:
            return str(getattr(strategy_mode, "value", strategy_mode)).lower()
        dp_mode = getattr(getattr(self, "dp", None), "runmode", None)
        if dp_mode is not None:
            return str(getattr(dp_mode, "value", dp_mode)).lower()
        return ""

    def _ensure_runmode_token(self) -> str:
        if not self._runmode_token:
            self._runmode_token = self._compute_runmode_token()
        return self._runmode_token

    def _is_backtest_like_runmode(self) -> bool:
        token = self._ensure_runmode_token() or ""
        return any(key in token for key in ("backtest", "hyperopt"))

    def _derive_informative_timeframes(
        self, manual: tuple[str, ...], inferred: tuple[str, ...]
    ) -> tuple[str, ...]:
        """  """
        ordered: list[str] = []
        for tf in (*manual, *inferred):
            if not tf or tf == self.timeframe:
                continue
            normalized = str(tf)
            if normalized not in ordered:
                ordered.append(normalized)
        return tuple(ordered)

    @staticmethod
    def _timeframe_suffix_token(timeframe: Optional[str]) -> str:
        token = (timeframe or "").strip()
        return token.replace("/", "_") if token else ""

    def _prepare_informative_frame(self, frame: Optional[pd.DataFrame], timeframe: str) -> Optional[pd.DataFrame]:
        if frame is None or frame.empty:
            return frame
        return frame

    def _register_informative_methods(self) -> None:
        if not self._informative_timeframes:
            return
        for tf in self._informative_timeframes:
            self.__class__._ensure_informative_method(tf)

    def _gc_informative_cache(self, current_whitelist: List[str]) -> None:
        """Drop cached informative frames for pairs no longer in the active whitelist."""
        if not current_whitelist:
            return
        allowed = {str(pair) for pair in current_whitelist if pair}
        if not allowed:
            return
        stale_pairs = [pair for pair in list(self._informative_cache.keys()) if pair not in allowed]
        for pair in stale_pairs:
            self._informative_cache.pop(pair, None)
            self._informative_last.pop(pair, None)

    @classmethod
    def _ensure_informative_method(cls, timeframe: str) -> None:
        func_name = f"populate_indicators_{timeframe.replace('/', '_')}"
        if hasattr(cls, func_name):
            return

        @informative(timeframe)
        def _informative_populator(self, dataframe, metadata, tf=timeframe):
            needs = getattr(
                self.__class__, "_indicator_requirements_map", {}).get(tf)
            return indicators.compute_indicators(
                dataframe,
                self.cfg,
                suffix=None,
                required=needs or set(),
                duplicate_ohlc=True,    # 保留原始 OHLC（列名保持 open/close/volume）
            )
        _informative_populator.__name__ = func_name
        setattr(cls, func_name, _informative_populator)

    def _get_informative_dataframe(self, pair: str, timeframe: str) -> Optional[pd.DataFrame]:
        cached = self._informative_cache.get(pair, {}).get(timeframe)
        if cached is not None and not cached.empty:
            return cached
        getter = getattr(self.dp, 'get_informative_dataframe', None)
        if callable(getter):
            result = getter(pair, timeframe)
            if isinstance(result, tuple):
                result = result[0]
            if isinstance(result, pd.DataFrame) and not result.empty:
                prepared = self._prepare_informative_frame(result.copy(), timeframe)
                self._informative_cache.setdefault(pair, {})[timeframe] = prepared
                return prepared
            return self._prepare_informative_frame(result, timeframe)
        result = self.dp.get_analyzed_dataframe(pair, timeframe)
        if isinstance(result, tuple):
            result = result[0]
        if isinstance(result, pd.DataFrame) and not result.empty:
            prepared = self._prepare_informative_frame(result.copy(), timeframe)
            self._informative_cache.setdefault(pair, {})[timeframe] = prepared
            return prepared
        return self._prepare_informative_frame(result, timeframe)

    def informative_pairs(self):
        try:
            whitelist = self.dp.current_whitelist()
        except Exception:
            whitelist = self.config.get(
                "exchange", {}).get("pair_whitelist", [])
        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        if whitelist and self._informative_timeframes:
            for pair in whitelist:
                for tf in self._informative_timeframes:
                    key = (pair, tf)
                    if key not in seen:
                        seen.add(key)
                        pairs.append(key)
        sensor_pairs = [("BTC/USDT", self.timeframe), ("ETH/USDT", self.timeframe)]
        for sp in sensor_pairs:
            if sp not in seen:
                seen.add(sp)
                pairs.append(sp)
        return pairs

    def bot_start(self, **kwargs) -> None:
        """ """
        self._runmode_token = self._compute_runmode_token()
        backtest_like = self._is_backtest_like_runmode()
        self._persist_enabled = not backtest_like
        try:
            self.logger.info(
                f"[tier] runmode={self._runmode_token or 'unknown'} backtest_like={backtest_like} persist={'on' if self._persist_enabled else 'off'}"
            )
        except Exception:
            pass
        if self._persist_enabled:
            self.persist.load_if_exists()
        else:
            self.persist = _NoopStateStore()
            self.cycle_agent.persist = self.persist
        if self.state.treasury.cycle_start_tick == 0:
            self.state.treasury.cycle_start_tick = self.state.bar_tick
            self.state.treasury.cycle_start_equity = self.eq_provider.get_equity()

    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """计算指标、生成候选信号并触发周期 finalize。
        Args:
            df: 原始 K 线数据。
            metadata: Freqtrade 提供的上下文字典，至少包含 pair。
        Returns:
            pd.DataFrame: 附加指标列后的数据帧。
        Notes:
            当 cfg.adx_len != 14 时会记录动态列名日志，对应 V29.1 修订 #1。
        """
        pair = metadata["pair"]
        print(">>> base_needs:", self._indicator_requirements.get(None))
        base_needs = self._indicator_requirements.get(None)
        df = indicators.compute_indicators(df, self.cfg, required=base_needs)
        informative_rows: Dict[str, pd.Series] = {}
        if self._informative_timeframes:
            for tf in self._informative_timeframes:
                try:
                    info_df = self._get_informative_dataframe(pair, tf)
                except Exception:
                    continue
                if info_df is None or info_df.empty:
                    continue
                informative_rows[tf] = info_df.iloc[-1].copy()
                cache = self._informative_cache.setdefault(pair, {})
                cache[tf] = info_df.copy()
        if informative_rows:
            self._informative_last[pair] = informative_rows
        elif pair in self._informative_last:
            self._informative_last.pop(pair, None)
            self._informative_cache.pop(pair, None)
        if len(df) == 0:
            return df

        if pair.upper().startswith("BTC"):
            eth_df = None
            try:
                eth_df = self._get_informative_dataframe("ETH/USDT", self.timeframe)
            except Exception:
                try:
                    eth_df = self.dp.get_analyzed_dataframe("ETH/USDT", self.timeframe)
                except Exception:
                    eth_df = None
            try:
                self.market_sensor.analyze(df, eth_df)
            except Exception:
                pass

        last = df.iloc[-1].copy()
        last_ts = float(pd.to_datetime(
            last.get("date", pd.Timestamp.utcnow())).timestamp())
        try:
            whitelist = self.dp.current_whitelist()
        except Exception:
            whitelist = [pair]
        self.cycle_agent.maybe_finalize(
            pair=pair,
            bar_ts=last_ts,
            whitelist=whitelist,
            timeframe_sec=self._tf_sec,
            eq_provider=self.eq_provider,
        )
        now_ts = time.time()
        if (now_ts - getattr(self, "_informative_gc_last_ts", 0.0)) >= getattr(self, "_informative_gc_interval_sec", 900):
            try:
                self._gc_informative_cache(whitelist)
            finally:
                self._informative_gc_last_ts = now_ts
        return df

    def get_informative_row(self, pair: str, timeframe: str) -> Optional[pd.Series]:
        """"""
        return self._informative_last.get(pair, {}).get(timeframe)

    def get_informative_value(
        self, pair: str, timeframe: str, column: str, default: Optional[float] = None
    ) -> Optional[float]:
        """"""
        row = self.get_informative_row(pair, timeframe)
        if row is None:
            return default
        try:
            if column in row:
                return row[column]
        except Exception:
            pass
        suffix = self._timeframe_suffix_token(timeframe)
        if suffix:
            alt_col = f"{column}_{suffix}"
            try:
                if alt_col in row:
                    return row[alt_col]
            except Exception:
                pass
        try:
            return row.get(column, default)  # pandas Series get fallback
        except Exception:
            return default

    def _eval_entry_on_row(
        self,
        row: pd.Series,
        inf_rows: Optional[Dict[str, pd.Series]],
        pst_state,
        history_close: Optional[List[float]] = None,
        global_snap: Any = None,
    ) -> Optional[schemas.Candidate]:
        """
        ??????????K??????????? Candidate
        """
        candidates = builder.build_candidates(
            row,
            self.cfg,
            informative=inf_rows,
            history_close=history_close,
            global_snap=global_snap,
        )
        policy = self.tier_mgr.get(pst_state.closs)
        return self.tier_agent.filter_best(policy, candidates)

    def _candidate_with_plan(
        self,
        pair: str,
        candidate: Optional[schemas.Candidate],
        row: pd.Series,
        inf_rows: Optional[Dict[str, pd.Series]],
    ) -> Optional[schemas.Candidate]:
        if not candidate or not getattr(self, "exit_facade", None):
            return candidate
        try:
            _, _, plan = self.exit_facade.resolve_entry_plan(pair, candidate, row, inf_rows or {})
        except Exception:
            plan = None
        if not plan:
            return candidate
        try:
            return replace(
                candidate,
                sl_pct=float(plan.sl_pct) if getattr(plan, "sl_pct", None) else candidate.sl_pct,
                tp_pct=float(plan.tp_pct) if getattr(plan, "tp_pct", None) else candidate.tp_pct,
                plan_timeframe=getattr(plan, "timeframe", getattr(candidate, "plan_timeframe", None)),
                plan_atr_pct=getattr(plan, "atr_pct", getattr(candidate, "plan_atr_pct", None)),
            )
        except Exception:
            return candidate

    def _aligned_informative_for_df(self, pair: str, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """一次性把所有 informative timeframe merge_asof 到主周期索引，避免每行重复合并。"""
        out = {}
        for tf in getattr(self, "_informative_timeframes", []):
            info_df = self._get_informative_dataframe(pair, tf)
            if info_df is None or info_df.empty:
                continue
            left = pd.DataFrame({"_t": pd.to_datetime(
                df["date"]) if "date" in df.columns else pd.to_datetime(df.index)})
            right = info_df.copy()
            right["_tinfo"] = pd.to_datetime(
                right["date"]) if "date" in right.columns else pd.to_datetime(right.index)
            merged = pd.merge_asof(left.sort_values("_t"),
                                   right.sort_values("_tinfo"),
                                   left_on="_t", right_on="_tinfo",
                                   direction="backward")
            merged.index = df.index
            out[tf] = merged
        return out

    def _informative_rows_for_index(self, aligned_info: Dict[str, pd.DataFrame], idx) -> Dict[str, pd.Series]:
        """"""
        rows: Dict[str, pd.Series] = {}
        for tf, frame in aligned_info.items():
            try:
                rows[tf] = frame.loc[idx]
            except Exception:
                continue
        return rows

    def _candidate_to_payload(self, candidate: schemas.Candidate) -> dict[str, Any]:
        return {
            "direction": candidate.direction,
            "kind": candidate.kind,
            "raw_score": candidate.raw_score,
            "rr_ratio": candidate.rr_ratio,
            "expected_edge": candidate.expected_edge,
            "win_prob": candidate.win_prob,
            "squad": candidate.squad,
            "sl_pct": candidate.sl_pct,
            "tp_pct": candidate.tp_pct,
            "score": getattr(candidate, "expected_edge", 0.0),
            "exit_profile": candidate.exit_profile,
            "recipe": candidate.recipe,
            "plan_timeframe": getattr(candidate, "plan_timeframe", None),
            "plan_atr_pct": getattr(candidate, "plan_atr_pct", None),
        }

    def _candidate_allowed_by_policy(self, policy, candidate: Optional[schemas.Candidate]) -> bool:
        if not policy or not candidate:
            return False
        if not policy.permits(kind=candidate.kind, squad=candidate.squad, recipe=candidate.recipe):
            return False
        if candidate.raw_score < policy.min_raw_score:
            return False
        if candidate.rr_ratio < policy.min_rr_ratio:
            return False
        if candidate.expected_edge < policy.min_edge:
            return False
        return True

    def _candidate_allowed_any_tier(self, candidate: Optional[schemas.Candidate]) -> bool:
        if not candidate:
            return False
        iter_policies = getattr(self.tier_mgr, "policies", None)
        policies = iter_policies() if callable(iter_policies) else ()
        if not policies:
            policies = (self.tier_mgr.get(0),)
        for policy in policies:
            if self._candidate_allowed_by_policy(policy, candidate):
                return True
        return False

    def _group_candidates_by_direction(
        self, candidates: list[schemas.Candidate]
    ) -> dict[str, list[schemas.Candidate]]:
        grouped: dict[str, list[schemas.Candidate]] = {"long": [], "short": []}
        for cand in candidates:
            grouped.setdefault(cand.direction, []).append(cand)
        return grouped

    def _trim_candidate_pool(
        self, grouped: dict[str, list[schemas.Candidate]]
    ) -> dict[str, list[schemas.Candidate]]:
        limited: dict[str, list[schemas.Candidate]] = {"long": [], "short": []}
        for direction, items in grouped.items():
            ordered = sorted(items, key=lambda c: (c.expected_edge, c.raw_score), reverse=True)
            limited[direction] = ordered[: self._candidate_pool_limit]
        return limited

    def _apply_entry_signal(
        self, df: pd.DataFrame, idx, candidates_by_dir: dict[str, list[schemas.Candidate]]
    ) -> None:
        has_long = bool(candidates_by_dir.get("long"))
        has_short = bool(candidates_by_dir.get("short"))
        df.at[idx, "enter_long"] = 1 if has_long else 0
        df.at[idx, "enter_short"] = 1 if has_short else 0
        if not has_long and not has_short:
            df.at[idx, "enter_tag"] = None
            return
        payload = {
            "version": 2,
            "candidates": {
                "long": [
                    self._candidate_to_payload(cand) for cand in candidates_by_dir.get("long", [])
                ],
                "short": [
                    self._candidate_to_payload(cand) for cand in candidates_by_dir.get("short", [])
                ],
            },
        }
        df.at[idx, "enter_tag"] = json.dumps(payload)

    def _update_last_signal(self, pair: str, candidate: Optional[schemas.Candidate], row: pd.Series) -> None:
        pst = self.state.get_pair_state(pair)
        try:
            pst.last_atr_pct = float(row.get("atr_pct", 0.0) or 0.0)
        except Exception:
            pst.last_atr_pct = 0.0
        if candidate:
            pst.last_dir = candidate.direction
            pst.last_kind = candidate.kind                           # ← 信号名（mean_rev_long / trend_short / …）
            pst.last_squad = getattr(candidate, "squad", None)       # ← 所属 squad（MRL / PBL / TRS）
            pst.last_score = candidate.expected_edge
            pst.last_sl_pct = candidate.sl_pct
            pst.last_tp_pct = candidate.tp_pct
            pst.last_exit_profile = candidate.exit_profile
            pst.last_recipe = candidate.recipe
        else:
            pst.last_dir = None
            pst.last_squad = None
            pst.last_score = 0.0
            pst.last_sl_pct = 0.0
            pst.last_tp_pct = 0.0
            pst.last_exit_profile = None
            pst.last_recipe = None
        self._last_signal[pair] = candidate

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata["pair"]
        if "BNB" in pair:
            print("BNB")
        df["enter_long"] = 0
        df["enter_short"] = 0
        df["enter_tag"] = None
        aligned_info = self._aligned_informative_for_df(pair, df)
        runmode = getattr(self.dp, "runmode", None)
        is_vector_pass = (RunMode and runmode in {
                          RunMode.BACKTEST, RunMode.HYPEROPT, RunMode.PLOT})
        pst_snapshot = copy.deepcopy(self.state.get_pair_state(pair))
        backend = getattr(self, "global_backend", None)
        try:
            global_snap = backend.get_snapshot() if backend else None
        except Exception:
            global_snap = None

        if is_vector_pass:
            for idx in df.index:
                row = df.loc[idx].copy()
                row["LOSS_TIER_STATE"] = pst_snapshot.closs
                inf_rows = self._informative_rows_for_index(aligned_info, idx)
                raw_candidates = builder.build_candidates(
                    row,
                    self.cfg,
                    informative=inf_rows,
                    global_snap=global_snap,
                )
                if not raw_candidates:
                    continue
                planned: list[schemas.Candidate] = []
                for candidate in raw_candidates:
                    cand_with_plan = self._candidate_with_plan(pair, candidate, row, inf_rows)
                    if cand_with_plan:
                        planned.append(cand_with_plan)
                if not planned:
                    continue
                if is_vector_pass:
                    planned = [
                        cand for cand in planned
                        if self._candidate_allowed_any_tier(cand)
                    ]
                    if not planned:
                        continue
                grouped = self._trim_candidate_pool(self._group_candidates_by_direction(planned))
                if not grouped.get("long") and not grouped.get("short"):
                    continue
                self._apply_entry_signal(df, idx, grouped)

                # pst_snapshot = advance_state(pst_snapshot, sig, row)

        actual_state = self.state.get_pair_state(pair)
        last_idx = df.index[-1]
        last_row = df.loc[last_idx].copy()
        last_row["LOSS_TIER_STATE"] = actual_state.closs
        last_inf_rows = self._informative_rows_for_index(aligned_info, last_idx)
        last_candidate = self._eval_entry_on_row(
            last_row,
            last_inf_rows,
            actual_state,
            history_close=df.get("close", pd.Series(dtype=float)).tolist() if hasattr(df, "get") else None,
            global_snap=global_snap,
        )
        if last_candidate:
            last_candidate = self._candidate_with_plan(pair, last_candidate, last_row, last_inf_rows)
        if not is_vector_pass and last_candidate:
            grouped = self._trim_candidate_pool(
                self._group_candidates_by_direction([last_candidate])
            )
            self._apply_entry_signal(df, last_idx, grouped)
        self._update_last_signal(pair, last_candidate, last_row)
        try:
            if last_candidate and getattr(self, "global_backend", None):
                self.global_backend.record_signal_score(
                    pair, float(getattr(last_candidate, "expected_edge", 0.0))
                )
        except Exception:
            pass
        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        通过 Exit Router 的 VectorExit 渠道应用“可插拔退出规则”。
        - 规则可以直接改 df 并返回 df，或返回一个包含列的 dict（exit_long/exit_short/exit_tag）
        - Router 负责 OR 聚合、exit_tag 覆盖策略（按优先级）
        """

        # 兜底：Freqtrade 要求存在列
        if "exit_long" not in df.columns:
            df["exit_long"] = 0
        if "exit_short" not in df.columns:
            df["exit_short"] = 0
        if "exit_tag" not in df.columns:
            df["exit_tag"] = None
        meta = {
            "pair": metadata.get("pair"),
            "timeframe": getattr(self, "timeframe", None) or self.timeframe,
        }

        # 交给 Router 聚合所有 vector exit 规则
        try:
            if self.exit_facade:
                df = self.exit_facade.apply_vector(
                    df=df,
                    metadata=meta,
                    state=self.state,
                    timeframe_col=None,
                )
            elif EXIT_ROUTER is not None:
                df = EXIT_ROUTER.apply_vector_exits(
                    df=df,
                    metadata=meta,
                    dp=self.dp,
                    cfg=self.cfg,
                    state=self.state,
                    timeframe_col=None,
                )
        except Exception:
            # 安全兜底：如果外部规则出错，不阻断主流程
            pass
        return df

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime,
                            entry_tag: str | None, side: str, **kwargs) -> bool:
        runmode_cfg = self.config.get("runmode", None)
        runmode_token = str(
            getattr(runmode_cfg, "value", runmode_cfg) or "").lower()
        dp_runmode = getattr(self.dp, "runmode", None)
        dp_mode_token = str(
            getattr(dp_runmode, "value", dp_runmode) or "").lower()
        backtest_modes = {"runmode.backtest",
                          "backtest", "hyperopt", "runmode.hyperopt"}
        is_backtest_mode = any(
            token in backtest_modes for token in (runmode_token, dp_mode_token) if token)
        pst = self.state.get_pair_state(pair)
        tier_pol = self.tier_mgr.get(pst.closs)
        try:
            self._tier_debug(
                f"cte called pair={pair} runmode_token={runmode_token} dp_token={dp_mode_token} backtest={is_backtest_mode} closs={pst.closs}"
            )
        except Exception:
            pass
        if is_backtest_mode and pst.cooldown_bars_left > 0:
            try:
                now_ts = float(current_time.timestamp())
            except Exception:
                now_ts = None
            if now_ts is not None:
                last_ts = getattr(pst, "_cooldown_last_ts", None)
                if last_ts is None:
                    pst._cooldown_last_ts = now_ts
                else:
                    bars_elapsed = int((now_ts - last_ts) // self._tf_sec)
                    if bars_elapsed > 0:
                        pst.cooldown_bars_left = max(
                            0, pst.cooldown_bars_left - bars_elapsed)
                        pst._cooldown_last_ts = now_ts
        # 冷却期一律禁止开新仓
        # print(f"pst.closs_{pst.closs}")
        if pst.cooldown_bars_left > 0:
            return False

        # 若当前 tier 配置了 single_position_only，则有仓位时禁止再开
        if tier_pol and getattr(tier_pol, "single_position_only", False) and pst.active_trades:
            return False

        # 实盘 / 干跑：仍用“最新一拍”的 _last_signal 语义
        requested_dir = "long" if side.lower() in ("buy", "long") else "short"
        payload = None
        candidate_from_tag: Optional[schemas.Candidate] = None
        if entry_tag:
            try:
                payload = json.loads(entry_tag)
            except Exception:
                payload = None
            if isinstance(payload, dict):
                candidate_from_tag = self._resolve_candidate_from_tag(pair, payload, side)
            elif payload is not None:
                normalized = self._normalize_entry_meta(payload, side)
                if normalized and pair not in self._pending_entry_meta:
                    self._pending_entry_meta[pair] = normalized

        if is_backtest_mode:
            try:
                self._tier_debug(
                    f"cte precheck pair={pair} candidate_present={bool(candidate_from_tag)} entry_tag_present={bool(entry_tag)}"
                )
                if not candidate_from_tag:
                    tier_name = getattr(tier_pol, "name", None) if tier_pol else None
                    try:
                        self.logger.info(
                            f"[tier] bt-entry-skip pair={pair} closs={pst.closs} tier={tier_name} reason=no_candidate"
                        )
                    except Exception:
                        pass
                    self._tier_debug(
                        f"bt-entry-skip pair={pair} closs={pst.closs} tier={tier_name} reason=no_candidate"
                    )
                    return False
                if pair not in self._pending_entry_meta:
                    self._pending_entry_meta[pair] = self._candidate_meta_from_candidate(candidate_from_tag)
                tier_name = getattr(tier_pol, "name", None) if tier_pol else None
                try:
                    self.logger.info(
                        f"[tier] bt-entry pair={pair} closs={pst.closs} tier={tier_name} recipe={getattr(candidate_from_tag, 'recipe', None)} kind={getattr(candidate_from_tag, 'kind', None)} sl={getattr(candidate_from_tag, 'sl_pct', None)} tp={getattr(candidate_from_tag, 'tp_pct', None)}"
                    )
                except Exception:
                    pass
                self._tier_debug(
                    f"bt-entry pair={pair} closs={pst.closs} tier={tier_name} recipe={getattr(candidate_from_tag, 'recipe', None)} kind={getattr(candidate_from_tag, 'kind', None)} sl={getattr(candidate_from_tag, 'sl_pct', None)} tp={getattr(candidate_from_tag, 'tp_pct', None)}"
                )
                return True
            except Exception as exc:
                self._tier_debug(f"bt-entry-error pair={pair} err={exc}")
                return False

        sig = self._last_signal.get(pair)
        if not sig or sig.direction != requested_dir:
            return False
        self._pending_entry_meta[pair] = {
            "sl_pct": sig.sl_pct,
            "tp_pct": sig.tp_pct,
            "dir": sig.direction,
            "exit_profile": sig.exit_profile,
            "recipe": sig.recipe,
            "plan_timeframe": getattr(sig, "plan_timeframe", None),
            "atr_pct": getattr(sig, "plan_atr_pct", None),
        }
        return True

    @staticmethod
    def _dir_from_side(side: str) -> str:
        return "long" if str(side).lower() in ("buy", "long") else "short"

    def _candidate_meta_from_candidate(self, candidate: schemas.Candidate) -> dict[str, Any]:
        return {
            "dir": candidate.direction,
            # 【新增】补上这两个字段，财政部才能识别信号
            "kind": candidate.kind,
            "squad": getattr(candidate, "squad", None),
            
            "sl_pct": candidate.sl_pct,
            "tp_pct": candidate.tp_pct,
            "exit_profile": candidate.exit_profile,
            "recipe": candidate.recipe,
            "plan_timeframe": getattr(candidate, "plan_timeframe", None),
            "atr_pct": getattr(candidate, "plan_atr_pct", None),
            "expected_edge":candidate.expected_edge,
            "raw_score":candidate.raw_score,
            "score": getattr(candidate, "expected_edge", None),
        }

    def _normalize_entry_meta(self, meta: Optional[Dict[str, Any]], side: str) -> Optional[Dict[str, Any]]:
        """  """
        if meta is None:
            return None
        normalized = dict(meta)
        sl_val = normalized.get('sl_pct')
        if sl_val is None and 'sl' in normalized:
            sl_val = normalized.get('sl')
        try:
            normalized['sl_pct'] = float(sl_val) if sl_val is not None else 0.0
        except Exception:
            normalized['sl_pct'] = 0.0
        tp_val = normalized.get('tp_pct')
        if tp_val is None and 'tp' in normalized:
            tp_val = normalized.get('tp')
        try:
            normalized['tp_pct'] = float(tp_val) if tp_val is not None else 0.0
        except Exception:
            normalized['tp_pct'] = 0.0
        direction = normalized.get('dir')
        if not direction:
            direction = self._dir_from_side(side)
        normalized['dir'] = direction
        if 'plan_timeframe' in normalized:
            ptf = normalized.get('plan_timeframe')
            if ptf is None:
                normalized['plan_timeframe'] = None
            else:
                token = str(ptf).strip()
                normalized['plan_timeframe'] = token or None
        atr_val = normalized.get('atr_pct')
        score_val = normalized.get('score')
        if score_val is None:
            score_val = normalized.get('expected_edge', normalized.get('raw_score'))
        try:
            normalized['score'] = float(score_val) if score_val is not None else 0.0
        except Exception:
            normalized['score'] = 0.0

        if atr_val is not None:
            try:
                normalized['atr_pct'] = float(atr_val)
            except Exception:
                normalized['atr_pct'] = None
        else:
            normalized['atr_pct'] = None
        return normalized

    def _resolve_candidate_from_tag(
        self, pair: str, payload: Dict[str, Any], side: str
    ) -> Optional[schemas.Candidate]:
        candidates_pool = payload.get("candidates")
        if not isinstance(candidates_pool, dict):
            return None
        direction = self._dir_from_side(side)
        raw_list = candidates_pool.get(direction) or []
        if not raw_list:
            return None
        hydrated: list[schemas.Candidate] = []
        for raw in raw_list:
            try:
                plan_atr_val = raw.get("plan_atr_pct")
                plan_atr = None
                if plan_atr_val not in (None, ""):
                    plan_atr = float(plan_atr_val)
                hydrated.append(
                    schemas.Candidate(
                        direction=direction,
                        kind=str(raw["kind"]),
                        raw_score=float(raw["raw_score"]),
                        rr_ratio=float(raw["rr_ratio"]),
                        win_prob=float(raw.get("win_prob", 0.0)),
                        expected_edge=float(raw["expected_edge"]),
                        squad=str(raw["squad"]),
                        sl_pct=float(raw["sl_pct"]),
                        tp_pct=float(raw["tp_pct"]),
                        exit_profile=raw.get("exit_profile"),
                        recipe=raw.get("recipe"),
                        plan_timeframe=raw.get("plan_timeframe"),
                        plan_atr_pct=plan_atr,
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        if not hydrated:
            return None
        pst = self.state.get_pair_state(pair)
        tier_pol = self.tier_mgr.get(pst.closs)
        best = self.tier_agent.filter_best(tier_pol, hydrated)
        if best is None:
            try:
                tier_name = getattr(tier_pol, "name", None) if tier_pol else None
                recipes = [getattr(h, "recipe", None) for h in hydrated]
                kinds = [getattr(h, "kind", None) for h in hydrated]
                self.logger.info(
                    f"[tier] reject_by_tier pair={pair} closs={pst.closs} tier={tier_name} recipes={recipes} kinds={kinds}"
                )
                self._tier_debug(
                    f"reject_by_tier pair={pair} closs={pst.closs} tier={tier_name} recipes={recipes} kinds={kinds}"
                )
            except Exception:
                pass
        return best

    def confirm_trade_exit(
        self,
        pair: str,
        trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime | None = None,
        **kwargs,
    ) -> bool:
        return True

    def _extract_entry_meta(
        self, pair: str, entry_tag: Optional[str], side: str
    ) -> Optional[Dict[str, Any]]:
        meta: Optional[Dict[str, Any]] = None
        parsed = None
        if entry_tag:
            try:
                parsed = json.loads(entry_tag)
            except Exception:
                parsed = None
            if isinstance(parsed, dict) and "candidates" in parsed:
                selected = self._resolve_candidate_from_tag(pair, parsed, side)
                if selected:
                    meta = self._normalize_entry_meta(self._candidate_meta_from_candidate(selected), side)
            elif parsed is not None:
                meta = self._normalize_entry_meta(parsed, side)

        if meta is None:
            cached = self._pending_entry_meta.get(pair)
            if cached is not None:
                meta = self._normalize_entry_meta(cached, side)

        if meta is None:
            sig = self._last_signal.get(pair)
            if sig:
                meta = self._normalize_entry_meta(
                    {
                        "sl_pct": float(sig.sl_pct),
                        "tp_pct": float(sig.tp_pct),
                        "dir": sig.direction,
                        "exit_profile": sig.exit_profile,
                        "recipe": sig.recipe,
                        "plan_timeframe": getattr(sig, "plan_timeframe", None),
                        "atr_pct": getattr(sig, "plan_atr_pct", None),
                        "score": getattr(sig, "raw_score", getattr(sig, "expected_edge", None)),
                    },
                    side,
                )
        return meta

    def _format_gatekeeper_detail(self, result: GateResult, score: float) -> str:
        thresholds = result.thresholds or {}
        th_min = thresholds.get("min_score")
        detail_parts: list[str] = []
        if th_min is not None and score < float(th_min):
            detail_parts.append(f"score {score:.4f} < {float(th_min):.4f}")

        if result.debt is not None:
            try:
                detail_parts.append(f"debt={float(result.debt):.4f}")
            except Exception:
                pass
        return ", ".join(part for part in detail_parts if part)

    def _check_gatekeeping(self, pair: str, meta: Dict[str, Any]) -> GateResult:
        pst = self.state.get_pair_state(pair)
        try:
            score = float(meta.get("score", meta.get("expected_edge", meta.get("raw_score", 0.0))))
        except Exception:
            score = 0.0
        try:
            raw = self.treasury_agent.evaluate_signal_quality(pair, score, closs=pst.closs)
        except Exception:
            raw = {}

        result = GateResult.from_mapping(raw, default_closs=pst.closs) if raw else GateResult(
            allowed=True, thresholds={}, reason="", debt=None, closs=pst.closs
        )
        if not result.allowed:
            detail = self._format_gatekeeper_detail(result, score)
            msg = f"Global Gatekeeper: Rejected {pair} - {result.reason or 'rejected'}"
            if detail:
                msg = f"{msg} ({detail})"
            try:
                self.logger.info(msg)
            except Exception:
                print(msg)
        return result

    def _reserve_backend_risk(self, pair: str, risk: float) -> bool:
        cap_abs = 0.0
        try:
            equity_now = self.eq_provider.get_equity()
            cap_pct = self.state.get_dynamic_portfolio_cap_pct(equity_now)
            cap_abs = cap_pct * equity_now
            reserved = bool(self.global_backend.add_risk_usage(risk, cap_abs))
        except Exception as exc:
            reserved = False
            try:
                self.logger.error(
                    f"Global backend reservation failed for {pair}, risk={risk:.4f}, cap_abs={cap_abs:.4f}",
                    exc_info=exc,
                )
            except Exception:
                print(f"[backend] failed to reserve {pair}: {exc}")

        if not reserved:
            msg = f"Global Gatekeeper: CAP reached for {pair}, risk={risk:.4f}, cap_abs={cap_abs:.4f}"
            try:
                self.logger.info(msg)
            except Exception:
                print(msg)
            return False
        return True

    def _reserve_risk_resources(
        self,
        pair: str,
        stake: float,
        risk: float,
        bucket: str,
        sl: float,
        tp: float,
        direction: str,
        current_rate: float,
        meta: Dict[str, Any],
    ) -> bool:
        meta_payload = dict(meta or {})
        meta_payload.update(
            {
                "sl_pct": sl,
                "tp_pct": tp,
                "stake_final": stake,
                "risk_final": risk,
                "bucket": bucket,
                "entry_price": current_rate,
                "dir": direction,
            }
        )

        if self._is_backtest_like_runmode():
            self._pending_entry_meta[pair] = meta_payload
            return True

        if getattr(self, "global_backend", None):
            backend_reserved = self._reserve_backend_risk(pair, risk)
            if not backend_reserved:
                return False

        rid = f"{pair}:{bucket}:{uuid.uuid4().hex}"
        self.reservation.reserve(pair, rid, risk, bucket)
        meta_payload["reservation_id"] = rid
        self._pending_entry_meta[pair] = meta_payload
        return True

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """Compute margin stake for Freqtrade while keeping nominal sizing internal.

        self.sizer.compute(ctx) returns:
            - stake_margin: amount passed back to Freqtrade as stake_amount (margin)
            - real_risk: max loss (USDT) at current stop based on the nominal position

        When recording trades via state.record_trade_open(), always use nominal position sizes;
        do not confuse stake_margin with any *_nominal field.
        """
        # 1. 先处理可能遗漏的退出（保持原样）
        # self._reconcile_missed_exits(pair)

        # ==================== 【修复开始】手动喂信号 ====================
        # 解析 Freqtrade 传入的 entry_tag (JSON 字符串)，必须在 _backtest_catch_up 之前
        meta = self._extract_entry_meta(pair, entry_tag, side)

        # 仅在回测/Hyperopt 模式下，且成功解析出 meta 时执行状态注入
        if self._is_backtest_like_runmode() and meta:
            pst = self.state.get_pair_state(pair)

            # 强制更新 pair_state，模拟实盘中 populate_entry_trend 的效果
            # 这样 TreasuryAgent.plan() 就能读到非零的 last_score 了
            try:
                pst.last_score = float(meta.get("score", meta.get("expected_edge", 0.0)))
                pst.last_dir = meta.get("dir")
                pst.last_sl_pct = float(meta.get("sl_pct", 0.0))
                pst.last_tp_pct = float(meta.get("tp_pct", 0.0))

                # 补充必要字段，防止 Treasury 因字段缺失而过滤
                # 如果 meta 里没这些字段，给个默认值确保能过 Gatekeeper
                pst.last_kind = str(meta.get("kind", "backtest_signal"))
                pst.last_squad = str(meta.get("squad", "NBX"))
                pst.last_exit_profile = meta.get("exit_profile")
                pst.last_recipe = meta.get("recipe")
            except Exception:
                pass
        # ==================== 【修复结束】 ====================

        # 2. 时光机倒带：补齐债务衰减、冷却，并重新运行 Treasury.plan()
        # 此时 Treasury 已经能看到上面注入的 last_score    
        self._backtest_catch_up(current_time)

        # 3. 后续标准流程（保持原样）
        if not meta:
            return 0.0

        gate_result = self._check_gatekeeping(pair, meta)
        if not gate_result.allowed:
            return 0.0

        sl = float(meta.get("sl_pct", 0.0))
        tp = float(meta.get("tp_pct", 0.0))
        direction = str(meta.get("dir"))
        try:
            score_val = float(meta.get("score", meta.get("expected_edge", meta.get("raw_score", 0.0))))
        except Exception:
            score_val = 0.0

        stake, risk, bucket = self.sizer.compute(
            pair=pair,
            sl_pct=sl,
            tp_pct=tp,
            direction=direction,
            proposed_stake=proposed_stake,
            min_stake=min_stake,
            max_stake=max_stake,
            leverage=leverage,
            plan_atr_pct=meta.get("atr_pct"),
            exit_profile=meta.get("exit_profile"),
            bucket=meta.get("bucket"),
            current_rate=current_rate,
            score=score_val,
        )
        if stake <= 0 or risk <= 0:
            return 0.0

        reserved = self._reserve_risk_resources(
            pair=pair,
            stake=stake,
            risk=risk,
            bucket=bucket,
            sl=sl,
            tp=tp,
            direction=direction,
            current_rate=current_rate,
            meta=meta,
        )
        if not reserved:
            return 0.0
        return float(stake)

    def leverage(self, *args, **kwargs) -> float:
        """返回配置中的强制杠杆倍数，供 Freqtrade 下单使用。"""
        return self.cfg.trading.sizing.enforce_leverage

    def order_filled(self, pair: str, trade, order, current_time: datetime, **kwargs) -> None:
        """处理开仓或平仓成交事件，更新风险账本、预约状态并触发持久化。"""
        trade_id = str(getattr(trade, "trade_id", getattr(trade, "id", "NA")))
        side = getattr(order, "ft_order_side", None)

        # Freqtrade 约定：entry_side / exit_side 与 ft_order_side 匹配
        is_entry = side == getattr(trade, "entry_side", None)
        is_exit  = side == getattr(trade, "exit_side", None)

        if is_entry:
            # 对 entry：尽量带上 meta，但没有也要能工作
            meta = self._pending_entry_meta.pop(pair, None)

            opened = self.execution.on_open_filled(
                pair=pair,
                trade=trade,
                order=order,
                pending_meta=meta,
                tier_mgr=self.tier_mgr,
            )
            if opened:
                try:
                    self.state.get_pair_state(pair)._cooldown_last_ts = float(current_time.timestamp())
                except Exception:
                    pass
            if opened and self._persist_enabled:
                self.persist.save()

        elif is_exit:
            processed = getattr(self, "_bt_closed_trades", None)
            if processed is None:
                processed = set()
                self._bt_closed_trades = processed

            if self._is_backtest_like_runmode() and trade_id in processed:
                return

            if self._is_backtest_like_runmode() and current_time:
                self._backtest_catch_up(current_time)

            profit_abs = getattr(trade, "close_profit_abs", 0.0)
            if profit_abs is None or profit_abs == 0.0:
                fill_price = getattr(order, "average", getattr(order, "price", getattr(trade, "close_rate", 0.0)))
                profit_abs = trade.calc_profit(fill_price)

            if profit_abs != 0:
                try:
                    trade.close_profit_abs = profit_abs
                    trade.profit_abs = profit_abs
                except Exception:
                    pass

            closed = self.execution.on_close_filled(
                pair=pair,
                trade=trade,
                order=order,
                tier_mgr=self.tier_mgr,
            )

            if closed:
                try:
                    self.state.get_pair_state(pair)._cooldown_last_ts = float(current_time.timestamp())
                except Exception:
                    pass

                reason = getattr(trade, "exit_reason", "unknown_fill")
                fill_price = getattr(order, "average", getattr(order, "price", getattr(trade, "close_rate", 0.0)))

                self._log_full_exit_state(
                    pair=pair,
                    trade=trade,
                    reason=reason,
                    pnl_abs=profit_abs,
                    exit_rate=fill_price,
                    exit_time=current_time,
                )

            if self._is_backtest_like_runmode():
                processed.add(trade_id)

            if closed and self._persist_enabled:
                self.persist.save()
        else:
            # 其它类型（比如 reduce_only / 手动改单等），可以先忽略或简单打个日志
            return

    def order_cancelled(self, pair: str, trade, order, current_time: datetime, **kwargs) -> None:
        """当订单被撤销时，释放对应预约并记录分析日志，符合 V29.1 修订 #5（仅释放预约，不回灌财政字段）。"""
        released, meta = self._handle_cancel_or_reject(pair)
        if released:
            trade_id = str(getattr(trade, "trade_id", getattr(trade, "id", meta.get(
                "reservation_id") if meta else "NA"))) if trade else meta.get("reservation_id", "NA") if meta else "NA"
            tag = ExitTags.ORDER_CANCELLED if ExitTags else "order_cancelled"
            self.analytics.log_exit(pair, trade_id, tag)

    def order_rejected(self, pair: str, trade, order, current_time: datetime, **kwargs) -> None:
        """当订单被拒绝时，与撤单路径一致，仅释放预约并写入分析日志。"""
        released, meta = self._handle_cancel_or_reject(pair)
        if released:
            trade_id = str(getattr(trade, "trade_id", getattr(trade, "id", meta.get(
                "reservation_id") if meta else "NA"))) if trade else meta.get("reservation_id", "NA") if meta else "NA"
            tag = ExitTags.ORDER_REJECTED if ExitTags else "order_rejected"
            self.analytics.log_exit(pair, trade_id, tag)

    def _handle_cancel_or_reject(self, pair: str) -> tuple[bool, Optional[dict[str, Any]]]:
        """内部工具：统一处理撤单/拒单路径，释放预约并返回是否释放及原始元数据快照。"""
        meta = self._pending_entry_meta.get(pair)
        meta_snapshot = dict(meta) if meta else None
        rid = meta.get("reservation_id") if meta else None
        released = self.execution.on_cancel_or_reject(pair, rid)
        if released:
            self._pending_entry_meta.pop(pair, None)
            if self._persist_enabled:
                self.persist.save()
        return released, meta_snapshot


    @staticmethod
    def _apply_leverage_pct(pct: Optional[float], trade) -> Optional[float]:
        """
        将基于“价格百分比”的 pct（例如 0.02 表示价格±2%）
        转换为 freqtrade 期望的“含杠杆 ROI 百分比”。

        - 对于期货：roi_pct = pct * trade.leverage
        - 对于现货：leverage=1，相当于不变
        """
        if pct is None:
            return None
        try:
            pct = float(pct)
        except (TypeError, ValueError):
            return None
        if pct <= 0:
            return None

        lev = getattr(trade, "leverage", 1.0) or 1.0
        try:
            lev = float(lev)
        except (TypeError, ValueError):
            lev = 1.0
        if lev <= 0:
            lev = 1.0

        return pct * lev


    def custom_stoploss(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> Optional[float]:
        """Delegate stoploss calculation to ExitRouter rules."""
        if EXIT_ROUTER is None or SLContext is None:
            return self.stoploss
        try:
            ctx = SLContext(
                pair=pair,
                trade=trade,
                now=current_time,
                profit=float(current_profit or 0.0),
                dp=self.dp,
                cfg=self.cfg,
                state=self.state,
                strategy=self,
            )
            sl_pct = EXIT_ROUTER.sl_best(ctx, base_sl_pct=None)
        except Exception:
            sl_pct = None

        # 转成含杠杆 ROI 百分比
        sl_pct = self._apply_leverage_pct(sl_pct, trade)

        if sl_pct is None or sl_pct <= 0:
            return self.stoploss
        
        return -abs(float(sl_pct))

    def custom_exit(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        """
        自定义离场逻辑：
        1. 先走 ExitPolicyV29 的风控 / flip 等“策略级规则”。
        2. 如果没有触发，再用“持仓均价 ± N×ATR(当前K线)”价格触发离场。
        """
        # 1) 策略级风控 / flip / ICU / 债务风险 等
        tid = str(getattr(trade, "trade_id", getattr(trade, "id", "NA")))
        reason = self.exit_policy.decide(
            pair,
            tid,
            float(current_profit) if current_profit is not None else None,
            trade=trade,
        )

        # 2) 如果策略级没有给出退出，再用 ATR 价格逻辑
        if not reason:
            reason = self._atr_entry_exit_reason(
                pair,
                trade,
                current_time,
                current_rate,
                current_profit,
            )

        if not reason:
            return None

        # 3) 记录退出原因，方便回测 / 复盘分析
        try:
            if hasattr(trade, "exit_reason"):
                setattr(trade, "exit_reason", reason)
        except Exception:
            pass
        try:
            if hasattr(trade, "set_custom_data"):
                trade.set_custom_data("exit_tag", reason)
        except Exception:
            pass

        try:
            self.analytics.log_exit(pair, tid, reason)
        except Exception:
            pass

        return reason


    def custom_roi(
        self,
        pair: str,
        trade,
        current_time: datetime,
        trade_duration: int,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> Optional[float]:
        """
        自定义 ROI 回调：
        - 由 freqtrade 在每次循环、每个未平仓交易上调用
        - 返回当前应使用的最小 ROI 阈值（比例，如 0.05 表示 +5%）
        - 实际阈值由 ExitRouter/ExitFacade 中的 TP 规则聚合得到
        """
        # 如果 Router / TPContext 尚未初始化，则回退到 minimal_roi 逻辑
        if EXIT_ROUTER is None or TPContext is None:
            return None

        try:
            # 当前我们的 TP 规则并未使用 ctx.profit，因此这里给一个占位值即可
            ctx = TPContext(
                pair=pair,
                trade=trade,
                now=current_time,
                profit=0.0,
                dp=self.dp,
                cfg=self.cfg,
                state=self.state,
                strategy=self,
            )
            # 从所有 TP 规则中取“最保守的”（最小的正值）tp_pct
            tp_pct = EXIT_ROUTER.tp_best(ctx, base_tp_pct=None)
        except Exception:
            tp_pct = None

        # 把 tp_pct 换算成“账户收益百分比”的格式（考虑杠杆）
        tp_pct = self._apply_leverage_pct(tp_pct, trade)

        # 返回 None 表示“保持原有 minimal_roi”
        if tp_pct is None or tp_pct <= 0:
            return None

        # freqtrade 期望的是一个 ROI 阈值（比例），由 min_roi_reached() 与当前利润对比后决定是否离场
        return  float(tp_pct)

    def _router_sl_tp_pct(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_profit: float,
    ) -> tuple[Optional[float], Optional[float]]:
        """
        使用 EXIT_ROUTER + ExitFacade 计算当前 trade 对应的 sl_pct / tp_pct。
        返回值为“相对开仓价的百分比（ΔP / entry_price）”，不含杠杆。
        """
        if EXIT_ROUTER is None or SLContext is None or TPContext is None:
            return None, None

        try:
            sl_ctx = SLContext(
                pair=pair,
                trade=trade,
                now=current_time,
                profit=float(current_profit or 0.0),
                dp=self.dp,
                cfg=self.cfg,
                state=self.state,
                strategy=self,
            )
            sl_pct = EXIT_ROUTER.sl_best(sl_ctx, base_sl_pct=None)
        except Exception:
            sl_pct = None

        try:
            tp_ctx = TPContext(
                pair=pair,
                trade=trade,
                now=current_time,
                profit=float(current_profit or 0.0),
                dp=self.dp,
                cfg=self.cfg,
                state=self.state,
                strategy=self,
            )
            tp_pct = EXIT_ROUTER.tp_best(tp_ctx, base_tp_pct=None)
        except Exception:
            tp_pct = None

        # 这里刻意不调用 self._apply_leverage_pct，保留为“价格百分比”
        return (float(sl_pct) if sl_pct and sl_pct > 0 else None,
                float(tp_pct) if tp_pct and tp_pct > 0 else None)

    def _get_current_atr_abs(
        self,
        pair: str,
        current_time: datetime,
        profile: Optional[Any] = None,
    ) -> Optional[float]:
        """
        从 DataProvider 的指标表里，按 current_time 精确地取这一根K线的 ATR 绝对值。

        优先使用 profile.atr_timeframe；否则退回到策略 / 配置的主 timeframe。
        """
        dp = getattr(self, "dp", None)
        if dp is None:
            return None

        # 1) 决定 ATR 使用的 timeframe
        atr_tf = None
        if profile is not None:
            atr_tf = getattr(profile, "atr_timeframe", None)
        if not atr_tf:
            # cfg.timeframe 一般等于策略主周期
            atr_tf = getattr(self.cfg, "timeframe", None) or self.timeframe

        try:
            analyzed = dp.get_analyzed_dataframe(pair, atr_tf)
        except Exception:
            return None

        df = analyzed[0] if isinstance(analyzed, (list, tuple)) else analyzed
        if df is None or df.empty:
            return None

        # 2) 和 atr_pct_from_dp 一样的时间对齐逻辑：df.loc[:current_time]
        try:
            upto = df.loc[:current_time] if current_time is not None else df
        except Exception:
            try:
                ct = (
                    current_time.replace(tzinfo=None)
                    if getattr(current_time, "tzinfo", None)
                    else current_time
                )
                upto = df.loc[:ct]
            except Exception:
                upto = df

        if upto is None or upto.empty:
            return None

        row = upto.iloc[-1]

        # 3) 优先直接用 ATR 列；没有的话用 atr_pct * close 反推
        try:
            if "atr" in row.index:
                atr_val = float(row["atr"])
                if math.isfinite(atr_val) and atr_val > 0:
                    return atr_val
        except Exception:
            pass

        try:
            atr_pct = float(row.get("atr_pct", 0.0))
            close = float(row.get("close", 0.0))
            if atr_pct > 0 and close > 0:
                atr_val = atr_pct * close
                if math.isfinite(atr_val) and atr_val > 0:
                    return atr_val
        except Exception:
            return None

        return None


    def _atr_entry_exit_reason(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
    ) -> Optional[str]:
        """
        仅根据“持仓均价 ± N×ATR(当前K线)”判断是否离场：
        - 多单: SL = entry - atr_mul_sl * ATR(t), TP = entry + atr_mul_tp * ATR(t)
        - 空单: SL = entry + atr_mul_sl * ATR(t), TP = entry - atr_mul_tp * ATR(t)

        N (atr_mul_sl / atr_mul_tp) 完全由 exit_profiles + ExitFacade 决定。
        """
        if self.exit_facade is None:
            return None

        # 1) 解析当前 trade 使用的 exit_profile
        try:
            profile_name, profile_def, _ = self.exit_facade.resolve_trade_plan(
                pair, trade, current_time
            )
        except Exception:
            profile_def = None

        if not profile_def:
            return None

        # 2) 读取配置中的 ATR 倍数
        k_sl = float(getattr(profile_def, "atr_mul_sl", 0.0) or 0.0)
        k_tp = getattr(profile_def, "atr_mul_tp", None)
        if k_tp is None or k_tp <= 0:
            # 和 compute_plan_from_atr 保持一致：没配置 atr_mul_tp 就 >= 2×SL
            k_tp = max(k_sl * 2.0, 0.0)
        k_tp = float(k_tp)

        # 3) 按 current_time 从原始指标数据里取这一根K线的 ATR 绝对值
        atr_abs = self._get_current_atr_abs(pair, current_time, profile_def)
        if atr_abs is None or atr_abs <= 0:
            return None

        entry = float(trade.open_rate)
        price = float(current_rate)
        is_short = bool(getattr(trade, "is_short", False))

        sl_abs = k_sl * atr_abs if k_sl > 0 else None
        tp_abs = k_tp * atr_abs if k_tp > 0 else None

        # 4) 按“持仓均价 ± N×ATR”比较当前价格
        if not is_short:
            # 多单
            if sl_abs is not None:
                sl_price = entry - sl_abs
                if price <= sl_price:
                    # print(f"price_{price},atr_abs{atr_abs} sl_abs{sl_abs} current_time{current_time}\n")
                    return "atr_entry_sl"
            if tp_abs is not None:
                tp_price = entry + tp_abs
                if price >= tp_price:
                    # print(f"price_{price},atr_abs{atr_abs} tp_price{tp_price} current_time{current_time}\n")
                    return "atr_entry_tp"
        else:
            # 空单
            if sl_abs is not None:
                sl_price = entry + sl_abs
                if price >= sl_price:
                    # print(f"price_{price},atr_abs{atr_abs} sl_abs{sl_abs} current_time{current_time}\n")
                    return "atr_entry_sl"
            if tp_abs is not None:
                tp_price = entry - tp_abs
                if price <= tp_price:
                    # print(f"price_{price},atr_abs{atr_abs} tp_price{tp_price} current_time{current_time}\n")
                    return "atr_entry_tp"

        return None

    def _backtest_catch_up(self, current_time: datetime) -> None:
        """

        在回测模式下模拟时间推进，驱动债务衰减和财政周期更新。

        修复了 populate_indicators 导致的“未来时间”阻断逻辑运行的问题。

        """
        if not self._is_backtest_like_runmode():
            return

        # 获取当前回测时间戳
        now_ts = current_time.timestamp()
        last_ts = self.state.last_finalized_bar_ts

        # ==================== [核心修复] 时间重置逻辑 ====================
        # 1. last_ts 为 None: 说明刚启动，从未记录过。
        # 2. last_ts > now_ts: 说明 last_ts 被 populate_indicators 设为了数据集的末尾时间(未来)，
        #    而当前回测指针(now_ts)刚从过去开始跑。此时产生了时间倒挂。
        # 3. 时间差距过大: 如果上次时间是 1970 (0.0)，也需要重置。
        if last_ts is None or last_ts > now_ts or last_ts == 0.0:
            # 将上次结算时间重置为“当前时间 - 1个周期”，确保本次能触发逻辑
            self.state.last_finalized_bar_ts = now_ts - self._tf_sec
            last_ts = self.state.last_finalized_bar_ts
            
            # 可选：如果是刚开始回测，初始化财政周期的起始时间
            if self.state.treasury.cycle_start_tick == 0:
                self.state.treasury.cycle_start_tick = self.state.bar_tick
                self.state.treasury.cycle_start_equity = self.eq_provider.get_equity()

        # ==================== 标准步进逻辑 ====================
        
        delta_seconds = now_ts - last_ts
        
        # 只有当时间流逝超过设定的 timeframe (例如5分钟) 才推进状态
        if delta_seconds < self._tf_sec:
            return

        # 计算过去了多少个 K 线周期
        bars_passed = int(delta_seconds // self._tf_sec)
        
        # 更新 Bar 计数
        self.state.bar_tick += bars_passed
        
        # [重要] 推进 last_finalized_bar_ts，保持步调一致
        # 不直接设为 now_ts 是为了避免 K 线时间对齐误差累积
        self.state.last_finalized_bar_ts += (bars_passed * self._tf_sec)

        # --- 以下为业务逻辑 (债务衰减 / 冷却 / 财政计划) ---

        decay_rate = getattr(getattr(self.cfg, "risk", None), "pain_decay_per_bar", 0.999)
        
        # 1. 中央债务池自然衰减
        if self.state.debt_pool > 0:
            self.state.debt_pool *= (decay_rate ** bars_passed)
            # 防止无限趋近 0 但不归零
            if self.state.debt_pool < 1.0: 
                self.state.debt_pool = 0.0

        # 2. 交易对冷却时间减少
        for pst in self.state.per_pair.values():
            if pst.cooldown_bars_left > 0:
                pst.cooldown_bars_left = max(0, pst.cooldown_bars_left - bars_passed)

        # 3. 财政周期 (Treasury Cycle) 检查
        cycle_len = int(self.cfg.cycle_len_bars)
        start_tick = self.state.treasury.cycle_start_tick
        
        if (self.state.bar_tick - start_tick) >= cycle_len:
            equity = self.eq_provider.get_equity()
            
            # 如果配置了“盈利周期清空债务”，执行清债
            if (equity - self.state.treasury.cycle_start_equity) >= 0 and self.cfg.risk.clear_debt_on_profitable_cycle:
                self.state.debt_pool = 0.0
                for pst in self.state.per_pair.values():
                    pst.local_loss = 0.0
                    pst.closs = 0
            
            # 重置周期起始点
            self.state.treasury.cycle_start_tick = self.state.bar_tick
            self.state.treasury.cycle_start_equity = equity

        # 4. 重新计算财政拨款 (Re-Plan)'
        # 注意：这里会利用修正后的债务池数据和权益数据
        try:
            snapshot = self.cycle_agent._build_snapshot()
            plan = self.treasury_agent.plan(snapshot, self.eq_provider.get_equity())
            self.state.treasury.k_long = plan.k_long
            self.state.treasury.k_short = plan.k_short
            self.state.treasury.theta = plan.theta
            self.state.treasury.final_r = plan.final_r
            self.state.treasury.available = plan.available
            self.state.treasury.bias = plan.bias
            self.state.treasury.volatility = plan.volatility
        except Exception:
            pass

    def _log_full_exit_state(
        self,
        pair: str,
        trade,
        reason: str,
        pnl_abs: float,
        exit_rate: float | None = None,
        exit_time: datetime | None = None,
    ):
        try:
            trade_id = str(getattr(trade, "trade_id", getattr(trade, "id", "NA")))
            pst = self.state.get_pair_state(pair)
            tier_pol = self.tier_mgr.get(pst.closs)
            tier_name = getattr(tier_pol, "name", "unknown")

            entry_price = getattr(trade, "open_rate", 0.0)
            amount = getattr(trade, "amount", 0.0)
            stake_amount = getattr(trade, "stake_amount", 0.0)
            open_date = getattr(trade, "open_date", None)

            final_exit_price = exit_rate if exit_rate is not None else getattr(trade, "close_rate", 0.0)
            final_exit_time = exit_time if exit_time is not None else getattr(trade, "close_date", None)

            # --- [新增] 获取 K 值 (根据方向) ---
            is_short = getattr(trade, "is_short", False)
            k_val = self.state.treasury.k_short if is_short else self.state.treasury.k_long

            # --- [新增] 获取 Backend 真实债务 ---
            backend_debt = 0.0
            if self.state.backend:
                try:
                    # 获取后端快照中的 debt_pool
                    backend_debt = float(self.state.backend.get_snapshot().debt_pool)
                except Exception:
                    pass

            details = {
                "pnl_abs": pnl_abs,
                "tier": tier_name,
                "closs": pst.closs,
                
                # 核心债务数据
                "local_loss": pst.local_loss,            # 1. 局部债务
                "central_debt_local": self.state.debt_pool, # 2. 中央债务 (本地衰减版)
                "central_debt_backend": backend_debt,    # 3. [新增] 中央债务 (后端刚性版)
                
                # 信号与参数
                "score": pst.last_score,                 # 4. [新增] 分数
                "k_val": k_val,                          # 5. [新增] K值 (拨款系数)

                "cooldown": pst.cooldown_bars_left,
                "entry_price": entry_price,
                "exit_price": final_exit_price,
                "amount": amount,
                "stake": stake_amount,
                "open_at": str(open_date) if open_date else "",
                "close_at": str(final_exit_time) if final_exit_time else "",
                "duration": str(final_exit_time - open_date) if (final_exit_time and open_date) else "",
            }
            self.analytics.log_exit(pair, trade_id, reason, **details)
        except Exception as e:
            print(f"Log exit state failed: {e}")
