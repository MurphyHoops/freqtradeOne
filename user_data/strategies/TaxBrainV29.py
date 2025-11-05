"""TaxBrainV29 策略路由与多代理协调层。

本模块实现 AGENTS.md 规定的 V29 架构：Signal/Tier/Treasury/Reservation/Risk/Execution 等代理在此由 TaxBrainV29 统筹调度，并保留 V29.1 的五项修订以确保兼容回测、监控与恢复流程。
"""

from __future__ import annotations

import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import types
from typing import Any, Dict, Optional

import pandas as pd

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

from freqtrade.strategy import IStrategy

from user_data.strategies.agents.analytics import AnalyticsAgent
from user_data.strategies.agents.cycle import CycleAgent
from user_data.strategies.agents.execution import ExecutionAgent
from user_data.strategies.agents.exit import ExitPolicyV29
from user_data.strategies.agents.persist import StateStore
from user_data.strategies.agents.reservation import ReservationAgent
from user_data.strategies.agents.risk import RiskAgent
from user_data.strategies.agents.signal import Candidate, compute_indicators, gen_candidates
from user_data.strategies.agents.sizer import SizerAgent
from user_data.strategies.agents.tier import TierAgent, TierManager
from user_data.strategies.agents.treasury import TreasuryAgent
from user_data.strategies.config.v29_config import V29Config, apply_overrides


@dataclass
class ActiveTradeMeta:
    """ActiveTradeMeta 用于记录在市订单在 TaxBrainV29 流程中的关键元数据，方便执行、退出与风控代理在不同 hook 间共享统一视图。
    
    Attributes:
        sl_pct: 建仓时设定的止损百分比（通常为负值，代表最大亏损距离）。
        tp_pct: 建仓时设定的止盈百分比。
        direction: 信号方向，取值为 'long' 或 'short'。
        entry_bar_tick: 建仓发生时的全局 bar_tick 序号，用于与 CycleAgent 对齐。
        entry_price: 建仓价格，供执行/退出时计算盈亏。
        bucket: 拨款桶信息，fast 或 slow。
        icu_bars_left: ICU 强制回收剩余 bar 数，None 表示当前不受 ICU 约束。
    """
    sl_pct: float
    tp_pct: float
    direction: str
    entry_bar_tick: int
    entry_price: float
    bucket: str
    icu_bars_left: Optional[int]


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
    last_squad: Optional[str] = None
    last_sl_pct: float = 0.0
    last_tp_pct: float = 0.0
    last_atr_pct: float = 0.0
    active_trades: Dict[str, ActiveTradeMeta] = field(default_factory=dict)


@dataclass
class TreasuryState:
    """TreasuryState 持久化 fast/slow 拨款结果以及财政周期起点，保证跨周期及崩溃恢复后的拨款连续性。
    
    Attributes:
        fast_alloc_risk: fast 拨款桶中每个交易对的名义风险额度。
        slow_alloc_risk: slow 拨款桶中每个交易对的名义风险额度。
        cycle_start_tick: 当前财政周期开始时记录的 bar_tick。
        cycle_start_equity: 当前财政周期开始时的权益快照。
    """
    fast_alloc_risk: Dict[str, float] = field(default_factory=dict)
    slow_alloc_risk: Dict[str, float] = field(default_factory=dict)
    cycle_start_tick: int = 0
    cycle_start_equity: float = 0.0

    def to_snapshot(self) -> Dict[str, Any]:
        """导出财政子模块的可序列化快照数据，供 StateStore 写入磁盘。
        
        Returns:
            Dict[str, Any]: fast/slow 拨款与周期起点组成的字典。
        """
        return {
            "fast_alloc_risk": self.fast_alloc_risk,
            "slow_alloc_risk": self.slow_alloc_risk,
            "cycle_start_tick": self.cycle_start_tick,
            "cycle_start_equity": self.cycle_start_equity,
        }

    def restore_snapshot(self, payload: Dict[str, Any]) -> None:
        """从 StateStore 读取的字典中恢复财政拨款状态。
        
        Args:
            payload: 包含 fast/slow 拨款及周期起点信息的字典。
        """
        self.fast_alloc_risk = {k: float(v) for k, v in payload.get("fast_alloc_risk", {}).items()}
        self.slow_alloc_risk = {k: float(v) for k, v in payload.get("slow_alloc_risk", {}).items()}
        self.cycle_start_tick = int(payload.get("cycle_start_tick", 0))
        self.cycle_start_equity = float(payload.get("cycle_start_equity", 0.0))


class GlobalState:
    """GlobalState 负责维护 TaxBrainV29 的组合级状态，包括债务池、各交易对风险、财政拨款以及 bar_tick 进度。
    
    该容器为多个代理提供读写接口，并承担快照持久化与恢复的职责。
    """
    def __init__(self, cfg: V29Config) -> None:
        """初始化全局状态容器。
        
        Args:
            cfg: 当前加载的 V29Config，用于读取 CAP、税率等参数。
        """
        self.cfg = cfg
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

    def get_pair_state(self, pair: str) -> PairState:
        """获取指定交易对的 PairState，如不存在则初始化默认实例。
        
        Args:
            pair: 交易对名称，例如 'BTC/USDT'。
        
        Returns:
            PairState: 可供调用方修改的状态对象。
        """
        if pair not in self.per_pair:
            self.per_pair[pair] = PairState()
        return self.per_pair[pair]

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
        base = self.cfg.portfolio_cap_pct_base
        if equity <= 0:
            return base * 0.5
        if (self.debt_pool / equity) > self.cfg.drawdown_threshold_pct:
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
        cap = tier_pol.per_pair_risk_cap_pct * equity
        used = self.pair_risk_open.get(pair, 0.0) + reserved
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
    ) -> None:
        """记录新建仓的风险占用，并创建 ActiveTradeMeta 以便执行与退出逻辑后续引用。
        
        Args:
            pair: 交易对名称。
            trade_id: Freqtrade 分配的交易标识。
            real_risk: 根据止损距离计算出的真实风险敞口。
            sl_pct: 建仓时的止损百分比。
            tp_pct: 建仓时的止盈百分比。
            direction: 方向标识 'long' 或 'short'。
            bucket: 当前订单使用的拨款桶（fast/slow）。
            entry_price: 建仓价格。
            tier_pol: 所属 TierPolicy，用于初始化 ICU 倒计时与冷却时长。
        """
        pst = self.get_pair_state(pair)
        self.trade_risk_ledger[trade_id] = float(real_risk)
        self.pair_risk_open[pair] = self.pair_risk_open.get(pair, 0.0) + float(real_risk)

        icu_left = tier_pol.icu_force_exit_bars if tier_pol.icu_force_exit_bars > 0 else None
        pst.active_trades[trade_id] = ActiveTradeMeta(
            sl_pct=float(sl_pct),
            tp_pct=float(tp_pct),
            direction=str(direction),
            entry_bar_tick=self.bar_tick,
            entry_price=float(entry_price),
            bucket=str(bucket),
            icu_bars_left=icu_left,
        )
        pst.cooldown_bars_left = max(pst.cooldown_bars_left, tier_pol.cooldown_bars)

    def record_trade_close(self, pair: str, trade_id: str, profit_abs: float, tier_mgr) -> None:
        """在平仓时回收风险、更新债务池，并依据盈亏调整冷却与连续亏损计数。
        
        Args:
            pair: 交易对名称。
            trade_id: 交易标识。
            profit_abs: 本次交易的绝对盈亏（正数表示盈利）。
            tier_mgr: TierManager，用于在状态更新后获取新的策略参数。
        """
        pst = self.get_pair_state(pair)
        was_risk = self.trade_risk_ledger.pop(trade_id, 0.0)
        self.pair_risk_open[pair] = max(0.0, self.pair_risk_open.get(pair, 0.0) - was_risk)
        if self.pair_risk_open.get(pair, 0.0) <= 1e-12:
            self.pair_risk_open[pair] = 0.0
        pst.active_trades.pop(trade_id, None)

        if profit_abs >= 0:
            tax = profit_abs * self.cfg.tax_rate_on_wins
            self.debt_pool = max(0.0, self.debt_pool - tax)
            pst.local_loss = max(0.0, pst.local_loss - profit_abs)
            pst.closs = max(0, pst.closs - 1)
            pol = tier_mgr.get(pst.closs)
            pst.cooldown_bars_left = max(pst.cooldown_bars_left, pol.cooldown_bars_after_win)
        else:
            loss = abs(profit_abs)
            pst.local_loss += loss
            self.debt_pool += loss
            pst.closs += 1
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
                "last_squad": pst.last_squad,
                "last_sl_pct": pst.last_sl_pct,
                "last_tp_pct": pst.last_tp_pct,
                "last_atr_pct": pst.last_atr_pct,
                "active_trades": {
                    tid: {
                        "sl_pct": meta.sl_pct,
                        "tp_pct": meta.tp_pct,
                        "direction": meta.direction,
                        "entry_bar_tick": meta.entry_bar_tick,
                        "entry_price": meta.entry_price,
                        "bucket": meta.bucket,
                        "icu_bars_left": meta.icu_bars_left,
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
        self.trade_risk_ledger = {k: float(v) for k, v in payload.get("trade_risk_ledger", {}).items()}
        self.pair_risk_open = {k: float(v) for k, v in payload.get("pair_risk_open", {}).items()}
        self.bar_tick = int(payload.get("bar_tick", 0))
        self.current_cycle_ts = payload.get("current_cycle_ts")
        self.last_finalized_bar_ts = payload.get("last_finalized_bar_ts")
        self.last_finalize_walltime = float(payload.get("last_finalize_walltime", time.time()))

        self.per_pair = {}
        for pair, pst_payload in payload.get("per_pair", {}).items():
            pst = PairState(
                closs=int(pst_payload.get("closs", 0)),
                local_loss=float(pst_payload.get("local_loss", 0.0)),
                cooldown_bars_left=int(pst_payload.get("cooldown_bars_left", 0)),
                last_dir=pst_payload.get("last_dir"),
                last_score=float(pst_payload.get("last_score", 0.0)),
                last_squad=pst_payload.get("last_squad"),
                last_sl_pct=float(pst_payload.get("last_sl_pct", 0.0)),
                last_tp_pct=float(pst_payload.get("last_tp_pct", 0.0)),
                last_atr_pct=float(pst_payload.get("last_atr_pct", 0.0)),
            )
            for tid, meta_payload in pst_payload.get("active_trades", {}).items():
                pst.active_trades[tid] = ActiveTradeMeta(
                    sl_pct=float(meta_payload.get("sl_pct", 0.0)),
                    tp_pct=float(meta_payload.get("tp_pct", 0.0)),
                    direction=str(meta_payload.get("direction", "")),
                    entry_bar_tick=int(meta_payload.get("entry_bar_tick", 0)),
                    entry_price=float(meta_payload.get("entry_price", 0.0)),
                    bucket=str(meta_payload.get("bucket", "slow")),
                    icu_bars_left=(
                        int(meta_payload["icu_bars_left"])
                        if meta_payload.get("icu_bars_left") is not None
                        else None
                    ),
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
        self.equity_current = float(payload.get("equity_current", self.equity_current))

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
    """TaxBrainV29 是遵循 AGENTS.md 规范实现的多代理路由策略类。
    
    该类负责在 Freqtrade 各个 hook 中调度信号、财政、风险、预约、执行等子代理，并保持 V29.1 的五项修订（动态 ADX、盈利周期清债、timeframe/startup 覆盖、早锁盈兜底、仅释放预约）持续生效。
    """
    timeframe = V29Config().timeframe
    can_short = True
    startup_candle_count = V29Config().startup_candle_count

    minimal_roi = {"0": 10_000.0}
    stoploss = -0.99
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = False
    exit_profit_only = False
    ignore_buy_signals = False
    ignore_sell_signals = True

    def __init__(self, config: Dict[str, Any]) -> None:
        """构造策略实例并初始化所有代理、状态容器及持久化组件。
        
        Args:
            config: Freqtrade 传入的策略配置字典。
        
        Notes:
            - 通过 apply_overrides 应用 strategy_params；
            - 同步 timeframe/startup_candle_count 至实例与类属性（V29.1 修订 #3）；
            - 搭建 Treasury/Risk/Reservation 等代理并初始化 StateStore。
        """
        super().__init__(config)
        base_cfg = V29Config()
        self.cfg = apply_overrides(base_cfg, config.get("strategy_params", {}))

        self.timeframe = self.cfg.timeframe
        self.startup_candle_count = self.cfg.startup_candle_count
        try:
            self.__class__.timeframe = self.cfg.timeframe
            self.__class__.startup_candle_count = self.cfg.startup_candle_count
        except Exception:
            pass

        print(f"[TaxBrainV29] timeframe sync cfg={self.cfg.timeframe} instance={self.timeframe} class={getattr(self.__class__, 'timeframe', 'NA')}")
        print(f"[TaxBrainV29] ADX_{self.cfg.adx_len} active")

        self.tier_mgr = TierManager()
        self.tier_agent = TierAgent()
        user_data_dir = Path(config.get("user_data_dir", "."))
        initial_equity = float(config.get("dry_run_wallet", self.cfg.dry_run_wallet_fallback))
        self.eq_provider = EquityProvider(initial_equity)
        self.state = GlobalState(self.cfg)

        self.analytics = AnalyticsAgent(user_data_dir / "logs")
        self.reservation = ReservationAgent(self.cfg, analytics=self.analytics)
        self.treasury_agent = TreasuryAgent(self.cfg, self.tier_mgr)
        self.risk_agent = RiskAgent(self.cfg, self.reservation, self.tier_mgr)
        state_file = (user_data_dir / "taxbrain_v29_state.json").resolve()
        self.persist = StateStore(
            filepath=str(state_file),
            state=self.state,
            eq_provider=self.eq_provider,
            reservation_agent=self.reservation,
        )

        self.cycle_agent = CycleAgent(
            cfg=self.cfg,
            state=self.state,
            reservation=self.reservation,
            treasury=self.treasury_agent,
            risk=self.risk_agent,
            analytics=self.analytics,
            persist=self.persist,
            tier_mgr=self.tier_mgr,
        )
        self.sizer = SizerAgent(self.state, self.reservation, self.eq_provider, self.cfg, self.tier_mgr)
        self.exit_policy = ExitPolicyV29(self.state, self.eq_provider, self.cfg)
        self.execution = ExecutionAgent(self.state, self.reservation, self.eq_provider, self.cfg)

        self._last_signal: Dict[str, Optional[Candidate]] = {}
        self._pending_entry_meta: Dict[str, Dict[str, Any]] = {}
        self._tf_sec = self._tf_to_sec(self.cfg.timeframe)

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

    def bot_start(self, **kwargs) -> None:
        """Freqtrade bot 启动阶段加载持久化状态，并在需要时初始化财政周期起点。"""
        self.persist.load_if_exists()
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
        pst = self.state.get_pair_state(pair)
        df = compute_indicators(df, self.cfg)

        if self.cfg.adx_len != 14 and "adx" in df.columns:
            self.analytics.log_debug(
                "adx_dynamic",
                f"Using ADX_{self.cfg.adx_len}",
                {"pair": pair},
            )

        if len(df) == 0:
            return df
        last = df.iloc[-1]
        last_ts = float(pd.to_datetime(last.get("date", pd.Timestamp.utcnow())).timestamp())

        candidates = gen_candidates(last)
        policy = self.tier_mgr.get(pst.closs)
        best = self.tier_agent.filter_best(policy, candidates)
        if best:
            pst.last_dir = best.direction
            pst.last_score = best.expected_edge
            pst.last_squad = best.kind
            pst.last_sl_pct = best.sl_pct
            pst.last_tp_pct = best.tp_pct
            pst.last_atr_pct = float(last.get("atr_pct", 0.0))
        else:
            pst.last_dir = None
            pst.last_squad = None
            pst.last_score = 0.0
            pst.last_sl_pct = 0.0
            pst.last_tp_pct = 0.0
        self._last_signal[pair] = best

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
        return df

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """将最近一次筛选出的信号方向写入 Freqtrade 期望的 enter_long/enter_short 列，仅对最后一根 K 线生效。"""
        df["enter_long"] = 0
        df["enter_short"] = 0
        pair = metadata["pair"]
        sig = self._last_signal.get(pair)
        if sig and len(df) > 0:
            if sig.direction == "long":
                df.loc[df.index[-1], "enter_long"] = 1
            else:
                df.loc[df.index[-1], "enter_short"] = 1
        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """策略使用 custom_exit 进行完整的离场判断，此处按接口要求返回全零列。"""
        df["exit_long"] = 0
        df["exit_short"] = 0
        return df


    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        """在发单前执行冷却、持仓与方向一致性校验，并缓存后续下单所需的止损/止盈元数据。
        
        Args:
            pair: 交易对名称。
            order_type: Freqtrade 传入的订单类型。
            amount: 计划下单数量。
            rate: 当前价格。
            time_in_force: 有效期设定。
            current_time: 当前钩子调用时间。
            entry_tag: 自定义标签。
            side: 'buy' 或 'sell'，与信号方向对应。
        
        Returns:
            bool: 是否允许继续下单。
        """
        pst = self.state.get_pair_state(pair)
        if pst.cooldown_bars_left > 0:
            return False
        if len(pst.active_trades) > 0:
            return False
        sig = self._last_signal.get(pair)
        if not sig or sig.direction != side.lower():
            return False
        self._pending_entry_meta[pair] = {
            "sl_pct": sig.sl_pct,
            "tp_pct": sig.tp_pct,
            "dir": sig.direction,
        }
        return True

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """委托 SizerAgent 计算名义仓位并通过 ReservationAgent 锁定风险额度，供随后的 order_filled 使用。
        
        Args:
            pair: 交易对名称。
            current_time: 当前时间。
            current_rate: 当前价格。
            proposed_stake: Freqtrade 根据余额计算的原始仓位建议。
        
        Returns:
            float: 实际用于下单的仓位金额。
        
        Notes:
            会缓存 reservation_id、风险敞口与拨款桶等信息以便后续钩子引用。
        """
        if abs(leverage - self.cfg.enforce_leverage) > 1e-6:
            return 0.0
        meta = self._pending_entry_meta.get(pair)
        if not meta:
            return 0.0
        if "stake_final" in meta:
            return float(meta["stake_final"])

        sl = float(meta["sl_pct"])
        tp = float(meta["tp_pct"])
        stake, risk, bucket = self.sizer.compute(
            pair=pair,
            sl_pct=sl,
            tp_pct=tp,
            direction=meta["dir"],
            min_stake=min_stake,
            max_stake=max_stake,
        )
        if stake <= 0 or risk <= 0:
            return 0.0

        rid = f"{pair}:{bucket}:{uuid.uuid4().hex}"
        self.reservation.reserve(pair, rid, risk, bucket)
        meta.update(
            {
                "stake_final": stake,
                "risk_final": risk,
                "reservation_id": rid,
                "bucket": bucket,
                "entry_price": current_rate,
            }
        )
        return float(stake)

    def leverage(self, *args, **kwargs) -> float:
        """返回配置中的强制杠杆倍数，供 Freqtrade 下单使用。"""
        return self.cfg.enforce_leverage

    def order_filled(self, pair: str, trade, order, current_time: datetime, **kwargs) -> None:
        """处理开仓或平仓成交事件，更新风险账本、预约状态并触发持久化。"""
        trade_id = str(getattr(trade, "trade_id", getattr(trade, "id", "NA")))
        is_open = bool(getattr(trade, "is_open", True))
        if is_open:
            meta = self._pending_entry_meta.get(pair)
            if not meta:
                return
            opened = self.execution.on_open_filled(pair, trade, order, meta, self.tier_mgr)
            if opened:
                self._pending_entry_meta.pop(pair, None)
                self.persist.save()
        else:
            closed = self.execution.on_close_filled(pair, trade, order, self.tier_mgr)
            if closed:
                self.analytics.log_exit(pair, trade_id, "close_filled")
                self.persist.save()

    def order_cancelled(self, pair: str, trade, order, current_time: datetime, **kwargs) -> None:
        """当订单被撤销时，释放对应预约并记录分析日志，符合 V29.1 修订 #5（仅释放预约，不回灌财政字段）。"""
        released, meta = self._handle_cancel_or_reject(pair)
        if released:
            trade_id = str(getattr(trade, "trade_id", getattr(trade, "id", meta.get("reservation_id") if meta else "NA"))) if trade else meta.get("reservation_id", "NA") if meta else "NA"
            self.analytics.log_exit(pair, trade_id, "order_cancelled")

    def order_rejected(self, pair: str, trade, order, current_time: datetime, **kwargs) -> None:
        """当订单被拒绝时，与撤单路径一致，仅释放预约并写入分析日志。"""
        released, meta = self._handle_cancel_or_reject(pair)
        if released:
            trade_id = str(getattr(trade, "trade_id", getattr(trade, "id", meta.get("reservation_id") if meta else "NA"))) if trade else meta.get("reservation_id", "NA") if meta else "NA"
            self.analytics.log_exit(pair, trade_id, "order_rejected")

    def _handle_cancel_or_reject(self, pair: str) -> tuple[bool, Optional[dict[str, Any]]]:
        """内部工具：统一处理撤单/拒单路径，释放预约并返回是否释放及原始元数据快照。"""
        meta = self._pending_entry_meta.get(pair)
        meta_snapshot = dict(meta) if meta else None
        rid = meta.get("reservation_id") if meta else None
        released = self.execution.on_cancel_or_reject(pair, rid)
        if released:
            self._pending_entry_meta.pop(pair, None)
            self.persist.save()
        return released, meta_snapshot

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
        """结合 trade 元信息、ActiveTradeMeta 以及早锁盈策略计算自定义止损距离。
        
        Args:
            pair: 交易对名称。
            trade: 当前交易对象。
            current_time: 当前时间。
            current_rate: 即时价格。
            current_profit: 当前收益率。
            after_fill: Freqtrade 是否在开仓后立即调用。
            **kwargs: 预留兼容参数。
        
        Returns:
            Optional[float]: 若返回 None 则沿用默认止损，否则为新的止损百分比（负值）。
        
        Notes:
            遵循 V29.1 修订 #4，tp_pct 会依次从 trade.custom_data、trade.user_data 和 ActiveTradeMeta 中获取，实现三重兜底。
        """
        pst = self.state.get_pair_state(pair)
        tid = str(getattr(trade, "trade_id", getattr(trade, "id", "NA")))
        meta = pst.active_trades.get(tid)

        sl_pct = None
        tp_pct = None
        atr_pct_hint = 0.0

        try:
            sl_pct = getattr(trade, "get_custom_data", lambda *_: None)("sl_pct")
            tp_pct = getattr(trade, "get_custom_data", lambda *_: None)("tp_pct")
        except Exception:
            pass
        try:
            if (sl_pct is None or sl_pct <= 0) and hasattr(trade, "user_data"):
                sl_pct = float(trade.user_data.get("sl_pct", 0.0))
            if (tp_pct is None or tp_pct <= 0) and hasattr(trade, "user_data"):
                tp_pct = float(trade.user_data.get("tp_pct", 0.0))
        except Exception:
            pass
        if (sl_pct is None or sl_pct <= 0) and meta:
            sl_pct = meta.sl_pct
        if (tp_pct is None or tp_pct <= 0) and meta:
            tp_pct = meta.tp_pct
        if meta:
            atr_pct_hint = pst.last_atr_pct

        if sl_pct is None or sl_pct <= 0:
            return -0.03

        if tp_pct and current_profit and current_profit > 0:
            try:
                if current_profit >= float(tp_pct) * float(self.cfg.breakeven_lock_frac_of_tp):
                    try:
                        analyzed = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                        pairdf = analyzed[0] if isinstance(analyzed, (list, tuple)) else analyzed
                        if len(pairdf) > 0 and "atr" in pairdf.columns:
                            atr_pct_hint = float(pairdf["atr"].iloc[-1] / max(pairdf["close"].iloc[-1], 1e-12))
                    except Exception:
                        pass
                    dist = self.exit_policy.early_lock_distance(trade, current_rate, current_profit, atr_pct_hint)
                    if dist is not None:
                        self.analytics.log_exit(pair, tid, "breakeven_lock")
                        return max(-abs(float(sl_pct)), dist)
            except Exception:
                pass

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
        """调用 ExitPolicyV29 判断是否需要触发自定义离场，并记录原因供分析。
        
        Args:
            pair: 交易对名称。
            trade: 当前交易对象。
            current_time: 当前时间。
            current_rate: 当前价格。
            current_profit: 当前收益率。
            **kwargs: 预留参数。
        
        Returns:
            Optional[str]: 若返回字符串则表示触发指定退出原因，否则 None。
        """
        tid = str(getattr(trade, "trade_id", getattr(trade, "id", "NA")))
        reason = self.exit_policy.decide(
            pair, tid, float(current_profit) if current_profit is not None else None
        )
        if reason:
            self.analytics.log_exit(pair, tid, reason)
        return reason






