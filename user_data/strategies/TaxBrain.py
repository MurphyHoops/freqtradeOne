# TaxBrainV29.1.py
# 在 V29 基础上的最小修订包（5 处）：
# 1) ADX 动态列名，随 cfg.adx_len 自动取列
# 2) 盈利周期清债：finalize 内实现 cycle 计数与清债触发
# 3) timeframe / startup_candle_count 实例覆盖（并同步到类属性，以适配不同 Freqtrade 版本）
# 4) 早锁盈兜底 tp_pct：确保任意路径都能取到 tp_pct 以触发保本抬止损
# 5) 撤/拒单不回灌财政表：仅释放预约，由下一拍重算财政

from __future__ import annotations

from freqtrade.strategy import IStrategy
from typing import Optional, Dict, Any, Tuple, List, Literal
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import pandas_ta as ta
import json, os, time, uuid, math

# -------------------------------
# 0) 配置
# -------------------------------
@dataclass
class V29Config:
    timeframe: str = "5m"
    startup_candle_count: int = 210

    # 组合 CAP 与压力降档
    portfolio_cap_pct_base: float = 0.20
    drawdown_threshold_pct: float = 0.15

    # 财政
    treasury_fast_split_pct: float = 0.30
    fast_topK_squads: int = 10
    slow_universe_pct: float = 0.90
    min_injection_nominal_fast: float = 30.0
    min_injection_nominal_slow: float = 7.0

    # 债务/税/衰减
    tax_rate_on_wins: float = 0.20
    pain_decay_per_bar: float = 0.999
    clear_debt_on_profitable_cycle: bool = True
    cycle_len_bars: int = 288  # 5m*24h

    # 早锁盈
    breakeven_lock_frac_of_tp: float = 0.5
    breakeven_lock_eps_atr_pct: float = 0.1  # 乘以 ATR%（非绝对ATR）

    # 预约/Finalize
    force_finalize_mult: float = 1.5
    reservation_ttl_bars: int = 6  # 预约超时自动回收

    # 指标
    ema_fast: int = 50
    ema_slow: int = 200
    rsi_len: int = 14
    atr_len: int = 14
    adx_len: int = 14

    # 行为开关
    suppress_baseline_when_stressed: bool = True

    # 运行
    dry_run_wallet_fallback: float = 1000.0
    enforce_leverage: float = 1.0


# -------------------------------
# 1) TierPolicy / TierManager
# -------------------------------
@dataclass
class TierPolicy:
    name: str
    allowed_kinds: set[str]
    min_raw_score: float
    min_rr_ratio: float
    min_edge: float
    sizing_algo: Literal["BASELINE", "TARGET_RECOVERY"]
    k_mult_base_pct: float
    recovery_factor: float
    cooldown_bars: int
    cooldown_bars_after_win: int
    per_pair_risk_cap_pct: float
    max_stake_notional_pct: float
    icu_force_exit_bars: int

class TierManager:
    def __init__(self):
        self._t0 = TierPolicy(
            name="T0_healthy",
            allowed_kinds={"mean_rev_long","pullback_long","trend_short"},
            min_raw_score=0.20, min_rr_ratio=1.2, min_edge=0.002,
            sizing_algo="BASELINE",
            k_mult_base_pct=0.005, recovery_factor=1.0,
            cooldown_bars=5, cooldown_bars_after_win=2,
            per_pair_risk_cap_pct=0.03, max_stake_notional_pct=0.15,
            icu_force_exit_bars=0
        )
        self._t12 = TierPolicy(
            name="T12_recovery",
            allowed_kinds={"pullback_long","trend_short"},
            min_raw_score=0.15, min_rr_ratio=1.4, min_edge=0.003,
            sizing_algo="TARGET_RECOVERY",
            k_mult_base_pct=0.004, recovery_factor=1.5,
            cooldown_bars=10, cooldown_bars_after_win=4,
            per_pair_risk_cap_pct=0.02, max_stake_notional_pct=0.12,
            icu_force_exit_bars=30
        )
        self._t3p = TierPolicy(
            name="T3p_ICU",
            allowed_kinds={"trend_short","mean_rev_long"},
            min_raw_score=0.20, min_rr_ratio=1.6, min_edge=0.004,
            sizing_algo="TARGET_RECOVERY",
            k_mult_base_pct=0.003, recovery_factor=2.0,
            cooldown_bars=20, cooldown_bars_after_win=6,
            per_pair_risk_cap_pct=0.01, max_stake_notional_pct=0.10,
            icu_force_exit_bars=20
        )

    def get(self, closs: int) -> TierPolicy:
        if closs <= 0: return self._t0
        if closs <= 2: return self._t12
        return self._t3p


# -------------------------------
# 2) 数据结构
# -------------------------------
@dataclass
class Candidate:
    direction: str
    kind: str
    sl_pct: float
    tp_pct: float
    raw_score: float
    rr_ratio: float
    win_prob: float
    expected_edge: float
    squad: str

@dataclass
class ActiveTradeMeta:
    sl_pct: float
    tp_pct: float
    direction: str
    entry_bar_tick: int
    entry_price: float
    bucket: str
    icu_bars_left: Optional[int]

@dataclass
class PairState:
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
class Treasury:
    fast_alloc_risk: Dict[str, float] = field(default_factory=dict)
    slow_alloc_risk: Dict[str, float] = field(default_factory=dict)
    cycle_start_tick: int = 0
    cycle_start_equity: float = 0.0
    def to_snapshot(self) -> Dict[str, Any]:
        return {
            "fast_alloc_risk": self.fast_alloc_risk,
            "slow_alloc_risk": self.slow_alloc_risk,
            "cycle_start_tick": self.cycle_start_tick,
            "cycle_start_equity": self.cycle_start_equity,
        }
    def restore_snapshot(self, snap: Dict[str, Any]) -> None:
        self.fast_alloc_risk = {k: float(v) for k,v in snap.get("fast_alloc_risk", {}).items()}
        self.slow_alloc_risk = {k: float(v) for k,v in snap.get("slow_alloc_risk", {}).items()}
        self.cycle_start_tick = int(snap.get("cycle_start_tick", 0))
        self.cycle_start_equity = float(snap.get("cycle_start_equity", 0.0))

class EquityProvider:
    def __init__(self, initial_equity: float):
        self.equity_current = float(initial_equity)
    def to_snapshot(self) -> Dict[str, Any]:
        return {"equity_current": self.equity_current}
    def restore_snapshot(self, snap: Dict[str, Any]) -> None:
        self.equity_current = float(snap.get("equity_current", self.equity_current))
    def get_equity(self) -> float:
        return self.equity_current
    def on_trade_closed_update(self, profit_abs: float) -> None:
        self.equity_current += float(profit_abs)


# -------------------------------
# 3) 全局状态（账本/财政/预约）
# -------------------------------
class GlobalState:
    def __init__(self, cfg: V29Config):
        self.cfg = cfg
        self.per_pair: Dict[str, PairState] = {}
        self.debt_pool: float = 0.0

        self.trade_risk_ledger: Dict[str, float] = {}
        self.pair_risk_open: Dict[str, float] = {}

        # reservation_id -> (pair, risk, bucket, ttl_bars)
        self.reservations: Dict[str, Tuple[str, float, str, int]] = {}
        self.reserved_pair_risk: Dict[str, float] = {}
        self.reserved_portfolio_risk: float = 0.0
        self.reserved_bucket_risk: Dict[str, float] = {"fast":0.0, "slow":0.0}

        self.treasury = Treasury()

        # bar 时钟
        self.bar_tick: int = 0
        self.current_cycle_ts: Optional[float] = None
        self.last_finalized_bar_ts: Optional[float] = None
        self.reported_pairs_for_current_cycle: set[str] = set()
        self.last_finalize_walltime: float = time.time()

    # ---- PairState ----
    def get_pair_state(self, pair: str) -> PairState:
        if pair not in self.per_pair:
            self.per_pair[pair] = PairState()
        return self.per_pair[pair]

    # ---- 快照 ----
    def to_snapshot(self) -> Dict[str, Any]:
        per_pair_snap = {}
        for p, st in self.per_pair.items():
            per_pair_snap[p] = {
                "closs": st.closs, "local_loss": st.local_loss,
                "cooldown_bars_left": st.cooldown_bars_left,
                "last_dir": st.last_dir, "last_score": st.last_score,
                "last_squad": st.last_squad, "last_sl_pct": st.last_sl_pct,
                "last_tp_pct": st.last_tp_pct, "last_atr_pct": st.last_atr_pct,
                "active_trades": {
                    tid: {
                        "sl_pct": m.sl_pct, "tp_pct": m.tp_pct, "direction": m.direction,
                        "entry_bar_tick": m.entry_bar_tick, "entry_price": m.entry_price,
                        "bucket": m.bucket, "icu_bars_left": m.icu_bars_left
                    } for tid, m in st.active_trades.items()
                }
            }
        return {
            "per_pair": per_pair_snap,
            "debt_pool": self.debt_pool,
            "trade_risk_ledger": self.trade_risk_ledger,
            "pair_risk_open": self.pair_risk_open,
            "reservations": self.reservations,
            "reserved_pair_risk": self.reserved_pair_risk,
            "reserved_portfolio_risk": self.reserved_portfolio_risk,
            "reserved_bucket_risk": self.reserved_bucket_risk,
            "treasury": self.treasury.to_snapshot(),
            "bar_tick": self.bar_tick,
            "current_cycle_ts": self.current_cycle_ts,
            "last_finalized_bar_ts": self.last_finalized_bar_ts,
            "reported_pairs_for_current_cycle": list(self.reported_pairs_for_current_cycle),
            "last_finalize_walltime": self.last_finalize_walltime,
        }

    def restore_snapshot(self, snap: Dict[str, Any]) -> None:
        self.per_pair = {}
        for p, st in snap.get("per_pair", {}).items():
            atmap = {}
            for tid, meta in st.get("active_trades", {}).items():
                atmap[tid] = ActiveTradeMeta(
                    sl_pct=float(meta["sl_pct"]), tp_pct=float(meta["tp_pct"]),
                    direction=str(meta["direction"]), entry_bar_tick=int(meta["entry_bar_tick"]),
                    entry_price=float(meta.get("entry_price", 0.0)),
                    bucket=str(meta.get("bucket","slow")), icu_bars_left=meta.get("icu_bars_left")
                )
            self.per_pair[p] = PairState(
                closs=int(st.get("closs", 0)),
                local_loss=float(st.get("local_loss", st.get("local_debt", 0.0))),
                cooldown_bars_left=int(st.get("cooldown_bars_left", 0)),
                last_dir=st.get("last_dir"), last_score=float(st.get("last_score", 0.0)),
                last_squad=st.get("last_squad"),
                last_sl_pct=float(st.get("last_sl_pct", 0.0)),
                last_tp_pct=float(st.get("last_tp_pct", 0.0)),
                last_atr_pct=float(st.get("last_atr_pct", 0.0)),
                active_trades=atmap
            )
        self.debt_pool = float(snap.get("debt_pool", snap.get("global_loss_bucket", 0.0)))
        self.trade_risk_ledger = {k: float(v) for k,v in snap.get("trade_risk_ledger", {}).items()}
        self.pair_risk_open = {k: float(v) for k,v in snap.get("pair_risk_open", {}).items()}

        # reservations 兼容旧格式
        self.reservations = {}
        for rid, val in snap.get("reservations", {}).items():
            if isinstance(val, (list, tuple)):
                if len(val) == 4:
                    pair, risk, bucket, ttl = val
                    self.reservations[rid] = (pair, float(risk), str(bucket), int(ttl))
                elif len(val) == 3:
                    pair, risk, bucket = val
                    self.reservations[rid] = (pair, float(risk), str(bucket), int(self.cfg.reservation_ttl_bars))
                elif len(val) == 2:
                    pair, risk = val
                    self.reservations[rid] = (pair, float(risk), "slow", int(self.cfg.reservation_ttl_bars))
        self.reserved_pair_risk = {k: float(v) for k,v in snap.get("reserved_pair_risk", {}).items()}
        self.reserved_portfolio_risk = float(snap.get("reserved_portfolio_risk", 0.0))
        self.reserved_bucket_risk = {k: float(v) for k,v in snap.get("reserved_bucket_risk", {"fast":0.0,"slow":0.0}).items()}

        self.treasury.restore_snapshot(snap.get("treasury", {}))
        self.bar_tick = int(snap.get("bar_tick", 0))
        self.current_cycle_ts = snap.get("current_cycle_ts", None)
        self.last_finalized_bar_ts = snap.get("last_finalized_bar_ts", None)
        self.reported_pairs_for_current_cycle = set(snap.get("reported_pairs_for_current_cycle", []))
        self.last_finalize_walltime = float(snap.get("last_finalize_walltime", time.time()))

    def reset_cycle_after_restore(self) -> None:
        self.reported_pairs_for_current_cycle = set()
        self.current_cycle_ts = None
        self.last_finalize_walltime = time.time()

    # ---- CAP ----
    def get_dynamic_portfolio_cap_pct(self, equity: float) -> float:
        base = self.cfg.portfolio_cap_pct_base
        if equity <= 0: return base * 0.5
        if (self.debt_pool / equity) > self.cfg.drawdown_threshold_pct:
            return base * 0.5
        return base

    def get_total_risk(self, include_reserved: bool=True) -> float:
        s = sum(self.pair_risk_open.values())
        if include_reserved: s += self.reserved_portfolio_risk
        return s

    def get_pair_risk(self, pair: str, include_reserved: bool=True) -> float:
        s = self.pair_risk_open.get(pair, 0.0)
        if include_reserved: s += self.reserved_pair_risk.get(pair, 0.0)
        return s

    def per_pair_cap_room(self, pair: str, equity: float, tier_pol: TierPolicy) -> float:
        cap = tier_pol.per_pair_risk_cap_pct * equity
        used = self.get_pair_risk(pair, include_reserved=True)
        return max(0.0, cap - used)

    # ---- 预约 ----
    def reserve_risk(self, pair: str, reservation_id: str, risk_amount: float, bucket: str) -> None:
        if reservation_id in self.reservations: return
        ttl = int(self.cfg.reservation_ttl_bars)
        risk = float(risk_amount)
        self.reservations[reservation_id] = (pair, risk, bucket, ttl)
        self.reserved_portfolio_risk += risk
        self.reserved_pair_risk[pair] = self.reserved_pair_risk.get(pair, 0.0) + risk
        self.reserved_bucket_risk[bucket] = self.reserved_bucket_risk.get(bucket, 0.0) + risk

    def release_reservation(self, reservation_id: str) -> Tuple[str, float, str]:
        if reservation_id not in self.reservations: return ("", 0.0, "slow")
        pair, risk, bucket, _ttl = self.reservations.pop(reservation_id)
        self.reserved_portfolio_risk -= risk
        self.reserved_pair_risk[pair] = max(0.0, self.reserved_pair_risk.get(pair, 0.0) - risk)
        self.reserved_bucket_risk[bucket] = max(0.0, self.reserved_bucket_risk.get(bucket, 0.0) - risk)
        if self.reserved_pair_risk.get(pair, 0.0) <= 1e-12:
            self.reserved_pair_risk[pair] = 0.0
        return (pair, risk, bucket)

    def tick_reservation_ttl(self) -> None:
        # 在 finalize 时调用
        expired: List[str] = []
        for rid,(pair,risk,bucket,ttl) in list(self.reservations.items()):
            ttl -= 1
            self.reservations[rid] = (pair,risk,bucket,ttl)
            if ttl <= 0: expired.append(rid)
        for rid in expired:
            self.release_reservation(rid)

    # ---- 账本推进 ----
    def record_trade_open(self, pair: str, trade_id: str, real_risk: float,
                          sl_pct: float, tp_pct: float, direction: str,
                          bucket: str, entry_price: float, tier_pol: TierPolicy) -> None:
        pst = self.get_pair_state(pair)
        self.trade_risk_ledger[trade_id] = float(real_risk)
        self.pair_risk_open[pair] = self.pair_risk_open.get(pair, 0.0) + float(real_risk)

        icu_left = tier_pol.icu_force_exit_bars if tier_pol.icu_force_exit_bars > 0 else None
        pst.active_trades[trade_id] = ActiveTradeMeta(
            sl_pct=float(sl_pct), tp_pct=float(tp_pct), direction=str(direction),
            entry_bar_tick=self.bar_tick, entry_price=float(entry_price),
            bucket=str(bucket), icu_bars_left=icu_left
        )
        # 冷却
        pst.cooldown_bars_left = max(pst.cooldown_bars_left, tier_pol.cooldown_bars)

    def record_trade_close(self, pair: str, trade_id: str, profit_abs: float, tier_mgr: TierManager) -> None:
        pst = self.get_pair_state(pair)
        was_risk = self.trade_risk_ledger.pop(trade_id, 0.0)
        self.pair_risk_open[pair] = max(0.0, self.pair_risk_open.get(pair, 0.0) - was_risk)
        if self.pair_risk_open.get(pair, 0.0) <= 1e-12: self.pair_risk_open[pair] = 0.0
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

    # ---- finalize ----
    def decrement_timers_and_decay(self) -> None:
        for pst in self.per_pair.values():
            if pst.cooldown_bars_left > 0: pst.cooldown_bars_left -= 1
            for m in pst.active_trades.values():
                if m.icu_bars_left is not None and m.icu_bars_left > 0:
                    m.icu_bars_left -= 1
        self.debt_pool *= self.cfg.pain_decay_per_bar
        self.tick_reservation_ttl()

    def plan_treasury_allocations(self, eq: float, tier_mgr: TierManager) -> None:
        cap_pct = self.get_dynamic_portfolio_cap_pct(eq)
        port_cap = cap_pct * eq
        used = self.get_total_risk(include_reserved=True)
        free = max(0.0, port_cap - used)

        needed_risk = min(self.debt_pool, free)
        if needed_risk <= 0:
            self.treasury.fast_alloc_risk, self.treasury.slow_alloc_risk = {}, {}
            return

        fast_budget = needed_risk * self.cfg.treasury_fast_split_pct
        slow_budget = needed_risk - fast_budget

        scored: List[Tuple[str,float,str]] = []
        for pair, pst in self.per_pair.items():
            if pst.cooldown_bars_left > 0: continue
            if len(pst.active_trades) > 0: continue
            if pst.last_score > 0 and pst.last_dir and pst.last_squad:
                pol = tier_mgr.get(pst.closs)
                if pst.last_squad in pol.allowed_kinds:
                    # 疼痛加权
                    pain = 1.0
                    if eq > 0:
                        pain += min(3.0, pst.local_loss / (0.02 * eq))
                    scored.append((pair, pst.last_score * pain, pst.last_squad))
        scored.sort(key=lambda x: x[1], reverse=True)

        # fast: 每个 squad 选分数最高的 pair
        best_by_squad: Dict[str, Tuple[str,float]] = {}
        for p, score, squad in scored:
            if (squad not in best_by_squad) or (score > best_by_squad[squad][1]):
                best_by_squad[squad] = (p, score)
        top_squads = sorted(best_by_squad.items(), key=lambda kv: kv[1][1], reverse=True)[:max(1, self.cfg.fast_topK_squads)]
        fast_pairs = [v[0] for (k,v) in top_squads]
        k = max(1, len(fast_pairs))
        fast_each = fast_budget / k

        # slow: 前 M 覆盖
        m = max(1, int(len(scored) * self.cfg.slow_universe_pct))
        slow_list = scored[:m]
        slow_each = slow_budget / m

        fast: Dict[str,float] = {}
        slow: Dict[str,float] = {}
        for p in fast_pairs: fast[p] = fast_each
        for p,_score,_sq in slow_list: slow[p] = slow_each

        # 单票 CAP 与最小注入（名义 → VaR 下限）
        for p in list(fast.keys()):
            pst = self.get_pair_state(p)
            pol = tier_mgr.get(pst.closs)
            min_risk = max(0.0, self.cfg.min_injection_nominal_fast * max(1e-9, pst.last_sl_pct))
            room = self.per_pair_cap_room(p, eq, pol)
            fast[p] = min(max(fast[p], min_risk), room)
            if fast[p] <= 0: fast.pop(p)
        for p in list(slow.keys()):
            pst = self.get_pair_state(p)
            pol = tier_mgr.get(pst.closs)
            min_risk = max(0.0, self.cfg.min_injection_nominal_slow * max(1e-9, pst.last_sl_pct))
            room = self.per_pair_cap_room(p, eq, pol)
            slow[p] = min(max(slow[p], min_risk), room)
            if slow[p] <= 0: slow.pop(p)

        self.treasury.fast_alloc_risk = fast
        self.treasury.slow_alloc_risk = slow

    def finalize_bar_cycle(self, eqprov: EquityProvider, tier_mgr: TierManager) -> None:
        self.bar_tick += 1
        self.decrement_timers_and_decay()
        self.plan_treasury_allocations(eqprov.get_equity(), tier_mgr)

        # (2) 盈利周期清债：周期到达时对比权益变化，盈利则清空债务与本地亏损
        try:
            if self.treasury.cycle_start_tick == 0:
                self.treasury.cycle_start_tick = self.bar_tick
                self.treasury.cycle_start_equity = eqprov.get_equity()
            if (self.bar_tick - self.treasury.cycle_start_tick) >= int(self.cfg.cycle_len_bars):
                pnl = eqprov.get_equity() - float(self.treasury.cycle_start_equity)
                if pnl >= 0 and bool(self.cfg.clear_debt_on_profitable_cycle):
                    self.debt_pool = 0.0
                    for pst in self.per_pair.values():
                        pst.local_loss = 0.0
                # 重新起一个周期
                self.treasury.cycle_start_tick = self.bar_tick
                self.treasury.cycle_start_equity = eqprov.get_equity()
        except Exception:
            pass

    def maybe_finalize_bar_cycle(self, pair: str, bar_ts: float,
                                 whitelist: List[str], timeframe_sec: int,
                                 persist_callback, eqprov: EquityProvider,
                                 tier_mgr: TierManager) -> None:
        now = time.time()
        if self.current_cycle_ts is None or bar_ts > float(self.current_cycle_ts):
            self.current_cycle_ts = float(bar_ts)
            self.reported_pairs_for_current_cycle = set()
        self.reported_pairs_for_current_cycle.add(pair)

        all_reported = all(p in self.reported_pairs_for_current_cycle for p in whitelist)
        newer_than_last = (self.last_finalized_bar_ts is None) or (float(self.current_cycle_ts) > float(self.last_finalized_bar_ts))
        timeout = (now - self.last_finalize_walltime) >= (self.cfg.force_finalize_mult * timeframe_sec)

        if newer_than_last and (all_reported or timeout):
            self.finalize_bar_cycle(eqprov, tier_mgr)
            self.last_finalize_walltime = now
            self.last_finalized_bar_ts = float(self.current_cycle_ts)
            # 不变式检查（日志告警）
            try:
                eq = eqprov.get_equity()
                cap = self.get_dynamic_portfolio_cap_pct(eq) * eq
                if self.get_total_risk(True) > cap + 1e-6:
                    print("[WARN] Portfolio risk exceeds cap after finalize.")
            except Exception:
                pass
            persist_callback()
            self.reported_pairs_for_current_cycle = set()


# -------------------------------
# 4) 状态存储
# -------------------------------
class StateStore:
    def __init__(self, filepath: str, g: GlobalState, eq: EquityProvider):
        self.file = filepath; self.g=g; self.eq=eq
    def save(self) -> None:
        try:
            snap = {"global_state": self.g.to_snapshot(), "equity_provider": self.eq.to_snapshot()}
            tmp = self.file + ".tmp"
            with open(tmp,"w") as f: json.dump(snap,f)
            os.replace(tmp, self.file)
        except Exception as e:
            print(f"[CRITICAL] save failed: {e}")
    def load_if_exists(self) -> None:
        if not os.path.isfile(self.file): return
        try:
            with open(self.file,"r") as f: snap=json.load(f)
            self.g.restore_snapshot(snap.get("global_state",{}))
            self.eq.restore_snapshot(snap.get("equity_provider",{}))
            self.g.reset_cycle_after_restore()
        except Exception as e:
            print(f"[WARN] restore failed: {e}")


# -------------------------------
# 5) Sizer
# -------------------------------
class PositionSizerV29:
    def __init__(self, g: GlobalState, eq: EquityProvider, cfg: V29Config, tier_mgr: TierManager):
        self.g=g; self.eq=eq; self.cfg=cfg; self.tier_mgr=tier_mgr

    def compute(self, pair: str, sl_pct: float, tp_pct: float, direction: str,
                min_stake: Optional[float], max_stake: float) -> Tuple[float,float,str]:
        equity = self.eq.get_equity()
        pst = self.g.get_pair_state(pair)
        pol = self.tier_mgr.get(pst.closs)

        # baseline VaR
        base_risk = pol.k_mult_base_pct * equity
        # stressed 抑制 baseline
        if self.cfg.suppress_baseline_when_stressed:
            if equity > 0 and (self.g.debt_pool / equity) > self.cfg.drawdown_threshold_pct:
                base_risk = 0.0

        # recovery 需求
        if pol.sizing_algo == "TARGET_RECOVERY" and tp_pct > 0:
            want_rec = pst.local_loss * pol.recovery_factor
            stake_rec = want_rec / tp_pct
            risk_rec = stake_rec * sl_pct
            risk_local_need = max(base_risk, risk_rec)
        else:
            risk_local_need = base_risk

        # 财政拨款
        fast = self.g.treasury.fast_alloc_risk.get(pair, 0.0)
        slow = self.g.treasury.slow_alloc_risk.get(pair, 0.0)
        alloc_total = fast + slow
        bucket = "fast" if fast >= slow else "slow"

        # 合并需求
        risk_wanted = max(risk_local_need, alloc_total)

        # CAP 约束
        port_cap_pct = self.g.get_dynamic_portfolio_cap_pct(equity)
        port_cap = port_cap_pct * equity
        port_room = max(0.0, port_cap - self.g.get_total_risk(include_reserved=True))
        pair_room = self.g.per_pair_cap_room(pair, equity, pol)
        risk_room = max(0.0, min(port_room, pair_room))

        risk_final = min(risk_wanted, risk_room)
        if risk_final <= 0 or sl_pct <= 0: return (0.0,0.0,bucket)

        stake_nominal = risk_final / sl_pct
        stake_cap_notional = pol.max_stake_notional_pct * equity
        stake_nominal = min(stake_nominal, stake_cap_notional, float(max_stake))
        if min_stake is not None:
            stake_nominal = max(stake_nominal, float(min_stake))
        if stake_nominal <= 0: return (0.0,0.0,bucket)

        real_risk = stake_nominal * sl_pct
        if real_risk > risk_room + 1e-12:
            return (0.0,0.0,bucket)

        return (stake_nominal, real_risk, bucket)


# -------------------------------
# 6) Exit/Risk
# -------------------------------
class ExitPolicyV29:
    def __init__(self, g: GlobalState, eq: EquityProvider, cfg: V29Config):
        self.g=g; self.eq=eq; self.cfg=cfg

    def decide(self, pair: str, trade_id: str, current_profit_pct: float) -> Optional[str]:
        pst = self.g.get_pair_state(pair)
        tmeta = pst.active_trades.get(trade_id)
        if not tmeta: return None

        if current_profit_pct is not None and tmeta.tp_pct > 0 and current_profit_pct >= tmeta.tp_pct:
            return "tp_hit"

        if (tmeta.icu_bars_left is not None) and (tmeta.icu_bars_left <= 0):
            return "icu_timeout"

        if pst.last_dir and pst.last_dir != tmeta.direction and pst.last_score > 0.01:
            return f"flip_{pst.last_dir}"

        eq = self.eq.get_equity()
        stress = (self.g.debt_pool / eq) if eq > 0 else 999.0
        if stress > self.cfg.drawdown_threshold_pct and (current_profit_pct is not None) and current_profit_pct < 0:
            return "risk_off"

        return None

    def early_lock_distance(self, trade, current_rate: float, current_profit: float,
                            atr_pct_hint: float) -> Optional[float]:
        # 返回 freqtrade 负距离
        try: open_rate = float(trade.open_rate)
        except Exception: return None
        is_long = bool(getattr(trade, "is_long", not getattr(trade,"is_short", False)))
        eps_pct = self.cfg.breakeven_lock_eps_atr_pct * max(1e-9, atr_pct_hint)
        if is_long:
            target = open_rate * (1.0 + eps_pct)
            dist = (current_rate - target) / max(current_rate,1e-12)
            return -max(0.0005, dist)
        else:
            target = open_rate * (1.0 - eps_pct)
            dist = (target - current_rate) / max(current_rate,1e-12)
            return -max(0.0005, dist)


# -------------------------------
# 7) 主策略
# -------------------------------
class TaxBrain(IStrategy):
    # 类属性先给默认值，构造后会用实例 cfg 覆盖（向下兼容不同 Freqtrade 版本）
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
        super().__init__(config)
        self.cfg = V29Config()
        # 覆盖
        for k,v in config.get("strategy_params", {}).items():
            if hasattr(self.cfg, k): setattr(self.cfg, k, v)

        # (3) 同步实例与类属性，避免调度器与实例心跳不一致
        self.timeframe = self.cfg.timeframe
        self.startup_candle_count = self.cfg.startup_candle_count
        try:
            self.__class__.timeframe = self.cfg.timeframe
            self.__class__.startup_candle_count = self.cfg.startup_candle_count
        except Exception:
            pass

        self.tier_mgr = TierManager()
        self.eqp = EquityProvider(float(config.get("dry_run_wallet", self.cfg.dry_run_wallet_fallback)))
        self.state = GlobalState(self.cfg)
        statedir = config.get("user_data_dir",".")
        self.store = StateStore(os.path.join(statedir,"taxbrain_v29_state.json"), self.state, self.eqp)

        self.sizer = PositionSizerV29(self.state, self.eqp, self.cfg, self.tier_mgr)
        self.exitp = ExitPolicyV29(self.state, self.eqp, self.cfg)

        self._last_signal: Dict[str, Optional[Candidate]] = {}
        self._pending_entry_meta: Dict[str, Dict[str, Any]] = {}
        self._tf_sec = self._tf_to_sec(self.cfg.timeframe)

    # ---- 工具 ----
    @staticmethod
    def _tf_to_sec(tf: str) -> int:
        if tf.endswith("m"): return int(tf[:-1]) * 60
        if tf.endswith("h"): return int(tf[:-1]) * 3600
        if tf.endswith("d"): return int(tf[:-1]) * 86400
        return 300

    def bot_start(self, **kwargs) -> None:
        self.store.load_if_exists()
        if self.state.treasury.cycle_start_tick == 0:
            self.state.treasury.cycle_start_tick = self.state.bar_tick
            self.state.treasury.cycle_start_equity = self.eqp.get_equity()

    # ---- 信号 ----
    def _candidates(self, row: pd.Series) -> List[Candidate]:
        out: List[Candidate] = []
        close = float(row["close"]); ema_f = float(row["ema_fast"]); ema_s=float(row["ema_slow"])
        rsi=float(row["rsi"]); adx=float(row["adx"]); atr_pct=float(row["atr_pct"])
        if any(math.isnan(v) for v in [close,ema_f,ema_s,rsi,adx,atr_pct]): return out

        # mean_rev_long
        if rsi < 25 and close < ema_f * 0.985:
            sl=atr_pct*1.2; tp=atr_pct*2.4
            raw=max(0.0,(25-rsi)/25.0); rr=tp/max(sl,1e-9)
            win=min(0.9, max(0.5, 0.52+0.4*raw))
            edge=win*tp-(1-win)*sl
            out.append(Candidate("long","mean_rev_long",sl,tp,raw,rr,win,edge,"MRL"))
        # pullback_long
        if ema_f > ema_s and adx > 20 and close < ema_f*0.99:
            sl=atr_pct*1.0; tp=atr_pct*2.0
            raw=0.5*max(0,(ema_f/max(ema_s,1e-9)-1.0))+0.5*max(0,(adx-20)/20)
            rr=tp/max(sl,1e-9); win=min(0.95,max(0.5,0.55+0.4*raw))
            edge=win*tp-(1-win)*sl
            out.append(Candidate("long","pullback_long",sl,tp,raw,rr,win,edge,"PBL"))
        # trend_short
        if ema_f < ema_s and adx > 25 and close > ema_f*1.01:
            sl=atr_pct*1.2; tp=atr_pct*2.4
            raw=0.5*max(0,(adx-25)/25)+0.5*max(0,(1.0-ema_f/max(ema_s,1e-9)))
            rr=tp/max(sl,1e-9); win=min(0.95,max(0.5,0.50+0.4*raw))
            edge=win*tp-(1-win)*sl
            out.append(Candidate("short","trend_short",sl,tp,raw,rr,win,edge,"TRS"))
        return out

    def _filter_rank(self, pol: TierPolicy, cands: List[Candidate]) -> Optional[Candidate]:
        ok: List[Candidate] = []
        for c in cands:
            if c.kind not in pol.allowed_kinds: continue
            if c.raw_score < pol.min_raw_score: continue
            if c.rr_ratio < pol.min_rr_ratio: continue
            if c.expected_edge < pol.min_edge: continue
            ok.append(c)
        if not ok: return None
        ok.sort(key=lambda x: (x.expected_edge, x.raw_score), reverse=True)
        return ok[0]

    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata["pair"]
        pst = self.state.get_pair_state(pair)

        df["ema_fast"] = ta.ema(df["close"], length=self.cfg.ema_fast)
        df["ema_slow"] = ta.ema(df["close"], length=self.cfg.ema_slow)
        df["rsi"] = ta.rsi(df["close"], length=self.cfg.rsi_len)
        atr = ta.atr(df["high"], df["low"], df["close"], length=self.cfg.atr_len)
        df["atr"] = atr
        df["atr_pct"] = df["atr"] / df["close"]
        # (1) ADX 动态列名
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=self.cfg.adx_len)
        col = f"ADX_{self.cfg.adx_len}"
        if isinstance(adx_df, pd.DataFrame) and col in adx_df.columns:
            df["adx"] = adx_df[col]
        else:
            df["adx"] = 20.0

        if len(df) == 0: return df
        last = df.iloc[-1]
        last_ts = float(pd.to_datetime(last.get("date", pd.Timestamp.utcnow())).timestamp())

        pol = self.tier_mgr.get(pst.closs)
        best = self._filter_rank(pol, self._candidates(last))
        if best:
            pst.last_dir = best.direction
            pst.last_score = best.expected_edge
            pst.last_squad = best.kind
            pst.last_sl_pct = best.sl_pct
            pst.last_tp_pct = best.tp_pct
            pst.last_atr_pct = float(last.get("atr_pct", 0.0))
        else:
            pst.last_dir=pst.last_squad=None
            pst.last_score=0.0; pst.last_sl_pct=pst.last_tp_pct=0.0
        self._last_signal[pair] = best

        try: wl = self.dp.current_whitelist()
        except Exception: wl = [pair]

        self.state.maybe_finalize_bar_cycle(
            pair=pair, bar_ts=last_ts, whitelist=wl, timeframe_sec=self._tf_sec,
            persist_callback=lambda: self.store.save(),
            eqprov=self.eqp, tier_mgr=self.tier_mgr
        )
        return df

    # ---- Entry ----
    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df["enter_long"]=0; df["enter_short"]=0
        pair = metadata["pair"]
        sig = self._last_signal.get(pair)
        if sig and len(df)>0:
            if sig.direction=="long": df.loc[df.index[-1],"enter_long"]=1
            else: df.loc[df.index[-1],"enter_short"]=1
        return df

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime,
                            entry_tag: Optional[str], side: str, **kwargs) -> bool:
        pst = self.state.get_pair_state(pair)
        if pst.cooldown_bars_left > 0: return False
        if len(pst.active_trades) > 0: return False
        sig = self._last_signal.get(pair)
        if not sig or sig.direction != side.lower(): return False
        self._pending_entry_meta[pair] = {
            "sl_pct": sig.sl_pct, "tp_pct": sig.tp_pct,
            "dir": sig.direction
        }
        return True

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        if abs(leverage - self.cfg.enforce_leverage) > 1e-6: return 0.0
        meta = self._pending_entry_meta.get(pair)
        if not meta: return 0.0
        if "stake_final" in meta: return float(meta["stake_final"])

        sl=float(meta["sl_pct"]); tp=float(meta["tp_pct"]); _dir=meta["dir"]

        stake, risk, bucket = self.sizer.compute(
            pair=pair, sl_pct=sl, tp_pct=tp, direction=_dir,
            min_stake=min_stake, max_stake=max_stake
        )
        if stake <= 0 or risk <= 0: return 0.0

        rid = f"{pair}:{bucket}:{uuid.uuid4().hex}"
        self.state.reserve_risk(pair, rid, risk, bucket)
        meta.update({
            "stake_final": stake, "risk_final": risk, "reservation_id": rid,
            "bucket": bucket, "entry_price": current_rate
        })
        return float(stake)

    def leverage(self, *args, **kwargs) -> float:
        return self.cfg.enforce_leverage

    # ---- Exit/SL ----
    def custom_stoploss(self, pair: str, trade, current_time: datetime,
                        current_rate: float, current_profit: float,
                        after_fill: bool, **kwargs) -> Optional[float]:
        pst = self.state.get_pair_state(pair)
        tid = str(getattr(trade,"trade_id", getattr(trade,"id","NA")))
        tmeta = pst.active_trades.get(tid)
        sl_pct = None; tp_pct_val=None; atr_pct_hint=0.0
        # 先从 trade 的自定义数据读取
        try:
            sl_pct = getattr(trade, "get_custom_data", lambda *_: None)("sl_pct")
            tp_pct_val = getattr(trade, "get_custom_data", lambda *_: None)("tp_pct")
        except Exception:
            pass
        # 再从 user_data 兜底
        try:
            if (sl_pct is None or sl_pct<=0) and hasattr(trade, "user_data"):
                sl_pct = float(trade.user_data.get("sl_pct", 0.0))
            if (tp_pct_val is None or tp_pct_val<=0) and hasattr(trade, "user_data"):
                tp_pct_val = float(trade.user_data.get("tp_pct", 0.0))
        except Exception:
            pass
        # 再从活动元信息兜底
        if (sl_pct is None or sl_pct<=0) and tmeta:
            sl_pct = tmeta.sl_pct
        if (tp_pct_val is None or tp_pct_val<=0) and tmeta:
            tp_pct_val = tmeta.tp_pct
        if tmeta:
            atr_pct_hint = pst.last_atr_pct

        if sl_pct is None or sl_pct<=0: return -0.03

        # 早锁盈（4）：达到阈值则抬到保本±ε(ATR%)
        if tp_pct_val and current_profit and current_profit > 0:
            try:
                if current_profit >= float(tp_pct_val) * float(self.cfg.breakeven_lock_frac_of_tp):
                    # 若有 data provider，可取更准确的 ATR%
                    try:
                        pairdf = self.dp.get_analyzed_dataframe(pair, self.timeframe)[0]
                        if len(pairdf)>0 and "atr" in pairdf.columns:
                            atr_pct_hint = float(pairdf["atr"].iloc[-1] / max(pairdf["close"].iloc[-1],1e-12))
                    except Exception:
                        pass
                    dist = self.exitp.early_lock_distance(trade, current_rate, current_profit, atr_pct_hint)
                    if dist is not None:
                        return max(-abs(float(sl_pct)), dist)
            except Exception:
                pass

        return -abs(float(sl_pct))

    def custom_exit(self, pair: str, trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        tid = str(getattr(trade,"trade_id", getattr(trade,"id","NA")))
        return self.exitp.decide(pair=pair, trade_id=tid, current_profit_pct=float(current_profit) if current_profit is not None else 0.0)

    # ---- 成交/撤单/拒单 ----
    def order_filled(self, pair: str, trade, order, current_time: datetime, **kwargs) -> None:
        pst = self.state.get_pair_state(pair)
        tid = str(getattr(trade,"trade_id", getattr(trade,"id","NA")))
        is_open = bool(getattr(trade, "is_open", True))

        profit_abs = 0.0
        if hasattr(trade, "close_profit_abs") and trade.close_profit_abs is not None:
            profit_abs = float(trade.close_profit_abs)
        elif hasattr(trade, "profit_abs") and trade.profit_abs is not None:
            profit_abs = float(trade.profit_abs)

        if is_open and tid not in pst.active_trades:
            meta = self._pending_entry_meta.get(pair)
            if not meta: return
            sl=meta["sl_pct"]; tp=meta["tp_pct"]; direction=meta["dir"]
            rid = meta.get("reservation_id"); bucket=meta.get("bucket","slow")
            risk = float(meta.get("risk_final", 0.0)); entry_price=float(meta.get("entry_price", 0.0))

            pol = self.tier_mgr.get(pst.closs)
            self.state.record_trade_open(pair, tid, risk, sl, tp, direction, bucket, entry_price, pol)
            if rid: self.state.release_reservation(rid)

            try:
                trade.set_custom_data("sl_pct", sl)
                trade.set_custom_data("tp_pct", tp)
            except Exception: pass
            try:
                if hasattr(trade,"user_data") and isinstance(trade.user_data, dict):
                    trade.user_data["sl_pct"]=sl; trade.user_data["tp_pct"]=tp
            except Exception: pass

            self._pending_entry_meta.pop(pair, None)
            self.store.save()
            return

        if (not is_open) and (tid in pst.active_trades):
            self.state.record_trade_close(pair, tid, profit_abs, self.tier_mgr)
            self.eqp.on_trade_closed_update(profit_abs)
            self.store.save()
            return

    def _cancel_or_reject_release(self, pair: str) -> None:
        meta = self._pending_entry_meta.get(pair)
        if meta and "reservation_id" in meta:
            # (5) 仅释放预约，不回灌财政，避免预算短暂膨胀；让下一拍统一重算
            self.state.release_reservation(meta["reservation_id"])
            self._pending_entry_meta.pop(pair, None)
            self.store.save()

    def order_cancelled(self, pair: str, trade, order, current_time: datetime, **kwargs) -> None:
        self._cancel_or_reject_release(pair)

    def order_rejected(self, pair: str, trade, order, current_time: datetime, **kwargs) -> None:
        self._cancel_or_reject_release(pair)
