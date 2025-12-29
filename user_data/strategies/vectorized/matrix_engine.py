from __future__ import annotations

from typing import Any, Dict, List, Optional
import sys
import math

import numpy as np
import pandas as pd

try:
    from freqtrade.enums import RunMode
except Exception:  # pragma: no cover
    RunMode = None

from ..agents.signals import builder, schemas, vectorized
from .pool_buffer import PoolBuffer, PoolSchema


class MatrixEngine:
    def __init__(self, strategy: Any, hub: Any, bridge: Any) -> None:
        self._strategy = strategy
        self._hub = hub
        self._bridge = bridge
        self._pool_schema = PoolSchema()

    def inject_features(self, df: pd.DataFrame, pair: str) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        runmode = getattr(getattr(self._strategy, "dp", None), "runmode", None)
        is_vector_pass = False
        module_runmode = None
        try:
            module_runmode = getattr(sys.modules.get(self._strategy.__class__.__module__, None), "RunMode", None)
        except Exception:
            module_runmode = None
        active_runmode = module_runmode or RunMode
        if active_runmode is not None:
            is_vector_pass = runmode in {
                active_runmode.BACKTEST,
                active_runmode.HYPEROPT,
                active_runmode.PLOT,
            }
        if not is_vector_pass:
            token = str(getattr(runmode, "value", runmode) or "").lower()
            is_vector_pass = any(key in token for key in ("backtest", "hyperopt", "plot"))
        system_cfg = getattr(self._strategy.cfg, "system", None)
        vectorized_bt = bool(getattr(system_cfg, "vectorized_entry_backtest", False)) if system_cfg else False
        merge_info = bool(getattr(system_cfg, "merge_informative_into_base", False)) if system_cfg else False
        informative_requires_merge = bool(self._strategy._informative_timeframes and not merge_info)
        use_vector_prefilter = bool(is_vector_pass and vectorized_bt and not informative_requires_merge)

        self._ensure_pool_columns(df)
        aligned_info = {} if use_vector_prefilter else self._bridge.align_informative(df, pair)

        pst_snapshot = self._strategy.state.get_pair_state(pair)
        payloads_long: list[list[dict]] = [[] for _ in range(len(df.index))]
        payloads_short: list[list[dict]] = [[] for _ in range(len(df.index))]

        if use_vector_prefilter:
            timeframes = (None, *getattr(self._strategy, "_informative_timeframes", ()))
            if self._strategy._derived_factor_columns_missing(df, timeframes):
                vectorized.add_derived_factor_columns(df, timeframes)
            df["LOSS_TIER_STATE"] = pst_snapshot.closs

            vec_specs = [
                spec for spec in self._strategy._enabled_signal_specs if vectorized.is_vectorizable(spec)
            ]
            fallback_specs = [
                spec for spec in self._strategy._enabled_signal_specs if spec not in vec_specs
            ]

            matrices = vectorized.build_signal_matrices(df, self._strategy.cfg, vec_specs)
            if is_vector_pass:
                iter_policies = getattr(self._strategy.tier_mgr, "policies", None)
                policies = list(iter_policies()) if callable(iter_policies) else []
                if not policies:
                    policies = [self._strategy.tier_mgr.get(0)]
                for mat in matrices:
                    allowed = None
                    for policy in policies:
                        if not policy.permits(
                            kind=mat["name"], squad=mat["squad"], recipe=mat["recipe"]
                        ):
                            continue
                        mask = (
                            (mat["raw_score"] >= policy.min_raw_score)
                            & (mat["rr_ratio"] >= policy.min_rr_ratio)
                            & (mat["expected_edge"] >= policy.min_edge)
                        )
                        allowed = mask if allowed is None else (allowed | mask)
                    if allowed is None:
                        mat["valid_mask"] = pd.Series(False, index=df.index)
                    else:
                        mat["valid_mask"] = mat["valid_mask"] & allowed

            payloads_long = self._select_topk_payloads(df, matrices, "long")
            payloads_short = self._select_topk_payloads(df, matrices, "short")
            idx_to_pos = {idx: pos for pos, idx in enumerate(df.index)}

            if fallback_specs:
                fallback_mask = vectorized.prefilter_signal_mask(
                    df, self._strategy.cfg, specs=fallback_specs
                )
                for idx in df.index[fallback_mask]:
                    row = df.loc[idx]
                    raw_candidates = builder.build_candidates(
                        row, self._strategy.cfg, specs=fallback_specs
                    )
                    if not raw_candidates:
                        continue
                    planned: list[schemas.Candidate] = []
                    for candidate in raw_candidates:
                        cand_with_plan = self._strategy._candidate_with_plan(
                            pair, candidate, row, None
                        )
                        if cand_with_plan:
                            planned.append(cand_with_plan)
                    if not planned:
                        continue
                    if is_vector_pass:
                        planned = [
                            cand
                            for cand in planned
                            if self._strategy._candidate_allowed_any_tier(cand)
                        ]
                        if not planned:
                            continue
                    grouped = self._strategy._trim_candidate_pool(
                        self._strategy._group_candidates_by_direction(planned)
                    )
                    pos = idx_to_pos.get(idx)
                    if pos is None:
                        continue
                    if grouped.get("long"):
                        extra = [self._candidate_to_payload(c) for c in grouped["long"]]
                        combined = payloads_long[pos] + extra
                        combined.sort(
                            key=lambda p: (p.get("expected_edge", 0.0), p.get("raw_score", 0.0)),
                            reverse=True,
                        )
                        payloads_long[pos] = combined[: self._strategy._candidate_pool_limit]
                    if grouped.get("short"):
                        extra = [self._candidate_to_payload(c) for c in grouped["short"]]
                        combined = payloads_short[pos] + extra
                        combined.sort(
                            key=lambda p: (p.get("expected_edge", 0.0), p.get("raw_score", 0.0)),
                            reverse=True,
                        )
                        payloads_short[pos] = combined[: self._strategy._candidate_pool_limit]

        elif is_vector_pass:
            for idx in df.index:
                df.at[idx, "LOSS_TIER_STATE"] = pst_snapshot.closs
                row = df.loc[idx]
                inf_rows = self._strategy._informative_rows_for_index(aligned_info, idx)
                raw_candidates = builder.build_candidates(
                    row,
                    self._strategy.cfg,
                    informative=inf_rows,
                    specs=self._strategy._enabled_signal_specs,
                )
                if not raw_candidates:
                    continue
                planned: list[schemas.Candidate] = []
                for candidate in raw_candidates:
                    cand_with_plan = self._strategy._candidate_with_plan(pair, candidate, row, inf_rows)
                    if cand_with_plan:
                        planned.append(cand_with_plan)
                if not planned:
                    continue
                planned = [
                    cand for cand in planned
                    if self._strategy._candidate_allowed_any_tier(cand)
                ]
                if not planned:
                    continue
                grouped = self._strategy._trim_candidate_pool(self._strategy._group_candidates_by_direction(planned))
                pos = df.index.get_loc(idx)
                if grouped.get("long"):
                    payloads_long[pos] = [self._candidate_to_payload(c) for c in grouped["long"]]
                if grouped.get("short"):
                    payloads_short[pos] = [self._candidate_to_payload(c) for c in grouped["short"]]
        else:
            last_idx = df.index[-1]
            df.at[last_idx, "LOSS_TIER_STATE"] = pst_snapshot.closs
            row = df.loc[last_idx]
            inf_rows = self._strategy._informative_rows_for_index(aligned_info, last_idx)
            raw_candidates = builder.build_candidates(
                row,
                self._strategy.cfg,
                informative=inf_rows,
                specs=self._strategy._enabled_signal_specs,
            )
            planned: list[schemas.Candidate] = []
            for candidate in raw_candidates:
                cand_with_plan = self._strategy._candidate_with_plan(pair, candidate, row, inf_rows)
                if cand_with_plan:
                    planned.append(cand_with_plan)
            policy = self._strategy.tier_mgr.get(pst_snapshot.closs)
            planned = [
                cand for cand in planned
                if self._strategy._candidate_allowed_by_policy(policy, cand)
            ]
            grouped = self._strategy._trim_candidate_pool(self._strategy._group_candidates_by_direction(planned))
            pos = df.index.get_loc(last_idx)
            if grouped.get("long"):
                payloads_long[pos] = [self._candidate_to_payload(c) for c in grouped["long"]]
            if grouped.get("short"):
                payloads_short[pos] = [self._candidate_to_payload(c) for c in grouped["short"]]

        pool = PoolBuffer.from_payloads(
            payloads_long,
            payloads_short,
            slots=self._strategy._candidate_pool_limit,
            signal_id_fn=self._payload_signal_id,
            schema=self._pool_schema,
        )
        self._apply_payloads(df, payloads_long, payloads_short)
        self._bridge.bind_df(pair, df)
        self._bridge.bind_pool_buffer(pair, pool)
        return df

    def _candidate_to_payload(self, candidate: schemas.Candidate) -> dict[str, Any]:
        return {
            "direction": candidate.direction,
            "kind": candidate.kind,
            "timeframe": getattr(candidate, "timeframe", None),
            "raw_score": candidate.raw_score,
            "rr_ratio": candidate.rr_ratio,
            "expected_edge": candidate.expected_edge,
            "sl_pct": candidate.sl_pct,
            "tp_pct": candidate.tp_pct,
            "plan_atr_pct": getattr(candidate, "plan_atr_pct", None),
        }

    def _select_topk_payloads(
        self, df: pd.DataFrame, matrices: list[dict], direction: str
    ) -> list[list[dict]]:
        if df is None or df.empty:
            return []
        specs = [m for m in matrices if m.get("direction") == direction]
        if not specs:
            return [[] for _ in range(len(df.index))]
        edges_list = []
        raw_list = []
        rr_list = []
        sl_list = []
        tp_list = []
        plan_atr_list = []
        for mat in specs:
            valid = mat["valid_mask"]
            edge = mat["expected_edge"].where(valid, -np.inf)
            raw = mat["raw_score"].where(valid, -np.inf)
            edges_list.append(edge.to_numpy(copy=False))
            raw_list.append(raw.to_numpy(copy=False))
            rr_list.append(mat["rr_ratio"].to_numpy(copy=False))
            sl_list.append(mat["sl_pct"].to_numpy(copy=False))
            tp_list.append(mat["tp_pct"].to_numpy(copy=False))
            plan_atr_list.append(mat["plan_atr_pct"].to_numpy(copy=False))
        edges = np.vstack(edges_list)
        raws = np.vstack(raw_list)
        rrs = np.vstack(rr_list)
        sls = np.vstack(sl_list)
        tps = np.vstack(tp_list)
        plan_atr = np.vstack(plan_atr_list)
        payloads: list[list[dict]] = [[] for _ in range(edges.shape[1])]
        k = self._strategy._candidate_pool_limit
        valid_any = np.isfinite(edges).any(axis=0)
        for col_idx in np.where(valid_any)[0]:
            row_edges = edges[:, col_idx]
            valid_idx = np.where(np.isfinite(row_edges))[0]
            if len(valid_idx) == 0:
                continue
            if len(valid_idx) > k:
                topk_idx = valid_idx[np.argpartition(row_edges[valid_idx], -k)[-k:]]
            else:
                topk_idx = valid_idx
            order = sorted(
                topk_idx,
                key=lambda i: (row_edges[i], raws[i, col_idx]),
                reverse=True,
            )
            for spec_idx in order:
                mat = specs[spec_idx]
                atr_val = float(plan_atr[spec_idx, col_idx])
                if not math.isfinite(atr_val):
                    atr_val = None
                payloads[col_idx].append(
                    {
                        "direction": direction,
                        "kind": mat["name"],
                        "timeframe": mat.get("timeframe"),
                        "raw_score": float(raws[spec_idx, col_idx]),
                        "rr_ratio": float(rrs[spec_idx, col_idx]),
                        "expected_edge": float(edges[spec_idx, col_idx]),
                        "sl_pct": float(sls[spec_idx, col_idx]),
                        "tp_pct": float(tps[spec_idx, col_idx]),
                        "plan_atr_pct": atr_val,
                    }
                )
        return payloads

    def _ensure_pool_columns(self, df: pd.DataFrame) -> None:
        for name in ("enter_long", "enter_short", "enter_tag", "_signal_id", "_signal_id_long", "_signal_id_short"):
            if name not in df.columns:
                df[name] = 0 if name in ("enter_long", "enter_short") else None

    def _payload_signal_id(self, payload: dict, direction: str) -> Optional[int]:
        name = payload.get("kind")
        if not name:
            return None
        if "timeframe" not in payload:
            return None
        timeframe = payload.get("timeframe")
        payload_dir = payload.get("direction")
        if not payload_dir:
            return None
        if payload_dir != direction:
            return None
        return self._hub.signal_id_for(name, timeframe, payload_dir)

    def _apply_payloads(
        self,
        df: pd.DataFrame,
        payloads_long: list[list[dict]],
        payloads_short: list[list[dict]],
    ) -> None:
        size = len(df.index)
        df["enter_long"] = 0
        df["enter_short"] = 0
        df["enter_tag"] = None

        top_long_id = np.full(size, np.nan)
        top_short_id = np.full(size, np.nan)
        top_long_edge = np.full(size, np.nan)
        top_short_edge = np.full(size, np.nan)

        for idx, pool in enumerate(payloads_long):
            if not pool:
                continue
            sig_id = self._payload_signal_id(pool[0], "long")
            if sig_id:
                top_long_id[idx] = float(sig_id)
                top_long_edge[idx] = float(pool[0].get("expected_edge", 0.0) or 0.0)
        for idx, pool in enumerate(payloads_short):
            if not pool:
                continue
            sig_id = self._payload_signal_id(pool[0], "short")
            if sig_id:
                top_short_id[idx] = float(sig_id)
                top_short_edge[idx] = float(pool[0].get("expected_edge", 0.0) or 0.0)

        long_mask = np.isfinite(top_long_id) & (top_long_id > 0)
        short_mask = np.isfinite(top_short_id) & (top_short_id > 0)
        df.loc[long_mask, "enter_long"] = 1
        df.loc[short_mask, "enter_short"] = 1

        df["_signal_id_long"] = top_long_id
        df["_signal_id_short"] = top_short_id

        selected = np.where(
            np.isnan(top_short_edge) | (top_long_edge >= top_short_edge),
            top_long_id,
            top_short_id,
        )
        df["_signal_id"] = selected
        df["enter_tag"] = pd.Series(
            [str(int(x)) if math.isfinite(x) and x > 0 else None for x in selected],
            index=df.index,
        )
