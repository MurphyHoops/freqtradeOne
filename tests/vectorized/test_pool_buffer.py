from __future__ import annotations

from user_data.strategies.vectorized.pool_buffer import PoolBuffer


def test_pool_buffer_candidates_for_row():
    payloads_long = [
        [
            {
                "direction": "long",
                "kind": "mean_rev_long",
                "timeframe": None,
                "raw_score": 0.5,
                "rr_ratio": 2.0,
                "expected_edge": 0.6,
                "sl_pct": 0.01,
                "tp_pct": 0.02,
                "plan_atr_pct": None,
            }
        ],
        [],
    ]
    payloads_short = [[], []]

    def _signal_id(payload, direction):
        if payload.get("direction") != direction:
            return None
        return 7

    pool = PoolBuffer.from_payloads(payloads_long, payloads_short, slots=2, signal_id_fn=_signal_id)
    candidates = pool.candidates_for(0, "buy")

    assert len(candidates) == 1
    assert candidates[0]["signal_id"] == 7
    assert candidates[0]["raw_score"] == 0.5
    assert candidates[0]["rr_ratio"] == 2.0
