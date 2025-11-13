# -*- coding: utf-8 -*-

"""źŲʹõͳһԼ



ģ鶨˺ѡźšԼźŹݽṹ

ָ㡢źעѡ֮䴫ݽṹϢ

ֶξ V29.1 汾ݣȷֺģԭЭͬ

"""



from __future__ import annotations



from dataclasses import dataclass

from typing import Any, Callable, Dict, List, Literal, Optional



Direction = Literal["long", "short"]





@dataclass(frozen=True)

class Candidate:

    """׼Ľֺѡ



    Attributes:

        direction: źŷ"long" ʾ࣬"short" ʾա

        kind: źƣעʱΨһʶغͷֲá

        raw_score: źԭʼ֣ԡ

        win_prob: ԤʤʣΧ 0~1

        expected_edge: 棬ʤӯȻõ

        squad: սСƣֲԻᰴֶιˡ

    """



    direction: Direction

    kind: str

    raw_score: float

    rr_ratio: float

    win_prob: float

    expected_edge: float

    squad: str

    exit_profile: Optional[str] = None

    recipe: Optional[str] = None





@dataclass(frozen=True)

class Condition:

    """źŴʽʾ



    Attributes:

        factor: Ƚϵƣ :class:FactorBank ṩ

        op: ȡ "<""<="">"">=""==""between""outside" ȡ

        value: Ƚֵ op Ϊʱ½硣

        value_hi: Ͻ磬 op Ϊ "between"  "outside" ʱʹá

        fn: ѡԶ庯ֵزֵʺϸ߼

    """



    factor: str

    op: Optional[str] = None

    value: Optional[float] = None

    value_hi: Optional[float] = None

    fn: Optional[Callable[[float], bool]] = None





@dataclass(frozen=True)

class SignalSpec:

    """עļ¼źŶ塣"""

    name: str

    direction: Direction

    squad: str

    conditions: List[Condition]

    raw_fn: Callable[[Dict[str, float], Any], float]

    win_prob_fn: Callable[[Dict[str, float], Any, float], float]

    sl_fn: Optional[Callable[[Dict[str, float], Any], float]] = None

    tp_fn: Optional[Callable[[Dict[str, float], Any], float]] = None

    min_rr: float = 0.0

    min_edge: float = 0.0

    required_factors: tuple[str, ...] = ()

    timeframe: Optional[str] = None

