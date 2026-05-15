"""Expected value per $1 contract for a Survivor elimination market.

Same shape as the tennis bot's ``ev.py`` so the dashboard code path
is identical. Net of slippage applied to the ask side.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvResult:
    ev_per_contract: float    # in $ (one $1 contract = $1 face value)
    breakeven_prob: float
    edge: float                # model_prob - breakeven_prob


def ev(model_prob: float, market_prob: float, slippage_pct: float
        ) -> EvResult:
    """EV of buying YES at the current market price, in $.

    ``market_prob`` is the YES implied probability (= ask in $).
    ``model_prob`` is our forecast of YES settling at $1. With a
    slippage haircut applied to the ask we approximate the realised
    fill price as market_prob * (1 + slippage_pct).
    """
    fill = market_prob * (1.0 + slippage_pct)
    fill = max(0.01, min(0.99, fill))
    breakeven = fill
    return EvResult(
        ev_per_contract=(model_prob * (1.0 - fill) - (1.0 - model_prob) * fill),
        breakeven_prob=breakeven,
        edge=model_prob - breakeven,
    )
