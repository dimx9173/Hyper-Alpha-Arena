"""Fuxi closed-loop backtest runner (single symbol, 5m master axis).

Event model per §6.1: every closed 5m bar reconciles open contracts, then the
champion prices the bar. A contract exits on first touch of its sealed TP/SL
(intrabar sweep = loss, §3.1) or at ``open_bar + M`` on the bar close.
Entries need ``EV > δ_min`` *and* ``p_cal > entry_threshold``; sizing comes
from ``PerpetualRiskAndSmoothingEngine`` against the ratcheted base.

Costs: taker fee on entry, maker on TP exit / taker on SL exit
(``config.fees.FeeSchedule`` bps), plus optional funding accrual per 8h mark
while a contract is open.

This runner is deliberately engine-independent: it prices Fuxi's own
contracts bar-by-bar so orphan isolation and the ratchet are exact, instead
of shoehorning ``(x, y, M)`` contracts into ``ProgramBacktestEngine``'s
TP/SL-pending-order model.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from config.fees import FeeSchedule  # noqa: E402

from .labels import MetaLabelingEngine  # noqa: E402
from .ratchet import AsymmetricCompoundingRatchet  # noqa: E402
from .risk import MIN_EV_THRESHOLD, PerpetualRiskAndSmoothingEngine  # noqa: E402
from .rolling import RollingClosedLoopController  # noqa: E402

EPS = 1e-8
FUNDING_INTERVAL_MS = 8 * 3_600_000


@dataclass
class FuxiTrade:
    exit_time_ms: int
    direction: int
    entry_price: float
    exit_price: float
    margin: float
    leverage: float
    gross_ret: float
    net_ret: float
    net_pnl_u: float
    exit_reason: str  # "tp" | "sl" | "expiry"
    model_id: str
    entry_time_ms: int = 0


@dataclass
class FuxiBacktestConfig:
    initial_capital: float = 1000.0
    reinvest_ratio: float = 0.60
    entry_threshold: float = 0.55
    target_leverage: float = 3.0
    kelly_c: float = 0.25
    max_cap: float = 0.20
    discrete_buckets: Optional[List[float]] = None
    funding_enabled: bool = True
    settle_every_bars: int = 2016  # ~7d of 5m bars; 0 disables periodic settle


@dataclass
class FuxiBacktestResult:
    trades: List[FuxiTrade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    total_equity: float = 0.0
    active_pool: float = 0.0
    reserve_vault: float = 0.0
    n_trades: int = 0
    win_rate: float = 0.0
    total_net_pnl: float = 0.0
    max_drawdown: float = 0.0

    def trade_returns(self) -> np.ndarray:
        """Per-trade net return on the notional that was at risk."""
        return np.array(
            [t.net_pnl_u / max(t.margin * t.leverage, EPS) for t in self.trades],
            dtype=float,
        )

    def exit_timestamps_ms(self) -> np.ndarray:
        return np.array([t.exit_time_ms for t in self.trades], dtype=np.int64)


class FuxiBacktestRunner:
    """Bar-by-bar Loop-1 simulator for one symbol and one champion."""

    def __init__(
        self,
        controller: RollingClosedLoopController,
        ratchet: AsymmetricCompoundingRatchet,
        risk: Optional[PerpetualRiskAndSmoothingEngine] = None,
        config: Optional[FuxiBacktestConfig] = None,
        fees: Optional[FeeSchedule] = None,
    ) -> None:
        if controller.champion is None:
            raise ValueError("controller has no champion — train one first (Loop 2)")
        self.controller = controller
        self.ratchet = ratchet
        self.risk = risk or PerpetualRiskAndSmoothingEngine()
        self.config = config or FuxiBacktestConfig()
        self.fees = fees or FeeSchedule()

    # ------------------------------------------------------------------ #
    def run(
        self,
        bars_5m: pd.DataFrame,
        features: pd.DataFrame,
        primary_signals: Optional[pd.Series] = None,
    ) -> FuxiBacktestResult:
        cfg = self.config
        champ = self.controller.champion
        assert champ is not None
        x = float(champ.params["x"])
        y = float(champ.params["y"])
        M = int(champ.params["M"])
        sl_pct = x / y

        idx = features.index.intersection(bars_5m.index)
        bars = bars_5m.loc[idx].sort_index()
        feats = features.loc[idx].sort_index()
        if primary_signals is None:
            raise ValueError("primary_signals is required (primary_dir is label-side, never a feature)")
        dirs = primary_signals.reindex(idx).fillna(0).to_numpy(dtype=int)

        closes = bars["close"].to_numpy(dtype=float)
        highs = bars["high"].to_numpy(dtype=float)
        lows = bars["low"].to_numpy(dtype=float)
        times_ms = (bars.index.astype("int64") // 10**6).to_numpy(dtype=np.int64)
        has_funding = "funding_rate" in bars.columns
        funding = bars["funding_rate"].fillna(0.0).to_numpy(dtype=float) if has_funding else np.zeros(len(bars))

        maker = self.fees.maker_fee_bps / 10000.0
        taker = self.fees.taker_fee_bps / 10000.0
        slip = self.fees.slippage_bps / 10000.0

        X = feats[champ.feature_cols] if all(c in feats.columns for c in champ.feature_cols) else feats
        try:
            p_cal_all = champ.calibrated_proba(feats)
        except Exception:
            p_cal_all = np.full(len(feats), 0.5)
        atr_all = feats["tf5m_norm_atr"].fillna(0.004).to_numpy(dtype=float) if "tf5m_norm_atr" in feats.columns else np.full(len(feats), 0.004)

        month_start = self.ratchet.get_status().total_equity
        month_hwm = month_start
        months = pd.DatetimeIndex(bars.index).tz_convert(None).to_period("M")
        cur_month = months[0] if len(bars) else None

        trades: List[FuxiTrade] = []
        equity_curve = np.empty(len(bars))
        open_pos: Optional[dict] = None
        pos_seq = 0

        def _settle_flat() -> None:
            self.ratchet.settle_rolling_period(is_flat=(open_pos is None))

        for i in range(len(bars)):
            # monthly reset
            bar_month = months[i]
            if bar_month != cur_month:
                cur_month = bar_month
                status = self.ratchet.get_status()
                month_start = status.total_equity
                month_hwm = month_start
            else:
                month_hwm = max(month_hwm, self.ratchet.get_status().total_equity)

            # ---- reconcile open contract against this closed bar ---- #
            if open_pos is not None:
                d = open_pos["direction"]
                tp, sl = open_pos["tp"], open_pos["sl"]
                if d == 1:
                    hit_tp, hit_sl = (highs[i] >= tp), (lows[i] <= sl)
                else:
                    hit_tp, hit_sl = (lows[i] <= tp), (highs[i] >= sl)
                expired = (i - open_pos["open_idx"]) >= M
                exit_px, reason, gross = None, None, None
                if hit_tp and hit_sl:
                    exit_px, reason, gross = closes[i], "sl", -sl_pct  # sweep = loss
                elif hit_tp:
                    exit_px, reason, gross = tp, "tp", x
                elif hit_sl:
                    exit_px, reason, gross = sl, "sl", -sl_pct
                elif expired:
                    exit_px, reason = closes[i], "expiry"
                    gross = (closes[i] - open_pos["entry"]) / (open_pos["entry"] + EPS) * d

                if reason is not None:
                    notional = open_pos["margin"] * open_pos["leverage"]
                    # Full round trip charged exactly once, at close, so the
                    # identity Σ(trade.net_pnl_u) == Δequity always holds.
                    exit_cost = self.risk.exit_cost_fraction(reason == "tp")
                    cost = (open_pos["entry_cost"] + exit_cost) * notional
                    fund_cost = 0.0
                    if self.config.funding_enabled and has_funding:
                        n_marks = max(0, (times_ms[i] - open_pos["entry_ms"]) // FUNDING_INTERVAL_MS)
                        fund_cost = float(d * funding[i] * n_marks) * notional
                    net_pnl = gross * notional - cost - fund_cost
                    self.ratchet.update_intraday_pnl(net_pnl)
                    self.controller.close_position(open_pos["pos_id"])
                    trades.append(FuxiTrade(
                        exit_time_ms=int(times_ms[i]), direction=d,
                        entry_price=open_pos["entry"], exit_price=float(exit_px),
                        margin=open_pos["margin"], leverage=open_pos["leverage"],
                        gross_ret=float(gross),
                        net_ret=float(net_pnl / max(notional, EPS)),
                        net_pnl_u=float(net_pnl), exit_reason=reason,
                        model_id=open_pos["model_id"], entry_time_ms=open_pos["entry_ms"],
                    ))
                    open_pos = None
                    if cfg.settle_every_bars and (i % cfg.settle_every_bars == 0):
                        _settle_flat()

            # ---- champion prices this bar (flat only, one position max) ---- #
            if open_pos is None and dirs[i] != 0 and not self.controller.pass_regime:
                p_cal = float(p_cal_all[i])
                if p_cal > cfg.entry_threshold:
                    status = self.ratchet.get_status()
                    decision = self.risk.compute_safe_sizing(
                        prob_cal=p_cal, direction=int(dirs[i]), x=x, y=y,
                        target_leverage=cfg.target_leverage,
                        ratcheted_base_capital=status.kelly_base_capital,
                        total_equity=status.total_equity,
                        month_start_equity=month_start, month_hwm_equity=month_hwm,
                        rolling_penalty_scalar=self.controller.kelly_penalty_scalar,
                        current_norm_atr=float(atr_all[i]),
                        funding_rate=float(funding[i]) if has_funding else 0.0,
                    )
                    if decision.approved and decision.ev > MIN_EV_THRESHOLD:
                        pos_seq += 1
                        pos_id = f"FX{pos_seq:05d}"
                        entry_px = float(closes[i])
                        contract = self.controller.register_new_position(
                            pos_id, int(dirs[i]), entry_px,
                            decision.margin, decision.leverage, i)
                        open_pos = {
                            "pos_id": pos_id, "model_id": contract.model_id,
                            "direction": int(dirs[i]), "entry": entry_px,
                            "margin": decision.margin, "leverage": decision.leverage,
                            "tp": contract.tp_price, "sl": contract.sl_price,
                            "open_idx": i, "entry_ms": int(times_ms[i]),
                            # carried to the close so the entry leg is charged
                            # exactly once and Σ(net_pnl) == Δequity holds
                            "entry_cost": self.risk.entry_cost_fraction(),
                        }

            equity_curve[i] = self.ratchet.get_status().total_equity

        # force-close any leftover at the last close (expiry accounting)
        if open_pos is not None:
            d = open_pos["direction"]
            gross = (closes[-1] - open_pos["entry"]) / (open_pos["entry"] + EPS) * d
            notional = open_pos["margin"] * open_pos["leverage"]
            cost = (open_pos["entry_cost"] + self.risk.exit_cost_fraction(False)) * notional
            fund_cost = 0.0
            if self.config.funding_enabled and has_funding:
                n_marks = max(0, (times_ms[-1] - open_pos["entry_ms"]) // FUNDING_INTERVAL_MS)
                fund_cost = float(d * funding[-1] * n_marks) * notional
            net_pnl = gross * notional - cost - fund_cost
            self.ratchet.update_intraday_pnl(net_pnl)
            self.controller.close_position(open_pos["pos_id"])
            trades.append(FuxiTrade(
                exit_time_ms=int(times_ms[-1]), direction=d,
                entry_price=open_pos["entry"], exit_price=float(closes[-1]),
                margin=open_pos["margin"], leverage=open_pos["leverage"],
                gross_ret=float(gross), net_ret=float(net_pnl / max(notional, EPS)),
                net_pnl_u=float(net_pnl), exit_reason="expiry",
                model_id=open_pos["model_id"], entry_time_ms=open_pos["entry_ms"]))
            open_pos = None
            equity_curve[-1] = self.ratchet.get_status().total_equity

        _settle_flat()
        final = self.ratchet.get_status()
        curve = pd.Series(equity_curve, index=bars.index)
        peak = curve.cummax()
        dd = float(((peak - curve) / np.maximum(peak, EPS)).max()) if len(curve) else 0.0
        wins = sum(1 for t in trades if t.net_pnl_u > 0)
        return FuxiBacktestResult(
            trades=trades, equity_curve=curve,
            total_equity=final.total_equity, active_pool=final.active_pool,
            reserve_vault=final.reserve_vault, n_trades=len(trades),
            win_rate=(wins / len(trades)) if trades else 0.0,
            total_net_pnl=float(sum(t.net_pnl_u for t in trades)), max_drawdown=dd)
