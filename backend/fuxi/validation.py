"""§7 機構級收益驗證四大關卡.

Gate 1 — Deflated Sharpe Ratio (Bailey & López de Prado): the estimated SR
must clear the expected maximum under the null of ``num_trials`` tries,
corrected for skew/kurtosis. ``dsr >= 0.95`` to promote.

Gate 2 — Monte Carlo trade reshuffling (B=10_000): 95th-percentile max
drawdown ≤ 20% and ruin (MDD ≥ 30%) probability < 0.5%.

Gate 3 — Friction stress: at 1.5× two-sided cost, annualized SR stays > 1.0
and net-profit retention ≥ 50%.

Gate 4 — Monthly consistency: winning-month share in [65%, 85%], max time
underwater ≤ 30 days, net retention after removing top-5 trades ≥ 50%.

``PromotionGate.evaluate`` ties all four to the §4.3 handoff rule.
"""

from __future__ import annotations



import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

MC_SIMULATIONS = 10_000
MC_MDD_LIMIT = 0.20
MC_RUIN_THRESHOLD = 0.30
MC_RUIN_LIMIT = 0.005

FRICTION_MULTIPLIER = 1.5
FRICTION_MIN_SR = 1.0
FRICTION_MIN_RETENTION = 0.50

MONTHLY_WIN_SHARE_LO = 0.65
MONTHLY_WIN_SHARE_HI = 0.85
MAX_TUW_DAYS = 30
TOP5_MIN_RETENTION = 0.50


    # ------------------------------------------------------------------ #
    # Gate 1: DSR


class ProfitValidationSuite:
    """The four institutional gates as pure functions of trade returns."""

    # ------------------------------------------------------------------ #
    # Gate 1: DSR
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_deflated_sharpe_ratio(
        returns: np.ndarray,
        num_trials: int = 100,
        trials_sharpe_std: float = 0.5,
    ) -> dict:
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
        T = len(r)
        if T < 10 or float(np.std(r, ddof=1)) == 0.0:
            return {"sr_est": 0.0, "sr_null": 0.0, "dsr": 0.0, "n": T}

        sr_est = float(np.mean(r) / np.std(r, ddof=1))
        sk = float(skew(r))
        kr = float(kurtosis(r, fisher=False))
        gamma = 0.5772156649
        K = max(2, int(num_trials))
        z1 = float(norm.ppf(1.0 - 1.0 / K))
        z2 = float(norm.ppf(1.0 - 1.0 / (K * np.e)))
        sr_null = float(trials_sharpe_std * ((1.0 - gamma) * z1 + gamma * z2))

        denom_var = 1.0 - sk * sr_est + ((kr - 1.0) / 4.0) * (sr_est**2)
        denom = float(np.sqrt(max(denom_var, 1e-8)))
        z = ((sr_est - sr_null) * np.sqrt(T - 1.0)) / denom
        return {"sr_est": sr_est, "sr_null": sr_null, "dsr": float(norm.cdf(z)), "n": T}

    # ------------------------------------------------------------------ #
    # Gate 2: Monte Carlo reshuffling
    # ------------------------------------------------------------------ #
    @staticmethod
    def run_monte_carlo_reshuffling(
        trade_returns: np.ndarray,
        n_simulations: int = MC_SIMULATIONS,
        ruin_threshold: float = MC_RUIN_THRESHOLD,
        seed: int = 42,
    ) -> dict:
        tr = np.asarray(trade_returns, dtype=float)
        tr = tr[np.isfinite(tr)]
        n = len(tr)
        if n == 0:
            return {"mdd_median": 0.0, "mdd_95th": 0.0, "risk_of_ruin": 0.0, "n": 0}

        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_simulations, n))
        sims = tr[idx]
        curves = np.cumprod(1.0 + sims, axis=1)
        peaks = np.maximum.accumulate(curves, axis=1)
        dds = (peaks - curves) / np.maximum(peaks, 1e-8)
        max_dds = np.max(dds, axis=1)
        return {
            "mdd_median": float(np.median(max_dds)),
            "mdd_95th": float(np.percentile(max_dds, 95)),
            "risk_of_ruin": float(np.mean(max_dds >= ruin_threshold)),
            "n": n,
        }

    # ------------------------------------------------------------------ #
    # Gate 3: friction stress
    # ------------------------------------------------------------------ #
    @staticmethod
    def friction_sensitivity(
        trade_returns: np.ndarray,
        cost_per_trade: float,
        multiplier: float = FRICTION_MULTIPLIER,
        trades_per_year: Optional[float] = None,
    ) -> dict:
        """Re-price every trade at ``multiplier``× cost; check SR and retention.

        The per-trade SR is scaled by √N with N = ``trades_per_year``. When N is
        unknown the sample itself is treated as ~1 trade-year, so SR values are
        comparable across same-sample comparisons but NOT true annualized SR.
        """
        tr = np.asarray(trade_returns, dtype=float)
        tr = tr[np.isfinite(tr)]
        base_total = float(np.sum(tr))
        if len(tr) == 0 or base_total <= 0:
            return {"passed": False, "sr": 0.0, "retention": 0.0, "stressed_total": 0.0}

        extra_cost = (multiplier - 1.0) * cost_per_trade
        stressed = tr - extra_cost
        stressed_total = float(np.sum(stressed))
        retention = stressed_total / base_total if base_total != 0 else 0.0

        per_trade = float(np.mean(stressed) / (np.std(stressed, ddof=1) + 1e-12))
        if trades_per_year is None:
            trades_per_year = float(len(tr))  # sample ≈ 1y; see docstring
        sr = per_trade * float(np.sqrt(max(trades_per_year, 1.0)))
        passed = bool(sr > FRICTION_MIN_SR and retention >= FRICTION_MIN_RETENTION)
        return {"passed": passed, "sr": float(sr), "retention": float(retention),
                "stressed_total": stressed_total}

    # ------------------------------------------------------------------ #
    # Gate 4: monthly consistency
    # ------------------------------------------------------------------ #
    @staticmethod
    def monthly_consistency(
        trade_returns: np.ndarray,
        trade_timestamps_ms: np.ndarray,
        max_tuw_days: int = MAX_TUW_DAYS,
    ) -> dict:
        tr = np.asarray(trade_returns, dtype=float)
        ts = np.asarray(trade_timestamps_ms, dtype=np.int64)
        if len(tr) == 0:
            return {"passed": False, "reason": "no trades"}

        order = np.argsort(ts)
        tr, ts = tr[order], ts[order]
        months = pd.to_datetime(ts, unit="ms", utc=True).tz_convert(None).to_period("M")
        monthly = pd.Series(tr).groupby(months).sum()
        n_months = len(monthly)
        win_share = float((monthly > 0).mean()) if n_months else 0.0

        equity = np.cumprod(1.0 + tr)
        peaks = np.maximum.accumulate(equity)
        underwater = equity < peaks
        # longest underwater stretch measured in calendar days
        tuw_days = 0.0
        if underwater.any():
            days = (ts - ts[0]) / 86_400_000.0
            best, start = 0.0, None
            for i, uw in enumerate(underwater):
                if uw and start is None:
                    start = days[i]
                if not uw and start is not None:
                    best = max(best, days[i] - start)
                    start = None
            if start is not None:
                best = max(best, days[-1] - start)
            tuw_days = float(best)

        total = float(np.sum(tr))
        top5 = float(np.sum(np.sort(tr)[-5:])) if len(tr) >= 5 else float(np.sum(tr[tr > 0]))
        retention = (total - top5) / total if total > 0 else 0.0

        checks = {
            "win_share_in_band": bool(MONTHLY_WIN_SHARE_LO <= win_share <= MONTHLY_WIN_SHARE_HI),
            "tuw_ok": bool(tuw_days <= max_tuw_days),
            "top5_retention_ok": bool(retention >= TOP5_MIN_RETENTION),
        }
        return {
            "passed": bool(all(checks.values())),
            "win_share": win_share,
            "n_months": int(n_months),
            "max_tuw_days": tuw_days,
            "top5_retention": float(retention),
            **checks,
        }


class PromotionGate:
    """§9 governance: all four gates must pass before a challenger promotes."""

    def __init__(
        self,
        dsr_min: float = 0.95,
        mdd95_max: float = MC_MDD_LIMIT,
        ruin_max: float = MC_RUIN_LIMIT,
        cost_per_trade: float = 0.0010,
        dsr_trials: int = 100,
        dsr_trials_std: float = 0.5,
    ) -> None:
        self.dsr_min = dsr_min
        self.mdd95_max = mdd95_max
        self.ruin_max = ruin_max
        self.cost_per_trade = cost_per_trade
        self.dsr_trials = dsr_trials
        self.dsr_trials_std = dsr_trials_std

    def evaluate(
        self, trade_returns: np.ndarray, trade_timestamps_ms: np.ndarray
    ) -> dict:
        dsr = ProfitValidationSuite.compute_deflated_sharpe_ratio(
            trade_returns, num_trials=self.dsr_trials, trials_sharpe_std=self.dsr_trials_std
        )
        mc = ProfitValidationSuite.run_monte_carlo_reshuffling(trade_returns)
        fr = ProfitValidationSuite.friction_sensitivity(trade_returns, self.cost_per_trade)
        mo = ProfitValidationSuite.monthly_consistency(trade_returns, trade_timestamps_ms)
        verdict = bool(
            dsr["dsr"] >= self.dsr_min
            and mc["mdd_95th"] <= self.mdd95_max
            and mc["risk_of_ruin"] < self.ruin_max
            and fr["passed"]
            and mo["passed"]
        )
        return {
            "passed": verdict,
            "dsr": dsr,
            "monte_carlo": mc,
            "friction": fr,
            "monthly": mo,
        }
