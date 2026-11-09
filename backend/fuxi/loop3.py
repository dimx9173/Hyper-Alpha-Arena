"""§6.2 + §4.1 Loop 3: parameter-plateau search over θ = (N, M, x, y).

The objective is the *plateau* score, not the peak score:

    μ_SR(θ) = mean SR over the neighbourhood N(θ)
    σ_SR(θ) = std  SR over the neighbourhood N(θ)
    J(θ)    = μ_SR(θ) − λ·σ_SR(θ)          (λ = 1.0)

Searching raw peak SR is how the legacy GA produced needle peaks (neighbourhood
retention ≈ −79%). This module deliberately scores the neighbourhood first, so
a lone spike cannot win.

§4.2 revenge-overfitting guard: if every candidate plateau has
``max_θ μ_SR(θ) < 0.5`` the market is in a noise regime and the caller must
switch to ``PASS_REGIME`` (flat, no trades) rather than trade a distorted
parameter set.

Loop-3 also reselects the feature subset: correlation pre-prune (threshold
0.80) followed by target-correlation ranking (SHAP substitute).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.fees import FeeSchedule

from .features import MultiTimeframeFeaturePipeline
from .labels import MetaLabelingEngine, build_model_frame
from .risk import round_trip_cost_fractions
# §4.2 floor is canonical in rolling (which also owns the controller's
# revenge-guard semantics); this module re-exports rather than redefining it.
from .rolling import PASS_REGIME_MEDIAN_SR, RollingClosedLoopController
from .validation import ProfitValidationSuite, infer_periods_per_year

LAMBDA_PENALTY = 1.0
DEFAULT_NEIGHBOUR_RATE = 0.10  # ±perturbation for (x, y); M swings ±1 bar

# 5m bars per year: 365d × 24h × 12 bars/h (review F4: annualized SR units).
BARS_PER_5M_YEAR = 365 * 288


def annualize_sharpe(
    sr_per_trade: float, n_trades: int, window_bars: int,
    bars_per_year: int = BARS_PER_5M_YEAR,
) -> float:
    """Scale a per-trade Sharpe to the annual units the PASS_REGIME floor uses.

    Generic helper (review F4): the bar-count form of per-year scaling. The
    live Loop-3 ``sr`` line itself reuses Gate-1's ``sr_unit * sqrt(ppy)``
    inline so plateau scoring and DSR share one annualization; this helper is
    the documented conversion for callers that only know trade counts.
    """
    if n_trades <= 0 or window_bars <= 0 or bars_per_year <= 0:
        return 0.0
    years = float(window_bars) / float(bars_per_year)
    return float(sr_per_trade) * float(np.sqrt(float(n_trades) / max(years, 1e-12)))


@dataclass
class PlateauResult:
    theta: Dict[str, float]
    mu_sr: float
    sigma_sr: float
    score: float
    neighbours: List[Tuple[Dict[str, float], float]] = field(default_factory=list)
    n_traded: int = 0

    def as_dict(self) -> dict:
        return {
            "theta": dict(self.theta),
            "mu_sr": self.mu_sr,
            "sigma_sr": self.sigma_sr,
            "score": self.score,
            "n_neighbours": len(self.neighbours),
            "n_traded": self.n_traded,
        }


def _neighbourhood(
    theta: Dict[str, float], rate: float = DEFAULT_NEIGHBOUR_RATE
) -> List[Dict[str, float]]:
    """1-D ± plus diagonal perturbations of (x, y) and ±1 bar swings on M."""
    x, y, m = float(theta["x"]), float(theta["y"]), int(theta["M"])
    out: List[Dict[str, float]] = []
    for fx, fy in (
        (1 - rate, 1.0), (1 + rate, 1.0),
        (1.0, 1 - rate), (1.0, 1 + rate),
        (1 - rate, 1 - rate), (1 + rate, 1 + rate),
    ):
        out.append({"x": x * fx, "y": y * fy, "M": m})
    for dm in (-1, 1):
        n = m + dm
        if n >= 2:
            out.append({"x": x, "y": y, "M": float(n)})
    return out


class ParameterPlateauSearch:
    """Loop 3: score (x, y, M) by neighbourhood mean SR minus dispersion."""

    def __init__(
        self,
        lambd: float = LAMBDA_PENALTY,
        neighbour_rate: float = DEFAULT_NEIGHBOUR_RATE,
        dsr_threshold: float = 0.95,
        dsr_trials: int = 100,
        dsr_trials_std: float = 0.5,
        fees: Optional[FeeSchedule] = None,
        seed: int = 42,
        entry_threshold: float = 0.55,
        min_trades: int = 20,
    ) -> None:
        self.lambd = float(lambd)
        self.neighbour_rate = float(neighbour_rate)
        self.dsr_threshold = float(dsr_threshold)
        self.dsr_trials = int(dsr_trials)
        self.dsr_trials_std = float(dsr_trials_std)
        self.fees: FeeSchedule = fees or FeeSchedule()
        self.seed = int(seed)
        self.entry_threshold = float(entry_threshold)
        self.min_trades = int(min_trades)

    # ------------------------------------------------------------------ #
    def evaluate_theta(
        self,
        theta: Dict[str, float],
        bars_5m: pd.DataFrame,
        features: pd.DataFrame,
        primary_signals: pd.Series,
        train_frac: float = 0.70,
        feature_cols: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Fit a champion at θ on the train slice, score its OOS trade returns."""
        if feature_cols is not None:
            # Review F5: Loop 3 freezes the kept subset; every θ is scored on
            # exactly those columns so no stray feature leaks back into search.
            missing = [c for c in feature_cols if c not in features.columns]
            if missing:
                raise ValueError(f"kept subset has {len(missing)} column(s) missing from features")
            features = features[feature_cols]
        eng = MetaLabelingEngine(x=float(theta["x"]), y=float(theta["y"]), M=int(theta["M"]))
        meta = eng.apply_triple_barrier_and_weights(bars_5m, primary_signals)
        X, y, w, r, expired = build_model_frame(features, meta)
        if len(X) < 200:
            return {"sr": 0.0, "ev": 0.0, "n": 0, "traded": False}

        cut = int(len(X) * train_frac)
        if cut < 100 or len(X) - cut < 50:
            return {"sr": 0.0, "ev": 0.0, "n": 0, "traded": False}

        ctrl = RollingClosedLoopController(dsr_threshold=self.dsr_threshold)
        ctrl.train_and_evaluate_challenger(
            X.iloc[:cut], y.iloc[:cut], w[:cut],
            X.iloc[cut:], y.iloc[cut:], r[cut:],
            candidate_params={"x": float(theta["x"]), "y": float(theta["y"]), "M": int(theta["M"])},
            fees=self.fees,
            seed=self.seed,
            val_expired=expired[cut:],
        )
        probs = ctrl.champion.calibrated_proba(X.iloc[cut:])
        # Trade only what the model would actually take (p > entry threshold).
        mask = probs > self.entry_threshold
        # Review F3/F-A: take-profit exits rest at the maker limit, every other
        # exit (stop or time) crosses the book — one fee source, both legs.
        cost_win, cost_loss = round_trip_cost_fractions(self.fees)
        gross = np.asarray(r[cut:], dtype=float)[mask]
        gross_expired = np.asarray(expired[cut:], dtype=bool)[mask]
        rets = gross - np.where(gross > 0.0 & ~gross_expired, cost_win, cost_loss)
        if len(rets) < self.min_trades:
            # A parameter set that cannot clear the net-EV gate has NO measured
            # edge. Scoring it as SR 0.0 would let "never trade" masquerade as
            # "neutral", so mark it untraded and exclude it from the plateau mean.
            return {
                "sr": 0.0,
                "ev": float(np.mean(rets)) if len(rets) else 0.0,
                "n": int(len(rets)),
                "traded": False,
            }

        # These are per-trade returns, so the annualization unit is trades/year.
        oos_index = X.iloc[cut:].index
        try:
            ppy = infer_periods_per_year(oos_index[mask])
        except ValueError:
            # Every selected exit shares one instant: scale the OOS bar rate by
            # the observed trades-per-bar ratio instead of guessing a constant.
            ppy = infer_periods_per_year(oos_index) * len(rets) / max(len(oos_index), 1)

        sr_unit = float(np.mean(rets) / (np.std(rets, ddof=1) + 1e-12))
        sr = float(sr_unit * np.sqrt(ppy))
        dsr = ProfitValidationSuite.compute_deflated_sharpe_ratio(
            rets,
            periods_per_year=ppy,
            num_trials=self.dsr_trials,
            trials_sharpe_std=self.dsr_trials_std,
        )
        return {
            "sr": sr,
            "ev": float(np.mean(rets)),
            "n": int(len(rets)),
            "dsr": float(dsr["dsr"]),
            "traded": True,
        }

    # ------------------------------------------------------------------ #
    def score_plateau(
        self,
        theta: Dict[str, float],
        bars_5m: pd.DataFrame,
        features: pd.DataFrame,
        primary_signals: pd.Series,
        include_self: bool = True,
        feature_cols: Optional[List[str]] = None,
    ) -> PlateauResult:
        """J(θ) = μ_SR − λ·σ_SR over θ and its neighbourhood."""
        points = ([theta] if include_self else []) + _neighbourhood(theta, self.neighbour_rate)
        scored: List[Tuple[Dict[str, float], float]] = []
        for padj in points:
            m = self.evaluate_theta(padj, bars_5m, features, primary_signals,
                                    feature_cols=feature_cols)
            scored.append((padj, float(m["sr"]), bool(m.get("traded", False))))
        measured = np.array([s for _, s, t in scored if t], dtype=float)
        n_traded = int(len(measured))
        if n_traded == 0:
            # Nothing in the neighbourhood clears the net-EV gate: no edge
            # measured, so the plateau sits well below the PASS_REGIME floor.
            return PlateauResult(theta=dict(theta), mu_sr=-1.0, sigma_sr=0.0,
                                 score=-1.0, neighbours=[(p, s) for p, s, _ in scored],
                                 n_traded=0)
        mu = float(np.mean(measured))
        sigma = float(np.std(measured, ddof=1)) if n_traded > 1 else 0.0
        return PlateauResult(
            theta=dict(theta),
            mu_sr=mu,
            sigma_sr=sigma,
            score=mu - self.lambd * sigma,
            neighbours=[(p, s) for p, s, _ in scored],
            n_traded=n_traded,
        )

    # ------------------------------------------------------------------ #
    def search(
        self,
        candidates: Sequence[Dict[str, float]],
        bars_5m: pd.DataFrame,
        features: pd.DataFrame,
        primary_signals: pd.Series,
        progress: bool = False,
        feature_cols: Optional[List[str]] = None,
    ) -> Tuple[PlateauResult, List[PlateauResult]]:
        """Score every candidate plateau; return best + all (ordered by score)."""
        results: List[PlateauResult] = []
        for i, theta in enumerate(candidates):
            res = self.score_plateau(theta, bars_5m, features, primary_signals,
                                     feature_cols=feature_cols)
            results.append(res)
            if progress:
                print(
                    f"[Loop3 {i + 1}/{len(candidates)}] θ={theta} "
                    f"μ={res.mu_sr:+.3f} σ={res.sigma_sr:.3f} J={res.score:+.3f}",
                    flush=True,
                )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[0], results

    # ------------------------------------------------------------------ #
    @staticmethod
    def is_pass_regime(results: Sequence[PlateauResult], floor: float = PASS_REGIME_MEDIAN_SR) -> bool:
        """§4.2: every plateau median SR (annualized) below the floor → sleep."""
        if not results:
            return True
        return max(r.mu_sr for r in results) < floor


def default_parameter_grid() -> List[Dict[str, float]]:
    """Compact (x, y, M) grid for a 5m axis.

    ``x`` spans 0.4%–3% (below ~0.4% the 0.08% round trip dominates), ``y``
    spans 1.0–4.0 (higher y lowers the break-even win rate), ``M`` spans
    12–72 bars (1h–6h of holding).
    """
    out: List[Dict[str, float]] = []
    for x in (0.004, 0.006, 0.010, 0.015, 0.020, 0.030):
        for y in (1.0, 1.5, 2.0, 3.0, 4.0):
            for m in (12, 24, 48, 72):
                out.append({"x": float(x), "y": float(y), "M": float(m)})
    return out


def stratified_grid_sample(
    grid: Sequence[Dict[str, float]], n: int, seed: int = 7
) -> List[Dict[str, float]]:
    """Stratified sample of the parameter grid, one tier per ``x`` (review F6).

    The old ``grid[:8]`` slice took all 8 samples from ``x = 0.4%`` — the
    width where the ~0.08% round trip dominates — so Loop 3 under-reported
    realizable edge and over-triggered PASS_REGIME. Strata are the distinct
    ``x`` values; each receives ``n // n_strata`` picks (first strata take the
    remainder) with a deterministic rotation by ``seed`` so the sample is
    reproducible without a caller-supplied RNG.
    """
    items = list(grid)
    if n <= 0:
        raise ValueError(f"loop3 sample size must be positive, got {n}")
    if n >= len(items):
        return list(items)
    tiers: Dict[float, List[Dict[str, float]]] = {}
    for t in items:
        tiers.setdefault(float(t["x"]), []).append(t)
    keys = sorted(tiers)
    base, rem = divmod(n, len(keys))
    out: List[Dict[str, float]] = []
    for i, k in enumerate(keys):
        tier = tiers[k]
        take = base + (1 if i < rem else 0)
        start = (seed + i) % len(tier)
        for j in range(take):
            out.append(tier[(start + j) % len(tier)])
    return out


def reselect_features(
    features: pd.DataFrame, meta: pd.DataFrame, top_k: int = 24, corr_threshold: float = 0.80
) -> List[str]:
    """Loop-3 feature rebuild: collinearity prune, then target-correlation rank."""
    idx = features.index.intersection(meta.index)
    X = features.loc[idx]
    y = meta.loc[idx, "meta_label"]
    kept = MultiTimeframeFeaturePipeline.prune_collinear_features(X, threshold=corr_threshold)
    return MultiTimeframeFeaturePipeline.select_by_target_correlation(X[kept], y, top_k=top_k)
