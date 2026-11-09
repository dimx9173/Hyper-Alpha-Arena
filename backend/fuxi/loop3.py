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
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .features import MultiTimeframeFeaturePipeline
from .labels import MetaLabelingEngine, build_model_frame
# §4.2 floor is canonical in rolling (which also owns the controller's
# revenge-guard semantics); this module re-exports rather than redefining it.
from .rolling import PASS_REGIME_MEDIAN_SR, RollingClosedLoopController
from .validation import ProfitValidationSuite

LAMBDA_PENALTY = 1.0
DEFAULT_NEIGHBOUR_RATE = 0.10  # ±perturbation for (x, y); M swings ±1 bar


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
        fee_per_trade: float = 0.0008,
        seed: int = 42,
        entry_threshold: float = 0.55,
        min_trades: int = 20,
    ) -> None:
        self.lambd = float(lambd)
        self.neighbour_rate = float(neighbour_rate)
        self.dsr_threshold = float(dsr_threshold)
        self.fee_per_trade = float(fee_per_trade)
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
    ) -> Dict[str, float]:
        """Fit a champion at θ on the train slice, score its OOS trade returns."""
        eng = MetaLabelingEngine(x=float(theta["x"]), y=float(theta["y"]), M=int(theta["M"]))
        meta = eng.apply_triple_barrier_and_weights(bars_5m, primary_signals)
        X, y, w, r = build_model_frame(features, meta)
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
            seed=self.seed,
        )
        probs = ctrl.champion.calibrated_proba(X.iloc[cut:])
        # Trade only what the model would actually take (p > entry threshold).
        mask = probs > self.entry_threshold
        rets = np.asarray(r[cut:], dtype=float)[mask] - self.fee_per_trade
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

        sr = float(np.mean(rets) / (np.std(rets, ddof=1) + 1e-12))
        dsr = ProfitValidationSuite.compute_deflated_sharpe_ratio(
            rets, num_trials=30, trials_sharpe_std=0.2
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
    ) -> PlateauResult:
        """J(θ) = μ_SR − λ·σ_SR over θ and its neighbourhood."""
        points = ([theta] if include_self else []) + _neighbourhood(theta, self.neighbour_rate)
        scored: List[Tuple[Dict[str, float], float]] = []
        for padj in points:
            m = self.evaluate_theta(padj, bars_5m, features, primary_signals)
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
    ) -> Tuple[PlateauResult, List[PlateauResult]]:
        """Score every candidate plateau; return best + all (ordered by score)."""
        results: List[PlateauResult] = []
        for i, theta in enumerate(candidates):
            res = self.score_plateau(theta, bars_5m, features, primary_signals)
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
        """§4.2: every plateau median SR below the floor → sleep, do not trade."""
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


def reselect_features(
    features: pd.DataFrame, meta: pd.DataFrame, top_k: int = 24, corr_threshold: float = 0.80
) -> List[str]:
    """Loop-3 feature rebuild: collinearity prune, then target-correlation rank."""
    idx = features.index.intersection(meta.index)
    X = features.loc[idx]
    y = meta.loc[idx, "meta_label"]
    kept = MultiTimeframeFeaturePipeline.prune_collinear_features(X, threshold=corr_threshold)
    return MultiTimeframeFeaturePipeline.select_by_target_correlation(X[kept], y, top_k=top_k)
