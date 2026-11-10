"""§2 跨週期無洩漏對齊與定常化特徵工程.

The 5m bar is the master axis ``t``. Higher timeframes H in {15m, 30m} are only
usable once their bar has *closed*: the value joined at ``t`` is the latest H
bar whose ``CloseTime`` is <= ``t``. That is what ``merge_asof(...,
direction="backward")`` on close-time labels gives us; no future bar can leak.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

EPS = 1e-8

BASE_PERIOD_MINUTES = 5
HIGHER_PERIODS_MINUTES = (15, 30)

# §2.1 warm-start: max(N, 60) * (30m / 5m) = 360 base bars before the first
# live inference, so EMA60 / RSI14 / ATR14 / Bollinger20 / RVOL20 are all warm.
WARM_START_BARS = 360

# §2.3.4 OI is optional: exchange history caps (~30d) make it unusable for
# long windows, so the family is emitted only when this share of the slice is
# populated.
OI_MIN_COVERAGE = 0.80


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean() + EPS
    return 100.0 - (100.0 / (1.0 + gain / loss))


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()


class MultiTimeframeFeaturePipeline:
    """§2.2 + §2.3 stationary features on 5m/15m/30m, joined without lookahead."""

    def __init__(self, lag_n: int = 5, rvol_window: int = 20) -> None:
        self.lag_n = int(lag_n)
        self.rvol_window = int(rvol_window)

    # ------------------------------------------------------------------ #
    # single timeframe
    # ------------------------------------------------------------------ #
    def _compute_single_tf_features(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        feats = pd.DataFrame(index=df.index)
        o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

        # A. bar geometry (§2.2)
        feats[f"{prefix}_body"] = (c - o) / (o + EPS)
        bar_range = h - l + EPS
        feats[f"{prefix}_upper_shadow"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / bar_range
        feats[f"{prefix}_lower_shadow"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / bar_range
        feats[f"{prefix}_norm_range"] = bar_range / (c + EPS)

        for k in range(1, self.lag_n + 1):
            feats[f"{prefix}_log_ret_lag_{k}"] = np.log((c + EPS) / (c.shift(k) + EPS))

        feats[f"{prefix}_rvol"] = v / (v.rolling(self.rvol_window).mean() + EPS)

        # B. stationary indicators (§2.3)
        ema_20 = c.ewm(span=20, adjust=False).mean()
        ema_60 = c.ewm(span=60, adjust=False).mean()
        feats[f"{prefix}_ema_bias_20"] = (c - ema_20) / (ema_20 + EPS)
        feats[f"{prefix}_ema_bias_60"] = (c - ema_60) / (ema_60 + EPS)
        feats[f"{prefix}_rsi_scaled"] = (_rsi(c, 14) - 50.0) / 50.0
        feats[f"{prefix}_norm_atr"] = _atr(df, 14) / (c + EPS)

        mid = c.rolling(20).mean()
        std = c.rolling(20).std()
        upper, lower = mid + 2.0 * std, mid - 2.0 * std
        feats[f"{prefix}_bb_pct"] = (c - lower) / (upper - lower + EPS)
        feats[f"{prefix}_bb_bw"] = (upper - lower) / (mid + EPS)

        # C. contract microstructure (§2.3.4) — only when the column is present
        if "funding_rate" in df.columns:
            fr = df["funding_rate"].ffill()
            fr_mean = fr.rolling(30).mean()
            fr_std = fr.rolling(30).std() + EPS
            feats[f"{prefix}_funding_z"] = (fr - fr_mean) / fr_std
        if "open_interest" in df.columns:
            oi = df["open_interest"]
            # Binance OI history caps at ~30d, so a long training window is
            # mostly NaN. Emit the feature only when the column is actually
            # populated for most of this slice; otherwise omit it entirely so
            # the NaN does not erase every training row downstream.
            if float(oi.notna().mean()) >= OI_MIN_COVERAGE:
                oi_f = oi.ffill()
                feats[f"{prefix}_oi_delta"] = np.log((oi_f + EPS) / (oi_f.shift(1) + EPS))

        return feats

    # ------------------------------------------------------------------ #
    # multi timeframe
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        for col in ("funding_rate", "open_interest"):
            if col in df.columns:
                agg[col] = "last"
        out = df.resample(f"{minutes}min", closed="left", label="left").agg(agg)
        out = out.dropna(subset=["open", "high", "low", "close"])
        # Label each aggregated bar by its CLOSE time so merge_asof(backward)
        # can only ever pick a bar that has finished at or before `t`.
        out.index = out.index + pd.Timedelta(minutes=minutes)
        return out

    def build_multi_tf_features(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        """Returns a frame indexed by 5m bar open time, one row per usable bar."""
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df_5m.columns)
        if missing:
            raise ValueError(f"df_5m missing columns: {sorted(missing)}")
        df_5m = df_5m.sort_index()

        feats_5m = self._compute_single_tf_features(df_5m, prefix="tf5m")
        base = feats_5m.copy()
        base["close_time"] = base.index + pd.Timedelta(minutes=BASE_PERIOD_MINUTES)

        merged = base.sort_values("close_time")
        for minutes in HIGHER_PERIODS_MINUTES:
            prefix = f"tf{minutes}m"
            high = self._resample(df_5m, minutes)
            if high.empty:
                continue
            feats_high = self._compute_single_tf_features(high, prefix=prefix)
            merged = pd.merge_asof(
                merged,
                feats_high.sort_index(),
                left_on="close_time",
                right_index=True,
                direction="backward",
                allow_exact_matches=True,
            )

        merged = merged.set_index(merged.index).drop(columns=["close_time"])
        merged.index.name = df_5m.index.name
        return merged

    # ------------------------------------------------------------------ #
    # §4 Loop 3 pre-processing
    # ------------------------------------------------------------------ #
    @staticmethod
    def prune_collinear_features(
        X: pd.DataFrame, threshold: float = 0.80, keep: Sequence[str] = ()
    ) -> List[str]:
        """Drop columns correlated above ``threshold`` with an earlier column.

        Columns listed in ``keep`` are never dropped (e.g. ``primary_dir``).
        """
        keep = list(keep)
        numeric = X.drop(columns=[c for c in keep if c in X.columns], errors="ignore")
        numeric = numeric.select_dtypes(include=[np.number])
        if numeric.shape[1] <= 1:
            return list(X.columns)

        corr = numeric.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = {col for col in upper.columns if (upper[col] > threshold).any()}
        return [c for c in X.columns if c not in drop]

    @staticmethod
    def select_by_target_correlation(
        X: pd.DataFrame, y: pd.Series, top_k: int = 24
    ) -> List[str]:
        """§4 Loop 3 SHAP substitute: rank by |corr(feature, label)| and keep top_k.

        SHAP itself needs a fitted booster; this ranking is the deterministic,
        dependency-free proxy used to rebuild the feature subset each Loop 3.
        """
        numeric = X.select_dtypes(include=[np.number])
        if numeric.empty or len(y) == 0:
            return list(numeric.columns)
        yv = pd.Series(np.asarray(y, dtype=float), index=numeric.index)
        scores = {}
        for col in numeric.columns:
            s = numeric[col]
            if s.std(skipna=True) == 0 or s.isna().all():
                scores[col] = 0.0
                continue
            scores[col] = float(abs(s.corr(yv)))
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [c for c, _ in ranked[:top_k] if not np.isnan(scores[c])]


def prepare_feature_frame(df_5m: pd.DataFrame, lag_n: int = 5) -> pd.DataFrame:
    """Convenience: build + drop warm-up rows that still contain NaN."""
    feats = MultiTimeframeFeaturePipeline(lag_n=lag_n).build_multi_tf_features(df_5m)
    return feats.dropna()
