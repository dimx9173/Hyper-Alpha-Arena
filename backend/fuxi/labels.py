"""§3 兩階段元標籤架構與樣本唯一性權重.

Stage 1 (primary direction ``d_t``): a high-recall rule on the 5m close series.
The default below is trend+mom (EMA12/36 cross with a 5-bar momentum gate),
matching the spec blueprint; replace ``generate_primary_signals`` with any
cross-timeframe breakout rule as long as the output stays in {-1, 0, +1}.

Stage 2 (meta label ``m_t``): directional triple barrier — first touch of the
take-profit boundary wins, stop-loss or anything else loses. A same-bar double
touch is a loss (§3.1 m_t=0 for 同根K線雙向掃損).

Sample weights (§3.2): average uniqueness over the bar span times |return|,
normalized to mean 1. Overlapping (high-concurrency) spans get down-weighted
so clustered bets cannot dominate the LGBM loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-8


class MetaLabelingEngine:
    """Directional triple-barrier labeller with concurrency weights."""

    def __init__(self, x: float = 0.03, y: float = 2.0, M: int = 15) -> None:
        if x <= 0:
            raise ValueError("x (take-profit width) must be > 0")
        if y <= 0:
            raise ValueError("y (reward:risk) must be > 0")
        if M <= 0:
            raise ValueError("M (max hold bars) must be > 0")
        self.x = float(x)
        self.y = float(y)
        self.M = int(M)

    # ------------------------------------------------------------------ #
    # stage 1
    # ------------------------------------------------------------------ #
    def generate_primary_signals(
        self,
        df: pd.DataFrame,
        mom_threshold: float = 0.002,
        fast_span: int = 12,
        slow_span: int = 36,
        mom_lag: int = 5,
    ) -> pd.Series:
        close = df["close"]
        ema_fast = close.ewm(span=fast_span, adjust=False).mean()
        ema_slow = close.ewm(span=slow_span, adjust=False).mean()
        mom = close.pct_change(mom_lag)
        signals = pd.Series(0, index=df.index, dtype=int)
        signals[(ema_fast > ema_slow) & (mom > mom_threshold)] = 1
        signals[(ema_fast < ema_slow) & (mom < -mom_threshold)] = -1
        return signals

    # ------------------------------------------------------------------ #
    # stage 2
    # ------------------------------------------------------------------ #
    def apply_triple_barrier_and_weights(
        self, df: pd.DataFrame, primary_signals: pd.Series
    ) -> pd.DataFrame:
        closes = df["close"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        sigs = primary_signals.reindex(df.index).fillna(0).to_numpy(dtype=int)
        n = len(df)
        sl_pct = self.x / self.y

        meta_labels = np.full(n, np.nan)
        t_end_idx = np.full(n, -1, dtype=int)
        realized_ret = np.zeros(n)

        for i in range(n - self.M):
            d = sigs[i]
            if d == 0:
                continue
            entry_p = closes[i]
            tp_price = entry_p * (1.0 + d * self.x)
            sl_price = entry_p * (1.0 - d * sl_pct)

            outcome = 0
            end_pos = i + self.M
            ret = (closes[end_pos] - entry_p) / (entry_p + EPS) * d

            for offset in range(1, self.M + 1):
                cur = i + offset
                h, l = highs[cur], lows[cur]
                if d == 1:
                    hit_tp, hit_sl = (h >= tp_price), (l <= sl_price)
                else:
                    hit_tp, hit_sl = (l <= tp_price), (h >= sl_price)

                if hit_tp and hit_sl:
                    # same-bar sweep counts as a loss (§3.1)
                    outcome, end_pos, ret = 0, cur, -sl_pct
                    break
                if hit_tp:
                    outcome, end_pos, ret = 1, cur, self.x
                    break
                if hit_sl:
                    outcome, end_pos, ret = 0, cur, -sl_pct
                    break

            meta_labels[i] = outcome
            t_end_idx[i] = end_pos
            realized_ret[i] = ret

        concurrency = np.zeros(n)
        valid = np.where(~np.isnan(meta_labels))[0]
        for idx in valid:
            concurrency[idx : t_end_idx[idx] + 1] += 1.0

        uniqueness = np.full(n, np.nan)
        raw_weights = np.full(n, np.nan)
        for idx in valid:
            span = concurrency[idx : t_end_idx[idx] + 1]
            u = float(np.mean(1.0 / np.maximum(span, 1.0)))
            uniqueness[idx] = u
            raw_weights[idx] = u * (abs(realized_ret[idx]) + 1e-5)

        mean_w = float(np.nanmean(raw_weights)) if len(valid) else 1.0
        norm_weights = raw_weights / (mean_w if mean_w > 0 else 1.0)

        res = pd.DataFrame(
            {
                "primary_dir": sigs,
                "meta_label": meta_labels,
                "t_end_pos": t_end_idx,
                "realized_ret": realized_ret,
                "uniqueness": uniqueness,
                "sample_weight": norm_weights,
            },
            index=df.index,
        )
        return res.dropna(subset=["meta_label"])


def build_model_frame(
    features: pd.DataFrame,
    meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
    """Inner-join feature rows with labels; split out y / w / realized returns."""
    idx = features.index.intersection(meta.index)
    X = features.loc[idx].copy()
    X["primary_dir"] = meta.loc[idx, "primary_dir"]
    y = meta.loc[idx, "meta_label"].astype(int)
    w = meta.loc[idx, "sample_weight"].to_numpy(dtype=float)
    r = meta.loc[idx, "realized_ret"].to_numpy(dtype=float)
    return X, y, w, r
