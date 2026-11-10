"""Binance USDT-M 5m/15m/30m feed for the Fuxi rolling loop.

Why Binance, not Hyperliquid, for training data (verified 2026-11-09):

* HL ``candleSnapshot`` caps at 5000 bars and, worse, history depth is
  period-bound: 5m ~17d, 15m ~50d. Neither trains a Loop-2 window.
* Binance ``/fapi/v1/klines`` serves 5m/15m/30m back to 2019-09 in 1500-bar
  pages; ``/fapi/v1/fundingRate`` is equally deep (from 2022-01).
* Binance ``openInterestHist`` is capped at ~30 days, so OI is an **optional**
  column: present for recent training, absent (NaN-tolerant pipeline) for
  older windows. The feature pipeline simply skips ``oi_delta`` when missing.

Disk cache layout (parquet when pyarrow exists, else CSV)::
    fuxi_cache/<SYMBOL>_<interval>_klines.{parquet,csv}
    fuxi_cache/<SYMBOL>_funding.{parquet,csv}
    fuxi_cache/<SYMBOL>_oi.{parquet,csv}

All timestamps are UTC open-time milliseconds in the raw payload, converted
to a tz-aware UTC DatetimeIndex named ``open_time``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

FAPI = "https://fapi.binance.com"
KLINE_LIMIT = 1500
FUNDING_LIMIT = 1000
OI_LIMIT = 500
OI_MAX_LOOKBACK_DAYS = 30

INTERVAL_MS = {"5m": 300_000, "15m": 900_000, "30m": 1_800_000}
FUXI_INTERVALS = ("5m", "15m", "30m")

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"


def _get(url: str, params: dict, timeout: int = 20, retries: int = 4) -> requests.Response:
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (418, 429):
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
        except Exception as exc:  # network blip → backoff, then raise
            last = exc
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} tries: {last}")


class BinanceFuxiDataSource:
    """Fetch + cache Binance USDT-M klines, funding, and (recent) OI."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # cache helpers
    # ------------------------------------------------------------------ #
    def _cache_path(self, name: str) -> Path:
        try:
            import pyarrow  # noqa: F401
            return self.cache_dir / f"{name}.parquet"
        except ImportError:
            return self.cache_dir / f"{name}.csv"

    @staticmethod
    def _read(path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        else:
            # CSV round-trips carry mixed fractional-second formats; infer per element.
            df = pd.read_csv(path)
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True, format="mixed")
        return df.set_index("open_time").sort_index()

    @staticmethod
    def _write(path: Path, df: pd.DataFrame) -> None:
        out = df.copy()
        out.index.name = "open_time"
        out = out.reset_index()
        if path.suffix == ".parquet":
            out.to_parquet(path, index=False)
        else:
            out.to_csv(path, index=False)

    # ------------------------------------------------------------------ #
    # klines
    # ------------------------------------------------------------------ #
    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        """Raw paged fetch of ``symbol`` (e.g. ``BTCUSDT``) klines in [start, end)."""
        if interval not in INTERVAL_MS:
            raise ValueError(f"interval must be one of {sorted(INTERVAL_MS)}")
        frames: list[pd.DataFrame] = []
        cursor = start_ms
        step = INTERVAL_MS[interval]
        while cursor < end_ms:
            params = {"symbol": symbol, "interval": interval, "startTime": cursor,
                      "endTime": min(cursor + KLINE_LIMIT * step, end_ms), "limit": KLINE_LIMIT}
            rows = _get(f"{FAPI}/fapi/v1/klines", params).json()
            if not rows:
                break
            kl = pd.DataFrame(rows, columns=[
                "open_ms", "open", "high", "low", "close", "volume",
                "close_ms", "quote_vol", "n_trades", "taker_base",
                "taker_quote", "ignore",
            ])
            kl["open_time"] = pd.to_datetime(kl["open_ms"].astype("int64"), unit="ms", utc=True)
            for col in ("open", "high", "low", "close", "volume"):
                kl[col] = kl[col].astype(float)
            frames.append(kl[["open_time", "open", "high", "low", "close", "volume"]])
            cursor = int(rows[-1][0]) + step
            if len(rows) < KLINE_LIMIT:
                break
            time.sleep(0.15)  # weight-friendly paging

        if not frames:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).set_index(
                pd.DatetimeIndex([], tz="UTC", name="open_time"))
        df = pd.concat(frames).drop_duplicates("open_time").set_index("open_time").sort_index()
        return df

    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        refresh: bool = False,
    ) -> pd.DataFrame:
        name = f"{symbol}_{interval}_klines"
        path = self._cache_path(name)
        if not refresh:
            cached = self._read(path)
            if cached is not None and not cached.empty:
                lo = int(cached.index[0].timestamp() * 1000)
                hi = int(cached.index[-1].timestamp() * 1000)
                if lo <= start_ms and hi >= end_ms - INTERVAL_MS[interval]:
                    mask = (cached.index.astype("int64") // 10**6 >= start_ms) & \
                           (cached.index.astype("int64") // 10**6 < end_ms)
                    return cached.loc[mask]
        fresh = self.fetch_klines(symbol, interval, start_ms, end_ms)
        if not fresh.empty:
            old = self._read(path)
            merged = pd.concat([old, fresh]).sort_index() if old is not None else fresh
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            self._write(path, merged)
            mask = (merged.index.astype("int64") // 10**6 >= start_ms) & \
                   (merged.index.astype("int64") // 10**6 < end_ms)
            return merged.loc[mask]
        return fresh

    # ------------------------------------------------------------------ #
    # funding
    # ------------------------------------------------------------------ #
    def get_funding(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Funding events → forward-filled 5m series (NaN when nothing fetched)."""
        name = f"{symbol}_funding"
        path = self._cache_path(name)
        cached = None if path is None else self._read(path)
        need_fetch = True
        if cached is not None and not cached.empty:
            lo = int(cached.index[0].timestamp() * 1000)
            hi = int(cached.index[-1].timestamp() * 1000)
            need_fetch = not (lo <= start_ms and hi >= end_ms - 8 * 3_600_000)
        if need_fetch:
            rows: list[dict] = []
            cursor = start_ms
            while True:
                params = {"symbol": symbol, "startTime": cursor,
                          "endTime": end_ms, "limit": FUNDING_LIMIT}
                batch = _get(f"{FAPI}/fapi/v1/fundingRate", params).json()
                if not batch:
                    break
                rows.extend(batch)
                cursor = int(batch[-1]["fundingTime"]) + 1
                if len(batch) < FUNDING_LIMIT:
                    break
                time.sleep(0.15)
            if rows:
                fr = pd.DataFrame(rows)
                fr["open_time"] = pd.to_datetime(fr["fundingTime"].astype("int64"), unit="ms", utc=True)
                fr["funding_rate"] = fr["fundingRate"].astype(float)
                fr = fr[["open_time", "funding_rate"]].drop_duplicates("open_time").set_index("open_time").sort_index()
                merged = pd.concat([cached, fr]).sort_index() if cached is not None else fr
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                self._write(path, merged)
                cached = merged
        if cached is None or cached.empty:
            return pd.DataFrame(columns=["funding_rate"])
        mask = (cached.index.astype("int64") // 10**6 >= start_ms) & \
               (cached.index.astype("int64") // 10**6 < end_ms)
        return cached.loc[mask]

    # ------------------------------------------------------------------ #
    # open interest (best-effort, ~30d cap)
    # ------------------------------------------------------------------ #
    def get_open_interest(self, symbol: str, period: str = "5m") -> pd.DataFrame:
        """Most recent ~30d of OI; empty frame when the API refuses the range."""
        name = f"{symbol}_oi"
        path = self._cache_path(name)
        cached = self._read(path)
        try:
            rows = _get(f"{FAPI}/futures/data/openInterestHist",
                        {"symbol": symbol, "period": period, "limit": OI_LIMIT}).json()
        except RuntimeError:
            rows = []
        if rows and isinstance(rows, list):
            oi = pd.DataFrame(rows)
            oi["open_time"] = pd.to_datetime(oi["timestamp"].astype("int64"), unit="ms", utc=True)
            oi["open_interest"] = oi["sumOpenInterest"].astype(float)
            oi = oi[["open_time", "open_interest"]].drop_duplicates("open_time").set_index("open_time").sort_index()
            merged = pd.concat([cached, oi]).sort_index() if cached is not None else oi
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            self._write(path, merged)
            return merged
        return cached if cached is not None else pd.DataFrame(columns=["open_interest"])

    # ------------------------------------------------------------------ #
    # assembled 5m frame
    # ------------------------------------------------------------------ #
    def get_training_frame(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        with_oi: bool = False,
    ) -> pd.DataFrame:
        """5m OHLCV + funding_z source (+ OI when ``with_oi`` and available)."""
        bars = self.get_klines(symbol, "5m", start_ms, end_ms)
        if bars.empty:
            return bars
        fr = self.get_funding(symbol, start_ms, end_ms)
        if not fr.empty:
            idx_5m = bars.index.floor("5min")
            fr_5m = fr["funding_rate"].reindex(
                pd.DatetimeIndex(sorted(set(idx_5m) | set(fr.index))).sort_values()
            ).ffill().reindex(bars.index, method="ffill")
            bars = bars.copy()
            bars["funding_rate"] = fr_5m.to_numpy(dtype=float, na_value=float("nan"))
        if with_oi:
            oi = self.get_open_interest(symbol)
            if not oi.empty:
                bars = bars.copy()
                bars["open_interest"] = oi["open_interest"].reindex(bars.index, method="ffill")
        return bars
