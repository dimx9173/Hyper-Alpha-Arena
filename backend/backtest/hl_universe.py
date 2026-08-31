import time
import logging
import requests
from typing import List

logger = logging.getLogger(__name__)

STABLES = {"USDC", "USDT", "USDS", "USDH", "FEUSD", "USDE", "USDY", "USDD", "DAI", "TUSD", "BUSD", "FRAX", "GUSD", "PYUSD", "USDP", "EURI", "EURC"}

_FALLBACK_DEFAULT = ["BTC", "ETH", "SOL"]

_cache_symbols: List[str] | None = None
_cache_ts: float = 0
_CACHE_TTL = 300


def get_hl_top30(limit: int = 30, exclude_stables: bool = True, sort_by: str = "dayNtlVlm", timeout: int = 10, fallback: List[str] = ["BTC", "ETH", "SOL"]) -> List[str]:
    fb = list(fallback) if fallback is not None else list(_FALLBACK_DEFAULT)
    try:
        resp = requests.post("https://api.hyperliquid.xyz/info", json={"type": "metaAndAssetCtxs"}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or len(data) < 2:
            raise ValueError("invalid metaAndAssetCtxs response")
        meta, ctxs = data[0], data[1]
        universe = meta.get("universe", []) if isinstance(meta, dict) else []
        if not isinstance(ctxs, list):
            raise ValueError("invalid ctxs")
        candidates: list[tuple[float, str]] = []
        for idx, asset in enumerate(universe):
            if idx >= len(ctxs):
                break
            name = str(asset.get("name", "")).upper() if isinstance(asset, dict) else ""
            if not name:
                continue
            if exclude_stables and name in STABLES:
                continue
            ctx = ctxs[idx] if isinstance(ctxs[idx], dict) else {}
            try:
                vol = float(ctx.get("dayNtlVlm", 0) or 0)
                oi = float(ctx.get("openInterest", 0) or 0)
            except Exception:
                vol, oi = 0, 0
            if vol == 0 and oi == 0:
                continue
            raw = ctx.get(sort_by, None)
            if raw is None:
                try:
                    mark = float(ctx.get("markPx", 0) or 0)
                    oi2 = float(ctx.get("openInterest", 0) or 0)
                    val = mark * oi2
                except Exception:
                    val = 0.0
            else:
                try:
                    val = float(raw)
                except Exception:
                    val = 0.0
            try:
                from services.exchanges.symbol_mapper import SymbolMapper  # type: ignore[import-not-found]
                internal = SymbolMapper.to_internal(name, "hyperliquid")
            except Exception:
                try:
                    from backend.services.exchanges.symbol_mapper import SymbolMapper as _SM2  # type: ignore[import-not-found]
                    internal = _SM2.to_internal(name, "hyperliquid")
                except Exception:
                    internal = name
            internal = str(internal).upper()
            candidates.append((val, internal))
        candidates.sort(key=lambda x: x[0], reverse=True)
        result = [s for _, s in candidates[:limit]]
        if not result:
            raise ValueError("empty top30 after filtering")
        return result
    except Exception as e:
        logger.warning(f"get_hl_top30 fallback to {fb}: {e}")
        return list(fb)


def get_hl_top30_cached(limit: int = 30, exclude_stables: bool = True, sort_by: str = "dayNtlVlm", timeout: int = 10, fallback: List[str] = ["BTC", "ETH", "SOL"]) -> List[str]:
    global _cache_symbols, _cache_ts
    now = time.time()
    if _cache_symbols is not None and (now - _cache_ts) < _CACHE_TTL:
        return list(_cache_symbols[:limit])
    result = get_hl_top30(limit=limit, exclude_stables=exclude_stables, sort_by=sort_by, timeout=timeout, fallback=fallback)
    _cache_symbols = list(result)
    _cache_ts = now
    return list(result)
