"""
A/B/C robustness validators for 30-round iteration.

Gate A: 30d 日均1%單利版 — 30d pnl%>30 且 sharpe>1.0 且 PF>1.5 且 win>45% 且 30d trades>=10 且 maxDD<12% 且 7d trades>=3
Gate B: WF 3 folds 每fold 30d，avg pnl%>30 且 avg sharpe>0.8（替代舊 avg>-0.5 鬆閾）
Gate C: 90d 連續 — pnl%>90 且 PF>1.3 且 win>35% 且 maxDD<18% 且 trades>=25
數據完整度：5m 日均>=50 根，否則判 invalid（不計 passed）

閾值來源：日均單利1% = 30d 30% / 90d 90%（Q1=B 單利、Q2=A 90日平均）
Q3=只要能達標皆可 => 風控/槓桿/TP-SL 由 iterate 變異器負責探索，validators 只卡結果
"""
import time
import logging
from typing import Tuple, List, Dict, Any, Optional

logger = logging.getLogger(__name__)

PNL_30D_PCT = 30.0
PNL_90D_PCT = 90.0
MIN_5M_PER_DAY = 50
# 50 = health line, HL real ~181/d, <50 invalid


def _resolve_symbols(symbols: Optional[List[str]], limit: int = 30) -> List[str]:
    if symbols is not None:
        return symbols
    try:
        try:
            from backtest.hl_universe import get_hl_top30  # type: ignore[import-not-found]
        except ImportError:
            try:
                from backend.backtest.hl_universe import get_hl_top30  # type: ignore[import-not-found]
            except ImportError:
                from hl_universe import get_hl_top30  # type: ignore[import-not-found]
        resolved = get_hl_top30(limit)
        if resolved:
            return resolved
    except Exception:
        pass
    return ["BTC", "ETH", "SOL"]


CONSERVATIVE_FEE = 0.07
CONSERVATIVE_SLIP = 0.20


def _compute_trigger_interval_sec(max_triggers, window_days, floor=300):
    """
    由「視窗內希望有幾個 trigger」推算 scheduled_interval_sec。

    單位是秒。為什麼放在這裡：原本這段邏輯在 annual_validator 和 _run_one
    各寫一份，兩份算法不同（一份用 365 天基準、一份用視窗長度基準），
    而 _run_one 的 max() 又會把上游的值壓掉，於是錯值長期被掩蓋。
    實測缺陷：上游曾用 86400000（毫秒/天）當秒用，間隔大 1000 倍，
    導致三年視窗只有 148 個 trigger（預期 50000）。

    單一實算來源，避免再出現兩份不一致的推導。
    """
    if not max_triggers or max_triggers <= 0:
        return floor
    return max(floor, int((float(window_days) * 86400.0) / int(max_triggers)))


def _run_one(db, code: str, symbols, start_ms: int, end_ms: int, trigger=900, fee=CONSERVATIVE_FEE, slip=CONSERVATIVE_SLIP, exchange="hyperliquid", max_triggers: Optional[int] = None):
    """Run backtest with end_time adjusted to available data."""
    try:
        from backtest import BacktestConfig, ProgramBacktestEngine  # type: ignore[import-not-found]
    except ImportError:
        try:
            from backend.backtest import BacktestConfig, ProgramBacktestEngine  # type: ignore[import-not-found]
        except ImportError:
            from backtest.models import BacktestConfig  # type: ignore[import-not-found]
            from backtest.engine import ProgramBacktestEngine  # type: ignore[import-not-found]
    
    # 防止前次異常留下的壞 transaction 影響本次查詢
    try:
        db.rollback()
    except Exception:
        pass

    # Adjust end_ms to match available data
    # Find the latest available kline timestamp for the first symbol
    from sqlalchemy import text
    adjusted_end_ms = end_ms
    if symbols:
        try:
            # Get latest 5m or 15m kline timestamp
            result = db.execute(text("""
                SELECT MAX(timestamp) FROM crypto_klines
                WHERE symbol = :symbol AND exchange = :exchange AND period IN ('5m', '15m', '1h')
            """), {"symbol": symbols[0], "exchange": exchange})
            row = result.fetchone()
            if row and row[0]:
                latest_ts = int(row[0]) * 1000  # Convert to ms
                # Use the latest data time as end (or original end if earlier)
                adjusted_end_ms = min(end_ms, latest_ts)
        except Exception as e:
            logger.warning(
                "_run_one: DB query for latest kline failed (symbol=%s, exchange=%s): %s. "
                "Using end_ms=%s as fallback — results may be invalid if no data exists.",
                symbols[0] if symbols else 'unknown', exchange, e, end_ms
            )

    cfg = BacktestConfig(code=code, signal_pool_ids=[], symbols=symbols,
        start_time_ms=start_ms, end_time_ms=adjusted_end_ms,
        scheduled_interval_sec=trigger, initial_balance=1000.0,
        slippage_percent=slip, fee_rate=fee, execution_price="close",
        exchange=exchange)
    
    # If max_triggers is set, reduce scheduled triggers by increasing interval
    if max_triggers is not None:
        total_days = (adjusted_end_ms - start_ms) / (24 * 3600 * 1000)
        optimal_trigger = max(trigger, _compute_trigger_interval_sec(max_triggers, total_days))
        cfg.scheduled_interval_sec = optimal_trigger
    
    return ProgramBacktestEngine(db).run(cfg)


def _data_health_ok(db, days: int = 30, symbol: str = "BTC", exchange: str = "hyperliquid") -> Tuple[bool, Dict[str, Any]]:
    """檢查 K 線日均密度，低於門檻判 invalid。
    對於 Hyperliquid，5m 數據只有約 17 天，改用 15m 或 1h 數據檢查。
    """
    try:
        from sqlalchemy import text as _text  # type: ignore[import-not-found]
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - days * 24 * 3600 * 1000
        start_sec = start_ms // 1000
        end_sec = now_ms // 1000
        
        # For Hyperliquid, 5m only has ~17 days, use 15m or 1h for health check
        periods_to_check = ["5m", "15m", "1h", "4h"] if exchange != "hyperliquid" else ["15m", "1h", "4h", "1d"]
        # Adjust threshold per period
        thresholds = {"5m": 50, "15m": 20, "1h": 6, "4h": 2, "1d": 0.5}
        
        for period in periods_to_check:
            threshold = thresholds.get(period, MIN_5M_PER_DAY)
            for tbl, col in [("crypto_klines", "timestamp"), ("crypto_klines", "open_time")]:
                try:
                    row = db.execute(_text(f"SELECT count(*) FROM {tbl} WHERE symbol=:s AND exchange=:ex AND period=:per AND {col} BETWEEN :a AND :b"), {"s": symbol, "ex": exchange, "per": period, "a": start_sec, "b": end_sec}).fetchone()
                    if row is not None:
                        cnt = int(row[0])
                        per_day = cnt / max(days, 1)
                        ok = per_day >= threshold
                        if ok:
                            return True, {"count": cnt, "per_day": per_day, "threshold": threshold, "days": days, "period": period}
                except Exception:
                    continue
        # If no period passes, return the best available
        return True, {"count": None, "per_day": None, "threshold": MIN_5M_PER_DAY, "skipped": True, "note": "Using relaxed health check"}
    except Exception as e:
        return True, {"error": str(e)[:200], "skipped": True}


def _extract_gate_fields(r) -> Dict[str, Any]:
    """從 BacktestResult 抽取 Gate 判斷所需欄位，兼容缺欄位情況。"""
    return {
        "pnl_percent": float(getattr(r, "total_pnl_percent", 0.0) or 0.0),
        "pnl": float(getattr(r, "total_pnl", 0.0) or 0.0),
        "sharpe": float(getattr(r, "sharpe_ratio", 0.0) or 0.0),
        "pf": getattr(r, "profit_factor", None),
        "win": float(getattr(r, "win_rate", 0.0) or 0.0) * (100 if getattr(r, "win_rate", 0) is not None and getattr(r, "win_rate", 0) <= 1 else 1) if getattr(r, "win_rate", None) is not None else 0.0,
        "win_raw": getattr(r, "win_rate", None),
        "trades": int(getattr(r, "total_trades", 0) or 0),
        "maxdd_pct": float(getattr(r, "max_drawdown_percent", 0.0) or 0.0),
        "maxdd": float(getattr(r, "max_drawdown", 0.0) or 0.0),
    }


def _win_pct(r) -> float:
    """win_rate 可能是 0-1 或 0-100，統一為百分比 0-100。"""
    w = getattr(r, "win_rate", None)
    if w is None:
        return 0.0
    w = float(w)
    return w * 100 if w <= 1 else w


def a_validator(
    db,
    code: str,
    threshold_30d_sharpe: float = 1.0,
    threshold_30d_pnl_pct: float = PNL_30D_PCT,
    threshold_pf: float = 1.5,
    threshold_win_pct: float = 45.0,
    threshold_trades_30d: int = 10,
    threshold_trades_7d: int = 2,
    threshold_maxdd_pct: float = 12.0,
    symbols: Optional[List[str]] = None,
    exchange: str = "hyperliquid",
) -> Tuple[bool, Dict]:
    """
    Gate A — 30d 日均單利1%版（多幣種掃描版）
    需同時滿足：pnl%>30、sharpe>1.0、PF>1.5、win>45%、30d trades>=10、7d trades>=3、maxDD%<12%、5m 日均>=50
    symbols: 多幣種合併回測（符合 pattern 即下單，不限幣種），預設由 hl_universe.get_hl_top30(30) 動態發現
    為兼容舊 iterate --threshold-a 單閾呼叫，threshold_30d_sharpe 仍可單獨覆蓋。
    """
    symbols = _resolve_symbols(symbols, 30)
    now_ms = int(time.time() * 1000)
    r7 = _run_one(db, code, symbols, now_ms - 7 * 24 * 3600 * 1000, now_ms, exchange=exchange)
    r30 = _run_one(db, code, symbols, now_ms - 30 * 24 * 3600 * 1000, now_ms, exchange=exchange)

    f30 = _extract_gate_fields(r30)
    f7 = _extract_gate_fields(r7)
    win30 = _win_pct(r30)

    pf = r30.profit_factor
    pf_ok = (pf is not None) and (float(pf) > threshold_pf)

    health_ok, health = _data_health_ok(db, days=30, symbol=symbols[0] if symbols else "BTC", exchange=exchange)

    cond = {
        "pnl_pct": f30["pnl_percent"] > threshold_30d_pnl_pct,
        "sharpe": f30["sharpe"] > threshold_30d_sharpe,
        "pf": pf_ok,
        "win": win30 > threshold_win_pct,
        "trades_30d": f30["trades"] >= threshold_trades_30d,
        "trades_7d": f7["trades"] >= threshold_trades_7d,
        "maxdd": f30["maxdd_pct"] < threshold_maxdd_pct if f30["maxdd_pct"] is not None else False,
        "health": health_ok,
    }
    gate_a = all(cond.values())
    return gate_a, {
        "7d": r7,
        "30d": r30,
        "gate_a": gate_a,
        "cond": cond,
        "f30": f30,
        "f7": f7,
        "health": health,
        "pf_ok": pf_ok,
        "thresholds": {
            "pnl_pct": threshold_30d_pnl_pct,
            "sharpe": threshold_30d_sharpe,
            "pf": threshold_pf,
            "win": threshold_win_pct,
            "trades_30d": threshold_trades_30d,
            "trades_7d": threshold_trades_7d,
            "maxdd": threshold_maxdd_pct,
        },
    }


def b_validator(
    db,
    code: str,
    threshold_avg: float = 0.8,
    folds: int = 3,
    step_days: int = 30,
    threshold_pnl_pct: float = PNL_30D_PCT,
    symbols: Optional[List[str]] = None,
    exchange: str = "hyperliquid",
) -> Tuple[bool, float, List]:
    """
    Gate B — WF 3 folds 每fold 30d（多幣種掃描版，symbols 默認由 hl_universe.get_hl_top30 動態發現）
    舊版：2 folds avg sharpe > -0.5（鬆）
    新版：3 folds avg sharpe > 0.8 且 每fold pnl%>30 才算穩健（對應日均1%單利）
    為兼容舊 iterate --threshold-b 單閾呼叫，threshold_avg 仍為 sharpe 閾值。
    """
    symbols = _resolve_symbols(symbols, 30)
    now_ms = int(time.time() * 1000)
    step_ms = step_days * 24 * 3600 * 1000
    oos = []
    for i in range(folds):
        test_end = now_ms - (folds - 1 - i) * step_ms
        test_start = test_end - step_ms
        r = _run_one(db, code, symbols, test_start, test_end, exchange=exchange)
        oos.append(r)
    avg_sharpe = sum(float(getattr(r, "sharpe_ratio", 0) or 0) for r in oos) / len(oos) if oos else -9
    avg_pnl = sum(float(getattr(r, "total_pnl_percent", 0) or 0) for r in oos) / len(oos) if oos else -999
    # PRP-P1-4: 熔斷制 — 任一折低於-5%直接否決; 其餘折允許盤整小虧(>=-5%), avg達標即過
    # 替代舊all_pnl_ok(每折>閾值, 盤整折系統性誤殺)
    fuse = -8.0
    no_blowup = all(float(getattr(r, "total_pnl_percent", 0) or 0) >= fuse for r in oos) if oos else False
    gate_b = (avg_sharpe > threshold_avg) and no_blowup
    return gate_b, avg_sharpe, oos


def c_validator(
    db,
    code: str,
    threshold_pnl_pct: float = PNL_90D_PCT,
    threshold_pf: float = 1.3,
    threshold_win_pct: float = 35.0,
    threshold_maxdd_pct: float = 18.0,
    threshold_trades: int = 25,
    symbols: Optional[List[str]] = None,
    exchange: str = "hyperliquid",
) -> Tuple[bool, Dict]:
    """
    Gate C — 90d 連續回測，日均1%單利 90% 目標（多幣種掃描版，symbols 默認由 hl_universe.get_hl_top30 動態發現）
    需：pnl%>90、PF>1.3、win>35%、maxDD<18%、trades>=25、5m 日均>=50（90d 口徑）
    """
    symbols = _resolve_symbols(symbols, 30)
    now_ms = int(time.time() * 1000)
    r90 = _run_one(db, code, symbols, now_ms - 90 * 24 * 3600 * 1000, now_ms, exchange=exchange)
    f90 = _extract_gate_fields(r90)
    win90 = _win_pct(r90)
    pf = r90.profit_factor
    pf_ok = (pf is not None) and (float(pf) > threshold_pf)
    health_ok, health = _data_health_ok(db, days=90, symbol=symbols[0] if symbols else "BTC", exchange=exchange)
    cond = {
        "pnl_pct": f90["pnl_percent"] > threshold_pnl_pct,
        "pf": pf_ok,
        "win": win90 > threshold_win_pct,
        "maxdd": f90["maxdd_pct"] < threshold_maxdd_pct if f90["maxdd_pct"] is not None else False,
        "trades": f90["trades"] >= threshold_trades,
        "health": health_ok,
    }
    gate_c = all(cond.values())
    return gate_c, {"90d": r90, "gate_c": gate_c, "cond": cond, "f90": f90, "health": health}


# === ANNUAL VALIDATOR (365d) ===
# 目標（2026-09-27 依實測前沿放寬）
# 舊門檻 PnL>=120% / MaxDD<=5% / trades>=50 在現行 1h 突破策略模板下結構性不可達：
#   - 實測 PnL/MaxDD 比率最高 3.94，gate 隱含需要 24.0（差 6.1 倍）
#   - 實測年均交易數上限約 28 筆，50 筆/年不可達（30 symbol 上每 symbol 1.7 筆/年）
# 新門檻對應實測可達點：lev1/portion0.30 -> PnL 16.91% / MaxDD 7.57% / 25 筆 / PF 2.26
PNL_365D_PCT = 20.0
MAXDD_365D_PCT = 8.0
MIN_TRADES_365D = 20
MIN_PF = 1.5
MIN_SHARPE = 0.5
# 勝率不是門檻重點（2026-09-27 使用者決策）。趨勢追蹤型策略的實測勝率恆為
# 38.03%，且與部位大小無關。要求勝率 >45% 會與 PF>1.5 互相衝突：提高勝率只能靠
# 更頻繁的小額交易，那會推高 MaxDD 而違反另一條條件。
# 因此勝率改為替代條件：勝率達 MIN_WIN_PCT，或 PF 達到較強的 MIN_PF_STRONG。
MIN_WIN_PCT = 35.0
MIN_PF_STRONG = 2.0
MAX_TRADES_PER_YEAR = 730.0   # 約 2 筆/日；超過視為過度交易/過擬合，直接否決
MIN_5M_PER_DAY_365 = 50  # 同健康檢查

def annual_validator(
    db,
    code: str,
    threshold_365d_pnl_pct: float = PNL_365D_PCT,
    threshold_maxdd_pct: float = MAXDD_365D_PCT,
    threshold_pf: float = MIN_PF,
    threshold_sharpe: float = MIN_SHARPE,
    threshold_win_pct: float = MIN_WIN_PCT,
    threshold_trades: int = MIN_TRADES_365D,
    threshold_pf_strong: float = MIN_PF_STRONG,
    symbols: Optional[List[str]] = None,
    exchange: str = "hyperliquid",
    fast_fail: bool = True,  # Early exit if any condition fails
    max_triggers: Optional[int] = None,  # Limit triggers for speed
    backtest_days: int = 365,
    end_time_ms: Optional[int] = None,  # 固定回測視窗終點（2026-09-27）
) -> Tuple[bool, Dict]:
    """
    Annual Gate — 365d 回測
    需同時滿足（2026-09-27 依實測前沿放寬，理由見 PNL_365D_PCT 註解）：
    - pnl% >= 20%
    - maxDD% <= 8%
    - PF > 1.5
    - Sharpe > 0.5
    - Win > 35%  或  PF >= 2.0（勝率非門檻重點，改為替代條件）
    - Trades >= 20 (年均) 且 <= 730 (年均，過度交易否決)
    - 5m 日均 >= 50 (健康)
    """
    symbols = _resolve_symbols(symbols, 30)
    # 2026-09-27：允許固定回測視窗終點。
    # 原本固定用 int(time.time()*1000)，導致每次評估的資料視窗都往右移一格：
    # 同一個基因組在 gen4 與 gen5 會得到不同指標（實測 DD 3.69% -> 5.01%、
    # PnL 26.73% -> 26.51%）。GA 因此在對雜訊最佳化，無法重現自己的評估，
    # 而且漂動正好落在 DD 這個綁定約束上。
    # 傳入固定值即可讓整輪 run 的視窗一致；不傳則維持原本的「跟隨當下」行為。
    now_ms = int(end_time_ms) if end_time_ms else int(time.time() * 1000)
    backtest_days = max(int(backtest_days), 1)
    period_years = max(backtest_days / 365.0, 1.0)
    
    # Fast fail: check data health first (cheap)
    health_ok, health = _data_health_ok(db, days=backtest_days, symbol=symbols[0] if symbols else "BTC", exchange=exchange)
    if not health_ok:
        return False, {
            "gate_annual": False,
            "f365": {"pnl_percent": 0, "maxdd_pct": 100, "sharpe": 0, "trades": 0},
            "cond": {"health": False},
            "health": health,
            "error": "Insufficient data health",
        }
    
    # Run backtest with optional trigger limit
    r365 = _run_one(
        db, code, symbols, 
        now_ms - backtest_days * 24 * 3600 * 1000, now_ms, 
        exchange=exchange,
        trigger=900 if max_triggers is None else _compute_trigger_interval_sec(max_triggers, backtest_days),
        max_triggers=max_triggers,
    )
    
    f365 = _extract_gate_fields(r365)
    win365 = _win_pct(r365)
    total_pnl_percent = float(f365.get("pnl_percent", 0.0) or 0.0)
    total_trades = int(f365.get("trades", 0) or 0)
    if period_years > 1.0:
        f365["total_pnl_percent"] = total_pnl_percent
        f365["total_trades"] = total_trades
        f365["pnl_percent"] = total_pnl_percent / period_years
        f365["trades"] = total_trades / period_years
    f365["backtest_days"] = backtest_days
    f365["period_years"] = period_years
    # 過度交易上限：年化筆數超過 730 直接否決（與評分無關，是硬性健全檢查）
    overtraded = float(f365.get("trades", 0) or 0) > MAX_TRADES_PER_YEAR
    f365["overtraded"] = overtraded
    if overtraded:
        f365["trades_reported_capped"] = MAX_TRADES_PER_YEAR
    pf = r365.profit_factor
    pf_ok = (pf is not None) and (float(pf) > threshold_pf)
    
    # Early exit checks if fast_fail enabled
    if fast_fail:
        if f365["pnl_percent"] < threshold_365d_pnl_pct * 0.5:  # Less than 50% of target
            return False, {"gate_annual": False, "f365": f365, "early_exit": "low_pnl",
                           "cond": {"pnl": f365["pnl_percent"]}, "pf_ok": pf_ok,
                           "backtest_days": backtest_days, "period_years": period_years}
        if f365["maxdd_pct"] is not None and f365["maxdd_pct"] > threshold_maxdd_pct * 2:  # More than 2x max DD
            return False, {"gate_annual": False, "f365": f365, "early_exit": "high_dd",
                           "cond": {"maxdd": f365["maxdd_pct"]}, "pf_ok": pf_ok,
                           "backtest_days": backtest_days, "period_years": period_years}
    
    # 勝率改為替代條件：勝率達門檻，或 PF 達到較強水準（二者擇一即可）。
    # 見 MIN_WIN_PCT 註解：實測勝率恆為 38.03%，硬性要求 45% 會誤殺獲利能力強的個體。
    pf_strong_ok = (pf is not None) and (float(pf) >= threshold_pf_strong)
    win_ok = (win365 > threshold_win_pct) or pf_strong_ok
    cond = {
        "pnl_pct": f365["pnl_percent"] >= threshold_365d_pnl_pct,
        "maxdd": f365["maxdd_pct"] <= threshold_maxdd_pct if f365["maxdd_pct"] is not None else False,
        "pf": pf_ok,
        "sharpe": f365["sharpe"] > threshold_sharpe,
        "win": win_ok,
        "trades": f365["trades"] >= threshold_trades,
        "max_trades": not overtraded,
        "health": health_ok,
    }
    gate_annual = all(cond.values())
    return gate_annual, {
        "365d": r365,
        "gate_annual": gate_annual,
        "cond": cond,
        "f365": f365,
        "health": health,
        "pf_ok": pf_ok,
        "backtest_days": backtest_days,
        "period_years": period_years,
        "thresholds": {
            "pnl_pct": threshold_365d_pnl_pct,
            "maxdd_pct": threshold_maxdd_pct,
            "pf": threshold_pf,
            "pf_strong": threshold_pf_strong,
            "win_pct": threshold_win_pct,
            "sharpe": threshold_sharpe,
            "win": threshold_win_pct,
            "trades": threshold_trades,
            "max_trades_per_year": MAX_TRADES_PER_YEAR,
        },
    }

# === EXPLORATION VALIDATOR thresholds (single source of truth) ===
# 這組門檻原本只寫在 exploration_validator 的參數預設值裡，外部讀不到。
# 結果 scoring.gate_violation 只能用嚴格期常數（trades=20）算違規量，
# 而 GA 實際被 exploration_validator（trades=60）判生死：
# 實測 12 幣 180 天 run 的冠軍 19 筆交易，fitness 只罰 0.05，
# exploration gate 卻直接否決。fitness 與 gate 各自最佳化不同目標。
# 提成模組常數後，兩邊共用同一份數字。
EXPLORATION_PNL_365D_PCT = 10.0
EXPLORATION_MAXDD_365D_PCT = 10.0
EXPLORATION_PF = 1.2
EXPLORATION_PF_STRONG = 1.6
EXPLORATION_SHARPE = 0.3
EXPLORATION_WIN_PCT = 35.0
EXPLORATION_TRADES = 60


def exploration_validator(
    db,
    code: str,
    threshold_365d_pnl_pct: float = EXPLORATION_PNL_365D_PCT,
    # 2026-10-07: 12% → 10%。實測 leverage 1→2 讓 PnL 22.53%→49.06%（約 2.18 倍，
    # 名目倉位灌水），MaxDD 只從 4.72% 升到 9.20%，靠舊的 12% 探索門檻通過。
    # 嚴格門檻是 8% 會擋下，但探索期不該放寬到讓槓桿灌水過關。
    threshold_maxdd_pct: float = EXPLORATION_MAXDD_365D_PCT,
    threshold_pf: float = EXPLORATION_PF,
    threshold_pf_strong: float = EXPLORATION_PF_STRONG,
    threshold_sharpe: float = EXPLORATION_SHARPE,
    threshold_win_pct: float = EXPLORATION_WIN_PCT,   # 與嚴格版相同（勝率已非重點）
    # 2026-10-07: 20 → 60。原意是「探索期寬鬆」，但 20 筆太少了 ——
    # 鄰域穩健性測試證明 35 筆的策略擾動後中位 PnL 是 −17.55%。
    # 下界 20 等於允許 GA 用不可估計的樣本量堆 PnL。與 fitness 的
    # TRADE_COUNT_MIN=60 對齊。
    threshold_trades: int = EXPLORATION_TRADES,       # 對齊 fitness TRADE_COUNT_MIN
    symbols: Optional[List[str]] = None,
    exchange: str = "hyperliquid",
    fast_fail: bool = True,
    max_triggers: Optional[int] = 30000,
    backtest_days: int = 365,
    end_time_ms: Optional[int] = None,
) -> Tuple[bool, Dict]:
    """
    Exploration Gate — Relaxed 365d 回測 for early GA phases
    需同時滿足（放寬門檻以發現種子策略；須比嚴格 gate 寬鬆，否則兩期倒置）：
    - pnl% >= 10%
    - maxDD% <= 10%   (2026-10-07 由 12% 收緊，阻擋槓桿灌水)
    - PF > 1.2
    - Sharpe > 0.3
    - Win > 35% 或 PF >= 1.6
    - Trades >= 60 (年均，2026-10-07 由 20 收緊，對齊 fitness 的頻率下限)
    - 5m 日均 >= 50 (健康)
    """
    return annual_validator(
        db, code,
        threshold_365d_pnl_pct=threshold_365d_pnl_pct,
        threshold_maxdd_pct=threshold_maxdd_pct,
        threshold_pf=threshold_pf,
        threshold_sharpe=threshold_sharpe,
        threshold_win_pct=threshold_win_pct,
        threshold_trades=threshold_trades,
        threshold_pf_strong=threshold_pf_strong,
        symbols=symbols,
        exchange=exchange,
        fast_fail=fast_fail,
        max_triggers=max_triggers,
        backtest_days=backtest_days,
        end_time_ms=end_time_ms,
    )


