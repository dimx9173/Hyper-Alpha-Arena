"""
A+B robustness validators for 30-round iteration.
Gate A: 30d sharpe > -0.3 and 30d PF>0 and 7d trades>=2
Gate B: WF 30d rolling 2 folds avg sharpe > -0.5 (quick version for iteration loop)
"""
import time, json
from typing import Tuple, List, Dict, Any

def _run_one(db, code: str, symbols, start_ms: int, end_ms: int, trigger=3700, fee=0.035, slip=0.05, exchange="binance"):
    try:
        from backtest import BacktestConfig, ProgramBacktestEngine
    except ImportError:
        try:
            from backend.backtest import BacktestConfig, ProgramBacktestEngine
        except ImportError:
            from backtest.models import BacktestConfig
            from backtest.engine import ProgramBacktestEngine
    cfg = BacktestConfig(code=code, signal_pool_ids=[], symbols=symbols, start_time_ms=start_ms, end_time_ms=end_ms, scheduled_interval_sec=trigger, initial_balance=10000.0, slippage_percent=slip, fee_rate=fee, exchange=exchange)
    return ProgramBacktestEngine(db).run(cfg)

def a_validator(db, code: str, threshold_30d_sharpe=-0.3) -> Tuple[bool, Dict]:
    now_ms=int(time.time()*1000)
    r7=_run_one(db, code, ["BTC"], now_ms-7*24*3600*1000, now_ms)
    r30=_run_one(db, code, ["BTC"], now_ms-30*24*3600*1000, now_ms)
    pf = r30.profit_factor
    pf_ok = (pf is None) or (pf > 0)
    gate_a = (r30.sharpe_ratio > threshold_30d_sharpe) and pf_ok and (r7.total_trades >= 2)
    return gate_a, {"7d": r7, "30d": r30, "gate_a": gate_a, "pf_ok": pf_ok}

def b_validator(db, code: str, threshold_avg=-0.5, folds=2, step_days=30) -> Tuple[bool, float, List]:
    now_ms=int(time.time()*1000)
    step_ms=step_days*24*3600*1000
    oos=[]
    for i in range(folds):
        test_end=now_ms - (folds-1-i)*step_ms
        test_start=test_end - step_ms
        r=_run_one(db, code, ["BTC"], test_start, test_end)
        oos.append(r)
    avg=sum(r.sharpe_ratio for r in oos)/len(oos) if oos else -9
    gate_b = avg > threshold_avg
    return gate_b, avg, oos
