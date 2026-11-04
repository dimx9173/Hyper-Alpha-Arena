"""
Offline Hyper AI Pipeline
=========================
Wrap the 4 Hyper AI sub-agents (Prompt AI → Program AI → Signal AI → Attribution AI)
into a single callable function that runs without SSE streaming, returns the final
strategy code that can be evaluated by evaluate_spec_individual / ProgramBacktestEngine.

Usage:
    from backtest.hyper_ai_pipeline import run_hyper_ai_offline
    code, trace = run_hyper_ai_offline(
        user_request="EMA cross + Donchian breakout + RSI band on BTC 1h",
        exchange="hyperliquid",
        symbols=["BTC"]
    )
    # code -> strategy python code string
    # trace -> dict with intermediate outputs for debugging/analysis
"""

from __future__ import annotations

import json
import os
import re
import sys
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple

# Ensure backend is on path
BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from database.connection import SessionLocal
from database.models import Account, TradingProgram, SignalPool
from services.hyper_ai_service import get_llm_config


@contextmanager
def _db_session():
    """Context manager for DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _run_stream_to_completion(generator: Generator[str, None, None]) -> Dict[str, Any]:
    """Run a streaming generator to completion, return the final 'done' event data."""
    last_done = None
    for chunk in generator:
        try:
            # chunk is SSE format: "event: X\ndata: {...}\n\n"
            if chunk.startswith("event: done"):
                # Extract JSON after "data: "
                lines = chunk.strip().split("\n")
                for line in lines:
                    if line.startswith("data: "):
                        data_str = line[6:]
                        last_done = json.loads(data_str)
                        break
        except (json.JSONDecodeError, IndexError):
            continue
    if last_done is None:
        raise RuntimeError("Stream did not produce a 'done' event")
    return last_done


def run_prompt_ai_offline(
    db,
    user_request: str,
    exchange: str = "hyperliquid",
    conversation_id: Optional[int] = None,
    prompt_id: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Run Prompt AI offline, return (prompt_text, full_trace)."""
    from services.ai_prompt_generation_service import generate_prompt_with_ai_stream
    from services.hyper_ai_service import get_llm_config

    llm_config = get_llm_config(db)
    if not llm_config.get("configured"):
        raise RuntimeError("LLM not configured")

    gen = generate_prompt_with_ai_stream(
        db=db,
        user_message=user_request,
        conversation_id=conversation_id,
        prompt_id=prompt_id,
        llm_config=llm_config,
    )
    done = _run_stream_to_completion(gen)
    prompt_text = done.get("prompt_result", "")
    return prompt_text, done


def run_program_ai_offline(
    db,
    prompt_text: str,
    exchange: str = "hyperliquid",
    symbols: Optional[List[str]] = None,
    conversation_id: Optional[int] = None,
    program_id: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Run Program AI offline, return (strategy_code, full_trace)."""
    from services.ai_program_service import generate_program_with_ai_stream
    from services.hyper_ai_service import get_llm_config

    llm_config = get_llm_config(db)
    if not llm_config.get("configured"):
        raise RuntimeError("LLM not configured")

    # Sandbox 相容格式：必須輸出 Strategy + Decision + data.get_price/get_indicator/get_klines
    # 與 render_strategy_from_spec 完全一致（byte-identical block 形狀）
    task = f"""Generate a trading strategy program based on this prompt:
{prompt_text}

Exchange: {exchange}
Symbols: {symbols or ['BTC']}

CRITICAL OUTPUT FORMAT (must be sandbox-compatible):
Your code MUST follow this EXACT structure:

```python
class Strategy:
    def should_trade(self, data):
        symbol = data.trigger_symbol or "BTC"
        price = data.get_price(symbol)
        if price <= 0:
            return Decision(operation="hold", symbol=symbol, reason="no price")

        # Use get_indicator with EXACT keys: "EMA12"/"EMA55", "DONCHIAN55", "RSI14"
        ema_fast = data.get_indicator(symbol, "EMA12", "1h").get("value", 0)
        ema_slow = data.get_indicator(symbol, "EMA55", "1h").get("value", 0)
        donchian = data.get_indicator(symbol, "DONCHIAN55", "1h")
        donchian_high = donchian.get("upper", 0)
        rsi = data.get_indicator(symbol, "RSI14", "1h").get("value", 50)

        # Trading logic here (adjust periods/thresholds per the prompt)
        long_cond = (ema_fast > ema_slow and price > donchian_high * 0.99 and 30 <= rsi <= 70)

        positions = data.positions or {{}}
        has_position = symbol in positions and abs(positions[symbol].size) > 0

        if long_cond and not has_position:
            return Decision(
                operation="buy",
                symbol=symbol,
                target_portion_of_balance=0.35,
                leverage=2,
                max_price=price * 1.001,
                reason="your reason here",
                trading_strategy="your description here",
            )

        if has_position:
            pos = positions[symbol]
            if pos.side == "long":
                if ema_fast < ema_slow:  # exit condition
                    return Decision(
                        operation="close",
                        symbol=symbol,
                        target_portion_of_balance=1.0,
                        leverage=pos.leverage,
                        min_price=price * 0.999,
                        reason="exit reason",
                    )

        return Decision(operation="hold", symbol=symbol, reason="no signal")
```

Requirements:
- Class name MUST be exactly `Strategy` (not MultiIndicatorStrategy, not EMACrossStrategy)
- Method MUST be `should_trade(self, data)`
- MUST return Decision objects with operation="buy"/"sell"/"close"/"hold"
- MUST use data.get_price(symbol), data.get_indicator(symbol, "EMA12"/"EMA55"/"DONCHIAN55"/"RSI14", "1h"), data.positions
- MUST use Decision(operation=..., symbol=..., target_portion_of_balance=..., leverage=..., max_price/min_price=..., reason=..., trading_strategy=...)
- Do NOT use data.get_market_data, data.get_klines, data.get_flow, data.get_regime, or any other API
- Adjust EMA periods (12/55), Donchian lookback (55), threshold (0.99), RSI band (30/70), portion (0.35), leverage (2) per the prompt
- Output the code in a single ```python fence. No other text inside the fence.
"""
    gen = generate_program_with_ai_stream(
        db=db,
        user_message=task,
        conversation_id=conversation_id,
        program_id=program_id,
        llm_config=llm_config,
    )
    done = _run_stream_to_completion(gen)

    # 抽 code：優先找含 "class Strategy" 的 python fence，
    # 其次找最後一個 python fence（Program AI 常在前面放範例）。
    content = done.get("content", "") or ""
    code = _extract_strategy_code(content)
    return code, done


_PY_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_strategy_code(content: str) -> str:
    """從 LLM 內容抽出 sandbox 相容的 Strategy 類別。

    優先順序：
    1. 含 "class Strategy" 且含 "def should_trade" 的 fence
    2. 最後一個含 "def should_trade" 的 fence
    3. 最後一個 python fence
    4. 原文（讓 backtest 報錯，不靜默替換）
    """
    blocks = [m.group(1) for m in _PY_FENCE_RE.finditer(content)]
    if not blocks:
        return content
    for b in blocks:
        if "class Strategy" in b and "def should_trade" in b:
            return b
    for b in reversed(blocks):
        if "def should_trade" in b:
            return b
    return blocks[-1]


def run_signal_ai_offline(
    db,
    user_request: str,
    exchange: str = "hyperliquid",
    symbols: Optional[List[str]] = None,
    conversation_id: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run Signal AI offline, return (signal_config, full_trace)."""
    from services.ai_signal_generation_service import generate_signal_with_ai_stream
    from services.hyper_ai_service import get_llm_config

    llm_config = get_llm_config(db)
    if not llm_config.get("configured"):
        raise RuntimeError("LLM not configured")

    task = f"""Create signal pool for: {user_request}
Exchange: {exchange}
Symbols: {symbols or ['BTC']}
"""
    gen = generate_signal_with_ai_stream(
        db=db,
        user_message=task,
        conversation_id=conversation_id,
        llm_config=llm_config,
    )
    done = _run_stream_to_completion(gen)

    # Extract signal config
    signal_config = done.get("signal_config", {})
    return signal_config, done


def run_hyper_ai_offline(
    user_request: str,
    exchange: str = "hyperliquid",
    symbols: Optional[List[str]] = None,
    db_factory: Optional[Any] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Run the full Hyper AI pipeline offline (Prompt AI → Program AI).
    Returns (strategy_code, trace_dict).

    This is the main entry point for Training Agent to evaluate Hyper AI output.
    """
    db = (db_factory or SessionLocal)()
    trace = {
        "user_request": user_request,
        "exchange": exchange,
        "symbols": symbols or ["BTC"],
        "stages": {},
    }

    try:
        # Stage 1: Prompt AI
        prompt_text, prompt_trace = run_prompt_ai_offline(db, user_request, exchange)
        trace["stages"]["prompt_ai"] = {
            "prompt_text": prompt_text,
            "trace": prompt_trace,
        }

        # Stage 2: Program AI
        strategy_code, program_trace = run_program_ai_offline(
            db, prompt_text, exchange, symbols
        )
        trace["stages"]["program_ai"] = {
            "strategy_code": strategy_code,
            "trace": program_trace,
        }

        # Optional: Stage 3 - Signal AI (for trigger config)
        # signal_config, signal_trace = run_signal_ai_offline(db, user_request, exchange, symbols)
        # trace["stages"]["signal_ai"] = {"signal_config": signal_config, "trace": signal_trace}

        trace["status"] = "success"
        return strategy_code, trace

    except Exception as e:
        trace["status"] = "failed"
        trace["error"] = str(e)
        raise
    finally:
        db.close()


def strategy_code_to_spec(strategy_code: str) -> Dict[str, Any]:
    """
    Convert generated strategy code to a StrategySpec dict.
    This is a best-effort parser for the specific patterns our Program AI produces.
    Falls back to a generic spec if parsing fails.
    """
    import re
    from backtest.strategy_spec import StrategySpec, FilterSpec, RiskSpec, LogicSpec

    # Try to extract EMA/RSI/Donchian/ICT parameters from code
    spec_dict = {
        "filters": [],
        "risk": {
            "tp_pct": 0.08,
            "sl_pct": 0.04,
            "portion": 0.35,
            "leverage": 2,
            "max_positions": 3,
        },
        "logic": "all_pass",
    }

    # Extract ema_fast / ema_slow
    ema_fast_match = re.search(r"EMA\s*(\d+)", strategy_code, re.IGNORECASE)
    ema_slow_match = re.search(r"EMA\s*(\d+)", strategy_code, re.IGNORECASE)
    # More precise: look for ema_fast/ema_slow in params or comparisons
    ema_fast = re.search(r"ema_fast\s*[:=]\s*(\d+)", strategy_code)
    ema_slow = re.search(r"ema_slow\s*[:=]\s*(\d+)", strategy_code)
    if ema_fast and ema_slow:
        spec_dict["filters"].append({
            "kind": "ema_cross",
            "params": {
                "ema_fast": int(ema_fast.group(1)),
                "ema_slow": int(ema_slow.group(1)),
                "ema_tol_buy": 0.9985,
                "ema_tol_sell": 1.0008,
            },
        })

    # Extract Donchian
    dc_lookback = re.search(r"donchian.*?lookback\s*[:=]\s*(\d+)", strategy_code, re.IGNORECASE)
    dc_thr = re.search(r"donchian.*?threshold\s*[:=]\s*([\d.]+)", strategy_code, re.IGNORECASE)
    if dc_lookback and dc_thr:
        spec_dict["filters"].append({
            "kind": "donchian_breakout",
            "params": {
                "lookback": int(dc_lookback.group(1)),
                "threshold": float(dc_thr.group(1)),
            },
        })

    # Extract RSI
    rsi_low = re.search(r"rsi_low\s*[:=]\s*(\d+)", strategy_code, re.IGNORECASE)
    rsi_high = re.search(r"rsi_high\s*[:=]\s*(\d+)", strategy_code, re.IGNORECASE)
    if rsi_low and rsi_high:
        spec_dict["filters"].append({
            "kind": "rsi_band",
            "params": {
                "low": int(rsi_low.group(1)),
                "high": int(rsi_high.group(1)),
            },
        })

    # Extract ICT
    ict_w = re.search(r"ict_weight\s*[:=]\s*([\d.]+)", strategy_code, re.IGNORECASE)
    if ict_w:
        spec_dict["filters"].append({
            "kind": "ict_confirm",
            "params": {
                "ote_threshold": float(ict_w.group(1)),
                "require_alignment": 0,
                "require_bull_structure": 0,
                "require_discount": 0,
                "swing_len": 5,
                "weight": float(ict_w.group(1)),
            },
        })

    # Extract risk params
    portion = re.search(r"portion\s*[:=]\s*([\d.]+)", strategy_code, re.IGNORECASE)
    leverage = re.search(r"leverage\s*[:=]\s*(\d+)", strategy_code, re.IGNORECASE)
    tp = re.search(r"tp_pct\s*[:=]\s*([\d.]+)", strategy_code, re.IGNORECASE)
    sl = re.search(r"sl_pct\s*[:=]\s*([\d.]+)", strategy_code, re.IGNORECASE)
    if portion:
        spec_dict["risk"]["portion"] = float(portion.group(1))
    if leverage:
        spec_dict["risk"]["leverage"] = int(leverage.group(1))
    if tp:
        spec_dict["risk"]["tp_pct"] = float(tp.group(1))
    if sl:
        spec_dict["risk"]["sl_pct"] = float(sl.group(1))

    # Validate via StrategySpec.from_dict
    try:
        spec = StrategySpec.from_dict(spec_dict)
        return spec.to_dict()
    except Exception:
        return spec_dict


def evaluate_hyper_ai_output(
    user_request: str,
    exchange: str = "hyperliquid",
    symbols: Optional[List[str]] = None,
    backtest_days: int = 180,
    end_time_ms: Optional[int] = None,
    kline_period: str = "1h",
    db_factory: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Full offline evaluation: Hyper AI pipeline → StrategySpec → evaluate_spec_individual.
    Returns the evaluation result dict (fitness, gate, metrics, etc.).
    """
    from backtest.strategy_spec import evaluate_spec_individual

    # Run Hyper AI pipeline
    strategy_code, trace = run_hyper_ai_offline(
        user_request, exchange, symbols, db_factory
    )

    # Convert to StrategySpec
    spec_dict = strategy_code_to_spec(strategy_code)
    trace["derived_spec"] = spec_dict

    # Evaluate
    eval_result = evaluate_spec_individual(
        spec_dict,
        symbols or ["BTC"],
        exchange=exchange,
        backtest_days=backtest_days,
        end_time_ms=end_time_ms,
        kline_period=kline_period,
        generation=0,
        total_generations=1,
    )

    return {
        "strategy_code": strategy_code,
        "spec": spec_dict,
        "trace": trace,
        "evaluation": eval_result,
    }


if __name__ == "__main__":
    # Quick smoke test
    import time
    from datetime import datetime, timezone

    req = "EMA cross + Donchian breakout + RSI band on BTC 1h"
    end_ms = int(datetime(2026, 10, 31, 16, 0, tzinfo=timezone.utc).timestamp() * 1000)

    print(f"Testing Hyper AI offline pipeline: {req}")
    t0 = time.time()
    try:
        result = evaluate_hyper_ai_output(
            req, "hyperliquid", ["BTC"], backtest_days=180, end_time_ms=end_ms
        )
        ev = result["evaluation"]
        print(f"Done in {time.time()-t0:.1f}s")
        print(f"  fitness: {ev.get('fitness')}")
        print(f"  gate_annual: {ev.get('gate_annual')}")
        print(f"  pnl_365d: {ev.get('pnl_365d')}")
        print(f"  maxdd_365d: {ev.get('maxdd_365d')}")
        print(f"  sharpe_365d: {ev.get('sharpe_365d')}")
        print(f"  pf: {ev.get('pf')}")
        print(f"  trades_365d: {ev.get('trades_365d')}")
        print(f"  win_rate: {ev.get('win_rate')}")
        print(f"  spec_id: {ev.get('spec_id')}")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

def evaluate_strategy_code(
    strategy_code: str,
    symbols: Optional[List[str]] = None,
    exchange: str = "hyperliquid",
    generation: int = 0,
    total_generations: int = 100,
    backtest_days: int = 180,
    end_time_ms: Optional[int] = None,
    kline_period: str = "1h",
    db_factory: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Evaluate raw strategy code directly using the same path as evaluate_individual.
    This bypasses StrategySpec -> render, works with any valid sandbox strategy.
    """
    from pathlib import Path as _Path
    import sys as _sys
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

    from backtest.ab_validators import annual_validator, exploration_validator
    from backtest.ga_annual import (
        EXPLORATION_GENS_OVERRIDE,
        calculate_fitness,
        calculate_exploration_fitness,
        gate_violation,
    )
    from database.connection import SessionLocal

    override = EXPLORATION_GENS_OVERRIDE
    if override is not None and override > 0:
        exploration_gens = int(override)
    else:
        exploration_gens = max(3, int(total_generations * 0.5))
    exploration_gens = min(exploration_gens, max(1, total_generations))
    use_exploration = (generation - 1) < exploration_gens

    validate = exploration_validator if use_exploration else annual_validator
    calc = calculate_exploration_fitness if use_exploration else calculate_fitness

    db = (db_factory() if db_factory else SessionLocal())
    try:
        ga, info = validate(
            db, strategy_code, symbols=symbols or ["BTC"], exchange=exchange,
            fast_fail=True, max_triggers=8000,
            backtest_days=backtest_days, end_time_ms=end_time_ms
        )
    except Exception as e:
        return {
            "gate_annual": False,
            "pnl_365d": 0.0,
            "maxdd_365d": 100.0,
            "sharpe_365d": 0.0,
            "pf": None,
            "trades_365d": 0,
            "win_rate": 0.0,
            "fitness": -1000.0,
            "error": str(e)[:500],
        }
    finally:
        try:
            db.close()
        except Exception:
            pass

    f365 = info.get("f365", {}) if info else {}
    pnl = float(f365.get("pnl_percent", 0) or 0)
    dd = float(f365.get("maxdd_pct", 100) or 100)
    sharpe = float(f365.get("sharpe", 0) or 0)
    pf = f365.get("pf")
    pf_f = float(pf) if pf is not None else None
    trades = int(f365.get("trades", 0) or 0)
    win = float(f365.get("win", 0) or 0)

    class DummyResult:
        def __init__(self):
            self.pnl_365d = pnl
            self.maxdd_365d = dd
            self.sharpe_365d = sharpe
            self.pf = pf_f
            self.trades_365d = trades
            self.win_rate = win

    res = DummyResult()
    try:
        fitness = float(calc(res))
    except Exception:
        fitness = -1000.0

    return {
        "gate_annual": bool(ga),
        "pnl_365d": pnl,
        "maxdd_365d": dd,
        "sharpe_365d": sharpe,
        "pf": pf_f,
        "pf_ok": info.get("pf_ok", False) if info else False,
        "trades_365d": trades,
        "win_rate": win,
        "fitness": fitness,
        "spec_id": None,
        "f365": f365,
        "info": info,
    }
