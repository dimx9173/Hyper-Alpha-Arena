
"""
30-Round 策略迭代迴圈 harness

設計
====
全自動 30 輪策略演進：每輪以當前 best 策略為父本產生變體，經 A/B 雙閘門回測驗證，
僅雙閘門皆通過且指標優於歷史 best 才晉升。此迴圈讓使用者一句「繼續迭代策略」
即可執行 `PYTHONPATH=backend python -m backtest.iterate --rounds 30` 完成無人值守優化。

A/B 閘門定義
============
- Gate A (短期穩健)：30d sharpe > threshold-a (預設 -0.3) 且 PF>0 (或無交易時 None 視為通過)
  且 7d 交易數 >= 2。對應 ab_validators.a_validator。
- Gate B (跨期穩健)：Walk-Forward 2 folds、每 fold 30d，平均 sharpe > threshold-b
  (預設 -0.5)。對應 ab_validators.b_validator。

Pass 條件：gate_a and gate_b 皆為 True 才視為 candidate；否則 rejected。
晉升規則：candidate 且 sharpe_30d 與 PF 優於當前 best 才更新 best。

Backtest 參數
=============
BacktestConfig(scheduled_interval_sec=3700, fee=0.035, slippage=0.05) 與
Decision(operation="buy"/"sell"/"hold")、MarketData(data.get_indicator/get_factor/
get_klines/get_price 等) 介面保持一致；策略 sandbox 禁止 import os/sys/requests 等，
math/time 已注入。

輸出
====
- JSONL log：backend/backtest/iteration_log.jsonl 每輪一筆
- best_code.py：最終最佳策略原始碼
- summary.json：best sharpe、rounds、passed count 等摘要
- 支援 --resume：若 log 已存在，自動跳過已跑輪次
"""

import argparse
import ast
import json
import logging
import pathlib
import re
import sys
import time
import traceback
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


DEFAULT_BASELINE_CODE = """\
class Strategy:
    def should_trade(self, data):
        price = data.get_price("BTC")
        if price == 0:
            return Decision(operation="hold", symbol="BTC", reason="no price")
        klines = data.get_klines("BTC", "5m", 50)
        if len(klines) < 20:
            return Decision(operation="hold", symbol="BTC", reason="not enough klines")
        closes = [k.close for k in klines]
        ma_fast = sum(closes[-5:]) / 5
        ma_slow = sum(closes[-20:]) / 20
        if ma_fast > ma_slow * 1.001:
            return Decision(operation="buy", symbol="BTC", reason="ma cross up", target_portion_of_balance=0.3, leverage=3, max_price=price*1.005)
        elif ma_fast < ma_slow * 0.999:
            return Decision(operation="sell", symbol="BTC", reason="ma cross down", target_portion_of_balance=0.3, leverage=3, min_price=price*0.995)
        return Decision(operation="hold", symbol="BTC", reason="no signal")
"""

def _load_baseline(path: Optional[str]) -> str:
    if path:
        p = pathlib.Path(path)
        if p.exists():
            return p.read_text(encoding="utf-8")
        else:
            logger.warning(f"baseline file not found: {path}, fallback to built-in")
    return DEFAULT_BASELINE_CODE


def _validate_syntax(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        logger.warning(f"syntax error: {e}")
        return False


def _mutate_param_jitter(code: str, round_idx: int) -> str:
    seed = round_idx
    jitter_buy = 1.001 + ((seed % 7) - 3) * 0.0005
    jitter_sell = 0.999 + ((seed % 5) - 2) * 0.0005
    portion = 0.3 + ((seed % 9) - 4) * 0.02
    portion = max(0.1, min(0.9, portion))
    new_code = re.sub(r"ma_slow\s*\*\s*1\.\d+", f"ma_slow * {jitter_buy:.4f}", code)
    new_code = re.sub(r"ma_slow\s*\*\s*0\.\d+", f"ma_slow * {jitter_sell:.4f}", new_code)
    new_code = re.sub(r"target_portion_of_balance\s*=\s*0\.\d+", f"target_portion_of_balance={portion:.2f}", new_code)
    lever = 2 + (seed % 4)
    new_code = re.sub(r"leverage\s*=\s*\d+", f"leverage={lever}", new_code, count=2)
    return new_code


def _mutate_threshold(code: str, round_idx: int) -> str:
    if "get_indicator" in code and "rsi" in code.lower():
        # existing rsi filter — jitter thresholds (RSI14, buy when oversold rsi<, sell when overbought rsi>)
        th_buy = 68 + (round_idx % 5)   # 68..72
        th_sell = 32 - (round_idx % 5)  # 32..28
        new_code = re.sub(r"rsi\s*<\s*\d+", f"rsi < {th_buy}", code)
        new_code = re.sub(r"rsi\s*>\s*\d+", f"rsi > {th_sell}", new_code)
        # also fix legacy lowercase indicator name
        new_code = new_code.replace('"rsi"', '"RSI14"').replace("'rsi'", "'RSI14'")
        return new_code
    ind_block = (
        "        rsi_info = data.get_indicator(\"BTC\", \"RSI14\", \"5m\")\n"
        "        rsi = rsi_info.get(\"value\", 50) if isinstance(rsi_info, dict) else 50\n"
        "        if rsi is None:\n"
        "            rsi = 50\n"
     )
    if "rsi_info" not in code:
        new_code = code.replace(
            "        closes = [k.close for k in klines]",
            ind_block + "        closes = [k.close for k in klines]",
        )
        th = 70 - (round_idx % 4) * 2
        tl = 30 + (round_idx % 4) * 2
        new_code = new_code.replace(
            "if ma_fast > ma_slow *",
            f"if rsi < {th} and ma_fast > ma_slow *",
        )
        new_code = new_code.replace(
            "elif ma_fast < ma_slow *",
            f"elif rsi > {tl} and ma_fast < ma_slow *",
        )
        return new_code
    return code


def _mutate_add_filter(code: str, round_idx: int) -> str:
    use_regime = (round_idx % 2 == 0)
    if use_regime:
        if "get_regime" in code:
            return code
        regime_block = (
            "        regime_info = data.get_regime(\"BTC\", \"5m\")\n"
            "        regime = regime_info.regime if hasattr(regime_info, \"regime\") else \"noise\"\n"
        )
        new_code = code.replace(
            "        closes = [k.close for k in klines]",
            regime_block + "        closes = [k.close for k in klines]",
        )
        new_code = new_code.replace(
            "if ma_fast > ma_slow *",
            "if regime in (\"breakout\", \"continuation\", \"noise\") and ma_fast > ma_slow *",
        )
        return new_code
    else:
        if "get_factor" in code:
            return code
        factor_block = (
            "        factor_info = data.get_factor(\"BTC\", \"momentum_5m\", \"5m\")\n"
            "        fval = factor_info.get(\"value\") if isinstance(factor_info, dict) else None\n"
        )
        new_code = code.replace(
            "        closes = [k.close for k in klines]",
            factor_block + "        closes = [k.close for k in klines]",
        )
        if "fval is not None" not in new_code:
            new_code = new_code.replace(
                "if ma_fast > ma_slow *",
                "if (fval is None or fval > -0.01) and ma_fast > ma_slow *",
            )
        return new_code


def generate_variant(base_code: str, round_idx: int) -> str:
    # TODO: LLM propose - replace heuristics with LLM-generated mutation when API available
    # e.g., prompt LLM with best_code + metrics -> return mutated code
    kind = round_idx % 3
    if kind == 0:
        variant = _mutate_param_jitter(base_code, round_idx)
    elif kind == 1:
        variant = _mutate_threshold(base_code, round_idx)
    else:
        variant = _mutate_add_filter(base_code, round_idx)
    if not _validate_syntax(variant):
        return base_code
    return variant


def _get_db():
    try:
        try:
            from database.connection import SessionLocal
        except ImportError:
            from backend.database.connection import SessionLocal
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
        except Exception:
            try:
                from sqlalchemy import text as _text
                db.execute(_text("SELECT 1"))
            except Exception:
                pass
        return db
    except Exception as e:
        logger.warning(f"DB connection failed: {e} - running in degraded mode")
        return None


try:
    from sqlalchemy import text
except ImportError:
    text = None


def _safe_validators():
    try:
        from backtest.ab_validators import a_validator, b_validator
        return a_validator, b_validator
    except ImportError:
        try:
            from backend.backtest.ab_validators import a_validator, b_validator
            return a_validator, b_validator
        except ImportError:
            from ab_validators import a_validator, b_validator
            return a_validator, b_validator


def _extract_metrics(info_a: Dict, avg_b: float):
    try:
        r30 = info_a.get("30d")
        r7 = info_a.get("7d")
        sharpe_30d = float(getattr(r30, "sharpe_ratio", 0.0)) if r30 is not None else 0.0
        pf = getattr(r30, "profit_factor", None) if r30 is not None else None
        if pf is not None:
            try:
                pf = float(pf)
            except Exception:
                pf = None
        trades_7d = int(getattr(r7, "total_trades", 0)) if r7 is not None else 0
        return sharpe_30d, pf, trades_7d, float(avg_b) if avg_b is not None else -9.0
    except Exception:
        return 0.0, None, 0, float(avg_b) if avg_b is not None else -9.0


def main():
    parser = argparse.ArgumentParser(description="30-round strategy iteration harness")
    parser.add_argument("--rounds", type=int, default=30, help="number of rounds")
    parser.add_argument("--exchange", type=str, default="binance", choices=["binance", "hyperliquid"], help="exchange")
    parser.add_argument("--symbols", type=str, default="BTC", help="comma separated symbols, e.g. BTC or BTC,ETH")
    parser.add_argument("--baseline", type=str, default=None, help="path to baseline .py file")
    parser.add_argument("--threshold-a", type=float, default=-0.3, help="Gate A sharpe threshold")
    parser.add_argument("--threshold-b", type=float, default=-0.5, help="Gate B avg sharpe threshold")
    parser.add_argument("--log-file", type=str, default=None, help="log file path")
    parser.add_argument("--resume", action="store_true", help="resume from existing log")
    parser.add_argument("--best-code", type=str, default=None, help="output best code path")
    parser.add_argument("--summary", type=str, default=None, help="output summary path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    default_log = pathlib.Path(__file__).parent / "iteration_log.jsonl"
    log_file = pathlib.Path(args.log_file) if args.log_file else default_log
    best_code_path = pathlib.Path(args.best_code) if args.best_code else (pathlib.Path(__file__).parent / "best_code.py")
    summary_path = pathlib.Path(args.summary) if args.summary else (pathlib.Path(__file__).parent / "summary.json")

    completed = set()
    best_code = _load_baseline(args.baseline)
    best_sharpe = float("-inf")
    best_pf = None
    passed = 0
    total_logged = 0

    if log_file.exists():
        if args.resume:
            try:
                with log_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        line=line.strip()
                        if not line:
                            continue
                        try:
                            rec=json.loads(line)
                            r=rec.get("round")
                            if isinstance(r,int):
                                completed.add(r)
                            if rec.get("status")=="promoted":
                                passed+=1
                                s=rec.get("sharpe_30d")
                                if isinstance(s,(int,float)) and s>best_sharpe:
                                    best_sharpe=s
                                    best_pf=rec.get("pf")
                                    cp=rec.get("code_preview")
                        except Exception:
                            continue
                total_logged=len(completed)
                if completed:
                    last_promoted=None
                    with log_file.open("r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                rec=json.loads(line)
                                if rec.get("status")=="promoted":
                                    last_promoted=rec
                            except Exception:
                                continue
                    if last_promoted and last_promoted.get("code_preview"):
                        pass
                logger.info(f"resume: {len(completed)} rounds already logged, best_sharpe={best_sharpe}")
            except Exception as e:
                logger.warning(f"resume read failed: {e}")
        else:
            logger.info(f"log exists at {log_file}, use --resume to skip completed rounds or remove file")

    try:
        a_validator, b_validator = _safe_validators()
    except Exception as e:
        logger.error(f"failed to import validators: {e}")
        a_validator = None
        b_validator = None

    if not _validate_syntax(best_code):
        logger.warning("baseline syntax invalid, fallback to default")
        best_code = DEFAULT_BASELINE_CODE

    if args.resume and completed and best_code_path.exists():
        try:
            loaded = best_code_path.read_text(encoding="utf-8")
            if loaded.strip():
                full_best_code_try = loaded
                import ast as _ast2
                _ast2.parse(full_best_code_try)
                best_code = full_best_code_try
        except Exception:
            pass

    if best_sharpe == float("-inf"):
        best_sharpe = -9.0

    full_best_code = best_code

    start_round = 1
    if args.resume and completed:
        start_round = max(completed) + 1

    log_file.parent.mkdir(parents=True, exist_ok=True)

    total_rounds = args.rounds
    actually_run = 0

    for rnd in range(start_round, total_rounds + 1):
        if rnd in completed:
            continue
        t0 = time.time()
        variant = generate_variant(full_best_code, rnd)
        gate_a = False
        gate_b = False
        sharpe_30d = -9.0
        pf = None
        trades_7d = 0
        avg_b = -9.0
        status = "rejected"
        err_msg = None

        if a_validator is None or b_validator is None:
            status = "skipped"
            err_msg = "validators unavailable"
        else:
            db = None
            try:
                try:
                    from database.connection import SessionLocal as _SL
                except ImportError:
                    try:
                        from backend.database.connection import SessionLocal as _SL
                    except ImportError:
                        _SL = None
                if _SL is not None:
                    try:
                        db = _SL()
                    except Exception as e:
                        logger.warning(f"DB SessionLocal failed: {e}")
                        db = None
                else:
                    db = None
                if db is None:
                    status = "skipped"
                    err_msg = "db unavailable"
                else:
                    try:
                        ga, info_a = a_validator(db, variant, threshold_30d_sharpe=args.threshold_a)
                    except TypeError:
                        ga, info_a = a_validator(db, variant)
                    except Exception as e:
                        logger.warning(f"round {rnd} a_validator error: {e}")
                        ga, info_a = False, {"30d": None, "7d": None}
                        err_msg = str(e)[:500]
                    try:
                        gb, avg_b_val, _oos = b_validator(db, variant, threshold_avg=args.threshold_b)
                    except TypeError:
                        gb, avg_b_val, _oos = b_validator(db, variant)
                    except Exception as e:
                        logger.warning(f"round {rnd} b_validator error: {e}")
                        gb, avg_b_val = False, -9.0
                        if err_msg is None:
                            err_msg = str(e)[:500]
                    gate_a = bool(ga)
                    gate_b = bool(gb)
                    avg_b = float(avg_b_val) if avg_b_val is not None else -9.0
                    sharpe_30d, pf, trades_7d, avg_b = _extract_metrics(info_a, avg_b)
                    if gate_a and gate_b:
                        pf_ok = (pf is None) or (pf > 0)
                        if pf_ok:
                            is_better = False
                            if sharpe_30d > best_sharpe:
                                if pf is None and best_pf is None:
                                    is_better = True
                                elif pf is None:
                                    is_better = sharpe_30d > best_sharpe + 0.05
                                elif best_pf is None:
                                    is_better = True
                                elif pf > best_pf:
                                    is_better = True
                                elif pf == best_pf and sharpe_30d > best_sharpe:
                                    is_better = True
                                else:
                                    if sharpe_30d > best_sharpe + 0.1:
                                        is_better = True
                            if is_better:
                                status = "promoted"
                                full_best_code = variant
                                best_sharpe = sharpe_30d
                                best_pf = pf
                                passed += 1
                            else:
                                status = "candidate"
                                passed += 1
                        else:
                            status = "rejected"
                    else:
                        status = "rejected"
            except Exception as e:
                logger.warning(f"round {rnd} unexpected error: {e}\n{traceback.format_exc()}")
                status = "error"
                err_msg = str(e)[:500]
            finally:
                if db is not None:
                    try:
                        db.close()
                    except Exception:
                        pass

        elapsed_ms = int((time.time() - t0) * 1000)
        code_preview = variant[:2000] if len(variant) > 2000 else variant
        preview_one = code_preview.replace("\n", "\\n")[:600]

        rec = {
            "round": rnd,
            "exchange": args.exchange,
            "symbols": symbols,
            "gate_a": bool(gate_a),
            "gate_b": bool(gate_b),
            "sharpe_30d": sharpe_30d,
            "pf": pf,
            "trades_7d": trades_7d,
            "avg_sharpe_b": avg_b,
            "elapsed_ms": elapsed_ms,
            "status": status,
            "code_preview": preview_one,
            "threshold_a": args.threshold_a,
            "threshold_b": args.threshold_b,
        }
        if err_msg:
            rec["error"] = err_msg

        try:
            with log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"failed to write log: {e}")

        actually_run += 1
        logger.info(f"round {rnd}/{total_rounds} status={status} gate_a={gate_a} gate_b={gate_b} sharpe={sharpe_30d:.3f} pf={pf} trades7d={trades_7d} avg_b={avg_b:.3f} {elapsed_ms}ms")

    try:
        best_code_path.write_text(full_best_code, encoding="utf-8")
    except Exception as e:
        logger.error(f"write best_code failed: {e}")

    summary = {
        "rounds": total_rounds,
        "completed": len(completed) + actually_run,
        "passed": passed,
        "best_sharpe": best_sharpe if best_sharpe != float("-inf") else None,
        "best_pf": best_pf,
        "best_code_path": str(best_code_path),
        "log_file": str(log_file),
        "exchange": args.exchange,
        "symbols": symbols,
    }
    try:
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error(f"write summary failed: {e}")

    logger.info(f"done rounds={total_rounds} passed={passed} best_sharpe={best_sharpe}")


if __name__ == "__main__":
    main()
