#!/usr/bin/env python3
"""
Annual (365d) 策略迭代腳本

目標：月均 10% (年化 120%)，maxDD ≤ 5%
- Gate Annual: 365d pnl% ≥ 120%, maxDD% ≤ 5%, PF > 1.5, Sharpe > 0.5, Win > 45%, Trades ≥ 50, 5m 日均 ≥ 50
- 可選：保留 Gate A (30d) / Gate B (WF 3x30d) 作為額外穩健性檢查
"""

import argparse
import ast
import json
import logging
import pathlib
import sys
import time
import traceback
from typing import Optional, List

# Add backend to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from backtest.ab_validators import annual_validator, a_validator, b_validator

logger = logging.getLogger(__name__)

DEFAULT_BASELINE_CODE = """\
class Strategy:
    def should_trade(self, data):
        try:
            open_pos = data.positions or {{}}
        except Exception:
            open_pos = {{}}
        try:
            n_open = len(open_pos)
        except Exception:
            n_open = 0
        if n_open >= 3:
            return Decision(operation="hold", symbol="BTC", reason="max-3 guard")
        skip = set()
        try:
            skip = set(open_pos.keys())
        except Exception:
            skip = set()
        for symbol in ["BTC", "ETH", "SOL", "HYPE", "BNB", "AVAX", "NEAR", "SUI", "DOGE", "ARB", "AAVE", "TAO"]:
            if symbol in skip:
                continue
            price = data.get_price(symbol)
            if price == 0:
                continue
            klines = data.get_klines(symbol, "4h", 80)
            if len(klines) < 60:
                continue
            closes = [k.close for k in klines]
            if len(closes) < 55:
                continue
            e20 = sum(closes[-20:]) / 20
            e50 = sum(closes[-50:]) / 50
            e20p = sum(closes[-21:-1]) / 20
            e50p = sum(closes[-51:-1]) / 50
            if e20 <= e50 * 0.9985:
                continue
            if e20p > e50p * 1.0008:
                continue
            highs = [k.high for k in klines]
            hh = max(highs[-21:-1])
            if closes[-1] < hh * 0.98:
                continue
            return Decision(operation="buy", symbol=symbol, reason="ema-donch-4h", target_portion_of_balance=0.25, leverage=2, max_price=price*1.004, take_profit_price=price*1.08, stop_loss_price=price*0.92, sl_execution="market")
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


def generate_variant(base_code: str, round_idx: int) -> str:
    """Generate variant using existing mutators"""
    import re
    import hashlib
    
    # Use existing mutation strategies from iterate.py
    kind = (round_idx - 1) % 6  # 0..5
    
    if kind == 0:
        # param jitter
        seed = round_idx
        jitter_buy = 0.9985 + ((seed % 7) - 3) * 0.0005  # 0.9985 ± 0.0015
        jitter_sell = 1.0008 + ((seed % 5) - 2) * 0.0005  # 1.0008 ± 0.001
        portion = 0.25 + ((seed % 9) - 4) * 0.02
        portion = max(0.15, min(0.5, portion))
        lever = 2 + (seed % 2)  # 2 or 3
        
        new_code = re.sub(r"e50 \* 0\.\d+", f"e50 * {jitter_buy:.4f}", base_code)
        new_code = re.sub(r"e50p \* 1\.\d+", f"e50p * {jitter_sell:.4f}", new_code)
        new_code = re.sub(r"target_portion_of_balance\s*=\s*0\.\d+", f"target_portion_of_balance={portion:.2f}", new_code)
        new_code = re.sub(r"leverage\s*=\s*\d+", f"leverage={lever}", new_code, count=2)
        # TP/SL
        tp = 1.06 + (seed % 5) * 0.01  # 1.06..1.10
        sl = 0.96 - (seed % 5) * 0.01  # 0.96..0.92
        new_code = re.sub(r"take_profit_price=price\*1\.\d+", f"take_profit_price=price*{tp:.2f}", new_code, count=1)
        new_code = re.sub(r"stop_loss_price=price\*0\.\d+", f"stop_loss_price=price*{sl:.2f}", new_code, count=1)
        return new_code
        
    elif kind == 1:
        # Donchian threshold
        thr = 0.97 + (round_idx % 6) * 0.005  # 0.97..0.995
        new_code = re.sub(r"hh \* 0\.\d+", f"hh * {thr:.3f}", base_code)
        return new_code
        
    elif kind == 2:
        # RSI filter add/remove
        if "rsi" not in base_code.lower():
            # Add RSI filter
            m = re.search(r"(\n)([ ]+)closes = \[k\.close for k in klines\]", base_code)
            indent = m.group(2) if m else "            "
            rsi_block = (
                f"{indent}rsi_info = data.get_indicator(symbol, \"RSI14\", \"4h\")\n"
                f"{indent}rsi = rsi_info.get(\"value\", 50) if isinstance(rsi_info, dict) else 50\n"
                f"{indent}if rsi is None:\n"
                f"{indent}    rsi = 50\n"
            )
            target = f"{indent}closes = [k.close for k in klines]"
            new_code = base_code.replace(target, rsi_block + target, 1)
            rsi_th = 30 + (round_idx % 5) * 2  # 30..38
            new_code = new_code.replace(
                "if e20 <= e50 *",
                f"if rsi < {rsi_th} and e20 <= e50 *",
            )
            return new_code
        else:
            # Adjust RSI threshold
            th = 30 + (round_idx % 5) * 2
            new_code = re.sub(r"rsi < \d+", f"rsi < {th}", base_code)
            return new_code
            
    elif kind == 3:
        # Volume filter
        if "vol" not in base_code.lower():
            m = re.search(r"(\n)([ ]+)closes = \[k\.close for k in klines\]", base_code)
            indent = m.group(2) if m else "            "
            vol_block = (
                f"{indent}vol_info = data.get_indicator(symbol, \"VOLUME\", \"4h\")\n"
                f"{indent}vol = vol_info.get(\"value\", 0) if isinstance(vol_info, dict) else 0\n"
                f"{indent}vol_avg = sum(v for v in vol if v) / len(vol) if vol else 0\n"
            )
            target = f"{indent}closes = [k.close for k in klines]"
            new_code = base_code.replace(target, vol_block + target, 1)
            mult = 1.1 + (round_idx % 5) * 0.1  # 1.1..1.5
            new_code = new_code.replace(
                "if e20 <= e50 *",
                f"if vol > vol_avg * {mult:.2f} and e20 <= e50 *",
            )
            return new_code
        else:
            mult = 1.1 + (round_idx % 5) * 0.1
            new_code = re.sub(r"vol > vol_avg \* [0-9.]+", f"vol > vol_avg * {mult:.2f}", base_code)
            return new_code
            
    elif kind == 4:
        # TP/SL exploration
        tp = 1.06 + (round_idx % 8) * 0.01  # 1.06..1.13
        sl = 0.96 - (round_idx % 8) * 0.01  # 0.96..0.89
        new_code = re.sub(r"take_profit_price=price\*1\.\d+", f"take_profit_price=price*{tp:.2f}", base_code, count=1)
        new_code = re.sub(r"stop_loss_price=price\*0\.\d+", f"stop_loss_price=price*{sl:.2f}", new_code, count=1)
        return new_code
        
    else:
        # kind 5: EMA tolerance
        tol1 = 0.9985 + (round_idx % 5) * 0.001  # 0.9985..0.9995
        tol2 = 1.0008 + (round_idx % 5) * 0.001  # 1.0008..1.0018
        new_code = re.sub(r"e50 \* 0\.\d+", f"e50 * {tol1:.4f}", base_code)
        new_code = re.sub(r"e50p \* 1\.\d+", f"e50p * {tol2:.4f}", new_code)
        return new_code


def _sha256_short(s: str, n: int = 16) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def main():
    parser = argparse.ArgumentParser(description="Annual (365d) strategy iteration harness")
    parser.add_argument("--rounds", type=int, default=50, help="number of rounds")
    parser.add_argument("--exchange", type=str, default="hyperliquid", choices=["binance", "hyperliquid"])
    parser.add_argument("--symbols", type=str, default="BTC,ETH,SOL,HYPE,DOGE,SUI,ARB,AVAX,BNB,NEAR,AAVE,TAO", help="comma separated symbols")
    parser.add_argument("--baseline", type=str, default=None, help="path to baseline .py file")
    parser.add_argument("--log-file", type=str, default=None, help="log file path")
    parser.add_argument("--resume", action="store_true", help="resume from existing log")
    parser.add_argument("--best-code", type=str, default=None, help="output best code path")
    parser.add_argument("--summary", type=str, default=None, help="output summary path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    default_log = pathlib.Path(__file__).parent / "iteration_log_annual.jsonl"
    log_file = pathlib.Path(args.log_file) if args.log_file else default_log
    best_code_path = pathlib.Path(args.best_code) if args.best_code else (pathlib.Path(__file__).parent / "best_code_annual.py")
    summary_path = pathlib.Path(args.summary) if args.summary else (pathlib.Path(__file__).parent / "summary_annual.json")

    completed = set()
    best_code = _load_baseline(args.baseline)
    best_sharpe = float("-inf")
    best_pf = None
    best_pnl = float("-inf")
    passed = 0
    total_logged = 0

    best_any_sharpe = float("-inf")
    best_any_pf = None
    best_any_round = None
    best_any_code_hint = None
    if log_file.exists():
        if args.resume:
            try:
                with log_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            r = rec.get("round")
                            if isinstance(r, int):
                                completed.add(r)
                            s_any = rec.get("sharpe_365d")
                            if isinstance(s_any, (int, float)) and s_any > best_any_sharpe:
                                best_any_sharpe = s_any
                                best_any_pf = rec.get("pf")
                                best_any_round = r
                                best_any_code_hint = rec.get("code_preview")
                            if rec.get("status") == "promoted":
                                passed += 1
                        except Exception:
                            continue
                total_logged = len(completed)
                logger.info(f"resume: {len(completed)} rounds already logged, best_sharpe={best_sharpe}")
            except Exception as e:
                logger.warning(f"resume read failed: {e}")
        else:
            logger.info(f"log exists at {log_file}, use --resume to skip completed rounds or remove file")

    if not _validate_syntax(best_code):
        logger.warning("baseline syntax invalid, fallback to default")
        best_code = DEFAULT_BASELINE_CODE

    if args.resume and completed and best_code_path.exists():
        try:
            loaded = best_code_path.read_text(encoding="utf-8")
            if loaded.strip():
                import ast as _ast2
                _ast2.parse(loaded)
                best_code = loaded
        except Exception:
            pass

    if best_sharpe == float("-inf"):
        best_sharpe = -9.0
    if best_any_sharpe == float("-inf"):
        best_any_sharpe = -9.0

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
        
        # Duplicate check
        code_hash = _sha256_short(variant)
        duplicate = False
        if log_file.exists():
            try:
                with log_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line.strip())
                            if rec.get("code_hash") == code_hash:
                                duplicate = True
                                break
                        except Exception:
                            continue
            except Exception:
                pass
        
        if duplicate:
            gate_annual = False
            sharpe_365d = -9.0
            pf = None
            pnl_365d = 0.0
            maxdd_365d = 100.0
            trades_365d = 0
            status = "skipped_duplicate"
            err_msg = "duplicate variant"
            elapsed_ms = int((time.time() - t0) * 1000)
            code_preview = variant[:2000] if len(variant) > 2000 else variant
            preview_one = code_preview.replace("\n", "\\n")[:600]
            rec = {"round": rnd, "exchange": args.exchange, "symbols": symbols,
                   "gate_annual": False,
                   "sharpe_365d": sharpe_365d, "pnl_365d": float(pnl_365d),
                   "pf": pf, "maxdd_365d": maxdd_365d, "trades_365d": trades_365d,
                   "elapsed_ms": elapsed_ms, "status": status, "code_preview": preview_one,
                   "code_hash": code_hash}
            try:
                with log_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as ex:
                logger.error(f"failed to write log: {ex}")
            actually_run += 1
            logger.info(f"round {rnd}/{total_rounds} status={status} (duplicate skip)")
            continue
        
        gate_annual = False
        sharpe_365d = -9.0
        pf = None
        pnl_365d = 0.0
        maxdd_365d = 100.0
        trades_365d = 0
        status = "rejected"
        err_msg = None

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
                # Run Annual Validator
                try:
                    ga, info = annual_validator(db, variant, symbols=symbols)
                except TypeError:
                    try:
                        ga, info = annual_validator(db, variant)
                    except TypeError:
                        ga, info = annual_validator(db, variant)
                except Exception as e:
                    logger.warning(f"round {rnd} annual_validator error: {e}")
                    ga, info = False, {"365d": None}
                    err_msg = str(e)[:500]
                
                gate_annual = bool(ga)
                
                # Extract metrics
                if info.get("365d"):
                    r365 = info["365d"]
                    f365 = info.get("f365", {})
                    sharpe_365d = float(f365.get("sharpe", 0) or 0)
                    pnl_365d = float(f365.get("pnl_percent", 0) or 0)
                    maxdd_365d = float(f365.get("maxdd_pct", 100) or 100)
                    trades_365d = int(f365.get("trades", 0) or 0)
                    pf = info.get("pf_ok")
                    
        except Exception as e:
            logger.error(f"round {rnd} backtest failed: {e}")
            err_msg = str(e)[:500]
        
        if gate_annual:
            status = "promoted"
            passed += 1
            if sharpe_365d > best_sharpe:
                best_sharpe = sharpe_365d
                best_pf = pf
                best_pnl = pnl_365d
                full_best_code = variant
                # Write best code immediately
                try:
                    best_code_path.write_text(variant, encoding="utf-8")
                except Exception:
                    pass
        
        if sharpe_365d > best_any_sharpe:
            best_any_sharpe = sharpe_365d
            best_any_pf = pf
            best_any_round = rnd
            best_any_code_hint = variant[:2000]

        # Log
        elapsed_ms = int((time.time() - t0) * 1000)
        code_preview = variant[:2000] if len(variant) > 2000 else variant
        preview_one = code_preview.replace("\n", "\\n")[:600]
        rec = {"round": rnd, "exchange": args.exchange, "symbols": symbols,
               "gate_annual": gate_annual,
               "sharpe_365d": sharpe_365d, "pnl_365d": float(pnl_365d),
               "pf": pf, "maxdd_365d": maxdd_365d, "trades_365d": trades_365d,
               "elapsed_ms": elapsed_ms, "status": status, "code_preview": preview_one,
               "code_hash": code_hash}
        if err_msg:
            rec["err"] = err_msg
        try:
            with log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as ex:
            logger.error(f"failed to write log: {ex}")
        
        actually_run += 1
        logger.info(f"round {rnd}/{total_rounds} status={status} gate_annual={gate_annual} sharpe={sharpe_365d:+.3f} pnl={pnl_365d:+.2f}% dd={maxdd_365d:.1f}% pf={pf} trades={trades_365d}")

    # Write summary
    try:
        summary = {
            "rounds": total_rounds,
            "completed": len(completed) + actually_run,
            "actually_run": actually_run,
            "passed": passed,
            "best_sharpe": best_sharpe if best_sharpe != -9.0 else None,
            "best_pf": best_pf,
            "best_pnl": best_pnl,
            "best_any_sharpe": best_any_sharpe if best_any_sharpe != -9.0 else None,
            "best_any_pf": best_any_pf,
            "best_any_round": best_any_round,
            "best_code_path": str(best_code_path),
            "log_file": str(log_file),
            "exchange": args.exchange,
            "symbols": symbols,
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.error(f"failed to write summary: {e}")

    logger.info(f"=== Annual iteration complete ===")
    logger.info(f"total rounds: {total_rounds}, actually run: {actually_run}, passed: {passed}")
    logger.info(f"best sharpe: {best_sharpe:+.3f}, best pnl: {best_pnl:+.2f}%, best pf: {best_pf}")
    logger.info(f"best_any sharpe: {best_any_sharpe:+.3f} at round {best_any_round}")


if __name__ == "__main__":
    main()
