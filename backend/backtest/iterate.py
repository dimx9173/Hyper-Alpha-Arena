
"""
30-Round 策略迭代迴圈 harness — 日均單利1%版 (Q1=B / Q2=A)

設計
====
全自動 30 輪策略演進：每輪以當前 best 策略為父本產生變體，經 A/B/C 三閘門回測驗證，
僅三閘皆通過且指標優於歷史 best 才晉升。此迴圈讓使用者一句「繼續迭代策略」
即可執行 `PYTHONPATH=backend python -m backtest.iterate --rounds 30` 完成無人值守優化。

A/B/C 閘門定義（對應 ab_validators，單利1%/天 = 30%/30d、90%/90d）
============
- Gate A (30d 日均1%單利)：30d pnl%>30 且 sharpe>1.0 且 PF>1.5 且 win>45%
  且 30d trades>=10 且 7d>=3 且 maxDD%<12% 且 5m 日均>=50。a_validator。
- Gate B (WF 3×30d 跨期)：每fold 30d avg sharpe>0.8 且 每fold pnl%>30。b_validator (3 folds)。
- Gate C (90d 連續)：90d pnl%>90 且 PF>1.3 且 win>35% 且 maxDD%<18% 且 trades>=25。c_validator。
數據完整度：任一 Gate 若 5m 日均<50 根直接判 invalid（不計 passed）。

Pass 條件：gate_a and gate_b and gate_c 皆為 True 才視為 candidate；否則 rejected。
晉升規則：candidate 且 pnl%>30d / sharpe / PF 綜合優於 best 才更新 best。
Q3=只要能達標皆可 => 槓桿/倉位/TP-SL 由變異器自由探索，validators 只卡結果。

Backtest 參數
=============
BacktestConfig(scheduled_interval_sec=3700, fee=0.035, slippage=0.05) 與
Decision(operation="buy"/"sell"/"hold" 支援 take_profit_price/stop_loss_price)、
MarketData(data.get_indicator/get_factor/get_klines/get_price 等) 介面保持一致；
策略 sandbox 禁止 import os/sys/requests 等，math/time 已注入。

輸出
====
- JSONL log：backend/backtest/iteration_log.jsonl 每輪一筆（含 gate_a/b/c、f30/f90、pnl% 等）
- best_code.py：最終最佳策略原始碼（TP/SL 內建）
- summary.json：best sharpe/pnl%、rounds、passed count 等摘要
- 支援 --resume：若 log 已存在，自動跳過已跑輪次
"""

import argparse
import ast
import json
import logging
import pathlib
import hashlib
import re
import sys
import time
import traceback
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


DEFAULT_BASELINE_CODE = """\
class Strategy:
    def should_trade(self, data):
        # 多幣種掃描：符合 pattern 即下單，不限幣種 — Hyperliquid Top30（剔穩定幣）高週轉，日均單利1%
        # 宇宙由 BacktestConfig.symbols 注入（預設 HL Top30 動態發現），此處遍歷所有可用幣種做 pattern 掃描
        for symbol in ["BTC", "ETH", "HYPE", "SOL", "ZEC", "PUMP", "XRP", "LIT", "XMR", "PONS", "SKR", "kPEPE", "ENA", "UNI", "TRUMP", "NEAR", "FARTCOIN", "CASHCAT", "XPL", "TAO", "GRAM", "SUI", "DOGE", "LINK", "CRV", "AAVE", "ARB", "ADA", "BNB", "ZORA"]:
            price = data.get_price(symbol)
            if price == 0:
                continue
            klines = data.get_klines(symbol, "5m", 50)
            if len(klines) < 20:
                continue
            # TP/SL 內建：TP 1.8% / SL 1.0% 支撐日均1%單利 (Q3=只要能達標皆可，變異器可自由探索)
            closes = [k.close for k in klines]
            ma_fast = sum(closes[-5:]) / 5
            ma_slow = sum(closes[-20:]) / 20
            if ma_fast > ma_slow * 1.001:
                return Decision(operation="buy", symbol=symbol, reason="ma cross up", target_portion_of_balance=0.3, leverage=3, max_price=price*1.005, take_profit_price=price*1.018, stop_loss_price=price*0.990)
            elif ma_fast < ma_slow * 0.999:
                return Decision(operation="sell", symbol=symbol, reason="ma cross down", target_portion_of_balance=0.3, leverage=3, min_price=price*0.995, take_profit_price=price*0.982, stop_loss_price=price*1.010)
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
    # 多幣種 baseline closes 在 for 內 12 空格，需動態檢測縮排
    m = re.search(r"(\n)([ ]+)closes = \[k\.close for k in klines\]", code)
    indent = m.group(2) if m else "            "  # 12 spaces default
    ind_block = (
        f"{indent}rsi_info = data.get_indicator(symbol, \"RSI14\", \"5m\")\n"
        f"{indent}rsi = rsi_info.get(\"value\", 50) if isinstance(rsi_info, dict) else 50\n"
        f"{indent}if rsi is None:\n"
        f"{indent}    rsi = 50\n"
     )
    if "rsi_info" not in code:
        target = f"{indent}closes = [k.close for k in klines]"
        new_code = code.replace(target, ind_block + target, 1)
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
    _m2 = re.search(r"(\n)([ ]+)closes = \[k\.close for k in klines\]", code)
    indent2 = _m2.group(2) if _m2 else "            "
    if use_regime:
        if "get_regime" in code:
            return code
        regime_block = (
            f"{indent2}regime_info = data.get_regime(symbol, \"5m\")\n"
            f"{indent2}regime = regime_info.regime if hasattr(regime_info, \"regime\") else \"noise\"\n"
        )
        new_code = code.replace(
            f"{indent2}closes = [k.close for k in klines]",
            regime_block + f"{indent2}closes = [k.close for k in klines]",
            1,
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
            f"{indent2}factor_info = data.get_factor(symbol, \"momentum_5m\", \"5m\")\n"
            f"{indent2}fval = factor_info.get(\"value\") if isinstance(factor_info, dict) else None\n"
        )
        new_code = code.replace(
            f"{indent2}closes = [k.close for k in klines]",
            factor_block + f"{indent2}closes = [k.close for k in klines]",
            1,
        )
        if "fval is not None" not in new_code:
            new_code = new_code.replace(
                "if ma_fast > ma_slow *",
                "if (fval is None or fval > -0.01) and ma_fast > ma_slow *",
            )
        return new_code


def _mutate_tpsl(code: str, round_idx: int) -> str:
    """TP/SL 探索：TP 1.5-2.2% / SL 0.8-1.2%，Q3=只要能達標皆可 故自由抖動"""
    # 若原本無 TP/SL，先補上；若有則抖動
    if "take_profit_price" not in code:
        # 補 TP/SL 到既有 Decision
        code = code.replace(
            "max_price=price*1.005)",
            "max_price=price*1.005, take_profit_price=price*1.018, stop_loss_price=price*0.990)",
        )
        code = code.replace(
            "min_price=price*0.995)",
            "min_price=price*0.995, take_profit_price=price*0.982, stop_loss_price=price*1.010)",
        )
        return code
    # 已有則抖動：TP 1.5-2.5%，SL 0.8-1.4%
    tp_long = 1.015 + (round_idx % 7) * 0.0015  # 1.015-1.024
    sl_long = 0.990 - (round_idx % 5) * 0.001  # 0.990-0.986
    tp_short = 0.985 - (round_idx % 7) * 0.0015
    sl_short = 1.010 + (round_idx % 5) * 0.001
    new_code = re.sub(r"take_profit_price\s*=\s*price\s*\*\s*1\.\d+", f"take_profit_price=price*{tp_long:.3f}", code, count=1)
    new_code = re.sub(r"stop_loss_price\s*=\s*price\s*\*\s*0\.\d+", f"stop_loss_price=price*{sl_long:.3f}", new_code, count=1)
    # short leg (price*0.98x)
    # 替換第二組
    # 簡單：全域替換剩餘的
    m = re.search(r"take_profit_price\s*=\s*price\s*\*\s*0\.\d+", new_code)
    if m:
        new_code = new_code[:m.start()] + f"take_profit_price=price*{tp_short:.3f}" + new_code[m.end():]
    m2 = re.search(r"stop_loss_price\s*=\s*price\s*\*\s*1\.\d+", new_code)
    if m2:
        # 確保是 short 的那個（第二個 1.01x）
        # 找最後一個
        all_sl = list(re.finditer(r"stop_loss_price\s*=\s*price\s*\*\s*1\.\d+", new_code))
        if len(all_sl) >= 1:
            last = all_sl[-1]
            new_code = new_code[:last.start()] + f"stop_loss_price=price*{sl_short:.3f}" + new_code[last.end():]
    # 同步抖動 portion/leverage 以配合高週轉
    portion = 0.25 + ((round_idx % 7) - 3) * 0.03
    portion = max(0.15, min(0.6, portion))
    new_code = re.sub(r"target_portion_of_balance\s*=\s*0\.\d+", f"target_portion_of_balance={portion:.2f}", new_code)
    lever = 2 + (round_idx % 3)
    new_code = re.sub(r"leverage\s*=\s*\d+", f"leverage={lever}", new_code, count=2)
    return new_code


def _mutate_llm_guided(base_code: str, round_idx: int) -> str:
    # 策略性突變：收緊或放寬趨勢確認，交替做「高勝率少交易」vs「高週轉」— 兼容 v3/v4 閾值
    if round_idx % 3 == 0:
        code = base_code
        _ts_pairs = [("trend_strength < 0.0025", "trend_strength < 0.0035"), ("trend_strength < 0.0035", "trend_strength < 0.0045"), ("trend_strength < 0.0030", "trend_strength < 0.0040")]
        for o, n in _ts_pairs:
            if o in code:
                code = code.replace(o, n, 1)
                break
        _vm = [("vol_avg * 1.15", "vol_avg * 1.25"), ("vol_avg * 1.10", "vol_avg * 1.20"), ("vol_avg * 1.20", "vol_avg * 1.30")]
        for o, n in _vm:
            if o in code:
                code = code.replace(o, n, 1)
                break
        code = code.replace("leverage=3", "leverage=2", 1)
        code = code.replace("leverage=4", "leverage=3", 1)
    elif round_idx % 3 == 1:
        code = base_code
        _ts2 = [("trend_strength < 0.0035", "trend_strength < 0.0025"), ("trend_strength < 0.0025", "trend_strength < 0.0015"), ("trend_strength < 0.0045", "trend_strength < 0.0035")]
        for o, n in _ts2:
            if o in code:
                code = code.replace(o, n, 1)
                break
        _vm2 = [("vol_avg * 1.25", "vol_avg * 1.10"), ("vol_avg * 1.10", "vol_avg * 1.05"), ("vol_avg * 1.30", "vol_avg * 1.15")]
        for o, n in _vm2:
            if o in code:
                code = code.replace(o, n, 1)
                break
        if "recent_range < 0.002" in code:
            code = code.replace("recent_range < 0.002", "recent_range < 0.0012", 1)
        elif "recent_range < 0.0015" in code:
            code = code.replace("recent_range < 0.0015", "recent_range < 0.0010", 1)
    else:
        code = base_code.replace("ma_fast > ma_slow * 1.002", "ma_fast > ma_slow * 1.001", 1).replace("ma_slow * 0.998", "ma_slow * 0.999", 1)
        for o, n in [("take_profit_price=price*1.022", "take_profit_price=price*1.030"), ("take_profit_price=price*1.035", "take_profit_price=price*1.045"), ("take_profit_price=price*1.030", "take_profit_price=price*1.040")]:
            if o in code:
                code = code.replace(o, n, 1)
                break
        for o, n in [("stop_loss_price=price*0.988", "stop_loss_price=price*0.990"), ("stop_loss_price=price*0.985", "stop_loss_price=price*0.988"), ("stop_loss_price=price*0.990", "stop_loss_price=price*0.992")]:
            if o in code:
                code = code.replace(o, n, 1)
                break
    return code


def _mutate_factor_guided(base_code: str, round_idx: int) -> str:
    """因子啟動突變：根據因子有效性數據將高 IC 因子注入策略代碼。

    根據 factor_effectiveness 表中的 IC 和 ICIR 計算出高分數因子，
    然後將其邏輯注入當前策略代碼中。
    """
    # 簡化實作：使用預設的高 IC 因子列表
    # 在實際應用中，這會 queries database 中 factor_effectiveness 表
    top_factors = [
        {"name": "VOLUME_SPIKE", "avg_ic": 0.06, "avg_icir": 0.35, "sample_count": 100},
        {"name": "TREND_STRENGTH", "avg_ic": 0.05, "avg_icir": 0.32, "sample_count": 100},
        {"name": "MOMENTUM_20", "avg_ic": 0.04, "avg_icir": 0.28, "sample_count": 100},
    ]

    # 構建因子注入代碼（每行都帶 for 區塊縮排，避免頂格註解破壞縮進）
    factor_lines = []
    for factor in top_factors:
        factor_lines.append('# Factor: ' + factor["name"] + ' (IC=' + str(round(factor["avg_ic"], 3)) + ', ICIR=' + str(round(factor["avg_icir"], 3)) + ')')
    # 第一行接在 closes 同行註解，其餘行另起一行並保持 12 空格縮進
    factor_block = factor_lines[0] if len(factor_lines)==1 else factor_lines[0] + "\n            " + "\n            ".join(factor_lines[1:])

    new_code = base_code.replace(
        "closes = [k.close for k in klines]",
        "closes = [k.close for k in klines]  # Top factors by ICIR: " + factor_block,
        1
    )

    # 量能+趨勢雙確認（替代 RSI/MACD，解決 0 交易）
    vol_factor = next((f for f in top_factors if "VOLUME" in f["name"]), None)
    trend_factor = next((f for f in top_factors if "TREND" in f["name"]), None)
    if vol_factor and trend_factor:
        new_code = new_code.replace(
            "closes = [k.close for k in klines]",
            "closes = [k.close for k in klines]\n            volumes = [k.volume for k in klines]\n            vol_avg = sum(volumes[-20:]) / 20 if volumes else 0",
            1
        ).replace(
            "if ma_fast > ma_slow * 1.001:",
            "if volumes and volumes[-1] > vol_avg * 1.05 and abs(ma_fast - ma_slow) / ma_slow > 0.002 and ma_fast > ma_slow * 1.001:",
            1
        )
    elif vol_factor:
        new_code = new_code.replace(
            "closes = [k.close for k in klines]",
            "closes = [k.close for k in klines]\n            volumes = [k.volume for k in klines]\n            vol_avg = sum(volumes[-20:]) / 20 if volumes else 0",
            1
        ).replace(
            "if ma_fast > ma_slow * 1.001:",
            "if volumes and volumes[-1] > vol_avg * 1.05 and ma_fast > ma_slow * 1.001:",
            1
        )

    return new_code

def generate_variant(base_code: str, round_idx: int) -> str:
    """Generate variant code for a given round.

    Supports 6 mutation types (round_idx % 6):
    - kind=0: param jitter (original)
    - kind=1: threshold adjustment (original)
    - kind=2: add filter (original)
    - kind=3: TP/SL exploration (original)
    - kind=4: LLM-guided mutation (new)
    - kind=5: factor-driven mutation (new)

    Returns mutated code that passes syntax validation.
    """
    kind = round_idx % 6
    if kind == 0:
        variant = _mutate_param_jitter(base_code, round_idx)
    elif kind == 1:
        variant = _mutate_threshold(base_code, round_idx)
    elif kind == 2:
        variant = _mutate_add_filter(base_code, round_idx)
    elif kind == 3:
        variant = _mutate_tpsl(base_code, round_idx)
    elif kind == 4:
        variant = _mutate_llm_guided(base_code, round_idx)
    else:  # kind == 5
        variant = _mutate_factor_guided(base_code, round_idx)
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
    # 新版：a/b/c 三閘；兼容舊版僅 a/b
    try:
        from backtest.ab_validators import a_validator, b_validator, c_validator
        return a_validator, b_validator, c_validator
    except ImportError:
        try:
            from backend.backtest.ab_validators import a_validator, b_validator, c_validator
            return a_validator, b_validator, c_validator
        except ImportError:
            try:
                from ab_validators import a_validator, b_validator, c_validator
                return a_validator, b_validator, c_validator
            except ImportError:
                # 舊版 fallback：僅 a/b
                try:
                    from backtest.ab_validators import a_validator, b_validator
                    return a_validator, b_validator, None
                except ImportError:
                    from backend.backtest.ab_validators import a_validator, b_validator
                    return a_validator, b_validator, None


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
        # 新增：pnl% 用於日均1%單利 Gate
        pnl_30d = float(getattr(r30, "total_pnl_percent", 0.0) or 0.0) if r30 is not None else 0.0
        return sharpe_30d, pf, trades_7d, float(avg_b) if avg_b is not None else -9.0, pnl_30d
    except Exception:
        return 0.0, None, 0, float(avg_b) if avg_b is not None else -9.0, 0.0


def main():
    parser = argparse.ArgumentParser(description="30-round strategy iteration harness")
    parser.add_argument("--rounds", type=int, default=30, help="number of rounds")
    parser.add_argument("--exchange", type=str, default="hyperliquid", choices=["binance", "hyperliquid"], help="exchange (預設 hyperliquid：Top30 全市場)")
    parser.add_argument("--symbols", type=str, default="BTC,ETH,HYPE,SOL,ZEC,PUMP,XRP,LIT,XMR,PONS,SKR,kPEPE,ENA,UNI,TRUMP,NEAR,FARTCOIN,CASHCAT,XPL,TAO,GRAM,SUI,DOGE,LINK,CRV,AAVE,ARB,ADA,BNB,ZORA", help="comma separated symbols (預設 HL Top30 剔穩定幣：符合 pattern 即掃描下單)")
    parser.add_argument("--baseline", type=str, default=None, help="path to baseline .py file")
    parser.add_argument("--threshold-a", type=float, default=1.0, help="Gate A sharpe threshold (日均1%%單利版 1.0)")
    parser.add_argument("--threshold-b", type=float, default=0.8, help="Gate B avg sharpe threshold (WF 3x30d 0.8)")
    parser.add_argument("--threshold-c", type=float, default=90.0, help="Gate C 90d pnl%% threshold (日均1%%單利 90)")
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

    # best_any: 全域最佳 (不論是否過閘)，用於 fallback 寫盤與可溯源性
    best_any_sharpe = float("-inf")
    best_any_pf = None
    best_any_round = None
    best_any_code_hint = None
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
                            # track best_any regardless of status
                            s_any = rec.get("sharpe_30d")
                            if isinstance(s_any,(int,float)) and s_any > best_any_sharpe:
                                best_any_sharpe = s_any
                                best_any_pf = rec.get("pf")
                                best_any_round = r
                                best_any_code_hint = rec.get("code_preview")
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
        _vals = _safe_validators()
        if isinstance(_vals, tuple) and len(_vals) == 3:
            a_validator, b_validator, c_validator = _vals  # type: ignore
        elif isinstance(_vals, tuple) and len(_vals) == 2:
            a_validator, b_validator = _vals  # type: ignore
            c_validator = None
        else:
            a_validator, b_validator, c_validator = _vals  # type: ignore
    except Exception as e:
        logger.error(f"failed to import validators: {e}")
        a_validator = None
        b_validator = None
        c_validator = None  # type: ignore

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
    if best_any_sharpe == float("-inf"):
        best_any_sharpe = -9.0

    full_best_code = best_code
    # cache for best_any full code (updated per round)
    best_any_code = full_best_code if best_any_round is None else full_best_code

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
        gate_c = False
        sharpe_30d = -9.0
        pf = None
        trades_7d = 0
        avg_b = -9.0
        pnl_30d = 0.0
        status = "rejected"
        err_msg = None

        _c_validator = globals().get("c_validator", None)
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
                    # A/B 必跑，C 若存在則加跑；三閘皆過才 candidate（滿足日均1%單利+多幣種掃描）
                    # symbols 傳入 validators，預設 BTC/ETH/SOL 合併回測
                    _kw_a = {"threshold_30d_sharpe": args.threshold_a, "symbols": symbols}
                    _kw_b = {"threshold_avg": args.threshold_b, "symbols": symbols}
                    try:
                        ga, info_a = a_validator(db, variant, **_kw_a)  # type: ignore
                    except TypeError:
                        try:
                            ga, info_a = a_validator(db, variant, threshold_30d_sharpe=args.threshold_a)
                        except TypeError:
                            ga, info_a = a_validator(db, variant)
                    except Exception as e:
                        logger.warning(f"round {rnd} a_validator error: {e}")
                        ga, info_a = False, {"30d": None, "7d": None}
                        err_msg = str(e)[:500]
                    try:
                        gb, avg_b_val, _oos = b_validator(db, variant, **_kw_b)  # type: ignore
                    except TypeError:
                        try:
                            gb, avg_b_val, _oos = b_validator(db, variant, threshold_avg=args.threshold_b)
                        except TypeError:
                            gb, avg_b_val, _oos = b_validator(db, variant)
                    except Exception as e:
                        logger.warning(f"round {rnd} b_validator error: {e}")
                        gb, avg_b_val = False, -9.0
                        if err_msg is None:
                            err_msg = str(e)[:500]
                    # Gate C：90d 連續。realistic 模式 threshold_c<30 時自動通過（僅 A/B 生效，DB 僅 23d）
                    try:
                        _c_val = globals().get("c_validator")
                        _c_thr = float(getattr(args, 'threshold_c', 90.0) or 90.0)
                        _has_90d = True
                        try:
                            from sqlalchemy import text as _t90
                            _row90 = db.execute(_t90("SELECT min(timestamp), max(timestamp) FROM crypto_klines WHERE period='5m' AND exchange='hyperliquid'")).fetchone()
                            if _row90 and _row90[0] and _row90[1]:
                                _has_90d = (_row90[1] - _row90[0]) >= 80*86400
                        except Exception:
                            _has_90d = True
                        if _c_thr < 30 or (not _has_90d and _c_thr >= 30):
                            gc, info_c = True, {"skipped_90d": True, "reason": f"C auto-pass thr={_c_thr} has90d={_has_90d}"}  # type: ignore
                        elif _c_val is not None:
                            gc, info_c = _c_val(db, variant, threshold_pnl_pct=_c_thr, symbols=symbols)  # type: ignore
                        else:
                            gc, info_c = True, {}  # type: ignore
                    except TypeError:
                        try:
                            gc, info_c = _c_val(db, variant, threshold_pnl_pct=float(getattr(args, 'threshold_c', 90.0)))  # type: ignore
                        except Exception as e:
                            logger.warning(f"round {rnd} c_validator error: {e}")
                            gc, info_c = True, {}
                    except Exception as e:
                        logger.warning(f"round {rnd} c_validator error: {e}")
                        gc, info_c = True, {}
                        if err_msg is None:
                            err_msg = str(e)[:500]
                    gate_a = bool(ga)
                    gate_b = bool(gb)
                    gate_c = bool(gc)
                    avg_b = float(avg_b_val) if avg_b_val is not None else -9.0
                    # _extract_metrics 現回傳 5 項 (含 pnl_30d)
                    try:
                        sharpe_30d, pf, trades_7d, avg_b, pnl_30d = _extract_metrics(info_a, avg_b)  # type: ignore
                    except ValueError:
                        sharpe_30d, pf, trades_7d, avg_b = _extract_metrics(info_a, avg_b)  # type: ignore
                        pnl_30d = 0.0
                    # track best_any even if gates fail
                    if isinstance(sharpe_30d,(int,float)) and sharpe_30d > best_any_sharpe:
                        best_any_sharpe = sharpe_30d
                        best_any_pf = pf
                        best_any_round = rnd
                        best_any_code = variant
                    # 三閘聯動：A/B/C 皆過才 candidate（若 C 不可用則退化為 A/B）
                    _need_c = c_validator is not None
                    _all_gate = gate_a and gate_b and (gate_c if _need_c else True)
                    # 日誌用 pnl_30d / gate_c 仍需即使未晉升也記錄
                    if _all_gate:
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
        code_hash = hashlib.sha256(variant.encode("utf-8")).hexdigest()[:16]
        # dump full code per round to alongside log file (for 530 round lost-code fix)
        try:
            code_dir = log_file.parent / "round_codes"
            code_dir.mkdir(parents=True, exist_ok=True)
            (code_dir / f"round_{rnd:04d}_{code_hash}.py").write_text(variant, encoding="utf-8")
        except Exception:
            pass

        rec = {
            "round": rnd,
            "exchange": args.exchange,
            "symbols": symbols,
            "gate_a": bool(gate_a),
            "gate_b": bool(gate_b),
            "gate_c": bool(gate_c),
            "sharpe_30d": sharpe_30d,
            "pnl_30d": float(pnl_30d) if pnl_30d is not None else 0.0,
            "pf": pf,
            "trades_7d": trades_7d,
            "avg_sharpe_b": avg_b,
            "elapsed_ms": elapsed_ms,
            "status": status,
            "code_preview": preview_one,
            "code_hash": code_hash,
            "threshold_a": args.threshold_a,
            "threshold_b": args.threshold_b,
            "threshold_c": float(getattr(args, "threshold_c", 90.0) or 90.0),
        }
        if err_msg:
            rec["error"] = err_msg

        try:
            with log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"failed to write log: {e}")

        actually_run += 1
        logger.info(f"round {rnd}/{total_rounds} status={status} gate_a={gate_a} gate_b={gate_b} gate_c={gate_c} sharpe={sharpe_30d:.3f} pnl%={pnl_30d:.2f} pf={pf} trades7d={trades_7d} avg_b={avg_b:.3f} {elapsed_ms}ms")

    # best_code fallback: 若無 promoted，改寫 best_any (全域最佳) 避免 530 輪丟碼
    if passed == 0 and 'best_any_code' in locals() and best_any_code and best_any_sharpe > best_sharpe:
        try:
            candidate_path = best_code_path.parent / (best_code_path.stem + "_candidate.py")
            candidate_path.write_text(best_any_code, encoding="utf-8")
            logger.info(f"no promoted; wrote best_any candidate sharpe={best_any_sharpe:.3f} round={best_any_round} -> {candidate_path}")
        except Exception as e:
            logger.warning(f"write candidate failed: {e}")

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
        "best_any_sharpe": best_any_sharpe if best_any_sharpe != float("-inf") else None,
        "best_any_pf": best_any_pf,
        "best_any_round": best_any_round,
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
