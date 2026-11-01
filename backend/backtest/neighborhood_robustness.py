"""鄰域穩健性測試（HANDOFF issue #4 / STRATEGY_CORE §四之二）。

對最優基因組的每個基因做 ±5%/±10%/±20% 擾動，各抽 N 個鄰居，
量化「這個點周圍的地形」而不是「這個點的分數」。

判準（STRATEGY_CORE §四之二）：擾動後中位數掉 30% 以上 → 針峰。

用法：
    .venv/bin/python backtest/neighborhood_robustness.py <genome.json|seed_log.jsonl>

genome 可為單一 dict，或 iterate_annual_ga 的 jsonl（取 fitness 最高者）。
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

from backtest.ab_validators import exploration_validator  # noqa: E402
from backtest.ga_annual import (  # noqa: E402
    BOUNDS,
    BOOL_GENES,
    ICT_INT_GENES,
    Genome,
    clamp,
    render_strategy,
)
from database.connection import SessionLocal  # noqa: E402

DEFAULT_SYMBOLS = [
    "BTC", "ETH", "SOL", "HYPE", "DOGE", "SUI",
    "ARB", "AVAX", "BNB", "NEAR", "AAVE", "TAO",
]

# 整數基因：擾動後四捨五入並 clamp 回 BOUNDS。
INT_GENES = {"ema_fast", "ema_slow", "rsi_buy_thr", "rsi_sell_thr", "leverage"} | set(ICT_INT_GENES)


def load_genome(genome_path: Path) -> Dict[str, Any]:
    """從單一 genome dict 或 GA jsonl 讀出基因組。"""
    path = Path(genome_path)
    text = path.read_text().strip()
    # 單一 JSON 物件（genome 或含 genome 的評分紀錄）優先；
    # GA 的 .jsonl 每行都是一個獨立的 JSON 物件，整份 text 不能一次 loads。
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        if "genome" in obj and isinstance(obj["genome"], dict):
            return obj["genome"]
        return obj

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    evals = [r for r in rows if isinstance(r.get("genome"), dict) and "fitness" in r]
    if not evals:
        raise SystemExit(f"{path}: 找不到含 genome+fitness 的紀錄")
    # jsonl 是完整歷史，同一基因組會重複出現；取 fitness 最高者即可。
    return max(evals, key=lambda r: r["fitness"])["genome"]


def perturb(genome: Dict[str, Any], rate: float, rng: random.Random) -> Dict[str, Any]:
    """對每個基因獨立施加 ±rate 的相對擾動（整數基因取整數步進）。"""
    out = dict(genome)
    for key, value in genome.items():
        if key in BOOL_GENES:
            continue  # 布林基因沒有「±5%」語義，保持原值
        if isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue
        lo, hi = BOUNDS.get(key, (None, None))
        factor = 1.0 + rng.uniform(-rate, rate)
        newval = value * factor
        if lo is not None:
            newval = max(lo, min(hi, newval))
        if key in INT_GENES:
            newval = int(round(newval))
        out[key] = newval
    return out


def evaluate(genome: Dict[str, Any], symbols: List[str], days: int,
             end_ms: int, max_triggers: Optional[int]) -> Optional[Dict[str, Any]]:
    db = SessionLocal()
    try:
        code = render_strategy(Genome(**genome), symbols=symbols)
        ok, info = exploration_validator(
            db, code, symbols=symbols, exchange="hyperliquid",
            fast_fail=True, max_triggers=max_triggers,
            backtest_days=days, end_time_ms=end_ms,
        )
        f = info.get("f365") or {}
        return {
            "gate": bool(ok),
            "pnl": float(f.get("pnl_percent") or 0.0),
            "dd": float(f.get("maxdd_pct") or 100.0),
            "sharpe": float(f.get("sharpe") or 0.0),
            "pf": float(f.get("pf") or 0.0),
            "trades": int(f.get("trades") or 0),
        }
    except Exception as exc:  # 單一鄰居失敗不該中斷整批
        print(f"    eval failed: {exc}", file=sys.stderr)
        return None
    finally:
        db.close()


def summarize(label: str, base: Dict[str, Any], samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    pnls = [s["pnl"] for s in samples]
    sharpes = [s["sharpe"] for s in samples]
    trades = [s["trades"] for s in samples]
    med = statistics.median(pnls)
    base_pnl = base["pnl"]
    # 保留率：中位 PnL 相對基準的比例。基準 <= 0 時用差值表示（比例會失真）。
    retention = (med / base_pnl * 100.0) if base_pnl > 0 else None
    p90 = sorted(pnls)[int(len(pnls) * 0.9) - 1] if len(pnls) >= 10 else max(pnls)
    return {
        "label": label,
        "n": len(samples),
        "pnl_median": med,
        "pnl_p90": p90,
        "sharpe_median": statistics.median(sharpes),
        "trades_median": statistics.median(trades),
        "Trades_min": min(trades),
        "Trades_max": max(trades),
        "retention_pct": retention,
        "gate_pass": sum(1 for s in samples if s["gate"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="鄰域穩健性測試")
    ap.add_argument("genome", type=Path, help="genome JSON 或 GA jsonl")
    ap.add_argument("--neighbors", type=int, default=30, help="每種擾動率的鄰居數")
    ap.add_argument("--rates", default="0.05,0.10,0.20", help="擾動率，逗號分隔")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--backtest-days", type=int, default=180)
    ap.add_argument("--end-time-ms", type=int, default=None)
    ap.add_argument("--max-triggers", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    rates = [float(r) for r in args.rates.split(",") if r.strip()]
    if args.end_time_ms is None:
        args.end_time_ms = int(time.time() * 1000)

    genome = load_genome(args.genome)
    print(f"genome: {json.dumps(genome, ensure_ascii=False)}")
    print(f"symbols={len(symbols)} days={args.backtest_days} neighbors/rate={args.neighbors}")
    print()

    t0 = time.time()
    base = evaluate(genome, symbols, args.backtest_days, args.end_time_ms, args.max_triggers)
    if base is None:
        print("基準評估失敗，中止。", file=sys.stderr)
        return 1
    print(f"基準: pnl {base['pnl']:+.2f}%  dd {base['dd']:.2f}%  "
          f"sharpe {base['sharpe']:.3f}  pf {base['pf']:.3f}  trades {base['trades']}  "
          f"gate={base['gate']}  ({time.time()-t0:.0f}s)")
    print()

    rows = []
    rng = random.Random(args.seed)
    for rate in rates:
        samples: List[Dict[str, Any]] = []
        print(f"擾動 ±{rate*100:.0f}% ...")
        for i in range(args.neighbors):
            nb = perturb(genome, rate, rng)
            res = evaluate(nb, symbols, args.backtest_days, args.end_time_ms, args.max_triggers)
            if res is not None:
                samples.append(res)
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{args.neighbors}  elapsed {time.time()-t0:.0f}s")
        if samples:
            rows.append(summarize(f"±{rate*100:.0f}%", base, samples))

    print()
    print("=" * 96)
    print(f"{'擾動':<8}{'n':<5}{'PnL中位':>12}{'p90':>10}{'Sharpe中位':>13}"
          f"{'交易中位':>12}{'保留率':>12}{'過gate':>8}")
    print("-" * 96)
    bp = f"{base['pnl']:+.2f}%"
    print(f"{'基準':<8}{1:<5}{bp:>12}{'—':>10}{base['sharpe']:>13.3f}"
          f"{base['trades']:>12}{'100%':>12}{str(base['gate']):>8}")
    for r in rows:
        ret = f"{r['retention_pct']:.1f}%" if r["retention_pct"] is not None else "n/a"
        print(f"{r['label']:<8}{r['n']:<5}{r['pnl_median']:>11.2f}%{r['pnl_p90']:>9.2f}%"
              f"{r['sharpe_median']:>13.3f}"
              f"{r['trades_median']:>12.1f}{ret:>12}{r['gate_pass']:>8}")
    print("=" * 96)

    # 判準：任何擾動率下中位 PnL 相對基準掉超過 30% → 針峰。
    verdict_pin = any(
        r["retention_pct"] is not None and r["retention_pct"] < 70.0 for r in rows
    )
    if base["pnl"] <= 0:
        print("注意：基準 PnL <= 0，保留率無意義，僅看絕對分佈。")
    print("判定：針峰（不穩健）" if verdict_pin else "判定：鄰域穩健度可接受")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
