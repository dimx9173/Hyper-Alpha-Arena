#!/usr/bin/env python3
"""Refit 循環：一次輪迴 = 評估現任 → 檢視衰退 → GA 搜尋 → 鄰域驗證 → promote。

設計（見 STRATEGY_CORE §八：策略會失效，所以 refit 才是系統主功能）：
  1. 若 ledger 有現任 champion，在**新視窗**（--eval-days）重新評估。
     衰退差拼進 ledger.reeval，累積成「策略會失效」的實證資料。
  2. seed 池 = 現任 + 上任 N 個 champion，避免每次都從隨機基因從零開始。
  3. 跑 GA（--generations/--pop-size），seed 現任基因變異後進入初代。
  4. 最佳者做鄰域穩健性檢查（小鄰居數，快速版），通過才 promote 為新任。
     鄰域判定針峰也不丟掉 GA 成果：仍然 promote 但標 note="needle - invest at risk"，
     讓後續 reeval 決定它的命運，而不是靠一次測試判死刑。

用法範例：
  .venv/bin/python backtest/refit_cycle.py \\
      --period 2h --window-days 365 --eval-days 90 \\
      --generations 12 --pop-size 12 --workers 4 \\
      --ledger /tmp/champion_ledger.jsonl --symbols BTC,ETH,SOL \\
      --neighbors 10 --note "weekly refit"
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

from backtest.champion_ledger import ChampionLedger  # noqa: E402
from backtest.ga_annual import Genome, evaluate_individual  # noqa: E402
from database.connection import SessionLocal  # noqa: E402

DEFAULT_SYMBOLS = ("BTC,ETH,SOL,HYPE,DOGE,SUI,ARB,AVAX,BNB,NEAR,AAVE,TAO").split(",")


def _record_kind(record: dict) -> str:
    """ledger 記錄的策略型別。舊記錄沒有 kind，一律視為 genome。"""
    kind = record.get("kind")
    return str(kind) if kind else "genome"


def _payload(record: dict) -> dict:
    """取記錄的策略定義：spec 記錄用 spec，genome 記錄用 genome。"""
    if _record_kind(record) == "spec":
        return dict(record.get("spec") or {})
    return dict(record.get("genome") or {})


def _metrics(res: dict) -> dict:
    return {
        "pnl_percent": float(res.get("pnl_365d") or 0.0),
        "maxdd_pct": float(res.get("maxdd_365d") or 100.0),
        "sharpe": float(res.get("sharpe_365d") or 0.0),
        "pf": float(res.get("pf") or 0.0),
        "trades": int(res.get("trades_365d") or 0),
        "gate": bool(res.get("gate_annual")),
    }


def _reeval_champion(
    led: ChampionLedger, champ: dict, symbols: list, period: str,
    eval_days: int, end_ms: int, max_triggers: int,
    bars_5m: Optional[dict] = None,
) -> None:
    """Reevaluate the champion on a shorter window, dispatching on record kind."""
    if _record_kind(champ) == "spec":
        from backtest.strategy_spec import evaluate_lgbm_spec

        if not bars_5m:
            raise SystemExit("spec 記錄的 reeval 需要 bars_5m（5m K 線）")
        res = evaluate_lgbm_spec(
            _payload(champ), bars_5m, symbols=symbols,
            generation=1, total_generations=1,
            backtest_days=eval_days, end_time_ms=end_ms,
        )
    else:
        res = evaluate_individual(
            _payload(champ), symbols, "hyperliquid",
            generation=1, total_generations=1,
            backtest_days=eval_days, end_time_ms=end_ms,
            kline_period=period,
        )
    entry = led.reeval(
        champ, _metrics(res), float(res.get("fitness") or 0.0),
        eval_days, end_ms,
        note=f"refit reeval on {period}/{eval_days}d ({_record_kind(champ)})",
    )
    print(f"  現任 reeval：PnL {entry['metrics']['pnl_percent']:+.2f}%  "
          f"decay {entry['decay_pnl']:+.2f}%  gate={entry['metrics']['gate']}")


def _run_ga(tmp_seed_log: Path, args: argparse.Namespace, end_ms: int) -> list:
    """跑 iterate_annual_ga 子程序；完成後掃 log 取最佳 fitness。"""
    cmd = [
        sys.executable, str(_BACKEND / "backtest" / "iterate_annual_ga.py"),
        "--generations", str(args.generations),
        "--pop-size", str(args.pop_size),
        "--elite", str(args.elite),
        "--mut-rate", str(args.mut_rate),
        "--workers", str(args.workers),
        "--exchange", "hyperliquid",
        "--symbols", ",".join(args.symbols),
        "--backtest-days", str(args.window_days),
        "--end-time-ms", str(end_ms),
        "--exploration-gens", str(max(1, args.generations // 2)),
        "--early-stop", "4",
        "--kline-period", args.period,
        "--log-file", str(tmp_seed_log),
    ]
    if args.seed_log_inline:
        cmd += ["--seed-log", args.seed_log_inline]
    print("  GA:", " ".join(cmd[2:8]), "...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout[-1500:])
    if proc.returncode != 0:
        print(proc.stderr[-2000:])
        raise SystemExit(f"GA 子程序失敗 rc={proc.returncode}")
    rows = [
        json.loads(line) for line in tmp_seed_log.read_text().splitlines() if line.strip()
    ]
    evals = [r for r in rows if "fitness" in r and isinstance(r.get("genome"), dict)]
    if not evals:
        raise SystemExit("GA 沒有產出評估紀錄")
    return evals


def _best(rows: list) -> dict:
    return max(rows, key=lambda r: r.get("fitness") or -1e18)


def _load_bars_5m(path: str) -> dict:
    """讀 spec 模式的 5m K 線：{symbol: [{timestamp, open, ...}]} -> DataFrame。

    timestamp 是 period 起始秒（與 _query_klines_from_db 同語義）。
    """
    import pandas as _pd

    raw = json.loads(Path(path).read_text())
    out = {}
    for symbol, rows in raw.items():
        frame = _pd.DataFrame(rows)
        if "timestamp" in frame.columns:
            idx = _pd.to_datetime(frame["timestamp"].astype(int), unit="s", utc=True)
            frame = frame.drop(columns=["timestamp"]).set_index(idx)
            frame.index.name = "open_time"
        out[str(symbol)] = frame
    return out


def _default_lgbm_spec() -> dict:
    from backtest.strategy_spec import FILTER_DEFAULTS

    return {
        "filters": [{
            "kind": "lgbm_gate",
            "params": dict(FILTER_DEFAULTS["lgbm_gate"]),
        }],
        "risk": {"tp_pct": 0.08, "sl_pct": 0.04, "portion": 0.25,
                 "leverage": 2, "max_positions": 3},
        "logic": "all_pass",
    }


def _run_spec_refit(args: argparse.Namespace) -> int:
    """spec 模式的一輪：現任 reeval -> 候選評估 -> promote。

    不跑 GA、不跑鄰域子程序。ledger 用 kind=spec 區分，genome 鍵留空 dict。
    """
    from backtest.strategy_spec import StrategySpec, evaluate_lgbm_spec, is_lgbm_spec

    if not args.bars_5m_json:
        raise SystemExit("spec 模式需要 --bars-5m-json")
    bars = _load_bars_5m(args.bars_5m_json)
    led = ChampionLedger(args.ledger)
    end_ms = int(time.time() * 1000)
    t0 = time.time()
    print(f"[refit:spec] window={args.window_days}d symbols={list(bars)}")

    champ = led.current()
    if champ is not None and _record_kind(champ) == "spec":
        print(f"[refit:spec] 現任 {champ.get('genome_id')} reeval…")
        _reeval_champion(led, champ, args.symbols, "5m",
                         args.eval_days, end_ms, args.max_triggers,
                         bars_5m=bars)
    elif champ is not None:
        print("[refit:spec] 現任是 genome 記錄，跳過 reeval（跨型別首輪）")
    else:
        print("[refit:spec] ledger 無現任，跳過 reeval（首輪）")

    if args.spec_json:
        candidate = json.loads(Path(args.spec_json).read_text())
    elif champ is not None and _record_kind(champ) == "spec":
        candidate = _payload(champ)
    else:
        candidate = _default_lgbm_spec()
    spec = StrategySpec.from_dict(candidate)
    if not is_lgbm_spec(spec):
        raise SystemExit("spec 模式只接受含 lgbm_gate 的 spec")
    res = evaluate_lgbm_spec(
        candidate, bars, symbols=args.symbols or list(bars),
        generation=1, total_generations=1,
        backtest_days=args.window_days, end_time_ms=end_ms,
    )
    promo = res.get("promotion") or {}
    print(f"[refit:spec] 候選 gate={res['gate_annual']} "
          f"pnl {res.get('pnl_365d', 0):+.2f}%  trades {res.get('trades_365d')}  "
          f"fitness {res.get('fitness', 0):+.3f}  "
          f"promotion={promo.get('passed')} "
          f"(dsr={((promo.get('dsr') or {}).get('dsr') if isinstance(promo.get('dsr'), dict) else '?')})")
    if not res.get("promote_ok"):
        # fail-closed：四關任一不通過就不記 promote，只記 reeval 供後續追蹤。
        # 避免垃圾模型靠回測門檻混進 ledger。拒絕原因列出具體卡關，
        # 而非泛用字串（便於輪次間追蹤是哪一關在擋）。
        failed = []
        _dsr = promo.get("dsr") if isinstance(promo.get("dsr"), dict) else {}
        _mc = promo.get("monte_carlo") if isinstance(promo.get("monte_carlo"), dict) else {}
        _fr = promo.get("friction") if isinstance(promo.get("friction"), dict) else {}
        _mo = promo.get("monthly") if isinstance(promo.get("monthly"), dict) else {}
        if not (_dsr.get("dsr", 0.0) >= 0.95):
            failed.append(f"dsr={_dsr.get('dsr', 0.0):.3f}")
        if not (_mc.get("mdd_95th", 1.0) <= 0.20 and _mc.get("risk_of_ruin", 1.0) < 0.005):
            failed.append(f"mc95={_mc.get('mdd_95th', 1.0):.3f}")
        if not _fr.get("passed", False):
            failed.append("friction")
        if not _mo.get("passed", False):
            failed.append(f"monthly({_mo.get('reason', 'fail')})")
        why = ",".join(failed) if failed else "gate fail"
        print(f"[refit:spec] promote 拒絕：promotion 未通過（{why}）")
        return 1
    led._append({
        "event": "promote", "ts": int(time.time() * 1000),
        "kind": "spec", "spec": spec.to_dict(), "genome": {},
        "genome_id": spec.spec_id(), "period": "5m",
        "window_days": int(args.window_days), "end_time_ms": int(end_ms),
        "metrics": _metrics(res), "fitness": float(res.get("fitness") or 0.0),
        "promotion": res.get("promotion") or {},
        "run_id": f"refit_spec_{end_ms}", "note": args.note,
    })
    print(f"[refit:spec] promote 完成（{time.time()-t0:.0f}s）：spec {spec.spec_id()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="refit 循環：一輪重新擬合")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                    help="逗號分隔")
    ap.add_argument("--period", default="2h", help="K 線週期（2h 有整年）")
    ap.add_argument("--window-days", type=int, default=365)
    ap.add_argument("--eval-days", type=int, default=90,
                    help="現任 champion 的 reeval 視窗（短期衰退檢）")
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--pop-size", type=int, default=12)
    ap.add_argument("--elite", type=int, default=2)
    ap.add_argument("--mut-rate", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-triggers", type=int, default=8000)
    ap.add_argument("--ledger", type=Path, default=Path("/tmp/champion_ledger.jsonl"))
    ap.add_argument("--strategy-kind", default="genome", choices=("genome", "spec"),
                    help="genome 走 GA 路徑；spec 走 LGBM 分支（需 --bars-5m-json）")
    ap.add_argument("--bars-5m-json", type=str, default=None,
                    help="5m K 線 JSON（symbol -> records），spec 模式必填")
    ap.add_argument("--spec-json", type=str, default=None,
                    help="候選 spec JSON 檔；不傳則用 ledger 現任 spec 或預設 lgbm_gate")
    ap.add_argument("--seed-log-inline", type=str, default=None,
                    help="GA 初代 seed 用的 jsonl（通常是 champion 歷史）")
    ap.add_argument("--neighbors", type=int, default=10,
                    help="新 champion 的鄰域驗證鄰居數/每率")
    ap.add_argument("--note", type=str, default="")
    args = ap.parse_args()
    args.symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if args.strategy_kind == "spec":
        return _run_spec_refit(args)

    led = ChampionLedger(args.ledger)
    end_ms = int(time.time() * 1000)
    t0 = time.time()
    print(f"[refit] period={args.period} window={args.window_days}d "
          f"end={end_ms} symbols={len(args.symbols)}")

    # Step 1：現任 champion 在新視窗 reeval
    champ = led.current()
    if champ is not None:
        print(f"[refit] 現任 {champ.get('genome_id')} 在最近 {args.eval_days}d reeval…")
        _reeval_champion(led, champ, args.symbols, args.period,
                         args.eval_days, end_ms, args.max_triggers)
    else:
        print("[refit] ledger 無現任，跳過 reeval（首輪）")

    # Step 2：GA seed 池 = 現任 + 種子庫
    seed_rows: list = []
    if champ is not None:
        g = dict(champ["genome"])
        rng = random.Random(42)
        for i in range(args.pop_size):
            nb = dict(g)
            for k, v in g.items():
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    continue
                delta = v * rng.uniform(-0.1, 0.1)
                nb[k] = v + delta
            seed_rows.append({"genome": nb, "fitness": -999.0 + i * 0.001})
        print(f"[refit] seed 池：現任 ±10% 擾動 {len(seed_rows)} 隻")
    seed_path = Path(f"/tmp/refit_seed_{end_ms}.jsonl")
    with seed_path.open("w", encoding="utf-8") as fh:
        for r in seed_rows:
            fh.write(json.dumps(r) + "\n")

    # Step 3：GA 搜尋
    tmp = Path(f"/tmp/refit_ga_{end_ms}.jsonl")
    ga_args = argparse.Namespace(**{**vars(args),
                                    "seed_log_inline": str(seed_path) if seed_rows else None})
    evals = _run_ga(tmp, ga_args, end_ms)
    winner = _best(evals)
    print(f"[refit] GA 最佳：fit {winner['fitness']:+.3f}  "
          f"pnl {winner.get('pnl_365d', 0):+.2f}%  trades {winner.get('trades_365d')}  "
          f"gen {winner.get('generation')}")

    # Step 4：新 champion 鄰域驗證（快速版）
    print("[refit] 鄰域驗證…")
    nb_log = Path(f"/tmp/refit_neighbor_{end_ms}.log")
    genome_path = Path(f"/tmp/refit_winner_{end_ms}.json")
    genome_path.write_text(json.dumps(winner["genome"]), encoding="utf-8")
    proc = subprocess.run([
        sys.executable, str(_BACKEND / "backtest" / "neighborhood_robustness.py"),
        str(genome_path),
        "--neighbors", str(args.neighbors),
        "--rates", "0.05,0.10",
        "--symbols", ",".join(args.symbols),
        "--backtest-days", str(args.window_days),
        "--end-time-ms", str(end_ms),
        "--max-triggers", str(args.max_triggers),
        "--kline-period", args.period,
    ], capture_output=True, text=True)
    tail = (proc.stdout + proc.stderr)[-800:]
    print(tail)
    pin = "針峰" in (proc.stdout + proc.stderr)

    # Step 5：promote（針峰也 promote，但標注風險）
    note = (args.note + (" | 針峰-高風險" if pin else "")).strip(" |")
    led.promote(winner["genome"], _metrics(winner),
                float(winner["fitness"] or 0.0),
                args.period, args.window_days, end_ms,
                run_id=f"refit_{end_ms}",
                note=note)
    print(f"[refit] promote 完成（{time.time()-t0:.0f}s）：{'針峰-已標注' if pin else '鄰域可接受'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
