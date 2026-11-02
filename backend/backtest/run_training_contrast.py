#!/usr/bin/env python3
"""正式對照實驗：GA 冠軍當 baseline，training agent 跑 10 輪。

用法：
  cd backend
  export DATABASE_URL="postgresql://alpha_user:alpha_pass@/alpha_arena?host=/tmp"
  .venv/bin/python backtest/run_training_contrast.py \
    --baseline /tmp/best_r2_genome.json --rounds 10 --backtest-days 180 \
    --end-time-ms 1788000000000 --kline-period 1h \
    --symbols BTC,ETH,SOL,HYPE,DOGE,SUI --log-file /tmp/training_contrast.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--backtest-days", type=int, default=180)
    ap.add_argument("--end-time-ms", type=int, default=None)
    ap.add_argument("--kline-period", default="1h")
    ap.add_argument("--symbols", default="BTC,ETH,SOL,HYPE,DOGE,SUI")
    ap.add_argument("--log-file", default="/tmp/training_contrast.jsonl")
    args = ap.parse_args()

    from backtest.strategy_spec import genome_to_spec
    from backtest.training_agent import run_training_loop

    g = json.load(open(args.baseline))
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    base_spec = genome_to_spec(g).to_dict()

    t0 = time.time()
    out = run_training_loop(
        base_spec, syms, rounds=args.rounds,
        backtest_days=args.backtest_days, end_time_ms=args.end_time_ms,
        kline_period=args.kline_period,
    )
    out["elapsed_s"] = round(time.time() - t0, 1)
    out["baseline_genome_file"] = args.baseline
    out["symbols"] = syms
    out["kline_period"] = args.kline_period

    Path(args.log_file).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print("elapsed %.0fs baseline=%.3f best=%.3f" % (
        out["elapsed_s"], out["baseline_fitness"], out["best_metrics"]["fitness"]))
    for r in out["rounds"]:
        f = r["fitness"]
        print("round %d accepted=%s fitness=%s spec=%s %s" % (
            r["round"], r["accepted"],
            round(f, 3) if f is not None else None,
            r["spec_id"], (r.get("error") or "")[:60]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
