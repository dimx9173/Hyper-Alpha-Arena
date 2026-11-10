#!/usr/bin/env bash
# 順序跑 10 個獨立 GA run（不同 seed、不同視窗終點）。
# 為什麼要順序：先前用 `nohup ... & wait $!` 的迴圈，wait 沒真的阻塞，
# 10 個 run 同時起跑搶 10 核，每個都慢到幾乎停擺。單進程序列跑才有確定的吞吐。
set -u

cd "$(dirname "$0")/.." || exit 1
export DATABASE_URL="postgresql://alpha_user:alpha_pass@/alpha_arena?host=/tmp"

PY=".venv/bin/python"
SYMS="BTC,ETH,SOL,HYPE,DOGE,SUI,ARB,AVAX,BNB,NEAR,AAVE,TAO"
ITERS="${ITERS:-10}"
GENS="${GENS:-10}"
POP="${POP:-12}"
WORKERS="${WORKERS:-3}"

for i in $(seq 1 "$ITERS"); do
    end_ms=$("$PY" -c 'import time; print(int(time.time()*1000))')
    echo "=== ITER $i/$ITERS start $(date '+%H:%M:%S') end_ms=$end_ms ==="
    "$PY" backtest/iterate_annual_ga.py \
        --generations "$GENS" --pop-size "$POP" --elite 2 --mut-rate 0.3 \
        --workers "$WORKERS" --exchange hyperliquid --symbols "$SYMS" \
        --backtest-days 180 --end-time-ms "$end_ms" \
        --exploration-gens 5 --early-stop 4 \
        --log-file "/tmp/ga_r2_iter${i}.jsonl" \
        > "/tmp/ga_r2_iter${i}.log" 2>&1
    echo "=== ITER $i done rc=$? $(date '+%H:%M:%S') ==="
    grep -E "best fitness|Early stop" "/tmp/ga_r2_iter${i}.log" | tail -2
done

echo "=== ALL $ITERS ITERATIONS COMPLETE $(date '+%H:%M:%S') ==="
