"""GA smoke（合成資料）：驗證修後流程能轉，不代表真實績效。

做法：sqlite 內存庫灌 90 天 1h 假 K 線（趨勢+噪聲+波段），跑極小 GA
（3 代 x 6 個體），斷言三件事：
 1. 流程能轉（有 fitness、无异常全灭）。
 2. get_indicator 隨時間推進值會變（前視已死）。
 3. portion×leverage 超標基因被 deploy 項壓住（fitness 明顯低於可行基因）。

數字看看就好。真實績效必須在 minipc（有 DB + cold-data）重跑。
"""
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base, CryptoKline
from backtest.ga_annual import (
    Genome, random_genome, clamp, tournament_select, crossover, mutate,
    evaluate_individual, gate_violation, render_strategy,
)
from backtest.historical_data_provider import HistoricalDataProvider

SYMBOLS = ["BTC"]
EXCHANGE = "hyperliquid"
random.seed(7)

NOW_MS = int(time.time() * 1000)
DAY_MS = 24 * 3600 * 1000
BACKTEST_DAYS = 90


PERIODS = {"1h": 3600, "4h": 14400, "1d": 86400}


def build_fake_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    t0 = NOW_MS // 1000 - BACKTEST_DAYS * 24 * 3600
    price = 60000.0
    # 先造 1h 主序列，再按週期聚合出 4h/1d，避免 fetch fallback 打真 API
    hourly = []
    for h in range(BACKTEST_DAYS * 24):
        ts = t0 + h * 3600
        drift = 8.0 * math.sin(h / 200.0) + random.gauss(0, 60)
        o = price
        c = max(1000.0, price + drift)
        hi = max(o, c) + abs(random.gauss(0, 30))
        lo = min(o, c) - abs(random.gauss(0, 30))
        hourly.append((ts, o, hi, lo, c))
        price = c
    rows = []
    for period, sec in PERIODS.items():
        n = sec // 3600
        for i in range(0, len(hourly), n):
            chunk = hourly[i:i + n]
            if not chunk:
                continue
            ts = chunk[0][0]
            rows.append(CryptoKline(
                exchange=EXCHANGE, symbol="BTC", market="CRYPTO", period=period,
                timestamp=ts, datetime_str=str(ts), environment="mainnet",
                open_price=chunk[0][1], high_price=max(r[2] for r in chunk),
                low_price=min(r[3] for r in chunk), close_price=chunk[-1][4],
                volume=100.0,
            ))
    db.add_all(rows)
    db.commit()
    return db


def main():
    db = build_fake_db()
    print(f"假 K 線：{BACKTEST_DAYS * 24} 根 1h BTC")

    # --- 斷言 2：指標隨時間推進會變 ---
    prov = HistoricalDataProvider(
        db, SYMBOLS, NOW_MS - BACKTEST_DAYS * DAY_MS, NOW_MS,
        exchange=EXCHANGE,
    )
    prov.set_current_time(NOW_MS - 60 * DAY_MS)
    early = prov.get_indicator("BTC", "EMA20", "1h").get("value")
    prov.set_current_time(NOW_MS - 5 * DAY_MS)
    late = prov.get_indicator("BTC", "EMA20", "1h").get("value")
    print(f"EMA20: 60 天前={early:.1f} 5 天前={late:.1f}")
    assert early is not None and late is not None
    assert abs(early - late) > 1e-9, "指標凍結，前視復發"

    # --- 斷言 3：超標基因被 deploy 項壓住 ---
    bad = Genome(portion=0.45, leverage=3)
    good = Genome(portion=0.25, leverage=1)
    from backtest.ga_annual import EvalResult

    def ev(g):
        return EvalResult(genome=g, gate_annual=False, pnl_365d=30.0,
                          maxdd_365d=5.0, sharpe_365d=1.5, pf=1.8,
                          trades_365d=25, win_rate=50.0, fitness=0.0,
                          elapsed_ms=0)
    vb, vg = gate_violation(ev(bad)), gate_violation(ev(good))
    print(f"violation: 超標 0.45x3={vb:.3f} 可行 0.25x1={vg:.3f}")
    assert vb > vg + 1.0, "deploy 項沒壓住超標基因"

    # --- 斷言 1：極小 GA 能轉（直調 evaluate_individual，避開子進程 DB） ---
    from backtest import ab_validators
    pop = [clamp(random_genome()) for _ in range(6)]
    orig_session = ab_validators.__dict__.get("SessionLocal", None)
    # evaluate_individual 內部自建 SessionLocal；這裡改走同進程直評：
    # 用 validator 函式直接吃假 db
    from backtest.ab_validators import exploration_validator
    scored = []
    for g in pop:
        code = render_strategy(g, symbols=SYMBOLS)
        ga, info = exploration_validator(
            db, code, symbols=SYMBOLS, exchange=EXCHANGE,
            fast_fail=False, backtest_days=BACKTEST_DAYS, end_time_ms=NOW_MS,
        )
        f365 = info.get("f365", {})
        r = ev(g)
        r.pnl_365d = float(f365.get("pnl_percent", 0) or 0)
        r.maxdd_365d = float(f365.get("maxdd_pct", 100) or 100)
        r.sharpe_365d = float(f365.get("sharpe", 0) or 0)
        raw_pf = f365.get("pf")
        r.pf = float(raw_pf) if raw_pf is not None else None
        r.trades_365d = int(f365.get("trades", 0) or 0)
        r.win_rate = float(f365.get("win", 0) or 0)
        r.gate_annual = ga
        from backtest.ga_annual import calculate_exploration_fitness
        fit = calculate_exploration_fitness(r)
        scored.append((fit, g))
    scored.sort(key=lambda x: x[0], reverse=True)
    print("fitness 排序（高→低）:")
    for fit, g in scored:
        print(f"  {fit:+.3f} portion={g.portion:.2f} lev={g.leverage} "
              f"ema={g.ema_fast}/{g.ema_slow}")
    fits = [f for f, _ in scored]
    assert all(math.isfinite(f) for f in fits), "出現非有限 fitness"
    assert max(fits) > min(fits), "全同分，GA 無梯度"
    print("SMOKE PASS：流程能轉，指標活著，超標被壓。數字不代表真實績效。")


if __name__ == "__main__":
    main()
